"""
ssh_runner.py — execute commands on a remote EC2 over SSH.

This module replaces subprocess.Popen in training_tools.py when the MCP
server is running locally but training needs to happen on EC2.

All tools call run_remote() instead of subprocess.Popen — the interface
is identical from the tool's perspective. The SSH connection is reused
across calls via a persistent ControlMaster socket.

Config is loaded from ~/.finetune-agent-mcp/config.json or environment vars.
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CONFIG_PATH = Path.home() / ".finetune-agent-mcp" / "config.json"


@dataclass
class SSHConfig:
    host: str              # EC2 public IP or hostname
    user: str = "ubuntu"   # EC2 SSH user (ubuntu for Ubuntu AMIs)
    key_path: str = ""     # path to .pem file, e.g. ~/.ssh/my-key.pem
    port: int = 22
    remote_project_dir: str = "~/finetune-agent-mcp"  # where code lives on EC2
    control_path: str = str(Path.home() / ".ssh" / "cm-finetune-agent-%h")


def load_config() -> SSHConfig:
    """
    Load SSH config from file or environment variables.

    Environment variables take priority over config file:
      FINETUNE_EC2_HOST      — EC2 IP or hostname (required)
      FINETUNE_EC2_USER      — SSH user (default: ubuntu)
      FINETUNE_EC2_KEY_PATH  — path to .pem key file
      FINETUNE_EC2_PORT      — SSH port (default: 22)
      FINETUNE_EC2_DIR       — remote project directory
    """
    # Start with file config if it exists
    cfg = {}
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())

    # Env vars override file
    host = os.environ.get("FINETUNE_EC2_HOST", cfg.get("host", ""))
    if not host:
        raise ValueError(
            "EC2 host not configured. Set FINETUNE_EC2_HOST env var or "
            f"create {CONFIG_PATH} with a 'host' field."
        )

    return SSHConfig(
        host=host,
        user=os.environ.get("FINETUNE_EC2_USER", cfg.get("user", "ubuntu")),
        key_path=os.environ.get("FINETUNE_EC2_KEY_PATH", cfg.get("key_path", "")),
        port=int(os.environ.get("FINETUNE_EC2_PORT", cfg.get("port", 22))),
        remote_project_dir=os.environ.get(
            "FINETUNE_EC2_DIR",
            cfg.get("remote_project_dir", "~/finetune-agent-mcp")
        ),
    )


def _ssh_base_cmd(cfg: SSHConfig) -> list[str]:
    """Build the base ssh command with shared ControlMaster options."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={cfg.control_path}",
        "-o", "ControlPersist=120",   # keep connection alive 2 min after last use
        "-o", "ConnectTimeout=15",
        "-p", str(cfg.port),
    ]
    if cfg.key_path:
        cmd += ["-i", str(Path(cfg.key_path).expanduser())]
    cmd.append(f"{cfg.user}@{cfg.host}")
    return cmd


def check_connection() -> dict:
    """
    Test SSH connectivity to EC2. Call this from detect_hardware() before any real work.
    Returns dict with status and any error message.
    """
    try:
        cfg = load_config()
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    cmd = _ssh_base_cmd(cfg) + ["echo OK"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0 and "OK" in result.stdout:
            return {
                "status": "ok",
                "host": cfg.host,
                "user": cfg.user,
                "remote_dir": cfg.remote_project_dir,
            }
        return {
            "status": "error",
            "message": result.stderr.strip() or "SSH connection failed",
            "tip": "Check your FINETUNE_EC2_HOST, FINETUNE_EC2_KEY_PATH, and EC2 security group (port 22 open).",
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "SSH connection timed out after 15s"}
    except FileNotFoundError:
        return {"status": "error", "message": "ssh not found — install OpenSSH client"}


def run_remote_sync(remote_cmd: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    """
    Run a command on EC2 synchronously. Used for short operations (data prep, merge).
    Returns CompletedProcess with stdout, stderr, returncode.
    """
    cfg = load_config()
    ssh_cmd = _ssh_base_cmd(cfg) + [
        f"cd {cfg.remote_project_dir} && {remote_cmd}"
    ]
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def launch_remote_background(remote_cmd: str, log_path_remote: str) -> dict:
    """
    Launch a long-running command on EC2 via nohup so it survives SSH disconnect.
    Returns immediately with the remote PID.

    The command runs as:
        nohup bash -c '<remote_cmd>' >> <log_path_remote> 2>&1 &
        echo $!   # prints PID

    Claude polls the log file via tail_remote_log() to track progress.
    """
    cfg = load_config()

    # nohup + disown so it survives SSH session end
    launch = (
        f"cd {cfg.remote_project_dir} && "
        f"nohup bash -c '{remote_cmd}' >> {log_path_remote} 2>&1 & echo $!"
    )
    ssh_cmd = _ssh_base_cmd(cfg) + [launch]

    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr.strip() or "Failed to launch remote process",
        }

    try:
        remote_pid = int(result.stdout.strip())
    except ValueError:
        return {
            "status": "error",
            "message": f"Could not parse remote PID from: {result.stdout!r}",
        }

    return {
        "status": "running",
        "remote_pid": remote_pid,
        "log_path_remote": log_path_remote,
        "host": cfg.host,
    }


def check_remote_pid(remote_pid: int) -> bool:
    """Return True if a process with this PID is still running on EC2."""
    result = run_remote_sync(f"kill -0 {remote_pid} 2>/dev/null && echo ALIVE || echo DEAD", timeout=10)
    return "ALIVE" in result.stdout


def tail_remote_log(log_path_remote: str, n_lines: int = 40) -> list[str]:
    """Return the last N lines of a log file on EC2."""
    result = run_remote_sync(f"tail -n {n_lines} {log_path_remote}", timeout=15)
    if result.returncode != 0:
        return [f"[error reading log: {result.stderr.strip()}]"]
    return result.stdout.splitlines()


def read_remote_file(remote_path: str) -> Optional[str]:
    """Read a file from EC2 and return its contents as a string."""
    result = run_remote_sync(f"cat {remote_path}", timeout=15)
    if result.returncode != 0:
        return None
    return result.stdout


def write_config(host: str, user: str = "ubuntu",
                 key_path: str = "", port: int = 22,
                 remote_dir: str = "~/finetune-agent-mcp"):
    """
    Write SSH config to ~/.finetune-agent-mcp/config.json.
    Call this once during setup instead of setting env vars every session.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "host": host,
        "user": user,
        "key_path": key_path,
        "port": port,
        "remote_project_dir": remote_dir,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    return {"status": "ok", "config_path": str(CONFIG_PATH), "config": config}
