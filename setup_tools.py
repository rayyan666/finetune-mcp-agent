"""
Setup tools — one-time configuration for the local→EC2 architecture.

Tools:
  configure_ec2   — save SSH connection details, test connectivity
  check_ec2       — verify EC2 is reachable and project is set up correctly
"""

from ..ssh_runner import write_config, check_connection, run_remote_sync


async def configure_ec2(
    host: str,
    user: str = "ubuntu",
    key_path: str = "",
    port: int = 22,
    remote_dir: str = "~/finetune-agent-mcp",
) -> dict:
    """
    Save EC2 SSH connection details and verify connectivity.

    Run this ONCE when setting up the local MCP server to point at your EC2.
    Config is saved to ~/.finetune-agent-mcp/config.json and reused on every
    subsequent tool call — you don't need to pass connection details every time.

    Args:
        host:        EC2 public IP or hostname (e.g. "54.123.45.67")
        user:        SSH user. Ubuntu AMIs use "ubuntu", Amazon Linux uses "ec2-user".
        key_path:    Path to your .pem key file (e.g. "~/.ssh/my-ec2-key.pem")
        port:        SSH port (default: 22)
        remote_dir:  Where finetune-agent-mcp lives on EC2
                     (default: "~/finetune-agent-mcp")

    Returns:
        dict with connection test result and saved config path.
    """
    # Save config first
    result = write_config(host=host, user=user, key_path=key_path,
                          port=port, remote_dir=remote_dir)

    # Test connection immediately
    conn = check_connection()
    if conn["status"] != "ok":
        return {
            "status": "error",
            "message": f"Config saved but SSH test failed: {conn['message']}",
            "tip": conn.get("tip", ""),
            "config_path": result["config_path"],
        }

    # Check that the project exists on EC2
    check = run_remote_sync(
        f"test -d {remote_dir} && echo EXISTS || echo MISSING", timeout=10
    )
    project_exists = "EXISTS" in check.stdout

    return {
        "status": "ok",
        "ssh_connection": "verified",
        "host": host,
        "config_path": result["config_path"],
        "remote_project_exists": project_exists,
        "next_step": (
            "You're connected. All tools will now run on EC2 automatically."
            if project_exists
            else (
                f"SSH works but project not found at {remote_dir} on EC2. "
                f"Run on EC2: git clone https://github.com/yourname/finetune-agent-mcp {remote_dir}"
            )
        ),
    }


async def check_ec2() -> dict:
    """
    Verify EC2 connectivity and check that all required scripts are in place.

    Call this any time you want to confirm the local→EC2 link is healthy,
    or to diagnose why a tool call failed.

    Returns:
        dict with connection status, GPU info, disk space, and script presence.
    """
    conn = check_connection()
    if conn["status"] != "ok":
        return conn

    remote_dir = conn.get("remote_dir", "~/finetune-agent-mcp")

    # Run a multi-check script on EC2 in one SSH round-trip
    check_script = f"""
cd {remote_dir} 2>/dev/null || {{ echo "PROJECT_MISSING"; exit 1; }}
echo "PROJECT_OK"

# GPU check
python3 -c "
import torch, json
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU:' + json.dumps({{'name': p.name, 'vram_gb': round(p.total_memory/1024**3,1)}}))
else:
    print('GPU:null')
" 2>/dev/null || echo "GPU:null"

# Disk space
df -h . | tail -1 | awk '{{print "DISK:" $4 " free of " $2}}'

# Scripts present
for f in scripts/dataset_generator_bedrock.py scripts/data_preparation.py scripts/finetune_qlora.py scripts/merge_and_push.py; do
    [ -f "$f" ] && echo "SCRIPT_OK:$f" || echo "SCRIPT_MISSING:$f"
done

# Data dirs
[ -d data/prepared ] && echo "DATA_DIR:ok" || echo "DATA_DIR:missing"
[ -f data/prepared/train.jsonl ] && echo "TRAIN_DATA:ok" || echo "TRAIN_DATA:missing"
"""

    result = run_remote_sync(check_script, timeout=30)
    lines = result.stdout.splitlines()

    gpu_info = None
    disk_info = None
    scripts = {}
    warnings = []

    for line in lines:
        if line.startswith("GPU:") and line[4:] != "null":
            try:
                import json
                gpu_info = json.loads(line[4:])
            except Exception:
                pass
        elif line.startswith("DISK:"):
            disk_info = line[5:]
        elif line.startswith("SCRIPT_OK:"):
            scripts[line[10:]] = True
        elif line.startswith("SCRIPT_MISSING:"):
            scripts[line[15:]] = False
            warnings.append(f"Missing script: {line[15:]} — copy it to EC2")
        elif line == "PROJECT_MISSING":
            return {
                "status": "error",
                "message": f"Project not found on EC2. Clone it there first.",
            }

    return {
        "status": "ok",
        "host": conn["host"],
        "gpu": gpu_info or "not detected",
        "disk": disk_info,
        "scripts": scripts,
        "warnings": warnings or None,
        "all_scripts_present": all(scripts.values()) if scripts else False,
    }
