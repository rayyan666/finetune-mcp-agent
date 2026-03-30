"""
Training tools — hardware detection, model selection, and async QLoRA fine-tuning.

SSH-aware: if EC2 is configured via configure_ec2() or FINETUNE_EC2_HOST env var,
all heavy operations (hardware probe, training launch) run on EC2 automatically.
The MCP server itself stays local — only the subprocess hops to EC2 via SSH.

Tools:
  detect_hardware      — probe GPU on EC2 (or local fallback)
  select_model         — confirm model + hyperparameters, estimate VRAM
  run_finetune         — launch QLoRA training on EC2 (non-blocking, returns job_id)
  get_training_status  — poll a running job for current status + metrics
  tail_training_logs   — return last N lines of the training log from EC2
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from ..job_state import registry


def _try_ssh():
    """Return (ssh_module, conn) if EC2 is configured, else (None, None)."""
    try:
        from ..ssh_runner import check_connection, load_config
        conn = check_connection()
        if conn["status"] == "ok":
            import finetune_agent_mcp.ssh_runner as ssh_mod
            return ssh_mod, conn
    except Exception:
        pass
    return None, None


def _annotate_hardware(hw: dict) -> dict:
    """Add model recommendation and T4-specific notes to a hardware dict."""
    vram = hw.get("vram_gb", 0)
    gpu  = hw.get("gpu_name", "")
    bf16 = hw.get("bf16_supported", False)

    if vram >= 40:
        rec_model = "Qwen/Qwen2.5-Coder-14B-Instruct"
        rec_note  = f"{vram:.0f}GB VRAM — 14B viable, best quality"
    elif vram >= 14:
        rec_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        rec_note  = f"{vram:.0f}GB VRAM — 7B is the sweet spot (recommended)"
    elif vram >= 7:
        rec_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
        rec_note  = f"{vram:.0f}GB VRAM — use 3B, 7B will OOM"
    else:
        rec_model = None
        rec_note  = f"{vram:.0f}GB VRAM — too low for QLoRA fine-tuning"

    notes = []
    if "T4" in gpu:
        notes.append("T4: use fp16 not bf16 (no bf16 on Turing arch)")
        notes.append("flash_attention_2 NOT available — use eager attention")
        notes.append("Recommended: per_device_batch_size=1, gradient_accumulation=16")
    if not bf16:
        notes.append("bf16 not supported — training will use fp16 automatically")

    return {
        "status": "ok",
        **hw,
        "recommended_model": rec_model,
        "recommendation_reason": rec_note,
        "notes": notes,
    }


# ── Tool: detect_hardware ─────────────────────────────────────────────────────

async def detect_hardware() -> dict:
    """
    Probe GPU hardware. If EC2 is configured, queries the remote GPU.
    Otherwise probes local hardware (useful for testing).

    Always call this first — never assume VRAM availability.

    Returns:
        dict with gpu_name, vram_gb, bf16_supported, recommended_model, notes,
        and source (remote EC2 host or 'local').
    """
    ssh, conn = _try_ssh()

    if ssh:
        # Query EC2 GPU over SSH
        probe_cmd = (
            "python3 -c \""
            "import torch,json;"
            "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None;"
            "print(json.dumps({'gpu_available':bool(p),"
            "'gpu_name':p.name if p else None,"
            "'vram_gb':round(p.total_memory/1024**3,1) if p else 0,"
            "'cuda_version':torch.version.cuda,"
            "'bf16_supported':torch.cuda.is_bf16_supported() if p else False,"
            "'compute_capability':f'{p.major}.{p.minor}' if p else None,"
            "'gpu_count':torch.cuda.device_count()}))"
            "\""
        )
        result = ssh.run_remote_sync(probe_cmd, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            try:
                hw = json.loads(result.stdout.strip())
                hw["source"] = f"remote EC2 ({conn['host']})"
                return _annotate_hardware(hw)
            except json.JSONDecodeError:
                pass
        return {
            "status": "error",
            "message": "Could not probe EC2 GPU: " + result.stderr[:300],
            "tip": "Run check_ec2() to verify the connection and that torch is installed on EC2.",
        }

    # Local fallback
    probe = """
import json, sys
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"gpu_available": False}))
        sys.exit(0)
    p = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "gpu_available": True,
        "gpu_name": p.name,
        "vram_gb": round(p.total_memory / 1024**3, 1),
        "cuda_version": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "compute_capability": f"{p.major}.{p.minor}",
        "gpu_count": torch.cuda.device_count(),
    }))
except ImportError:
    print(json.dumps({"gpu_available": False, "error": "torch not installed"}))
"""
    result = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"status": "error", "message": result.stderr[:500]}
    try:
        hw = json.loads(result.stdout.strip())
        hw["source"] = "local"
        return _annotate_hardware(hw)
    except json.JSONDecodeError:
        return {"status": "error", "message": "Could not parse hardware probe output"}


# ── Tool: select_model ────────────────────────────────────────────────────────

async def select_model(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    gradient_accumulation: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
) -> dict:
    """
    Confirm a model + training config and estimate VRAM usage.

    Call after detect_hardware() to lock in hyperparameters before training.

    Args:
        model_name:            HuggingFace model ID.
        lora_r:                LoRA rank (8, 16, or 32).
        lora_alpha:            LoRA alpha. Standard: 2× lora_r.
        batch_size:            Per-device batch size. Use 1 on T4 16GB.
        gradient_accumulation: Effective batch = batch_size × gradient_accumulation.
        num_epochs:            Training epochs. 3 is right for <500 examples.
        learning_rate:         Starting LR. 2e-4 with cosine decay is standard.

    Returns:
        dict with config summary, VRAM estimate, and effective batch size.
    """
    vram_estimates = {
        "3B":  {"base_gb": 2.0, "lora_gb": 0.3, "activations_gb": 1.5},
        "7B":  {"base_gb": 4.5, "lora_gb": 0.6, "activations_gb": 3.0},
        "14B": {"base_gb": 8.5, "lora_gb": 1.2, "activations_gb": 5.0},
    }
    size_key = "7B"
    if "3B" in model_name: size_key = "3B"
    elif "14B" in model_name: size_key = "14B"

    est = vram_estimates[size_key]
    total_vram = est["base_gb"] + est["lora_gb"] + est["activations_gb"] * batch_size

    return {
        "status": "ok",
        "model": model_name,
        "config": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj"],
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": batch_size * gradient_accumulation,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lr_scheduler": "cosine",
            "optimizer": "paged_adamw_32bit",
            "quantization": "4-bit NF4",
            "max_seq_length": 4096,
        },
        "estimates": {
            "vram_gb_approx": round(total_vram, 1),
            "trainable_params": {"3B":"~15M","7B":"~40M","14B":"~80M"}[size_key],
            "compute_dtype": "fp16 (T4) or bf16 (A10G+)",
        },
        "next_step": "Call run_finetune() with these parameters to start training",
    }


# ── Tool: run_finetune ────────────────────────────────────────────────────────

async def run_finetune(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    train_path: str = "./data/prepared/train.jsonl",
    val_path: str = "./data/prepared/val.jsonl",
    output_dir: str = "./outputs/qwen-agentic-coder",
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    gradient_accumulation: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    use_wandb: bool = False,
    wandb_project: str = "finetune-agent-mcp",
    run_name: str = "qlora-run",
) -> dict:
    """
    Launch QLoRA fine-tuning as a background job. Returns a job_id immediately.

    If EC2 is configured, training launches on EC2 via SSH (nohup). Otherwise
    launches locally. Poll get_training_status(job_id) every 60 seconds.

    Args:
        model_name:           HuggingFace model to fine-tune.
        train_path:           Path to prepared train.jsonl (on EC2 if remote).
        val_path:             Path to prepared val.jsonl (on EC2 if remote).
        output_dir:           Where to save checkpoints (on EC2 if remote).
        lora_r:               LoRA rank.
        lora_alpha:           LoRA alpha (standard: 2× lora_r).
        batch_size:           Per-device batch size (1 recommended for T4).
        gradient_accumulation: Gradient accumulation steps.
        num_epochs:           Training epochs.
        learning_rate:        Peak learning rate.
        use_wandb:            Enable Weights & Biases logging.
        wandb_project:        W&B project name.
        run_name:             Name for this training run.

    Returns:
        dict with job_id, status, log_path. Poll get_training_status(job_id).
    """
    job_id   = f"ft-{run_name}-{str(uuid.uuid4())[:8]}"
    log_name = f"{job_id}.log"

    ssh, conn = _try_ssh()

    if ssh:
        # ── Remote EC2 path ───────────────────────────────────────────────────
        cfg = ssh.load_config()
        remote_dir = cfg.remote_project_dir
        log_path_remote = f"{remote_dir}/outputs/logs/{log_name}"

        # Build the python command to run on EC2
        no_wandb_flag = "" if use_wandb else "--no-wandb"
        remote_cmd = (
            f"mkdir -p {remote_dir}/outputs/logs && "
            f"cd {remote_dir} && "
            f"python3 scripts/finetune_qlora.py "
            f"--model-name '{model_name}' "
            f"--train-path '{train_path}' "
            f"--val-path '{val_path}' "
            f"--output-dir '{output_dir}' "
            f"--lora-r {lora_r} --lora-alpha {lora_alpha} "
            f"--batch-size {batch_size} "
            f"--gradient-accumulation {gradient_accumulation} "
            f"--num-epochs {num_epochs} "
            f"--learning-rate {learning_rate} "
            f"--run-name '{run_name}' "
            f"{no_wandb_flag}"
        )

        launch = ssh.launch_remote_background(remote_cmd, log_path_remote)
        if launch["status"] == "error":
            return launch

        # Store in registry so get_training_status can find it
        job = registry.create(
            job_id=job_id,
            script="remote:finetune_qlora.py",
            args=[],
            log_path=log_path_remote,   # remote path — read via SSH
            output_dir=output_dir,
        )
        # Tag it as remote so status polling knows to SSH
        job.metrics["remote_pid"] = launch["remote_pid"]
        job.metrics["remote_host"] = conn["host"]
        job.metrics["log_path_remote"] = log_path_remote
        job.status = "running"
        job.started_at = time.time()
        registry._save()

        return {
            "status": "running",
            "job_id": job_id,
            "remote_pid": launch["remote_pid"],
            "host": conn["host"],
            "log_path_remote": log_path_remote,
            "message": (
                f"Training launched on EC2 {conn['host']} (PID {launch['remote_pid']}). "
                "Call get_training_status(job_id) every 60s to monitor. "
                "Expected: 10-60 min on T4."
            ),
        }

    # ── Local fallback path ───────────────────────────────────────────────────
    for p, label in [(train_path, "train_path"), (val_path, "val_path")]:
        if not Path(p).exists():
            return {"status": "error",
                    "message": f"{label} not found: {p}. Run prepare_data() first."}

    script = Path(__file__).parent.parent.parent / "scripts" / "finetune_qlora.py"
    if not script.exists():
        return {"status": "error",
                "message": f"Training script not found at {script}. Copy finetune_qlora.py to scripts/"}

    log_dir = Path("./outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / log_name)

    cmd = [
        sys.executable, str(script),
        "--model-name", model_name,
        "--train-path", train_path, "--val-path", val_path,
        "--output-dir", output_dir,
        "--lora-r", str(lora_r), "--lora-alpha", str(lora_alpha),
        "--batch-size", str(batch_size),
        "--gradient-accumulation", str(gradient_accumulation),
        "--num-epochs", str(num_epochs),
        "--learning-rate", str(learning_rate),
        "--run-name", run_name,
    ]
    if not use_wandb:
        cmd.append("--no-wandb")

    registry.create(job_id=job_id, script=str(script), args=cmd[2:],
                    log_path=log_path, output_dir=output_dir)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    registry.start(job_id, proc)

    return {
        "status": "running",
        "job_id": job_id,
        "pid": proc.pid,
        "log_path": log_path,
        "source": "local",
        "message": f"Training launched locally (PID {proc.pid}). Poll get_training_status every 60s.",
    }


# ── Tool: get_training_status ─────────────────────────────────────────────────

async def get_training_status(job_id: str) -> dict:
    """
    Check the current status of a training job (local or remote EC2).

    Poll this every 60 seconds after calling run_finetune().

    Args:
        job_id: The job ID returned by run_finetune(). Pass 'list' to see all jobs.

    Returns:
        dict with status, elapsed_min, latest_train_loss, latest_eval_loss.
    """
    if job_id == "list":
        return {
            "status": "ok",
            "jobs": [{"job_id": j.job_id, "status": j.status,
                      "output_dir": j.output_dir} for j in registry.list_all()],
        }

    job = registry.refresh(job_id)
    if job is None:
        return {"status": "error", "message": f"Job '{job_id}' not found."}

    elapsed_min = None
    if job.started_at:
        elapsed_min = round(((job.finished_at or time.time()) - job.started_at) / 60, 1)

    # For remote jobs, check PID liveness and read log via SSH
    remote_pid = job.metrics.get("remote_pid")
    if remote_pid and job.status == "running":
        ssh, _ = _try_ssh()
        if ssh:
            alive = ssh.check_remote_pid(remote_pid)
            if not alive:
                job.status = "completed"
                job.finished_at = time.time()
                registry._save()

    # Parse metrics from log
    log_path = job.metrics.get("log_path_remote") or job.log_path
    metrics = {}
    if remote_pid:
        ssh, _ = _try_ssh()
        if ssh:
            lines = ssh.tail_remote_log(log_path, n_lines=80)
            metrics = _parse_log_metrics_from_lines(lines)
    else:
        metrics = _parse_log_metrics(job.log_path)

    response = {"status": job.status, "job_id": job_id,
                "elapsed_min": elapsed_min, "output_dir": job.output_dir, **metrics}

    if job.status == "running":
        response["message"] = "Training in progress. Poll again in 60 seconds."
    elif job.status == "completed":
        response["message"] = "Training complete!"
        response["next_step"] = "Call evaluate_model() then merge_adapters()."
    elif job.status == "failed":
        response["error"] = job.error_msg
        response["message"] = "Training failed. Call tail_training_logs() to diagnose."

    return response


def _parse_log_metrics(log_path: Optional[str]) -> dict:
    if not log_path or not Path(log_path).exists():
        return {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return _parse_log_metrics_from_lines(f.readlines())


def _parse_log_metrics_from_lines(lines) -> dict:
    train_loss = eval_loss = step = None
    for line in lines:
        if "'loss'" in line or '"loss"' in line:
            try:
                start = line.find("{")
                end   = line.rfind("}") + 1
                if start >= 0 and end > start:
                    d = json.loads(line[start:end].replace("'", '"'))
                    if "loss" in d:      train_loss = round(d["loss"], 4)
                    if "eval_loss" in d: eval_loss  = round(d["eval_loss"], 4)
                    if "step" in d:      step       = d["step"]
            except (json.JSONDecodeError, ValueError):
                pass
    return {k: v for k, v in
            {"current_step": step, "latest_train_loss": train_loss,
             "latest_eval_loss": eval_loss}.items() if v is not None}


# ── Tool: tail_training_logs ──────────────────────────────────────────────────

async def tail_training_logs(job_id: str, n_lines: int = 40) -> dict:
    """
    Return the last N lines of the training log (from EC2 or local).

    Args:
        job_id:   Job ID returned by run_finetune().
        n_lines:  Lines to return from end of log (default 40).

    Returns:
        dict with log_lines list.
    """
    job = registry.get(job_id)
    if job is None:
        return {"status": "error", "message": f"Job '{job_id}' not found"}

    remote_log = job.metrics.get("log_path_remote")
    if remote_log:
        ssh, _ = _try_ssh()
        if ssh:
            lines = ssh.tail_remote_log(remote_log, n_lines)
            return {"status": "ok", "job_id": job_id,
                    "job_status": job.status, "log_tail": lines}
        return {"status": "error", "message": "EC2 not reachable — run check_ec2()"}

    if not job.log_path or not Path(job.log_path).exists():
        return {"status": "error", "message": "Log file not yet created"}

    with open(job.log_path, encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    return {"status": "ok", "job_id": job_id, "job_status": job.status,
            "log_tail": [l.rstrip() for l in all_lines[-n_lines:]]}
