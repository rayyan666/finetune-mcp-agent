"""
Training tools — hardware detection, model selection, and async QLoRA fine-tuning.

The core design: run_finetune() launches training as a background subprocess
and returns a job_id immediately. Claude then polls get_training_status(job_id)
every ~60 seconds rather than blocking for 15-60 minutes.

Tools:
  detect_hardware      — probe GPU VRAM, CUDA version, bf16 support
  select_model         — recommend model given hardware constraints
  run_finetune         — launch QLoRA training (non-blocking, returns job_id)
  get_training_status  — poll a running job for current status + metrics
  tail_training_logs   — return last N lines of the training log
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


# ── Tool: detect_hardware ─────────────────────────────────────────────────────

async def detect_hardware() -> dict:
    """
    Probe available GPU hardware and return VRAM, CUDA version, and capabilities.

    Always call this first — never assume the user's hardware. The result
    determines which model size is safe to fine-tune.

    Returns:
        dict with gpu_name, vram_gb, cuda_version, bf16_supported,
        recommended_model, and notes.
    """
    probe = """
import json, sys
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"gpu_available": False}))
        sys.exit(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    bf16 = torch.cuda.is_bf16_supported()
    print(json.dumps({
        "gpu_available": True,
        "gpu_name": props.name,
        "vram_gb": round(vram_gb, 1),
        "cuda_version": torch.version.cuda,
        "bf16_supported": bf16,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_gpu": torch.cuda.device_count() > 1,
        "gpu_count": torch.cuda.device_count(),
    }))
except ImportError:
    print(json.dumps({"gpu_available": False, "error": "torch not installed"}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "message": "Failed to probe hardware: " + result.stderr[:500],
        }

    try:
        hw = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"status": "error", "message": "Could not parse hardware probe output"}

    if not hw.get("gpu_available"):
        return {
            "status": "ok",
            "gpu_available": False,
            "recommendation": "No GPU detected. Fine-tuning requires a CUDA GPU with ≥8GB VRAM.",
        }

    vram = hw["vram_gb"]
    bf16 = hw["bf16_supported"]
    gpu  = hw["gpu_name"]

    # Model recommendation logic
    if vram >= 40:
        rec_model = "Qwen/Qwen2.5-Coder-14B-Instruct"
        rec_note  = f"{vram:.0f}GB VRAM — 14B is viable, best quality"
    elif vram >= 14:
        rec_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        rec_note  = f"{vram:.0f}GB VRAM — 7B is the sweet spot (recommended)"
    elif vram >= 7:
        rec_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
        rec_note  = f"{vram:.0f}GB VRAM — use 3B, 7B will OOM"
    else:
        rec_model = None
        rec_note  = f"{vram:.0f}GB VRAM — too low for QLoRA fine-tuning"

    # T4-specific notes (your EC2 g4dn.xlarge)
    notes = []
    if "T4" in gpu:
        notes.append("T4 detected: use fp16 not bf16 (no bf16 support on Turing arch)")
        notes.append("flash_attention_2 NOT available (requires Ampere+) — use eager attention")
        notes.append("Recommend: per_device_batch_size=1, gradient_accumulation=16")
    if not bf16:
        notes.append("bf16 not supported — training will use fp16 automatically")

    return {
        "status": "ok",
        **hw,
        "recommended_model": rec_model,
        "recommendation_reason": rec_note,
        "notes": notes,
    }


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
    Confirm a model + training config and estimate VRAM usage and training time.

    Call this after detect_hardware() to lock in your hyperparameters before
    launching training. Returns a config summary Claude can show the user.

    Args:
        model_name:            HuggingFace model ID.
        lora_r:                LoRA rank (8, 16, or 32). Higher = more params, more VRAM.
        lora_alpha:            LoRA alpha. Standard: 2x lora_r.
        batch_size:            Per-device batch size. Use 1 on T4 16GB.
        gradient_accumulation: Steps to accumulate before optimizer step.
                               Effective batch = batch_size × gradient_accumulation.
        num_epochs:            Training epochs. 3 is right for <500 examples.
        learning_rate:         Starting LR. 2e-4 with cosine decay is standard.

    Returns:
        dict with config summary, VRAM estimate, and effective batch size.
    """
    # Rough VRAM estimates for NF4 QLoRA
    vram_estimates = {
        "3B":  {"base_gb": 2.0, "lora_gb": 0.3, "activations_gb": 1.5},
        "7B":  {"base_gb": 4.5, "lora_gb": 0.6, "activations_gb": 3.0},
        "14B": {"base_gb": 8.5, "lora_gb": 1.2, "activations_gb": 5.0},
    }

    size_key = "7B"
    if "3B" in model_name:
        size_key = "3B"
    elif "14B" in model_name:
        size_key = "14B"

    est = vram_estimates[size_key]
    # VRAM scales with batch size for activations
    total_vram_est = (
        est["base_gb"]
        + est["lora_gb"]
        + est["activations_gb"] * batch_size
    )

    effective_batch = batch_size * gradient_accumulation
    trainable_params_approx = {
        "3B": "~15M", "7B": "~40M", "14B": "~80M"
    }[size_key]

    # Target modules for Qwen2.5
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    return {
        "status": "ok",
        "model": model_name,
        "config": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "target_modules": target_modules,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": effective_batch,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lr_scheduler": "cosine",
            "optimizer": "paged_adamw_32bit",
            "quantization": "4-bit NF4",
            "max_seq_length": 4096,
        },
        "estimates": {
            "vram_gb_approx": round(total_vram_est, 1),
            "trainable_params": trainable_params_approx,
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

    Training runs in a subprocess — do NOT block waiting for it to finish.
    After calling this, poll get_training_status(job_id) every 60 seconds.

    The training script is finetune_qlora.py with all parameters passed as
    CLI args. Logs stream to ./outputs/{run_name}.log.

    Args:
        model_name:           HuggingFace model to fine-tune.
        train_path:           Path to prepared train.jsonl.
        val_path:             Path to prepared val.jsonl.
        output_dir:           Where to save checkpoints and final adapter.
        lora_r:               LoRA rank.
        lora_alpha:           LoRA alpha (standard: 2× lora_r).
        batch_size:           Per-device batch size (1 recommended for T4).
        gradient_accumulation: Gradient accumulation steps.
        num_epochs:           Training epochs (3 is right for <500 examples).
        learning_rate:        Peak learning rate.
        use_wandb:            Enable Weights & Biases logging.
        wandb_project:        W&B project name.
        run_name:             Name for this training run.

    Returns:
        dict with job_id, status, log_path. Poll get_training_status(job_id).
    """
    # Validate inputs before launching
    for p, label in [(train_path, "train_path"), (val_path, "val_path")]:
        if not Path(p).exists():
            return {
                "status": "error",
                "message": f"{label} not found: {p}. Run prepare_data() first.",
            }

    script = Path(__file__).parent.parent.parent / "scripts" / "finetune_qlora.py"
    if not script.exists():
        return {
            "status": "error",
            "message": f"Training script not found at {script}. "
                       "Copy finetune_qlora.py to scripts/",
        }

    job_id   = f"ft-{run_name}-{str(uuid.uuid4())[:8]}"
    log_dir  = Path("./outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"{job_id}.log")

    cmd = [
        sys.executable, str(script),
        "--model-name", model_name,
        "--train-path", train_path,
        "--val-path", val_path,
        "--output-dir", output_dir,
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--batch-size", str(batch_size),
        "--gradient-accumulation", str(gradient_accumulation),
        "--num-epochs", str(num_epochs),
        "--learning-rate", str(learning_rate),
        "--run-name", run_name,
    ]
    if not use_wandb:
        cmd.append("--no-wandb")
    else:
        cmd += ["--wandb-project", wandb_project]

    # Register job before launching subprocess
    registry.create(
        job_id=job_id,
        script=str(script),
        args=cmd[2:],
        log_path=log_path,
        output_dir=output_dir,
    )

    # Launch as background process, redirect stdout+stderr to log file
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    registry.start(job_id, proc)

    return {
        "status": "running",
        "job_id": job_id,
        "pid": proc.pid,
        "log_path": log_path,
        "output_dir": output_dir,
        "message": (
            f"Training launched (PID {proc.pid}). "
            "Call get_training_status(job_id) every 60s to monitor progress. "
            "Expected duration: 10-60 min depending on dataset size and GPU."
        ),
    }


# ── Tool: get_training_status ─────────────────────────────────────────────────

async def get_training_status(job_id: str) -> dict:
    """
    Check the current status of a training job.

    Poll this every 60 seconds after calling run_finetune().
    Parses the training log to extract the latest loss metrics.

    Args:
        job_id: The job ID returned by run_finetune().

    Returns:
        dict with status (queued|running|completed|failed), elapsed_min,
        latest_train_loss, latest_eval_loss, and estimated completion.
    """
    job = registry.refresh(job_id)
    if job is None:
        return {
            "status": "error",
            "message": f"Job '{job_id}' not found. "
                       "Check job ID or list jobs with get_training_status('list')",
        }

    # Handle the special 'list' job_id to show all jobs
    if job_id == "list":
        return {
            "status": "ok",
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "started_at": j.started_at,
                    "output_dir": j.output_dir,
                }
                for j in registry.list_all()
            ],
        }

    elapsed_min = None
    if job.started_at:
        elapsed_sec = (job.finished_at or time.time()) - job.started_at
        elapsed_min = round(elapsed_sec / 60, 1)

    # Parse log for latest metrics
    metrics = _parse_log_metrics(job.log_path)

    response = {
        "status": job.status,
        "job_id": job_id,
        "elapsed_min": elapsed_min,
        "output_dir": job.output_dir,
        **metrics,
    }

    if job.status == "running":
        response["message"] = "Training in progress. Poll again in 60 seconds."
    elif job.status == "completed":
        adapter_path = str(Path(job.output_dir) / "final-adapter")
        response["message"] = "Training complete!"
        response["adapter_path"] = adapter_path
        response["next_step"] = (
            f"Call evaluate_model(adapter_path='{adapter_path}') "
            "then merge_adapters() when satisfied."
        )
    elif job.status == "failed":
        response["error"] = job.error_msg
        response["message"] = "Training failed. Call tail_training_logs() to diagnose."

    return response


def _parse_log_metrics(log_path: Optional[str]) -> dict:
    """Extract the latest train/eval loss from a HuggingFace Trainer log."""
    if not log_path or not Path(log_path).exists():
        return {}

    train_loss = eval_loss = step = None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # HF Trainer logs JSON dicts like: {'loss': 1.23, 'step': 50}
                if "'loss'" in line or '"loss"' in line:
                    try:
                        # Extract the JSON-like dict from the log line
                        start = line.find("{")
                        end   = line.rfind("}") + 1
                        if start >= 0 and end > start:
                            d = json.loads(line[start:end].replace("'", '"'))
                            if "loss" in d:
                                train_loss = round(d["loss"], 4)
                            if "eval_loss" in d:
                                eval_loss = round(d["eval_loss"], 4)
                            if "step" in d:
                                step = d["step"]
                    except (json.JSONDecodeError, ValueError):
                        pass
    except OSError:
        pass

    return {
        k: v for k, v in {
            "current_step": step,
            "latest_train_loss": train_loss,
            "latest_eval_loss": eval_loss,
        }.items() if v is not None
    }


# ── Tool: tail_training_logs ──────────────────────────────────────────────────

async def tail_training_logs(
    job_id: str,
    n_lines: int = 40,
) -> dict:
    """
    Return the last N lines of the training log for a job.

    Use this to diagnose errors or see what the trainer is currently doing.

    Args:
        job_id:   The job ID returned by run_finetune().
        n_lines:  Number of lines to return from the end of the log (default 40).

    Returns:
        dict with log_lines list and log_path.
    """
    job = registry.get(job_id)
    if job is None:
        return {"status": "error", "message": f"Job '{job_id}' not found"}

    if not job.log_path or not Path(job.log_path).exists():
        return {"status": "error", "message": "Log file not yet created"}

    with open(job.log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    tail = [l.rstrip() for l in lines[-n_lines:]]

    return {
        "status": "ok",
        "job_id": job_id,
        "job_status": job.status,
        "log_path": job.log_path,
        "total_lines": len(lines),
        "log_tail": tail,
    }
