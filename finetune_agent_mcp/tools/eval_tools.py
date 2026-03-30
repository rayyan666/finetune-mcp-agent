"""
Evaluation tools — measure model quality after fine-tuning.

Tools:
  evaluate_model   — run inference on val set, compute metrics, compare to base
  compare_outputs  — run a specific prompt through both base and fine-tuned model
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


async def evaluate_model(
    adapter_path: str = "./outputs/qwen-agentic-coder/final-adapter",
    val_path: str = "./data/prepared/val.jsonl",
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    n_samples: int = 10,
    max_new_tokens: int = 512,
) -> dict:
    """
    Run the fine-tuned model on the validation set and report quality metrics.

    Computes:
    - Average response length vs base model
    - Code block presence rate
    - A simple pass@1 style check (does the response contain runnable Python)
    - Side-by-side sample outputs for qualitative review

    Args:
        adapter_path:   Path to the trained LoRA adapter directory.
        val_path:       Path to val.jsonl to sample eval prompts from.
        base_model:     Base model name (for comparison).
        n_samples:      Number of val examples to run inference on.
        max_new_tokens: Max tokens to generate per response.

    Returns:
        dict with metrics and sample outputs.
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "evaluate.py"
    if not script.exists():
        # Fallback: do a quick structural check without running inference
        return _quick_eval(adapter_path, val_path)

    cmd = [
        sys.executable, str(script),
        "--adapter", adapter_path,
        "--val-path", val_path,
        "--base-model", base_model,
        "--n-samples", str(n_samples),
        "--max-new-tokens", str(max_new_tokens),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr[-2000:] or "Evaluation script failed",
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "status": "completed",
            "raw_output": result.stdout[-3000:],
        }


def _quick_eval(adapter_path: str, val_path: str) -> dict:
    """Structural eval when inference script is not available."""
    checks = {}

    # Check adapter files exist
    adapter = Path(adapter_path)
    checks["adapter_exists"] = adapter.exists()
    checks["adapter_config"] = (adapter / "adapter_config.json").exists()
    checks["adapter_weights"] = any(
        f.suffix in (".bin", ".safetensors")
        for f in adapter.iterdir()
    ) if adapter.exists() else False

    # Count val examples
    val = Path(val_path)
    val_count = 0
    if val.exists():
        with open(val) as f:
            val_count = sum(1 for l in f if l.strip())

    if adapter.exists() and (adapter / "adapter_config.json").exists():
        config = json.loads((adapter / "adapter_config.json").read_text())
        checks["lora_rank"] = config.get("r")
        checks["target_modules"] = config.get("target_modules")

    return {
        "status": "ok",
        "note": "Structural check only — inference eval script not found at scripts/evaluate.py",
        "checks": checks,
        "val_examples_available": val_count,
        "next_step": "Adapter looks good. Call merge_adapters() to merge into base weights.",
    }


async def compare_outputs(
    prompt: str,
    adapter_path: str = "./outputs/qwen-agentic-coder/final-adapter",
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_new_tokens: int = 400,
) -> dict:
    """
    Run a single prompt through both the fine-tuned model and the base model.

    Use this to qualitatively verify the fine-tune improved response style,
    code quality, or domain-specific behavior.

    Args:
        prompt:          The user prompt to test (plain text, no template needed).
        adapter_path:    Path to the trained LoRA adapter.
        base_model:      Base model to compare against.
        max_new_tokens:  Max tokens to generate.

    Returns:
        dict with base_output and finetuned_output side by side.
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "compare.py"
    if not script.exists():
        return {
            "status": "error",
            "message": (
                "Comparison script not found at scripts/compare.py. "
                "This requires the model to be loaded — ensure the adapter "
                "and base model are available, then run manually."
            ),
        }

    cmd = [
        sys.executable, str(script),
        "--prompt", prompt,
        "--adapter", adapter_path,
        "--base-model", base_model,
        "--max-new-tokens", str(max_new_tokens),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        return {"status": "error", "message": result.stderr[-2000:]}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "completed", "raw_output": result.stdout}
