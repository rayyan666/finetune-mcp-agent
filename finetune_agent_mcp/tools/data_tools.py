"""
Data tools — generate synthetic training data and prepare it for fine-tuning.

Tools:
  generate_dataset  — call AWS Bedrock (Claude Haiku) to create JSONL training data
  prepare_data      — validate, deduplicate, apply chat template, split 90/10
  inspect_examples  — sample N examples so Claude can verify quality before training
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..job_state import registry


# ── Tool: generate_dataset ────────────────────────────────────────────────────

async def generate_dataset(
    num_examples: int = 300,
    domains: list[str] = None,
    temperature: float = 0.5,
    output_path: str = "./data/raw/synthetic_bedrock_haiku.jsonl",
    resume: bool = True,
) -> dict:
    """
    Generate a synthetic training dataset using AWS Bedrock (Claude 3 Haiku).

    Calls dataset_generator_bedrock.py with the given parameters.
    This runs synchronously and streams progress — expect 15-30 min for 300 examples.

    Args:
        num_examples:  Target number of training examples to generate.
                       Recommended: 300 (5x per taxonomy instruction).
        domains:       List of domain categories to include. If None, uses all.
                       Options: agentic_react, agentic_langchain, agentic_llamaindex,
                       agentic_autogen, agentic_tool_calling, agentic_memory,
                       agentic_multiagent, ml_pipeline, ml_feature_engineering,
                       ml_hyperparameter, ml_advanced, dl_architectures, dl_training,
                       dl_custom, dl_debugging, genai_finetuning, genai_rag,
                       genai_vectordb, genai_evaluation, genai_redteaming, genai_inference
        temperature:   Generation temperature (0.25=deterministic, 0.7=diverse).
                       Run multiple passes at different temps for variety.
        output_path:   Where to write the JSONL file.
        resume:        If True, skip already-generated examples (safe to re-run).

    Returns:
        dict with status, examples_generated, output_path, cost_estimate_usd
    """
    domains = domains or []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Count existing examples if resuming
    existing = 0
    if resume and output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())

    script = Path(__file__).parent.parent.parent / "scripts" / "dataset_generator_bedrock.py"
    if not script.exists():
        return {
            "status": "error",
            "message": f"Generator script not found at {script}. "
                       "Copy dataset_generator_bedrock.py to scripts/",
        }

    cmd = [
        sys.executable, str(script),
        "--num-examples", str(num_examples),
        "--temperature", str(temperature),
        "--output", output_path,
    ]
    if domains:
        cmd += ["--domains"] + domains
    if resume:
        cmd.append("--resume")

    # Estimate cost: ~$0.00082 per call at Haiku pricing
    remaining = max(0, num_examples - existing)
    cost_est = remaining * 0.00082

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Dataset generation timed out after 60 min"}
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}

    # Count final output
    final_count = 0
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            final_count = sum(1 for line in f if line.strip())

    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr[-2000:] or "Generator script failed",
            "stdout_tail": result.stdout[-500:],
        }

    return {
        "status": "completed",
        "examples_generated": final_count,
        "previously_existing": existing,
        "new_examples": final_count - existing,
        "output_path": output_path,
        "cost_estimate_usd": round(cost_est, 4),
        "next_step": "Call prepare_data() to validate and split this data",
    }


# ── Tool: prepare_data ────────────────────────────────────────────────────────

async def prepare_data(
    input_path: str = "./data/raw/synthetic_bedrock_haiku.jsonl",
    output_dir: str = "./data/prepared",
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    val_ratio: float = 0.10,
    max_tokens: int = 4096,
    min_chars: int = 300,
    require_code_blocks: bool = True,
) -> dict:
    """
    Validate, clean, deduplicate, apply chat template, and split training data.

    Runs data_preparation.py which performs:
    - Structural validation (user+assistant turns present, non-empty)
    - Quality filters (min length, code block presence, no truncation stubs)
    - Deduplication (hash on first 120 chars of first user turn)
    - Chat template application (Qwen2.5 ChatML format)
    - Token length check (drop if > max_tokens)
    - 90/10 train/val split

    Args:
        input_path:          Path to raw JSONL from generate_dataset.
        output_dir:          Where to write train.jsonl and val.jsonl.
        model_name:          HuggingFace model name for tokenizer + chat template.
        val_ratio:           Fraction held out for validation (default 0.10).
        max_tokens:          Drop examples longer than this many tokens.
        min_chars:           Minimum assistant response character length.
        require_code_blocks: If True, drop examples with no triple-backtick code.

    Returns:
        dict with counts at each stage, train/val sizes, and stats file path.
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "data_preparation.py"
    if not script.exists():
        return {
            "status": "error",
            "message": f"Preparation script not found at {script}. "
                       "Copy data_preparation.py to scripts/",
        }

    cmd = [
        sys.executable, str(script),
        "--input", input_path,
        "--output-dir", output_dir,
        "--model", model_name,
        "--val-ratio", str(val_ratio),
        "--max-tokens", str(max_tokens),
        "--min-chars", str(min_chars),
    ]
    if not require_code_blocks:
        cmd.append("--no-require-code")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Data preparation timed out"}

    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr[-2000:] or "Preparation script failed",
        }

    # Read the stats JSON that data_preparation.py writes
    stats_path = Path(output_dir) / "stats.json"
    stats = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())

    train_count = val_count = 0
    train_file = Path(output_dir) / "train.jsonl"
    val_file   = Path(output_dir) / "val.jsonl"
    if train_file.exists():
        with open(train_file) as f:
            train_count = sum(1 for l in f if l.strip())
    if val_file.exists():
        with open(val_file) as f:
            val_count = sum(1 for l in f if l.strip())

    warning = ""
    if val_count < 20:
        warning = (
            f"WARNING: Only {val_count} validation examples — val loss signal will be noisy. "
            "Consider generating more data (target: ≥30 val examples)."
        )

    return {
        "status": "completed",
        "train_examples": train_count,
        "val_examples": val_count,
        "stats": stats,
        "warning": warning or None,
        "train_path": str(train_file),
        "val_path": str(val_file),
        "next_step": "Call inspect_examples() to verify quality, then select_model()",
    }


# ── Tool: inspect_examples ────────────────────────────────────────────────────

async def inspect_examples(
    data_path: str = "./data/prepared/train.jsonl",
    n: int = 3,
    category_filter: Optional[str] = None,
    show_full_response: bool = False,
) -> dict:
    """
    Sample N examples from a JSONL file so Claude can verify data quality.

    Use this before training to catch: bad code, truncated responses,
    wrong format, or low-quality synthetic outputs.

    Args:
        data_path:          Path to train.jsonl or val.jsonl.
        n:                  Number of examples to return (default 3).
        category_filter:    Only show examples from this category (optional).
                            e.g. "genai_finetuning" or "agentic_react"
        show_full_response: If False (default), truncates assistant turn to 500 chars.

    Returns:
        dict with sampled examples and summary statistics.
    """
    import random

    path = Path(data_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {data_path}"}

    records = []
    categories: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                records.append(r)
                cat = r.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            except json.JSONDecodeError:
                continue

    if category_filter:
        records = [r for r in records if r.get("category") == category_filter]

    sample = random.sample(records, min(n, len(records)))

    examples = []
    for rec in sample:
        msgs = rec.get("messages", [])
        user_turn = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_turn = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        code_blocks = asst_turn.count("```")

        if not show_full_response:
            asst_turn = asst_turn[:500] + ("..." if len(asst_turn) > 500 else "")

        examples.append({
            "category": rec.get("category", "unknown"),
            "turns": rec.get("turns", 1),
            "user_prompt": user_turn[:200],
            "assistant_response_preview": asst_turn,
            "code_block_count": code_blocks // 2,
            "char_length": len(next(
                (m["content"] for m in msgs if m["role"] == "assistant"), ""
            )),
        })

    return {
        "status": "ok",
        "total_examples": len(records),
        "category_distribution": categories,
        "sampled": examples,
    }
