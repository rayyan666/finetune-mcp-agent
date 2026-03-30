# ── Data Preparation for Qwen2.5-Coder Fine-tuning ───────────────────────────
# Runs after dataset_generator_bedrock.py
#
# Input:  ./data/raw/synthetic_bedrock_haiku.jsonl
# Output: ./data/prepared/train.jsonl
#         ./data/prepared/val.jsonl
#         ./data/prepared/stats.json   ← summary of what was kept/dropped
#
# Setup:
#   pip install transformers
#
# Run:
#   python data_preparation.py

import json
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH    = "./data/raw/synthetic_bedrock_haiku.jsonl"
OUTPUT_DIR    = Path("./data/prepared")
MODEL_NAME    = "Qwen/Qwen2.5-Coder-7B-Instruct"   # for chat template
VAL_RATIO     = 0.10      # 10% validation
MAX_TOKENS    = 4096      # drop examples longer than this
MIN_CHARS     = 300       # minimum assistant response length
RANDOM_SEED   = 42

# ── QUALITY FILTER RULES ──────────────────────────────────────────────────────
# These strings in the assistant turn signal a bad example
STUB_PATTERNS = [
    "# TODO",
    "# implement",
    "pass  # ",
    "raise NotImplementedError",
    "...\n```",          # truncated code block
    "# Your code here",
    "# Add implementation",
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_raw(path: str) -> list[dict]:
    examples = []
    errors   = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ✗ Line {i}: JSON parse error — {e}")
                errors += 1
    print(f"  Loaded {len(examples)} records ({errors} parse errors)")
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STRUCTURAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_structure(examples: list[dict]) -> tuple[list[dict], int]:
    """
    Each record must have:
      - messages: list with at least one user turn and one assistant turn
      - assistant content must be a non-empty string
    """
    valid, dropped = [], 0
    for ex in examples:
        msgs = ex.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            dropped += 1
            continue

        roles    = [m.get("role") for m in msgs]
        has_user = "user" in roles
        has_asst = "assistant" in roles

        if not (has_user and has_asst):
            dropped += 1
            continue

        # Get assistant content
        asst_content = next(
            (m.get("content", "") for m in msgs if m.get("role") == "assistant"),
            ""
        )
        if not asst_content or not asst_content.strip():
            dropped += 1
            continue

        valid.append(ex)

    print(f"  Structural validation: {len(valid)} valid, {dropped} dropped")
    return valid, dropped


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — QUALITY FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def get_assistant_text(ex: dict) -> str:
    """Extract the full assistant response text."""
    msgs = ex.get("messages", [])
    parts = [
        m.get("content", "")
        for m in msgs
        if m.get("role") == "assistant"
    ]
    return "\n".join(parts)


def has_code_block(text: str) -> bool:
    """Must contain at least one non-empty triple-backtick code block."""
    import re
    blocks = re.findall(r"```(?:\w+)?\n(.+?)```", text, re.DOTALL)
    # At least one block with real content (not just whitespace)
    return any(b.strip() for b in blocks)


def has_stub(text: str) -> bool:
    """Returns True if response looks like an incomplete stub."""
    return any(pattern in text for pattern in STUB_PATTERNS)


def quality_filter(examples: list[dict]) -> tuple[list[dict], dict]:
    valid     = []
    drop_reasons: Counter = Counter()

    for ex in examples:
        asst = get_assistant_text(ex)

        if len(asst) < MIN_CHARS:
            drop_reasons["too_short"] += 1
            continue

        if not has_code_block(asst):
            drop_reasons["no_code_block"] += 1
            continue

        if has_stub(asst):
            drop_reasons["stub_or_placeholder"] += 1
            continue

        valid.append(ex)

    total_dropped = sum(drop_reasons.values())
    print(f"  Quality filter: {len(valid)} kept, {total_dropped} dropped")
    for reason, count in drop_reasons.most_common():
        print(f"    {reason}: {count}")
    return valid, dict(drop_reasons)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def get_instruction_key(ex: dict) -> str:
    """
    Use the first user turn (first 120 chars) as the dedup key.
    Lowercased + stripped for near-duplicate matching.
    """
    msgs = ex.get("messages", [])
    first_user = next(
        (m.get("content", "") for m in msgs if m.get("role") == "user"),
        ""
    )
    normalized = first_user.strip().lower()[:120]
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    seen    = set()
    unique  = []
    dropped = 0

    for ex in examples:
        key = get_instruction_key(ex)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(ex)

    print(f"  Deduplication: {len(unique)} unique, {dropped} duplicates removed")
    return unique, dropped


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — APPLY QWEN CHAT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def apply_chat_template(examples: list[dict]) -> list[dict]:
    """
    Apply Qwen2.5's ChatML template to each example.

    tokenize=False → returns formatted string, not token IDs.
    SFTTrainer handles tokenization internally.

    The formatted string looks like:
        <|im_start|>system
        You are an elite AI/ML engineer...
        <|im_end|>
        <|im_start|>user
        Build a ReAct agent...
        <|im_end|>
        <|im_start|>assistant
        Here's a complete implementation...
        <|im_end|>
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        print(f"  Tokenizer loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"  ✗ Could not load tokenizer: {e}")
        print(f"    Run: pip install transformers")
        print(f"    Skipping chat template — examples saved as-is")
        for ex in examples:
            ex["text"] = None
        return examples

    failed = 0
    for ex in examples:
        msgs = ex.get("messages", [])
        try:
            # apply_chat_template handles system/user/assistant roles natively
            ex["text"] = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(f"  ⚠  Template error: {e}")
            ex["text"] = None
            failed += 1

    succeeded = sum(1 for ex in examples if ex.get("text"))
    print(f"  Chat template applied: {succeeded} succeeded, {failed} failed")
    return [ex for ex in examples if ex.get("text")]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TOKEN LENGTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_length(examples: list[dict]) -> tuple[list[dict], dict]:
    """
    Tokenize the formatted text and drop anything over MAX_TOKENS.
    Also logs the length distribution so you can tune batch size.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
    except Exception:
        print("  ⚠  Tokenizer unavailable — skipping length filter")
        return examples, {}

    lengths = []
    valid   = []
    dropped = 0

    for ex in examples:
        text = ex.get("text", "")
        if not text:
            continue
        n_tokens = len(tokenizer.encode(text))
        ex["n_tokens"] = n_tokens
        lengths.append(n_tokens)

        if n_tokens > MAX_TOKENS:
            dropped += 1
        else:
            valid.append(ex)

    if lengths:
        dist = {
            "min":    min(lengths),
            "max":    max(lengths),
            "mean":   round(sum(lengths) / len(lengths)),
            "median": sorted(lengths)[len(lengths) // 2],
            "p90":    sorted(lengths)[int(len(lengths) * 0.90)],
            "p99":    sorted(lengths)[int(len(lengths) * 0.99)],
        }
        print(f"  Length filter: {len(valid)} kept, {dropped} dropped (>{MAX_TOKENS} tokens)")
        print(f"  Token distribution: min={dist['min']} mean={dist['mean']} "
              f"p90={dist['p90']} max={dist['max']}")
    else:
        dist = {}

    return valid, dist


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SHUFFLE, SPLIT, SAVE
# ══════════════════════════════════════════════════════════════════════════════

def split_and_save(examples: list[dict], output_dir: Path) -> dict:
    """
    Shuffle to break category ordering, then split 90/10.
    Saves train.jsonl and val.jsonl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    n_val   = max(1, int(len(examples) * VAL_RATIO))
    n_train = len(examples) - n_val
    train   = examples[:n_train]
    val     = examples[n_val:]   # technically [n_train:] but shuffle means same

    # Actually do it correctly
    val   = examples[:n_val]
    train = examples[n_val:]

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"

    def write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(train, train_path)
    write_jsonl(val,   val_path)

    # Category distribution in train set
    cat_counts: Counter = Counter(
        ex.get("category", "unknown") for ex in train
    )

    print(f"\n  Train: {len(train)} examples → {train_path}")
    print(f"  Val:   {len(val)} examples → {val_path}")
    print(f"\n  Category distribution (train):")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat:<30} {count}")

    return {
        "n_train":            len(train),
        "n_val":              len(val),
        "category_counts":    dict(cat_counts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    input_path: str = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
):
    print("Data Preparation Pipeline")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}\n")

    stats: dict = {"input_path": input_path}

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("Step 1 — loading raw data")
    examples = load_raw(input_path)
    stats["n_raw"] = len(examples)

    # ── Step 2: Structural validation ─────────────────────────────────────────
    print("\nStep 2 — structural validation")
    examples, n_struct_dropped = validate_structure(examples)
    stats["n_struct_dropped"] = n_struct_dropped

    # ── Step 3: Quality filters ───────────────────────────────────────────────
    print("\nStep 3 — quality filters")
    examples, drop_reasons = quality_filter(examples)
    stats["quality_drop_reasons"] = drop_reasons

    # ── Step 4: Deduplication ─────────────────────────────────────────────────
    print("\nStep 4 — deduplication")
    examples, n_dupes = deduplicate(examples)
    stats["n_duplicates_removed"] = n_dupes

    # ── Step 5: Chat template ─────────────────────────────────────────────────
    print("\nStep 5 — applying Qwen chat template")
    examples = apply_chat_template(examples)

    # ── Step 6: Token length ──────────────────────────────────────────────────
    print("\nStep 6 — token length check")
    examples, length_dist = filter_by_length(examples)
    stats["token_distribution"] = length_dist

    # ── Step 7: Split + save ──────────────────────────────────────────────────
    print("\nStep 7 — split and save")
    split_stats = split_and_save(examples, output_dir)
    stats.update(split_stats)

    # Save summary stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Final summary
    print(f"\n{'='*55}")
    print(f"  Raw examples:         {stats['n_raw']}")
    print(f"  After all filters:    {split_stats['n_train'] + split_stats['n_val']}")
    print(f"  Train set:            {split_stats['n_train']}")
    print(f"  Val set:              {split_stats['n_val']}")
    print(f"  Stats saved:          {stats_path}")
    print(f"\n  Next step:")
    print(f"    python finetune_qlora.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data preparation for QLoRA fine-tuning")
    parser.add_argument("--input",       default=INPUT_PATH,
                        help="Path to raw JSONL file from dataset generator")
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR),
                        help="Directory to write train.jsonl, val.jsonl, stats.json")
    parser.add_argument("--model",       default=MODEL_NAME,
                        help="HuggingFace model name for tokenizer + chat template")
    parser.add_argument("--val-ratio",   type=float, default=VAL_RATIO,
                        help="Fraction held out for validation (default: 0.10)")
    parser.add_argument("--max-tokens",  type=int,   default=MAX_TOKENS,
                        help="Drop examples longer than this many tokens")
    parser.add_argument("--min-chars",   type=int,   default=MIN_CHARS,
                        help="Minimum assistant response character length")
    parser.add_argument("--no-require-code", action="store_true",
                        help="Skip the code block requirement filter")
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    INPUT_PATH  = args.input
    OUTPUT_DIR  = Path(args.output_dir)
    MODEL_NAME  = args.model
    VAL_RATIO   = args.val_ratio
    MAX_TOKENS  = args.max_tokens
    MIN_CHARS   = args.min_chars

    prepare_dataset()
