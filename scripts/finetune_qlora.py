# ── QLoRA Fine-tuning: Qwen2.5-Coder-7B-Instruct ─────────────────────────────
# Trains LoRA adapters on top of a 4-bit quantized base model.
# Runs on a single 16GB GPU (A10G, T4 16GB, RTX 4080+).
#
# Prerequisites:
#   pip install torch transformers peft trl bitsandbytes accelerate wandb
#
# Input:  ./data/prepared/train.jsonl
#         ./data/prepared/val.jsonl
# Output: ./outputs/qwen-agentic-coder/   (checkpoints + final adapter)
#
# Run:
#   python finetune_qlora.py
#   python finetune_qlora.py --no-wandb    (skip W&B logging)

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


# ── CONFIG ────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Model
    model_name:    str  = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir:    str  = "./outputs/qwen-agentic-coder"

    # Data
    train_path:    str  = "./data/prepared/train.jsonl"
    val_path:      str  = "./data/prepared/val.jsonl"
    max_seq_length: int = 4096

    # LoRA — the adapter configuration
    lora_r:        int   = 16      # rank: higher = more capacity, more memory
    lora_alpha:    int   = 32      # scaling factor: alpha/r = 2.0 is standard
    lora_dropout:  float = 0.05
    # Target all projection matrices — more coverage = better quality
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ])

    # Quantization
    use_4bit:          bool  = True
    bnb_4bit_quant_type: str = "nf4"          # NF4 > FP4 for LLMs
    bnb_compute_dtype:   str = "bfloat16"     # bf16 for A10G/A100, fp16 for T4

    # Training
    num_epochs:             int   = 3
    per_device_batch_size:  int   = 4
    gradient_accumulation:  int   = 4          # effective batch = 4×4 = 16
    learning_rate:          float = 2e-4
    lr_scheduler:           str   = "cosine"
    warmup_ratio:           float = 0.05       # 5% of steps for warmup
    weight_decay:           float = 0.01
    max_grad_norm:          float = 0.3        # gradient clipping for stability
    eval_steps:             int   = 50
    save_steps:             int   = 50
    logging_steps:          int   = 10
    save_total_limit:       int   = 3          # keep only the 3 best checkpoints

    # W&B
    use_wandb:     bool = True
    wandb_project: str  = "qwen-agentic-coder"
    run_name:      str  = "qlora-7b-v1"


CFG = FinetuneConfig()


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> Dataset:
    """Load a .jsonl file into a HuggingFace Dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # SFTTrainer expects a "text" column containing the pre-formatted string.
    # data_preparation.py already applied the chat template and stored it in "text".
    texts = [r["text"] for r in records if r.get("text")]
    print(f"  Loaded {len(texts)} examples from {path}")
    return Dataset.from_dict({"text": texts})


# ══════════════════════════════════════════════════════════════════════════════
# 2. QUANTIZATION CONFIG (BitsAndBytes)
# ══════════════════════════════════════════════════════════════════════════════

def get_bnb_config() -> BitsAndBytesConfig:
    """
    4-bit NF4 quantization config.

    NF4 (NormalFloat4) is designed specifically for normally-distributed weights
    like those in transformer LLMs — it gives better precision than FP4 at the
    same bit width.

    compute_dtype=bfloat16 means matrix multiplications happen in BF16 even
    though weights are stored in 4-bit. This is the key QLoRA trick: storage
    is tiny, compute is fast and numerically stable.
    """
    compute_dtype = (
        torch.bfloat16
        if CFG.bnb_compute_dtype == "bfloat16"
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=CFG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # nested quantization → saves ~0.4 GB extra
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. LORA CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def get_lora_config() -> LoraConfig:
    """
    LoRA adapter configuration.

    r=16:     The rank of adapter matrices A and B. Higher rank = more
              parameters = more capacity but more memory. 16 is the sweet spot
              for code generation on a 16GB GPU.

    alpha=32: Scaling factor applied as (alpha/r) * BA. With r=16, alpha=32
              gives a scale of 2.0. This is the standard setting.

    dropout:  Light regularization (0.05) to reduce overfitting on small datasets.

    target_modules: We inject LoRA into all attention projections (q/k/v/o)
                    AND all MLP projections (gate/up/down). This full coverage
                    gives significantly better code quality than attention-only.

    bias="none": Don't train bias terms — they add parameters without
                 meaningfully improving code generation quality.
    """
    return LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=CFG.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    print(f"Loading model: {CFG.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        padding_side="right",   # right-pad for causal LM training
    )
    # Qwen2.5 uses <|endoftext|> as pad — set it explicitly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",          # spread across available GPUs automatically
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Required before injecting LoRA into a quantized model:
    # - disables the dropout in quantized layers (not compatible with backprop)
    # - casts layer norms to float32 for numerical stability
    # - sets input embeddings to require grad so the model can be trained
    model = prepare_model_for_kbit_training(model)

    # Inject LoRA adapters
    model = get_peft_model(model, get_lora_config())

    # Log trainable parameter count
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def get_training_args() -> SFTConfig:
    """
    SFTConfig extends TrainingArguments with SFT-specific options.

    Key decisions:
    - fp16/bf16: use bf16 on A10G/A100, fp16 on T4 (no bf16 support)
    - gradient_checkpointing: trades compute for memory — essential at 4096 seq len
    - optim=paged_adamw_32bit: paged optimizer keeps optimizer states in CPU RAM
      when GPU is full, then pages them back. Saves ~3GB GPU memory vs adamw.
    - group_by_length: batches similar-length sequences together → less padding
      → faster training (especially useful with variable-length code examples)
    """
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    return SFTConfig(
        output_dir=CFG.output_dir,

        # Epochs and steps
        num_train_epochs=CFG.num_epochs,
        per_device_train_batch_size=CFG.per_device_batch_size,
        per_device_eval_batch_size=CFG.per_device_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation,

        # Learning rate
        learning_rate=CFG.learning_rate,
        lr_scheduler_type=CFG.lr_scheduler,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        max_grad_norm=CFG.max_grad_norm,

        # Precision + memory
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        group_by_length=True,

        # Evaluation + saving
        evaluation_strategy="steps",
        eval_steps=CFG.eval_steps,
        save_strategy="steps",
        save_steps=CFG.save_steps,
        save_total_limit=CFG.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=CFG.logging_steps,
        report_to="wandb" if CFG.use_wandb else "none",
        run_name=CFG.run_name,

        # SFT-specific
        max_seq_length=CFG.max_seq_length,
        dataset_text_field="text",     # which column SFTTrainer reads
        packing=False,                  # don't pack multiple examples — cleaner loss
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOSS MASKING — train only on assistant tokens
# ══════════════════════════════════════════════════════════════════════════════

def get_data_collator(tokenizer):
    """
    DataCollatorForCompletionOnlyLM masks the loss on everything BEFORE the
    assistant response token, so the model only learns to generate assistant
    content — not to predict the system prompt or user message.

    For Qwen2.5's ChatML format, the assistant turn always starts with:
        <|im_start|>assistant\n

    We set response_template to this exact byte sequence. The collator finds
    this marker in each tokenized example and sets labels=-100 for all tokens
    before it, effectively ignoring them in the loss calculation.

    Without this, the model learns to predict your system prompt and user
    instructions, which wastes capacity and can cause the model to repeat
    them at inference time.
    """
    response_template = "<|im_start|>assistant\n"
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(use_wandb: bool = True):
    CFG.use_wandb = use_wandb

    if CFG.use_wandb:
        import wandb
        wandb.init(project=CFG.wandb_project, name=CFG.run_name)

    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = load_jsonl(CFG.train_path)
    val_dataset   = load_jsonl(CFG.val_path)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # ── Configure trainer ─────────────────────────────────────────────────────
    training_args  = get_training_args()
    data_collator  = get_data_collator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ── Print training summary ────────────────────────────────────────────────
    n_train        = len(train_dataset)
    steps_per_epoch = n_train // (CFG.per_device_batch_size * CFG.gradient_accumulation)
    total_steps    = steps_per_epoch * CFG.num_epochs

    print(f"\n{'='*55}")
    print(f"  Model:            {CFG.model_name}")
    print(f"  LoRA rank:        {CFG.lora_r}  alpha: {CFG.lora_alpha}")
    print(f"  Train examples:   {n_train}")
    print(f"  Effective batch:  {CFG.per_device_batch_size * CFG.gradient_accumulation}")
    print(f"  Steps/epoch:      {steps_per_epoch}")
    print(f"  Total steps:      {total_steps}")
    print(f"  LR:               {CFG.learning_rate}  ({CFG.lr_scheduler})")
    print(f"  Output:           {CFG.output_dir}")
    print(f"  W&B:              {'enabled' if CFG.use_wandb else 'disabled'}")
    print(f"{'='*55}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training...\n")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = os.path.join(CFG.output_dir, "final-adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"  Final adapter:   {final_path}")
    print(f"\n  Next step:")
    print(f"    python merge_and_push.py")
    print(f"{'='*55}")

    if CFG.use_wandb:
        import wandb
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning script")

    # Model + paths
    parser.add_argument("--model-name",   default=CFG.model_name)
    parser.add_argument("--train-path",   default=CFG.train_path)
    parser.add_argument("--val-path",     default=CFG.val_path)
    parser.add_argument("--output-dir",   default=CFG.output_dir)

    # LoRA
    parser.add_argument("--lora-r",       type=int,   default=CFG.lora_r)
    parser.add_argument("--lora-alpha",   type=int,   default=CFG.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=CFG.lora_dropout)

    # Training
    parser.add_argument("--batch-size",            type=int,   default=CFG.per_device_batch_size)
    parser.add_argument("--gradient-accumulation", type=int,   default=CFG.gradient_accumulation)
    parser.add_argument("--num-epochs",            type=int,   default=CFG.num_epochs)
    parser.add_argument("--learning-rate",         type=float, default=CFG.learning_rate)
    parser.add_argument("--max-seq-length",        type=int,   default=CFG.max_seq_length)
    parser.add_argument("--eval-steps",            type=int,   default=CFG.eval_steps)

    # W&B
    parser.add_argument("--no-wandb",       action="store_true")
    parser.add_argument("--wandb-project",  default=CFG.wandb_project)
    parser.add_argument("--run-name",       default=CFG.run_name)

    args = parser.parse_args()

    # Apply CLI args to config
    CFG.model_name              = args.model_name
    CFG.train_path              = args.train_path
    CFG.val_path                = args.val_path
    CFG.output_dir              = args.output_dir
    CFG.lora_r                  = args.lora_r
    CFG.lora_alpha              = args.lora_alpha
    CFG.lora_dropout            = args.lora_dropout
    CFG.per_device_batch_size   = args.batch_size
    CFG.gradient_accumulation   = args.gradient_accumulation
    CFG.num_epochs              = args.num_epochs
    CFG.learning_rate           = args.learning_rate
    CFG.max_seq_length          = args.max_seq_length
    CFG.eval_steps              = args.eval_steps
    CFG.save_steps              = args.eval_steps   # keep in sync
    CFG.wandb_project           = args.wandb_project
    CFG.run_name                = args.run_name

    # Verify data exists before loading the model
    for p in [CFG.train_path, CFG.val_path]:
        if not Path(p).exists():
            print(f"✗ Missing: {p}")
            print(f"  Run data_preparation.py first.")
            sys.exit(1)

    train(use_wandb=not args.no_wandb)
