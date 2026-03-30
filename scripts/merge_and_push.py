# ── Merge LoRA Adapters + Save / Share ───────────────────────────────────────
# Run this after finetune_qlora.py completes.
#
# Produces three output formats — pick the ones you need:
#   1. Merged full model  (safetensors) → push to HuggingFace Hub
#   2. GGUF quantized     (Q4_K_M)      → load into Ollama locally
#   3. Adapter only                      → lightweight share for researchers
#
# Prerequisites:
#   pip install transformers peft bitsandbytes accelerate huggingface_hub
#   # For GGUF conversion:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp && pip install -r requirements.txt
#
# Run:
#   python merge_and_push.py                        # all steps, interactive HF login
#   python merge_and_push.py --skip-hub             # merge + GGUF only, no upload
#   python merge_and_push.py --skip-gguf            # merge + hub only, no GGUF
#   python merge_and_push.py --adapter-only         # push adapter without merging

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME  = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH     = "./outputs/qwen-agentic-coder/final-adapter"
MERGED_PATH      = "./outputs/qwen-agentic-coder-merged"
GGUF_PATH        = "./outputs/qwen-agentic-coder.Q4_K_M.gguf"
LLAMA_CPP_DIR    = "./llama.cpp"             # path to your llama.cpp clone

# HuggingFace Hub settings
HF_REPO_ID       = "YOUR_HF_USERNAME/qwen-agentic-coder"    # ← change this
HF_PRIVATE       = False    # set True to keep the repo private initially


# ══════════════════════════════════════════════════════════════════════════════
# 1. MERGE ADAPTERS INTO BASE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def merge_adapter(adapter_path: str, merged_path: str) -> None:
    """
    Load base model in fp16, load LoRA adapter on top, call merge_and_unload(),
    then save the result as a standard HuggingFace model.

    Why fp16 here instead of 4-bit?
    During training we used 4-bit to save GPU memory. For merging we want full
    precision so the merged weights are as accurate as possible. The merged
    safetensors files will be fp16 (~14 GB), which is what you upload to the Hub.
    Others can re-quantize from there to whatever precision they need.

    merge_and_unload() computes:
        W_merged = W_base + (alpha / r) * B @ A
    for every LoRA-targeted layer, writes the result into the base model, and
    removes the adapter modules. After this call the model is a standard
    AutoModelForCausalLM with no PEFT dependency.
    """
    print(f"\nStep 1 — merging adapters")
    print(f"  Base model:  {BASE_MODEL_NAME}")
    print(f"  Adapter:     {adapter_path}")
    print(f"  Output:      {merged_path}")

    if Path(merged_path).exists():
        print(f"  Merged model already exists at {merged_path} — skipping merge")
        print(f"  (Delete the folder to re-merge)")
        return

    print(f"  Loading base model in fp16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print(f"  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model...")
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        merged_path,
        safe_serialization=True,   # saves as .safetensors (safer than .bin)
        max_shard_size="4GB",      # splits into 4 GB shards for easy upload
    )

    # Also save tokenizer alongside the model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)

    # Write a minimal model card so Hub renders it nicely
    _write_model_card(merged_path)

    print(f"  Merge complete → {merged_path}")
    _print_dir_size(merged_path)


def _write_model_card(output_dir: str) -> None:
    """Write a README.md that HuggingFace Hub renders as the model card."""
    card = f"""---
base_model: {BASE_MODEL_NAME}
tags:
  - qwen
  - qwen2.5
  - code
  - agentic-ai
  - langchain
  - qlora
  - fine-tuned
license: apache-2.0
---

# Qwen2.5-Coder-7B — Agentic AI Fine-tune

Fine-tuned from [{BASE_MODEL_NAME}](https://huggingface.co/{BASE_MODEL_NAME})
using QLoRA on a synthetic dataset of agentic AI, ML, deep learning, and
generative AI coding tasks generated with Claude 3 Haiku on AWS Bedrock.

## Domains covered
- Agentic AI (ReAct, LangChain, LlamaIndex, AutoGen, multi-agent systems)
- Machine Learning (sklearn pipelines, feature engineering, hyperparameter tuning)
- Deep Learning (PyTorch training loops, custom architectures, debugging)
- Generative AI (QLoRA fine-tuning, RAG pipelines, vector databases, vLLM)

## Training details
- Base model: `{BASE_MODEL_NAME}`
- Method: QLoRA (r=16, alpha=32, all attention + MLP projections)
- Quantization: 4-bit NF4 during training, merged to fp16 for this checkpoint
- Dataset: 75 train / 8 val synthetic examples
- Epochs: 3

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{HF_REPO_ID}",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_ID}")

messages = [
    {{"role": "system", "content": "You are an elite AI/ML engineer."}},
    {{"role": "user",   "content": "Build a ReAct agent from scratch in Python."}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PUSH MERGED MODEL TO HUGGINGFACE HUB
# ══════════════════════════════════════════════════════════════════════════════

def push_to_hub(merged_path: str) -> None:
    """
    Upload the merged model to HuggingFace Hub.

    Before running: huggingface-cli login
    Or set HF_TOKEN env var: export HF_TOKEN=hf_...

    The repo is created automatically if it doesn't exist.
    Uploading ~14 GB of safetensors shards takes 10-30 min depending
    on your upload bandwidth.
    """
    print(f"\nStep 2 — pushing to HuggingFace Hub")
    print(f"  Repo: {HF_REPO_ID}  (private={HF_PRIVATE})")

    from huggingface_hub import HfApi, login

    # Auth: prefer env var, fall back to interactive login
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print(f"  Authenticated via HF_TOKEN env var")
    else:
        print(f"  No HF_TOKEN found — launching interactive login...")
        login()

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=HF_REPO_ID,
            private=HF_PRIVATE,
            exist_ok=True,
        )
        visibility = "private" if HF_PRIVATE else "public"
        print(f"  Repo ready: https://huggingface.co/{HF_REPO_ID} ({visibility})")
    except Exception as e:
        print(f"  ✗ Could not create repo: {e}")
        return

    # Upload the whole folder — Hub handles sharded uploads automatically
    print(f"  Uploading {merged_path} ...")
    api.upload_folder(
        folder_path=merged_path,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add merged QLoRA fine-tuned model",
    )

    print(f"  Upload complete!")
    print(f"  View at: https://huggingface.co/{HF_REPO_ID}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONVERT TO GGUF FOR OLLAMA
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_gguf(merged_path: str, gguf_path: str) -> None:
    """
    Convert the merged fp16 model to GGUF format using llama.cpp's converter,
    then quantize to Q4_K_M (best quality/size tradeoff for 7B models).

    GGUF Q4_K_M:
    - ~4 GB file (vs ~14 GB fp16)
    - Runs on CPU or GPU with llama.cpp / Ollama
    - Quality very close to fp16 for code generation tasks
    - Q4_K_M uses "medium" k-quant which preserves important weight channels

    Other quantization options if you need different tradeoffs:
    - Q8_0:    ~7.5 GB  — highest quality, still fits in 8 GB RAM
    - Q4_K_M:  ~4.1 GB  — recommended (this script's default)
    - Q4_K_S:  ~3.8 GB  — slightly smaller, slightly lower quality
    - Q3_K_M:  ~3.1 GB  — aggressive, noticeable quality drop
    """
    print(f"\nStep 3 — converting to GGUF (Q4_K_M)")

    llama_cpp = Path(LLAMA_CPP_DIR)
    if not llama_cpp.exists():
        print(f"  ✗ llama.cpp not found at {LLAMA_CPP_DIR}")
        print(f"    Clone it with:")
        print(f"      git clone https://github.com/ggerganov/llama.cpp {LLAMA_CPP_DIR}")
        print(f"      cd {LLAMA_CPP_DIR} && pip install -r requirements.txt")
        return

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Older llama.cpp versions use a different name
        convert_script = llama_cpp / "convert.py"

    if not convert_script.exists():
        print(f"  ✗ Conversion script not found in {LLAMA_CPP_DIR}")
        return

    # Step A: convert to full-precision GGUF first
    fp16_gguf = str(gguf_path).replace(".Q4_K_M.gguf", ".f16.gguf")
    print(f"  Converting fp16 HF model → fp16 GGUF...")
    result = subprocess.run(
        [
            sys.executable, str(convert_script),
            merged_path,
            "--outtype", "f16",
            "--outfile", fp16_gguf,
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ✗ Conversion failed:\n{result.stderr}")
        return
    print(f"  fp16 GGUF saved → {fp16_gguf}")

    # Step B: quantize to Q4_K_M using llama-quantize
    quantize_bin = llama_cpp / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp / "quantize"  # older name
    if not quantize_bin.exists():
        print(f"  ✗ llama-quantize binary not found — build llama.cpp first:")
        print(f"    cd {LLAMA_CPP_DIR} && cmake -B build && cmake --build build -j")
        print(f"    Then re-run this script.")
        return

    print(f"  Quantizing to Q4_K_M...")
    result = subprocess.run(
        [str(quantize_bin), fp16_gguf, gguf_path, "Q4_K_M"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ✗ Quantization failed:\n{result.stderr}")
        return

    # Clean up intermediate fp16 GGUF
    Path(fp16_gguf).unlink(missing_ok=True)

    size_gb = Path(gguf_path).stat().st_size / 1e9
    print(f"  GGUF ready → {gguf_path}  ({size_gb:.1f} GB)")
    print(f"\n  To run with Ollama:")
    print(f"    # Create a Modelfile")
    _write_ollama_modelfile(gguf_path)
    print(f"    ollama create qwen-agentic-coder -f Modelfile")
    print(f"    ollama run qwen-agentic-coder")


def _write_ollama_modelfile(gguf_path: str) -> None:
    """Write an Ollama Modelfile next to the GGUF for easy loading."""
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    content = f"""FROM {gguf_path}

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM You are an elite AI/ML engineer specializing in agentic AI systems, \\
LangChain, LlamaIndex, AutoGen, PyTorch, and generative AI workflows. \\
Write complete, immediately runnable Python code with type hints, \\
inline comments, and working examples.
"""
    with open(modelfile_path, "w") as f:
        f.write(content)
    print(f"    Modelfile written → {modelfile_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PUSH ADAPTER ONLY (lightweight option)
# ══════════════════════════════════════════════════════════════════════════════

def push_adapter_only(adapter_path: str) -> None:
    """
    Push just the LoRA adapter weights (~160 MB) without merging.

    Useful when:
    - You want to share quickly without the merge step
    - Recipients are researchers who want to experiment with the adapter
    - You plan to release multiple adapters on the same base model

    Recipients load it like this:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        model = PeftModel.from_pretrained(base, "YOUR_HF_USERNAME/qwen-agentic-coder-adapter")
    """
    print(f"\nStep — pushing adapter only (no merge)")
    print(f"  Adapter size: ~160 MB")

    adapter_repo = HF_REPO_ID + "-adapter"

    from huggingface_hub import HfApi, login
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        login()

    api = HfApi()
    api.create_repo(repo_id=adapter_repo, private=HF_PRIVATE, exist_ok=True)

    # Write a minimal adapter card
    card_path = os.path.join(adapter_path, "README.md")
    with open(card_path, "w") as f:
        f.write(f"""---
base_model: {BASE_MODEL_NAME}
tags: [peft, lora, qwen, code, agentic-ai]
---

# LoRA adapter — Qwen2.5-Coder-7B Agentic AI

Adapter-only weights for [{BASE_MODEL_NAME}](https://huggingface.co/{BASE_MODEL_NAME}).

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_NAME}", torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "{adapter_repo}")
model = model.merge_and_unload()  # optional: merge for faster inference
```
""")

    api.upload_folder(
        folder_path=adapter_path,
        repo_id=adapter_repo,
        repo_type="model",
        commit_message="Add LoRA adapter weights",
    )
    print(f"  Adapter uploaded → https://huggingface.co/{adapter_repo}")


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def _print_dir_size(path: str) -> None:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"  Directory size: {total / 1e9:.1f} GB")


def _check_adapter_exists(adapter_path: str) -> bool:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required:
        if not (Path(adapter_path) / fname).exists():
            # Some versions save as adapter_model.bin
            if fname == "adapter_model.safetensors":
                if (Path(adapter_path) / "adapter_model.bin").exists():
                    continue
            print(f"  ✗ Missing: {adapter_path}/{fname}")
            print(f"    Run finetune_qlora.py first.")
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and save/push the model"
    )
    parser.add_argument(
        "--skip-hub",   action="store_true",
        help="Skip HuggingFace Hub upload"
    )
    parser.add_argument(
        "--skip-gguf",  action="store_true",
        help="Skip GGUF conversion"
    )
    parser.add_argument(
        "--adapter-only", action="store_true",
        help="Push adapter weights only (no merge, much faster)"
    )
    args = parser.parse_args()

    print("Merge + Save Pipeline")
    print(f"Adapter: {ADAPTER_PATH}\n")

    if not _check_adapter_exists(ADAPTER_PATH):
        sys.exit(1)

    if args.adapter_only:
        push_adapter_only(ADAPTER_PATH)
        return

    # Always merge first
    merge_adapter(ADAPTER_PATH, MERGED_PATH)

    # Hub upload
    if not args.skip_hub:
        if HF_REPO_ID.startswith("YOUR_HF_USERNAME"):
            print(f"\n  ✗ Set HF_REPO_ID in the script before pushing to Hub")
            print(f"    e.g.  HF_REPO_ID = 'johndoe/qwen-agentic-coder'")
        else:
            push_to_hub(MERGED_PATH)

    # GGUF conversion
    if not args.skip_gguf:
        convert_to_gguf(MERGED_PATH, GGUF_PATH)

    print(f"\n{'='*55}")
    print(f"  Done! Summary:")
    if Path(MERGED_PATH).exists():
        print(f"  Merged model  → {MERGED_PATH}")
    if Path(GGUF_PATH).exists():
        print(f"  GGUF          → {GGUF_PATH}")
    if not HF_REPO_ID.startswith("YOUR_HF_USERNAME") and not args.skip_hub:
        print(f"  HF Hub        → https://huggingface.co/{HF_REPO_ID}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
