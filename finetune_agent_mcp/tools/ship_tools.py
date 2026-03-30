"""
Ship tools — merge LoRA adapters, export formats, and publish to HuggingFace Hub.

Tools:
  merge_adapters      — merge LoRA into base weights (full model, safetensors)
  push_to_hub         — upload merged model to HuggingFace Hub
  export_gguf         — convert to GGUF for Ollama / llama.cpp
  generate_model_card — auto-generate a README.md for the HF Hub repo
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


async def merge_adapters(
    adapter_path: str = "./outputs/qwen-agentic-coder/final-adapter",
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    output_path: str = "./outputs/qwen-agentic-coder-merged",
) -> dict:
    """
    Merge LoRA adapter weights into the base model to produce a standalone model.

    After merging, the output directory contains a full model in safetensors
    format that can be loaded without PEFT or the original base model.

    This operation requires loading the base model (fp16) + adapter — expect
    ~15GB RAM/VRAM. It runs synchronously and takes 5-15 minutes.

    Args:
        adapter_path:  Path to the trained LoRA adapter directory.
        base_model:    HuggingFace base model name.
        output_path:   Where to save the merged full model.

    Returns:
        dict with status, merged model path, and model size estimate.
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "merge_and_push.py"
    if not script.exists():
        return {
            "status": "error",
            "message": "merge_and_push.py not found at scripts/. "
                       "Copy your merge_and_push.py to scripts/",
        }

    if not Path(adapter_path).exists():
        return {
            "status": "error",
            "message": f"Adapter not found at {adapter_path}. "
                       "Complete training first (run_finetune → wait → merge).",
        }

    cmd = [
        sys.executable, str(script),
        "--adapter-path", adapter_path,
        "--base-model", base_model,
        "--output-path", output_path,
        "--skip-hub",   # merge only, don't push yet
        "--skip-gguf",  # GGUF is a separate tool
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr[-2000:] or "Merge script failed",
        }

    # Estimate size of merged model
    size_gb = None
    merged = Path(output_path)
    if merged.exists():
        total_bytes = sum(f.stat().st_size for f in merged.rglob("*") if f.is_file())
        size_gb = round(total_bytes / 1024**3, 1)

    return {
        "status": "completed",
        "merged_model_path": output_path,
        "size_gb": size_gb,
        "message": "Adapter merged into base weights. Full model ready.",
        "next_steps": [
            "Call push_to_hub() to share on HuggingFace",
            "Call export_gguf() to convert for Ollama",
        ],
    }


async def push_to_hub(
    model_path: str = "./outputs/qwen-agentic-coder-merged",
    repo_id: str = "",
    private: bool = False,
    commit_message: str = "Upload fine-tuned model via finetune-agent-mcp",
    hf_token: Optional[str] = None,
) -> dict:
    """
    Push a merged model (or adapter) to HuggingFace Hub.

    Requires: huggingface-cli login OR hf_token argument OR HF_TOKEN env var.

    Args:
        model_path:       Local path to the merged model or adapter.
        repo_id:          HuggingFace repo in format "username/model-name".
                          e.g. "yourname/qwen-agentic-coder"
        private:          Create a private repo (default: public).
        commit_message:   Commit message for the upload.
        hf_token:         HuggingFace API token (optional if already logged in).

    Returns:
        dict with status and the HuggingFace repo URL.
    """
    if not repo_id:
        return {
            "status": "error",
            "message": "repo_id is required. Format: 'your-username/model-name'",
        }

    if not Path(model_path).exists():
        return {
            "status": "error",
            "message": f"Model path not found: {model_path}. Run merge_adapters() first.",
        }

    push_script = """
import sys, os
from huggingface_hub import HfApi

token   = sys.argv[1] if sys.argv[1] != "env" else os.environ.get("HF_TOKEN")
api     = HfApi(token=token)
repo_id = sys.argv[2]
private = sys.argv[3] == "true"
path    = sys.argv[4]
msg     = sys.argv[5]

api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
api.upload_folder(
    folder_path=path,
    repo_id=repo_id,
    commit_message=msg,
)
print(f"https://huggingface.co/{repo_id}")
"""

    token_arg = hf_token or "env"
    result = subprocess.run(
        [sys.executable, "-c", push_script,
         token_arg, repo_id, str(private).lower(), model_path, commit_message],
        capture_output=True, text=True, timeout=1800,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr[-2000:] or "Push failed",
            "tip": "Run 'huggingface-cli login' on the server or set HF_TOKEN env var",
        }

    repo_url = result.stdout.strip()
    return {
        "status": "completed",
        "repo_url": repo_url,
        "repo_id": repo_id,
        "message": f"Model pushed successfully to {repo_url}",
    }


async def export_gguf(
    merged_model_path: str = "./outputs/qwen-agentic-coder-merged",
    output_path: str = "./outputs/qwen-agentic-coder.Q4_K_M.gguf",
    quantization: str = "Q4_K_M",
    llama_cpp_dir: str = "./llama.cpp",
) -> dict:
    """
    Convert a merged model to GGUF format for use with Ollama or llama.cpp.

    Requires llama.cpp to be cloned and its dependencies installed:
      git clone https://github.com/ggerganov/llama.cpp
      cd llama.cpp && pip install -r requirements.txt

    Args:
        merged_model_path: Path to the merged safetensors model.
        output_path:       Where to write the .gguf file.
        quantization:      GGUF quantization level. Q4_K_M is recommended
                           (4-bit, ~4GB, good quality/size tradeoff).
                           Options: Q2_K, Q4_0, Q4_K_M, Q5_K_M, Q8_0
        llama_cpp_dir:     Path to llama.cpp clone.

    Returns:
        dict with status, gguf_path, file size, and Ollama Modelfile template.
    """
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    quantize_bin   = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        return {
            "status": "error",
            "message": (
                f"llama.cpp not found at {llama_cpp_dir}. "
                "Clone it: git clone https://github.com/ggerganov/llama.cpp"
            ),
        }

    # Step 1: Convert to fp16 GGUF
    fp16_path = output_path.replace(".gguf", ".fp16.gguf")
    result = subprocess.run(
        [sys.executable, str(convert_script),
         merged_model_path, "--outfile", fp16_path, "--outtype", "f16"],
        capture_output=True, text=True, timeout=1200,
    )
    if result.returncode != 0:
        return {"status": "error", "message": "Conversion failed: " + result.stderr[-1000:]}

    # Step 2: Quantize if the binary exists
    if quantize_bin.exists():
        result = subprocess.run(
            [str(quantize_bin), fp16_path, output_path, quantization],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            output_path = fp16_path  # fall back to fp16 if quantize fails
    else:
        output_path = fp16_path

    size_gb = None
    if Path(output_path).exists():
        size_gb = round(Path(output_path).stat().st_size / 1024**3, 1)

    modelfile = f"""FROM {output_path}

SYSTEM \"\"\"You are an expert AI/ML engineer specializing in agentic systems,
deep learning, and generative AI. Write complete, production-ready Python code
with type hints, error handling, and working examples.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.95
"""

    return {
        "status": "completed",
        "gguf_path": output_path,
        "quantization": quantization,
        "size_gb": size_gb,
        "ollama_modelfile": modelfile,
        "ollama_command": f"ollama create qwen-agentic-coder -f Modelfile",
    }


async def generate_model_card(
    adapter_path: str = "./outputs/qwen-agentic-coder/final-adapter",
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    repo_id: str = "yourname/qwen-agentic-coder",
    train_examples: int = 0,
    eval_loss: Optional[float] = None,
    tags: Optional[list[str]] = None,
    output_path: str = "./outputs/qwen-agentic-coder-merged/README.md",
) -> dict:
    """
    Generate a HuggingFace model card (README.md) for the fine-tuned model.

    Creates a well-structured README with training details, usage instructions,
    and evaluation results — ready to push alongside the model weights.

    Args:
        adapter_path:    Path to the adapter (to read config from).
        base_model:      Base model name.
        repo_id:         HuggingFace repo ID for the card.
        train_examples:  Number of training examples used.
        eval_loss:       Final evaluation loss (optional).
        tags:            Additional HF tags.
        output_path:     Where to write the README.md.

    Returns:
        dict with status and the generated README content.
    """
    tags = tags or ["code", "qlora", "fine-tuned", "agentic-ai", "qwen"]

    lora_config = {}
    config_file = Path(adapter_path) / "adapter_config.json"
    if config_file.exists():
        lora_config = json.loads(config_file.read_text())

    card = f"""---
base_model: {base_model}
tags:
{chr(10).join(f'- {t}' for t in tags)}
license: apache-2.0
---

# {repo_id.split('/')[-1]}

A QLoRA fine-tuned version of [{base_model}](https://huggingface.co/{base_model})
specialized for agentic AI, ML/DL, and GenAI engineering tasks.

Fine-tuned using [finetune-agent-mcp](https://github.com/yourname/finetune-agent-mcp) —
an open-source MCP server that orchestrates the full fine-tuning pipeline via Claude.

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | `{base_model}` |
| Method | QLoRA (4-bit NF4) |
| LoRA rank | {lora_config.get('r', 16)} |
| LoRA alpha | {lora_config.get('lora_alpha', 32)} |
| Target modules | {', '.join(lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))} |
| Training examples | {train_examples or 'see training logs'} |
| Eval loss | {eval_loss or 'see training logs'} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "You are an expert AI/ML engineer."}},
    {{"role": "user", "content": "Build a ReAct agent from scratch in Python."}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## With Ollama

```bash
ollama pull {repo_id.split('/')[-1]}  # if GGUF version available
# or build from local GGUF:
ollama create qwen-agentic-coder -f Modelfile
ollama run qwen-agentic-coder
```

## Domains covered

- **Agentic AI**: ReAct agents, LangChain, LlamaIndex, AutoGen, multi-agent systems
- **Machine Learning**: sklearn pipelines, feature engineering, hyperparameter tuning
- **Deep Learning**: PyTorch architectures, training loops, custom losses
- **Generative AI**: QLoRA fine-tuning, RAG pipelines, vector databases, evaluation

## Generated with

[finetune-agent-mcp](https://github.com/yourname/finetune-agent-mcp) —
tell Claude "fine-tune a coding assistant" and it handles everything.
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(card, encoding="utf-8")

    return {
        "status": "completed",
        "readme_path": output_path,
        "preview": card[:800] + "...",
        "message": "Model card written. Push it alongside the model weights.",
    }
