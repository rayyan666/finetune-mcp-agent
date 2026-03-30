# finetune-agent-mcp

An open-source MCP (Model Context Protocol) server that lets Claude orchestrate the **full LLM fine-tuning pipeline** — from synthetic data generation to HuggingFace Hub publishing — through natural conversation.

Tell Claude: *"Fine-tune a coding assistant on agentic AI tasks"* and it handles the rest.

---

## What it does

Claude calls MCP tools that wrap a complete QLoRA fine-tuning stack:

```
Claude (orchestrator)
    │
    ├── generate_dataset()    → AWS Bedrock (Claude Haiku) → JSONL training data
    ├── prepare_data()        → validate, deduplicate, chat template, 90/10 split
    ├── inspect_examples()    → sample-check data quality before training
    ├── detect_hardware()     → probe GPU VRAM, recommend model size
    ├── select_model()        → confirm hyperparameters, estimate VRAM
    ├── run_finetune()        → launch QLoRA training (async, returns job_id)
    ├── get_training_status() → poll running job for loss metrics
    ├── tail_training_logs()  → stream log lines for debugging
    ├── evaluate_model()      → run val set inference, compare to base
    ├── merge_adapters()      → merge LoRA into base weights (safetensors)
    ├── push_to_hub()         → upload to HuggingFace Hub
    ├── export_gguf()         → convert for Ollama / llama.cpp
    └── generate_model_card() → auto-write README.md for the HF repo
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourname/finetune-agent-mcp
cd finetune-agent-mcp

# MCP server only (no GPU required on the machine running Claude)
pip install -e .

# Full stack including training dependencies (on your GPU machine)
pip install -e ".[all]"
```

### 2. Copy your scripts

```bash
cp /path/to/dataset_generator_bedrock.py scripts/
cp /path/to/data_preparation.py          scripts/
cp /path/to/finetune_qlora.py            scripts/
cp /path/to/merge_and_push.py            scripts/
```

### 3. Connect to Claude

**Local (Claude Desktop):** Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "finetune-agent": {
      "command": "python",
      "args": ["-m", "finetune_agent_mcp.server"],
      "cwd": "/path/to/finetune-agent-mcp"
    }
  }
}
```

**Remote EC2 (claude.ai):**
```bash
# On your g4dn.xlarge:
python -m finetune_agent_mcp.server --http --port 8000

# In claude.ai MCP settings:
# URL: http://your-ec2-ip:8000/mcp
```

### 4. Talk to Claude

```
You:   Fine-tune a 7B coding assistant on agentic AI tasks.
       I have an EC2 g4dn.xlarge with 16GB VRAM.

Claude: I'll start by checking your hardware, then generate 300 training
        examples and run QLoRA fine-tuning. Let me begin...

        [calls detect_hardware()]  → T4 16GB confirmed, fp16, no flash_attn
        [calls generate_dataset()] → generating 300 examples via Bedrock...
        [calls prepare_data()]     → 270 train / 30 val after filtering
        [calls run_finetune()]     → training launched (job ft-qlora-a3f8b1c2)
        [polls every 60s...]       → step 150/270, train_loss=1.24, eval_loss=1.31
        [training complete]        → best checkpoint saved
        [calls merge_adapters()]   → 14GB merged model
        [calls push_to_hub()]      → yourname/qwen-agentic-coder
```

---

## Project structure

```
finetune-agent-mcp/
├── finetune_agent_mcp/
│   ├── server.py          ← MCP server entry point, tool registration
│   ├── job_state.py       ← async job registry (persistent subprocess tracking)
│   └── tools/
│       ├── data_tools.py      ← generate_dataset, prepare_data, inspect_examples
│       ├── training_tools.py  ← detect_hardware, select_model, run_finetune,
│       │                          get_training_status, tail_training_logs
│       ├── eval_tools.py      ← evaluate_model, compare_outputs
│       └── ship_tools.py      ← merge_adapters, push_to_hub, export_gguf,
│                                  generate_model_card
├── scripts/               ← your existing pipeline scripts (copy here)
│   ├── dataset_generator_bedrock.py
│   ├── data_preparation.py
│   ├── finetune_qlora.py
│   └── merge_and_push.py
├── data/
│   ├── raw/               ← generated JSONL before processing
│   └── prepared/          ← train.jsonl, val.jsonl after prepare_data
├── outputs/               ← checkpoints, logs, merged models
│   └── logs/              ← per-job training logs (polled by Claude)
├── tests/
├── pyproject.toml
└── README.md
```

---

## Hardware requirements

| GPU | VRAM | Model | Notes |
|-----|------|-------|-------|
| T4 (g4dn.xlarge) | 16GB | Qwen2.5-Coder-7B | fp16, batch=1, grad_accum=16 |
| A10G (g5.xlarge) | 24GB | Qwen2.5-Coder-7B | bf16, batch=4, grad_accum=4 |
| A100 (p3.2xlarge) | 40GB | Qwen2.5-Coder-14B | bf16, flash_attn_2 |
| T4 (g4dn) | 16GB | Qwen2.5-Coder-3B | fallback if 7B OOMs |

---

## Key design decisions

**Why async training?** Fine-tuning takes 10-60 minutes. MCP tools have a response timeout — blocking Claude for that long breaks the connection. `run_finetune()` launches a subprocess and returns `job_id` immediately. Claude polls `get_training_status()` every 60 seconds.

**Why subprocess over importlib?** Training scripts have global state (CUDA device, W&B init, HF model cache). Running them as subprocesses gives clean process isolation — a crash in training doesn't take down the MCP server.

**Why your existing scripts?** They're already tested and working. The MCP layer is pure orchestration — it calls your scripts via CLI args, reads their stdout/log files, and translates results into structured tool responses.

---

## Contributing

PRs welcome. Especially useful:
- Support for other base models (Llama 3, Mistral, Phi-3)
- `evaluate.py` script with HumanEval / MBPP metrics
- DPO fine-tuning support via `trl.DPOTrainer`
- Multi-GPU training support

---

## License

MIT — use it, modify it, ship it.
