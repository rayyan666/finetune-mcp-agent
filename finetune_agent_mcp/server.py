"""
finetune-agent-mcp  —  MCP server that orchestrates the full LLM fine-tuning pipeline.

Claude calls tools in this server to:
  1. Generate a synthetic training dataset via AWS Bedrock
  2. Validate, clean, and split the data
  3. Detect available hardware and select the right model
  4. Launch QLoRA fine-tuning and stream progress
  5. Evaluate the trained model
  6. Merge adapters and push to HuggingFace Hub

Run:
  python -m finetune_agent_mcp.server          # stdio transport (Claude Desktop / claude.ai)
  python -m finetune_agent_mcp.server --http   # HTTP transport (team deployments)
"""

import asyncio
import logging
from mcp.server.fastmcp import FastMCP

from .tools.data_tools import (
    generate_dataset,
    prepare_data,
    inspect_examples,
)
from .tools.training_tools import (
    detect_hardware,
    select_model,
    run_finetune,
    get_training_status,
    tail_training_logs,
)
from .tools.eval_tools import (
    evaluate_model,
    compare_outputs,
)
from .tools.ship_tools import (
    merge_adapters,
    push_to_hub,
    export_gguf,
    generate_model_card,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Create the MCP server ──────────────────────────────────────────────────────
mcp = FastMCP(
    name="finetune-agent-mcp",
    instructions="""
You are orchestrating a complete LLM fine-tuning pipeline.

TYPICAL WORKFLOW (call tools in this order):
1. detect_hardware()           — check GPU VRAM, recommend model size
2. generate_dataset(...)       — create synthetic training data via Bedrock
3. prepare_data(...)           — validate, deduplicate, split 90/10
4. inspect_examples(...)       — sanity-check a sample before training
5. select_model(...)           — confirm model choice given hardware
6. run_finetune(...)           — launch QLoRA training (returns job_id)
7. get_training_status(job_id) — poll until done (check every 60s)
8. tail_training_logs(job_id)  — stream recent log lines for progress
9. evaluate_model(...)         — run eval on the hold-out val set
10. merge_adapters(...)        — merge LoRA into base weights
11. push_to_hub(...)           — upload to HuggingFace Hub
12. export_gguf(...)           — optional: convert to GGUF for Ollama

KEY RULES:
- Always call detect_hardware() first — never assume VRAM
- Always call prepare_data() before run_finetune() — training fails on raw data
- poll get_training_status() every 60 seconds — do NOT call run_finetune() twice
- If a tool returns status='error', read the message and decide: retry, fix inputs, or stop
- For long-running jobs (finetune, merge) inform the user training has started
  and you'll report back when it completes
""",
)

# ── Register all tools ─────────────────────────────────────────────────────────
mcp.tool()(generate_dataset)
mcp.tool()(prepare_data)
mcp.tool()(inspect_examples)
mcp.tool()(detect_hardware)
mcp.tool()(select_model)
mcp.tool()(run_finetune)
mcp.tool()(get_training_status)
mcp.tool()(tail_training_logs)
mcp.tool()(evaluate_model)
mcp.tool()(compare_outputs)
mcp.tool()(merge_adapters)
mcp.tool()(push_to_hub)
mcp.tool()(export_gguf)
mcp.tool()(generate_model_card)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="finetune-agent-mcp server")
    parser.add_argument("--http", action="store_true", help="Use HTTP transport instead of stdio")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="http", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
