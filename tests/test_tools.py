"""Tests for finetune-agent-mcp tools."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Data tools ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_inspect_examples_missing_file():
    from finetune_agent_mcp.tools.data_tools import inspect_examples
    result = await inspect_examples(data_path="./nonexistent.jsonl")
    assert result["status"] == "error"
    assert "not found" in result["message"]


@pytest.mark.asyncio
async def test_inspect_examples_valid_file(tmp_path):
    from finetune_agent_mcp.tools.data_tools import inspect_examples

    # Write a minimal JSONL file
    data_file = tmp_path / "train.jsonl"
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are an expert."},
                {"role": "user", "content": "Write a ReAct agent."},
                {"role": "assistant", "content": "```python\nclass Agent:\n    pass\n```"},
            ],
            "category": "agentic_react",
            "turns": 1,
        }
        for _ in range(5)
    ]
    with open(data_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    result = await inspect_examples(data_path=str(data_file), n=2)
    assert result["status"] == "ok"
    assert result["total_examples"] == 5
    assert len(result["sampled"]) == 2
    assert "agentic_react" in result["category_distribution"]


@pytest.mark.asyncio
async def test_prepare_data_missing_script():
    from finetune_agent_mcp.tools.data_tools import prepare_data
    result = await prepare_data(input_path="./nonexistent.jsonl")
    # Should return error since script doesn't exist in test env
    assert result["status"] == "error"


# ── Training tools ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_detect_hardware_returns_structure():
    from finetune_agent_mcp.tools.training_tools import detect_hardware
    result = await detect_hardware()
    # Must return a dict with status
    assert "status" in result
    assert result["status"] in ("ok", "error")
    # If GPU available, must include recommended_model
    if result.get("gpu_available"):
        assert "recommended_model" in result


@pytest.mark.asyncio
async def test_select_model_returns_config():
    from finetune_agent_mcp.tools.training_tools import select_model
    result = await select_model(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        batch_size=1,
        gradient_accumulation=16,
    )
    assert result["status"] == "ok"
    assert result["config"]["effective_batch_size"] == 16
    assert result["config"]["lora_r"] == 16


@pytest.mark.asyncio
async def test_run_finetune_missing_data():
    from finetune_agent_mcp.tools.training_tools import run_finetune
    result = await run_finetune(
        train_path="./nonexistent_train.jsonl",
        val_path="./nonexistent_val.jsonl",
    )
    assert result["status"] == "error"
    assert "prepare_data" in result["message"]


@pytest.mark.asyncio
async def test_get_training_status_unknown_job():
    from finetune_agent_mcp.tools.training_tools import get_training_status
    result = await get_training_status("ft-nonexistent-job")
    assert result["status"] == "error"


# ── Ship tools ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_to_hub_missing_repo_id():
    from finetune_agent_mcp.tools.ship_tools import push_to_hub
    result = await push_to_hub(repo_id="")
    assert result["status"] == "error"
    assert "repo_id" in result["message"]


@pytest.mark.asyncio
async def test_generate_model_card(tmp_path):
    from finetune_agent_mcp.tools.ship_tools import generate_model_card

    # Create a fake adapter config
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    }))

    output = tmp_path / "README.md"
    result = await generate_model_card(
        adapter_path=str(adapter_dir),
        repo_id="testuser/test-model",
        train_examples=270,
        eval_loss=0.42,
        output_path=str(output),
    )

    assert result["status"] == "completed"
    assert output.exists()
    content = output.read_text()
    assert "testuser/test-model" in content
    assert "QLoRA" in content
    assert "270" in content


@pytest.mark.asyncio
async def test_merge_adapters_missing_adapter():
    from finetune_agent_mcp.tools.ship_tools import merge_adapters
    result = await merge_adapters(adapter_path="./nonexistent-adapter")
    assert result["status"] == "error"
    # Either script not found, or adapter path not found — both are valid errors
    assert any(w in result["message"].lower() for w in ["adapter", "training", "script", "not found"])
