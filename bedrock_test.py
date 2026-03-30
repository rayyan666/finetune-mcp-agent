# ── AWS Bedrock Model Discovery + Cost Test ───────────────────────────────────
# Run this FIRST to see which models are available in your region,
# test a quick generation, and get a live cost estimate per example.
#
# Setup on EC2:
#   pip install boto3
#   (IAM role on instance handles auth automatically — no aws configure needed)
#
# IAM permissions needed:
#   bedrock:ListFoundationModels
#   bedrock-runtime:InvokeModel

import boto3
import json
import time
from botocore.exceptions import ClientError

# ── CONFIG ────────────────────────────────────────────────────────────────────
REGION = "us-east-1"

# ── PRICE TABLE (per 1M tokens, us-east-1, on-demand) ────────────────────────
PRICE_TABLE = {
    "amazon.nova-micro":              (0.035,  0.14),
    "amazon.nova-lite":               (0.06,   0.24),
    "amazon.nova-pro":                (0.80,   3.20),
    "meta.llama3-70b":                (0.72,   0.72),
    "meta.llama3-8b":                 (0.22,   0.22),
    "anthropic.claude-3-haiku":       (0.25,   1.25),
    "anthropic.claude-3-5-haiku":     (0.80,   4.00),
    "anthropic.claude-3-5-sonnet":    (3.00,  15.00),
    "mistral.mistral-7b":             (0.15,   0.20),
    "mistral.mixtral-8x7b":           (0.45,   0.70),
    "mistral.mistral-small":          (0.10,   0.30),
    "qwen.qwen3-coder-30b":           (0.25,   1.00),   # estimate
    "qwen.qwen3-32b":                 (0.25,   1.00),
    "deepseek.v3":                    (0.27,   1.10),
    "google.gemma-3-27b":             (0.50,   1.50),
    "google.gemma-3-12b":             (0.20,   0.60),
}

# ── RECOMMENDED MODELS (from your actual model list) ──────────────────────────
# Best for CODE generation, in order of recommendation:
#   1. qwen.qwen3-coder-30b-a3b-v1:0          MoE — best code quality, cheap
#   2. meta.llama3-70b-instruct-v1:0           solid code, proven model
#   3. anthropic.claude-3-haiku-20240307-v1:0  best quality, ~$0.14 total
#   4. amazon.nova-lite-v1:0                   cheapest, ~$0.03 total
RECOMMENDED_MODEL = "qwen.qwen3-coder-30b-a3b-v1:0"

# Models to test in the comparison run
MODELS_TO_TEST = [
    "qwen.qwen3-coder-30b-a3b-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "amazon.nova-lite-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

TEST_PROMPT = (
    "Write a Python function that implements a simple ReAct agent loop "
    "with thought/action/observation. Keep it under 30 lines."
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LIST AVAILABLE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def list_available_models() -> list[dict]:
    client = boto3.client("bedrock", region_name=REGION)
    try:
        resp = client.list_foundation_models(byInferenceType="ON_DEMAND")
        return resp.get("modelSummaries", [])
    except ClientError as e:
        print(f"✗ Error listing models: {e}")
        return []


def print_model_table(models: list[dict]):
    print(f"\n{'─'*85}")
    print(f"  {'MODEL ID':<50} {'PROVIDER':<18} {'MODALITIES'}")
    print(f"{'─'*85}")
    providers: dict[str, list] = {}
    for m in models:
        providers.setdefault(m.get("providerName", "?"), []).append(m)
    for provider, plist in sorted(providers.items()):
        for m in sorted(plist, key=lambda x: x["modelId"]):
            mod = ", ".join(m.get("inputModalities", []))
            print(f"  {m['modelId']:<50} {provider:<18} {mod}")
    print(f"{'─'*85}")
    print(f"  Total: {len(models)} on-demand models in {REGION}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD REQUEST BODY — each provider has a different schema
# ══════════════════════════════════════════════════════════════════════════════

def build_body(model_id: str, prompt: str, max_tokens: int = 1200) -> dict:
    mid = model_id.lower()

    if "anthropic" in mid:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

    elif "meta.llama" in mid:
        return {
            "prompt": (
                f"<|begin_of_text|>"
                f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            ),
            "max_gen_len": max_tokens,
            "temperature": 0.25,
        }

    elif "amazon.nova" in mid:
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxNewTokens": max_tokens, "temperature": 0.25},
        }

    elif "qwen" in mid:
        # Qwen3 uses the same converse-style messages format as Nova
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxNewTokens": max_tokens, "temperature": 0.25},
        }

    elif "deepseek" in mid:
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.25,
        }

    elif "google.gemma" in mid:
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxNewTokens": max_tokens, "temperature": 0.25},
        }

    elif "mistral" in mid:
        return {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": max_tokens,
            "temperature": 0.25,
        }

    else:
        # Generic Nova-style fallback
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxNewTokens": max_tokens, "temperature": 0.25},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. PARSE RESPONSE — extract text + token counts
# ══════════════════════════════════════════════════════════════════════════════

def parse_response(model_id: str, body: dict) -> tuple[str, int, int]:
    mid = model_id.lower()
    text = ""
    inp, out = 0, 0

    try:
        if "anthropic" in mid:
            text = body["content"][0]["text"]
            inp  = body["usage"]["input_tokens"]
            out  = body["usage"]["output_tokens"]

        elif "meta.llama" in mid:
            text = body.get("generation", "")
            inp  = body.get("prompt_token_count", 0)
            out  = body.get("generation_token_count", 0)

        elif "amazon.nova" in mid:
            text = body["output"]["message"]["content"][0]["text"]
            inp  = body.get("usage", {}).get("inputTokens", 0)
            out  = body.get("usage", {}).get("outputTokens", 0)

        elif "qwen" in mid:
            # Try Nova-style output first, fall back to other shapes
            if "output" in body:
                text = body["output"]["message"]["content"][0]["text"]
                inp  = body.get("usage", {}).get("inputTokens", 0)
                out  = body.get("usage", {}).get("outputTokens", 0)
            elif "choices" in body:
                text = body["choices"][0]["message"]["content"]
                inp  = body.get("usage", {}).get("prompt_tokens", 0)
                out  = body.get("usage", {}).get("completion_tokens", 0)
            else:
                text = str(body)

        elif "deepseek" in mid:
            text = body["choices"][0]["message"]["content"]
            inp  = body.get("usage", {}).get("prompt_tokens", 0)
            out  = body.get("usage", {}).get("completion_tokens", 0)

        elif "google.gemma" in mid:
            text = body["output"]["message"]["content"][0]["text"]
            inp  = body.get("usage", {}).get("inputTokens", 0)
            out  = body.get("usage", {}).get("outputTokens", 0)

        elif "mistral" in mid:
            text = body["outputs"][0]["text"]
            # Mistral doesn't return counts — rough estimate
            inp  = len(TEST_PROMPT.split()) * 4 // 3
            out  = len(text.split()) * 4 // 3

    except (KeyError, IndexError, TypeError) as e:
        print(f"  ⚠  Parse warning: {e} — raw body keys: {list(body.keys())}")
        text = str(body)[:300]

    return text, inp, out


# ══════════════════════════════════════════════════════════════════════════════
# 4. COST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_cost(model_id: str, inp: int, out: int) -> float:
    for prefix, (ip, op) in PRICE_TABLE.items():
        if prefix in model_id:
            return (inp / 1_000_000 * ip) + (out / 1_000_000 * op)
    return 0.0


def project_cost(model_id: str, result: dict):
    cost  = result["cost_usd"]
    tps   = result["tok_per_sec"]
    n     = 83   # total calls in full dataset
    total = cost * n
    mins  = (result["output_tokens"] / max(tps, 1) * n) / 60

    print(f"\n{'═'*60}")
    print(f"  Cost projection — full dataset ({n} calls)")
    print(f"{'═'*60}")
    print(f"  Per call:          ${cost:.6f}")
    print(f"  Full dataset:      ${total:.4f}")
    print(f"  With batch (50%):  ${total*0.5:.4f}  ← enable in dataset script")
    print(f"  Est. time:         {mins:.0f} min")
    print(f"{'═'*60}")
    if total < 0.05:
        print(f"  ★  Essentially free — go for it")
    elif total < 0.50:
        print(f"  ✓  Very cheap — safe to run")
    else:
        print(f"  ⚠  Consider a cheaper model")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TEST A SINGLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def test_model(model_id: str, prompt: str = TEST_PROMPT) -> dict | None:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    body   = build_body(model_id, prompt)

    print(f"\n  Testing: {model_id}")
    try:
        t0       = time.time()
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        elapsed  = time.time() - t0
        rb       = json.loads(response["body"].read())
        text, inp, out = parse_response(model_id, rb)
        cost     = estimate_cost(model_id, inp, out)

        print(f"  ✓  {elapsed:.1f}s  |  {inp} in / {out} out tokens  |  ${cost:.6f}  |  {out/elapsed:.0f} tok/s")
        print(f"  Preview: {text[:200].strip().replace(chr(10), ' ')[:200]}...")

        return {
            "model_id":      model_id,
            "input_tokens":  inp,
            "output_tokens": out,
            "elapsed_s":     round(elapsed, 2),
            "cost_usd":      round(cost, 6),
            "tok_per_sec":   round(out / max(elapsed, 0.1), 1),
        }

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        if "AccessDenied" in code:
            print(f"  ✗  Access denied — enable in Bedrock console → Model access")
        elif "ValidationException" in code:
            print(f"  ✗  Validation: {msg[:120]}")
            print(f"     Tip: model may need cross-region inference prefix")
            print(f"     Try: us.{model_id}")
        else:
            print(f"  ✗  {code}: {msg[:120]}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("AWS Bedrock — Model Discovery + Cost Test")
    print(f"Region: {REGION}\n")

    # Step 1: list all models
    print("Step 1: Available models")
    models = list_available_models()
    if not models:
        print("No models found — check IAM permissions and Bedrock access.")
        exit(1)
    print_model_table(models)

    # Step 2: test recommended model
    print(f"Step 2: Testing recommended model → {RECOMMENDED_MODEL}")
    result = test_model(RECOMMENDED_MODEL)
    if result:
        project_cost(RECOMMENDED_MODEL, result)

    # Step 3: cost comparison table
    print(f"\n{'─'*68}")
    print(f"  {'MODEL PREFIX':<38} {'$/call':<14} {'~Full dataset'}")
    print(f"{'─'*68}")
    for prefix, (ip, op) in sorted(PRICE_TABLE.items(), key=lambda x: x[1][0]+x[1][1]):
        cost  = (800/1_000_000*ip) + (1200/1_000_000*op)
        total = cost * 83
        flag  = "★" if total < 0.05 else ("✓" if total < 0.50 else "")
        print(f"  {prefix:<38} ${cost:.6f}     ${total:.4f}  {flag}")
    print(f"{'─'*68}")
    print(f"  Estimate: 800 in + 1200 out tokens per call, 83 calls total")
    print(f"  ★ under $0.05   ✓ under $0.50\n")

    # Step 4: optionally test all candidates
    run_all = input("Test all candidate models? (y/N): ").strip().lower()
    if run_all == "y":
        print("\nRunning comparison...\n")
        results = []
        for mid in MODELS_TO_TEST:
            if mid == RECOMMENDED_MODEL and result:
                results.append(result)
                continue
            r = test_model(mid)
            if r:
                results.append(r)
            time.sleep(1)

        if results:
            print(f"\n{'─'*80}")
            print(f"  {'MODEL':<45} {'tok/s':<8} {'$/call':<12} {'quality hint'}")
            print(f"{'─'*80}")
            for r in sorted(results, key=lambda x: x["cost_usd"]):
                print(f"  {r['model_id']:<45} {r['tok_per_sec']:<8} ${r['cost_usd']:.6f}")
            print(f"{'─'*80}")