# ── Agentic AI + ML/DL/GenAI Dataset Generator (AWS Bedrock + Claude Haiku) ──
# Validated working on EC2 with instance IAM role auth.
# Model: anthropic.claude-3-haiku-20240307-v1:0
#   - 97 tok/s (measured), $0.000820/call, ~$0.14 for full 83-call dataset
#   - Produces correct, immediately runnable Python code
#
# Setup on EC2:
#   pip install boto3
#   (IAM role handles auth — no aws configure needed)
#
# IAM permissions needed:
#   bedrock-runtime:InvokeModel
#
# Run:
#   python dataset_generator_bedrock.py

import boto3
import json
import time
import os
from pathlib import Path
from botocore.exceptions import ClientError

# ── CONFIG ────────────────────────────────────────────────────────────────────
REGION     = "us-east-1"
MODEL_ID   = "anthropic.claude-3-haiku-20240307-v1:0"
MAX_TOKENS = 3000
TEMPERATURE = 0.25   # low = deterministic, consistent code
DELAY_BETWEEN    = 1.0   # seconds between single-turn calls
MULTI_TURN_DELAY = 1.5   # seconds between turns in a conversation

Path("./data/raw").mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BEDROCK CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def get_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def call_bedrock(messages: list[dict], system: str, retries: int = 3) -> str | None:
    """
    Call Claude Haiku on Bedrock using the Anthropic messages format.
    Retries on throttling with exponential backoff.
    """
    client = get_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system,
        "messages": messages,
    }

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            rb = json.loads(response["body"].read())
            return rb["content"][0]["text"].strip()

        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]

            if code == "ThrottlingException":
                wait = (2 ** attempt) * 5   # 5s, 10s, 20s
                print(f"  ⏳ Throttled — waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)

            elif code == "ValidationException":
                print(f"  ✗ Validation error: {msg[:120]}")
                return None

            elif "AccessDenied" in code:
                print(f"  ✗ Access denied — enable model in Bedrock console → Model access")
                return None

            else:
                print(f"  ✗ {code}: {msg[:120]} — attempt {attempt+1}/{retries}")
                time.sleep(3)

        except (KeyError, json.JSONDecodeError) as e:
            print(f"  ✗ Bad response format: {e}")
            return None

    print("  ✗ All retries exhausted — skipping")
    return None


def preflight_check() -> bool:
    """Send one token to confirm auth + model access before the full run."""
    print(f"Checking Bedrock access → {MODEL_ID} in {REGION}...")
    result = call_bedrock(
        messages=[{"role": "user", "content": "Reply with only the word: OK"}],
        system="You are a test assistant.",
        retries=1,
    )
    if result:
        print(f"  ✓ Model responding: '{result[:30]}'\n")
        return True
    print("  ✗ Preflight failed — check IAM role and Bedrock model access\n")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SINGLE_TURN_SYSTEM = """You are an elite AI/ML engineer specializing in:
- Agentic AI systems (LangChain, LlamaIndex, AutoGen, custom agents)
- Machine Learning (sklearn, XGBoost, LightGBM, feature engineering)
- Deep Learning (PyTorch, custom architectures, training loops)
- Generative AI (HuggingFace, LoRA fine-tuning, RAG, vector databases)

When writing code:
1. Write complete, immediately runnable Python code
2. Add clear inline comments explaining WHY not just WHAT
3. Use type hints throughout
4. Include proper error handling
5. Add a working example in if __name__ == '__main__'
6. Follow best practices for the specific framework
7. Make code production-ready, not just tutorial-quality"""

MULTI_TURN_SYSTEM = """You are an elite AI engineer in an ongoing technical conversation
building a complex AI system step by step.
Each response must:
- Build naturally on previous code in this conversation
- Reuse the same variable names, class names, and architecture
- Be complete and immediately runnable
- Add inline comments explaining key decisions"""


# ══════════════════════════════════════════════════════════════════════════════
# TASK TAXONOMY — 4 Domains × Multiple Task Types
# ══════════════════════════════════════════════════════════════════════════════

TAXONOMY = {

    # ── DOMAIN 1: Agentic AI ─────────────────────────────────────────────────
    "agentic_react": [
        "Build a ReAct agent from scratch in Python without any frameworks. It should have a thought/action/observation loop, support multiple tools (calculator, string reversal, word counter), handle errors, and stop when it reaches a final answer.",
        "Implement a ReAct agent using LangChain that can answer multi-step questions. Include tools for Wikipedia search, Python REPL, and a custom dictionary lookup. Add reasoning traces and tool call logging.",
        "Write a ReAct agent that autonomously debugs Python code. It identifies the error, searches for solutions, applies fixes, re-runs the code, and iterates until tests pass.",
    ],
    "agentic_langchain": [
        "Build a LangChain agent with custom tools for a data science workflow: load CSV files, run pandas operations, create matplotlib charts, and generate statistical summaries.",
        "Create a LangChain multi-tool research assistant agent that searches the web, summarizes documents, extracts key facts, and compiles a structured report. Include conversation memory.",
        "Implement a LangChain agent with long-term memory using ChromaDB. The agent stores conversation summaries and retrieves relevant context from past conversations.",
        "Write a LangChain agent pipeline that takes a user's ML problem, selects the appropriate algorithm, writes the training code, and explains the approach.",
    ],
    "agentic_llamaindex": [
        "Build a RAG pipeline using LlamaIndex that ingests PDF documents, creates a vector index with HuggingFace embeddings, and answers questions with source citations. Include metadata filtering.",
        "Implement a LlamaIndex agent with multiple document indexes for PDFs, web pages, and CSV files. Route queries to the appropriate index and synthesize answers from multiple sources.",
        "Create a LlamaIndex sub-question query engine that breaks complex questions into sub-questions, retrieves context for each, and synthesizes a final answer.",
    ],
    "agentic_autogen": [
        "Create an AutoGen multi-agent system for code review with a UserProxy, CodeWriter, CodeReviewer, and TestWriter. Agents debate and improve code through multiple rounds.",
        "Build an AutoGen research team with 4 agents: Planner, Researcher, Analyst, and Writer. Each agent has a specific role and they collaborate to produce a research report.",
        "Implement an AutoGen agent system that autonomously runs an ML experiment: one agent prepares data, one trains the model, one evaluates, and one decides to stop or continue.",
    ],
    "agentic_tool_calling": [
        "Write a function-calling agent using an OpenAI-compatible API with tools for file operations, shell commands, and HTTP requests. Include error handling and tool result validation.",
        "Implement a tool-calling agent that manages ML experiments: tools to load data, train models, log metrics to MLflow, compare experiments, and select the best model.",
        "Build a structured output agent using Pydantic that extracts ML experiment metadata from unstructured text and returns validated typed Python objects.",
    ],
    "agentic_memory": [
        "Implement a hierarchical memory system for an AI agent: working memory (last 5 turns as a deque), episodic memory (ChromaDB for past conversations), and semantic memory (a facts dictionary).",
        "Build a self-improving agent that stores successful code solutions in a FAISS vector store and retrieves similar past solutions when it encounters new problems.",
    ],
    "agentic_multiagent": [
        "Build a supervisor-worker multi-agent system where a supervisor decomposes a data science task and delegates to specialized workers: DataCleaner, FeatureEngineer, ModelTrainer, Evaluator.",
        "Create a debate-style multi-agent system where two agents argue for deep learning vs gradient boosting and a judge agent decides based on their code and arguments.",
        "Implement a 3-agent code generation pipeline: Architect designs, Coder writes, Critic reviews. Agents iterate until the critic approves or max rounds is reached.",
    ],

    # ── DOMAIN 2: Machine Learning ────────────────────────────────────────────
    "ml_pipeline": [
        "Build a complete sklearn pipeline for binary classification: load data, preprocessing with ColumnTransformer, model selection with GridSearchCV, and full evaluation metrics including ROC curve.",
        "Write an end-to-end sklearn Pipeline with ColumnTransformer for house price prediction. Include polynomial features, feature selection, and model stacking with a meta-learner.",
        "Create an MLflow experiment tracking setup comparing XGBoost, LightGBM, RandomForest, and Logistic Regression. Log params, metrics, and artifacts, then register the best model.",
    ],
    "ml_feature_engineering": [
        "Write a time series feature engineering pipeline: lag features, rolling mean/std/min/max, Fourier transforms for seasonality, and target encoding with leave-one-out cross-validation.",
        "Implement automated feature selection combining mutual information scores, RFECV, and SHAP values. Visualize feature importance and remove redundant features.",
        "Build a FeatureStore class that computes, caches with joblib, and serves features for train and inference, with built-in leakage prevention.",
    ],
    "ml_hyperparameter": [
        "Set up Optuna hyperparameter optimization for LightGBM with MedianPruner, 50 trials, and 15 hyperparameters. Plot optimization history, param importances, and retrain with best params.",
        "Implement Bayesian optimization with scikit-optimize for a PyTorch MLP. Compare convergence against random search and grid search with the same evaluation budget.",
    ],
    "ml_advanced": [
        "Implement stacking ensemble of XGBoost, LightGBM, CatBoost, and MLP using out-of-fold predictions for meta-features. Train a logistic regression meta-learner.",
        "Build an anomaly detection system combining Isolation Forest, Local Outlier Factor, and an Autoencoder. Fuse scores with weighted voting, evaluate with precision-recall AUC.",
        "Write a multi-label classification pipeline using classifier chains with LightGBM base classifiers. Handle class imbalance with custom class weights.",
    ],

    # ── DOMAIN 3: Deep Learning ───────────────────────────────────────────────
    "dl_architectures": [
        "Implement a Transformer encoder from scratch in PyTorch: multi-head self-attention with scaled dot-product, sinusoidal positional encoding, feed-forward sublayer, and pre-norm residual connections.",
        "Build a CNN with squeeze-and-excitation channel attention blocks in PyTorch for multi-class image classification. Include gradient checkpointing and model summary.",
        "Create a DDPM diffusion model in PyTorch with a simple U-Net backbone (down/mid/up blocks), cosine noise schedule, and iterative denoising sampling loop.",
    ],
    "dl_training": [
        "Write a production PyTorch training loop: AMP mixed precision with GradScaler, gradient accumulation, cosine annealing with linear warmup, early stopping with patience, and checkpoint management.",
        "Implement PyTorch DDP training: process group init, DistributedSampler, SyncBatchNorm conversion, and cleanup. Show how to launch with torchrun across 2 GPUs.",
        "Build a PyTorch Trainer class with pluggable callbacks: LRFinder, MixupAugmentation, LabelSmoothing, ProgressBar, and WandBLogger.",
    ],
    "dl_custom": [
        "Implement Focal Loss in PyTorch with alpha and gamma, verify gradients with torch.autograd.gradcheck, and integrate into a training pipeline for imbalanced classification.",
        "Write knowledge distillation in PyTorch: combined loss of hard cross-entropy and soft KL divergence with temperature scaling, and a training loop for teacher-student transfer.",
        "Implement SimCLR contrastive learning in PyTorch: NT-Xent loss with in-batch negatives, ResNet backbone, projection head, augmentation pipeline, and linear evaluation.",
    ],
    "dl_debugging": [
        "Write PyTorch training diagnostics: plot gradient flow (mean abs grad per layer), activation histograms with forward hooks, weight norm curves, and detect dead ReLU neurons.",
        "Build a PyTorch profiler wrapper using torch.profiler that captures CPU/GPU time, memory, and FLOPs per layer. Output a ranked bottleneck table.",
    ],

    # ── DOMAIN 4: Generative AI ───────────────────────────────────────────────
    "genai_finetuning": [
        "Write a complete QLoRA SFT script: load a 7B model in 4-bit NF4, inject LoRA into all attention and MLP projections, apply chat template with loss masking on prompt tokens, train with SFTTrainer.",
        "Implement DPO fine-tuning with TRL: load chosen/rejected preference pairs, configure DPOTrainer with beta=0.1, log reward margins to W&B, and run evaluation.",
        "Build a reward model for RLHF: load a pretrained LM, add a scalar regression head, train on pairwise preferences with Bradley-Terry loss, and evaluate ranking accuracy.",
    ],
    "genai_rag": [
        "Build a RAG pipeline with LangChain + ChromaDB + HuggingFace BGE embeddings: recursive text splitting, batch embedding, hybrid BM25+dense retrieval, and RAGAS faithfulness evaluation.",
        "Implement self-RAG: a router LLM decides if retrieval is needed, a retriever fetches top-k chunks, and a grader LLM scores chunk relevance before generation.",
        "Build an agentic RAG with LlamaIndex RouterQueryEngine: routes to SQL query engine for structured data and VectorStoreIndex for unstructured docs based on query classification.",
    ],
    "genai_vectordb": [
        "Implement a Qdrant pipeline: create a collection with cosine metric, batch upsert points with payloads, filtered vector search, and a FastAPI wrapper with search and insert endpoints.",
        "Build a FAISS vector store: IVF-PQ index for large-scale search, batch add with ID mapping, serialization to disk, and a Python class wrapping add/search/delete.",
        "Create a hybrid search system: dense retrieval with sentence-transformers + BM25 sparse retrieval, combine with Reciprocal Rank Fusion, and rerank with a cross-encoder.",
    ],
    "genai_evaluation": [
        "Build an LLM evaluation suite: BLEU/ROUGE for summarization, RAGAS (faithfulness + answer relevancy + context recall) for RAG, and an LLM-as-judge with a rubric scoring 1-5.",
        "Implement automated red-teaming: generate adversarial prompts across 5 categories (jailbreak, prompt injection, bias, hallucination, data extraction), test a target LLM, and report attack success rates.",
    ],
    "genai_inference": [
        "Write a vLLM server client with async request batching, SSE streaming support, connection pooling, and a FastAPI frontend with API key auth and per-user rate limiting.",
        "Implement speculative decoding: use a small draft model to propose K tokens, verify with the large target model in one forward pass, accept/reject per token, and track acceptance rate.",
    ],

    # ── CROSS-DOMAIN: Multi-turn ──────────────────────────────────────────────
    "cross_domain_multiturn": [
        {
            "turns": [
                ("user", "I want to build an AI agent that automatically analyzes a CSV dataset and recommends the best ML model. What's the architecture?"),
                ("assistant", "PLACEHOLDER"),
                ("user", "Write the data_analysis_tool function the agent will call."),
                ("assistant", "PLACEHOLDER"),
                ("user", "Now write the model_selection_tool that trains and compares 3 models."),
                ("assistant", "PLACEHOLDER"),
                ("user", "Wire both tools into a LangChain agent with a ReAct prompt."),
                ("assistant", "PLACEHOLDER"),
            ]
        },
        {
            "turns": [
                ("user", "I want to build a RAG system over my MLflow experiment logs so I can ask questions like 'which run had the best F1?'. How do I start?"),
                ("assistant", "PLACEHOLDER"),
                ("user", "Write the MLflow log ingestion and ChromaDB indexing code."),
                ("assistant", "PLACEHOLDER"),
                ("user", "Now write the RAG query function with source citation."),
                ("assistant", "PLACEHOLDER"),
                ("user", "Wrap it all in a LangChain agent with memory so it remembers context."),
                ("assistant", "PLACEHOLDER"),
            ]
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_single_turn(instruction: str, category: str) -> dict | None:
    messages = [{"role": "user", "content": instruction}]
    code = call_bedrock(messages, system=SINGLE_TURN_SYSTEM)
    if not code:
        return None
    return {
        "messages": [
            {"role": "system",    "content": SINGLE_TURN_SYSTEM},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": code},
        ],
        "category": category,
        "source":   "bedrock_claude-3-haiku_synthetic",
        "turns":    1,
    }


def generate_multi_turn(turn_template: dict, category: str) -> dict | None:
    """
    Fill in PLACEHOLDER turns sequentially.
    Each completed assistant turn is fed back as context for the next.
    """
    turns           = turn_template["turns"]
    messages_so_far = []   # grows as turns are completed
    final_messages  = [{"role": "system", "content": MULTI_TURN_SYSTEM}]

    for role, content in turns:
        if role == "user":
            messages_so_far.append({"role": "user", "content": content})
            final_messages.append( {"role": "user", "content": content})

        elif role == "assistant" and content == "PLACEHOLDER":
            reply = call_bedrock(messages_so_far, system=MULTI_TURN_SYSTEM)
            if not reply:
                print("  ✗ Failed on a turn — skipping conversation")
                return None
            messages_so_far.append({"role": "assistant", "content": reply})
            final_messages.append( {"role": "assistant", "content": reply})
            time.sleep(MULTI_TURN_DELAY)

    return {
        "messages": final_messages,
        "category": category,
        "source":   "bedrock_claude-3-haiku_synthetic_multiturn",
        "turns":    sum(1 for r, _ in turns if r == "assistant"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — resumable generation loop
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(output_path: str = "./data/raw/synthetic_bedrock_haiku.jsonl"):
    if not preflight_check():
        return

    # Resume: load already-generated instruction keys
    existing: set[str] = set()
    if Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex  = json.loads(line)
                    key = next(m["content"] for m in ex["messages"] if m["role"] == "user")
                    existing.add(key[:80])
                except Exception:
                    pass
        if existing:
            print(f"Resuming — {len(existing)} examples already saved\n")

    total, success, skipped = 0, 0, 0

    with open(output_path, "a", encoding="utf-8") as f:
        for category, items in TAXONOMY.items():
            is_multiturn = (category == "cross_domain_multiturn")
            print(f"\n── {category.upper()} {'(multi-turn)' if is_multiturn else ''} ──")

            if is_multiturn:
                for turn_template in items:
                    first_user = turn_template["turns"][0][1]
                    if first_user[:80] in existing:
                        print("  ↷ Already done — skipping")
                        skipped += 1; total += 1
                        continue

                    n = sum(1 for r, _ in turn_template["turns"] if r == "assistant")
                    print(f"  Generating {n}-turn conversation...")
                    example = generate_multi_turn(turn_template, category)
                    if example:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        f.flush()
                        success += 1
                        print("  ✓ Saved")
                    total += 1
                    time.sleep(MULTI_TURN_DELAY)

            else:
                for instruction in items:
                    if instruction[:80] in existing:
                        skipped += 1; total += 1
                        continue

                    example = generate_single_turn(instruction, category)
                    if example:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        f.flush()
                        success += 1
                        print(f"  ✓ {instruction[:70]}...")
                    total += 1
                    time.sleep(DELAY_BETWEEN)

    print(f"\n{'='*60}")
    print(f"Done!  Generated={success}  Skipped={skipped}  Failed={total-success-skipped}")
    print(f"Output: {output_path}")
    print(f"\nNext steps:")
    print(f"  python data_preparation.py  (clean + split)")
    print(f"  python preprocessing.py     (tokenize)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic AI Dataset Generator — Bedrock Claude 3 Haiku")
    parser.add_argument("--num-examples",  type=int,   default=83,
                        help="Target number of examples to generate")
    parser.add_argument("--temperature",   type=float, default=TEMPERATURE,
                        help="Generation temperature (0.25=deterministic, 0.7=diverse)")
    parser.add_argument("--output",        default="./data/raw/synthetic_bedrock_haiku.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--domains",       nargs="*",  default=[],
                        help="Restrict to specific taxonomy categories (default: all)")
    parser.add_argument("--resume",        action="store_true", default=True,
                        help="Skip already-generated examples (default: True)")
    parser.add_argument("--no-resume",     action="store_true",
                        help="Regenerate everything from scratch")
    args = parser.parse_args()

    # Apply CLI overrides
    TEMPERATURE = args.temperature

    per_call = 0.000820
    n_calls  = args.num_examples
    print("Agentic AI Dataset Generator — Bedrock Claude 3 Haiku")
    print(f"Model:       {MODEL_ID}")
    print(f"Region:      {REGION}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Target:      {n_calls} examples")
    print(f"Est. cost:   ${per_call * n_calls:.4f} (~${per_call * n_calls:.2f} total)")
    print(f"Est. speed:  ~97 tok/s  (~{max(1, n_calls // 6)} min total)\n")

    resume = args.resume and not args.no_resume
    generate_dataset(output_path=args.output)
