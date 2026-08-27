"""
chatbot_eval.py
================
The "Llama-Based Customer-Service Chatbot - No RAG" component — OLLAMA
version for macOS, now comparing the BASE model against your FINE-TUNED
model side by side on the same held-out test sample.

ARCHITECTURE
-------------
  user message
       |
       v
  [Model 2: LinearSVC intent router]   (trained in train_models.py)
       |
       v
  instruction-tuned prompt (system + user turns, sent straight to Ollama's
  chat endpoint — NO retrieval / NO external documents at inference time
  -> "No RAG")
       |
       v
  LLaMA generator — run for BOTH:
    "base"      -> llama3.2                  (zero-shot, base Instruct model)
    "finetuned" -> llama3.2-customer-service  (LoRA/SFT fine-tuned, from
                                                llama_finetune_recipe.py)
       |
       v
  generated response (per model)
       |
       v
  automatic evaluation: correctness, relevance, helpfulness, fluency,
  hallucination / error rate — computed identically for both models so the
  comparison is apples-to-apples.

WHAT THIS SCRIPT PRODUCES
----------------------------
  models/chatbot_eval_results_base.csv        per-example results, base model
  models/chatbot_eval_results_finetuned.csv   per-example results, fine-tuned
  models/chatbot_eval_comparison.csv          mean metrics, one row per model
  eda/chatbot_model_comparison.png            bar chart, base vs fine-tuned
All four are read directly by the Streamlit app's "Chatbot Evaluation" page.

SETUP:
    ollama serve &
    ollama pull llama3.2
    ollama pull nomic-embed-text
    # llama3.2-customer-service must already exist in `ollama list` — see
    # llama_finetune_recipe.py Steps 2-5 (mlx_lm.lora -> mlx_lm.fuse ->
    # GGUF convert -> ollama create).
    pip install ollama pandas scikit-learn matplotlib
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ollama
from sklearn.model_selection import train_test_split

from data_cleaning import extract_slots, RE_SLOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Resolve paths relative to THIS FILE's location, not the current working
# directory, so this works no matter where it's launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../PythonProject/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     # .../PythonProject

CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")
EDA_DIR = os.path.join(PROJECT_ROOT, "eda")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Both models compared side by side. "finetuned" must already exist in
# `ollama list` (created via llama_finetune_recipe.py Step 5) — it is NOT
# something you `ollama pull`.
MODEL_CONFIGS = {
    "base": {
        "name": "llama3.2",
        "label": "Base Llama 3.2 (zero-shot)",
        "pull_hint": "ollama pull llama3.2",
    },
    "finetuned": {
        "name": "llama3.2-customer-service",
        "label": "Fine-tuned (LoRA / SFT)",
        "pull_hint": (
            "ollama create llama3.2-customer-service -f Modelfile "
            "(see llama_finetune_recipe.py Step 5 — this is NOT `ollama pull`)"
        ),
    },
}
EMBEDDING_MODEL = "nomic-embed-text"

SYSTEM_PROMPT = (
    "You are a helpful, concise customer-service assistant for an "
    "e-commerce company. Answer only using information the customer "
    "provides in their message (order numbers, account details, etc). "
    "Never invent order numbers, dates, or account information."
)

ACTION_MARKERS = [
    "log in", "click", "visit", "navigate", "contact", "call", "select",
    "follow these steps", "go to", "confirm", "provide", "sign in",
]


def check_ollama_ready(model_keys=None):
    """Fail fast with a clear, per-model message if Ollama isn't running or
    a required model hasn't been pulled/created yet. Always prints the full
    available/required model lists on failure, since exception messages can
    get truncated when copy-pasted, and different versions of the `ollama`
    Python package return model info in slightly different shapes (older:
    plain dicts; newer: typed Model objects)."""
    model_keys = model_keys or list(MODEL_CONFIGS.keys())
    try:
        raw = ollama.list()
    except Exception as e:
        raise RuntimeError(
            "Could not reach the Ollama server. Is it running? Start it with "
            "`ollama serve &` in a terminal, then re-run this script."
        ) from e

    # Handle both API shapes:
    #   older ollama-python: ollama.list() -> {"models": [{"model": "llama3.2:latest", ...}, ...]}
    #   newer ollama-python: ollama.list() -> ListResponse(models=[Model(model="llama3.2:latest", ...), ...])
    if isinstance(raw, dict):
        raw_models = raw.get("models", [])
    else:
        raw_models = getattr(raw, "models", [])

    def extract_name(m):
        if isinstance(m, dict):
            return m.get("model") or m.get("name")
        return getattr(m, "model", None) or getattr(m, "name", None)

    available_full = [extract_name(m) for m in raw_models]
    available_full = [a for a in available_full if a]  # drop any None
    available = {a.split(":")[0] for a in available_full}

    required = {MODEL_CONFIGS[k]["name"] for k in model_keys} | {EMBEDDING_MODEL}
    missing = [m for m in required if m.split(":")[0] not in available]

    print(f"[diagnostic] Ollama reports these models installed: {available_full}")
    print(f"[diagnostic] This script requires: {sorted(required)}")

    if missing:
        hints = []
        for m in missing:
            match = next((c for c in MODEL_CONFIGS.values() if c["name"] == m), None)
            hints.append(f"  {m}: {match['pull_hint']}" if match else f"  {m}: ollama pull {m}")
        raise RuntimeError(f"Missing Ollama model(s): {missing}\n" + "\n".join(hints))
    print(f"Ollama ready — models available: {sorted(required)}")


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------
class OllamaResponseGenerator:
    """Generates customer-service responses directly from a local LLaMA
    model via Ollama — no retrieval step at inference time (No-RAG)."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, instruction: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ],
            options={"temperature": 0.0},  # deterministic, matches greedy decoding
        )
        return response["message"]["content"].strip()

    def generate_batch(self, instructions: list, progress_every: int = 10) -> list:
        """Ollama's Python client doesn't batch server-side, so this simply
        loops with progress logging."""
        results = []
        for i, instr in enumerate(instructions, 1):
            results.append(self.generate(instr))
            if i % progress_every == 0 or i == len(instructions):
                print(f"    [{self.model_name}] generated {i}/{len(instructions)}")
        return results


# ---------------------------------------------------------------------------
# Embeddings for correctness/relevance (shared embedding model across both
# generators, so correctness/relevance scores are directly comparable)
# ---------------------------------------------------------------------------
def ollama_embed(texts: list, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    vecs = []
    for t in texts:
        resp = ollama.embeddings(model=model_name, prompt=t)
        vecs.append(resp["embedding"])
    return np.array(vecs)


def cosine_sim(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Evaluation metrics (identical for both models -> fair comparison)
# ---------------------------------------------------------------------------
def score_correctness(gen_vec, ref_vec) -> float:
    """Semantic closeness of generated vs. gold reference response (0-1)."""
    sim = cosine_sim(gen_vec, ref_vec)
    return max(0.0, min(1.0, (sim + 1) / 2))


def score_relevance(gen_vec, instr_vec) -> float:
    """Does the response semantically address the user's instruction? (0-1)"""
    sim = cosine_sim(gen_vec, instr_vec)
    return max(0.0, min(1.0, (sim + 1) / 2))


def score_helpfulness(generated: str) -> float:
    """Rule-based proxy (0-1): presence of concrete, actionable guidance."""
    text = generated.lower()
    hits = sum(1 for m in ACTION_MARKERS if m in text)
    has_steps = bool(re.search(r"\b[1-5]\.\s", generated)) or "\n1." in generated
    return min(1.0, 0.15 * hits + (0.3 if has_steps else 0) + 0.1)


def score_fluency(generated: str) -> float:
    """Rule-based proxy (0-1): lexical diversity + absence of broken repeats."""
    tokens = re.findall(r"[a-zA-Z']+", generated.lower())
    if not tokens:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    repeat_ratio = 1 - (len(set(bigrams)) / max(1, len(bigrams)))
    return max(0.0, min(1.0, 0.6 * ttr + 0.4 * (1 - repeat_ratio)))


def score_hallucination(generated: str, instruction: str) -> float:
    """Error rate (0-1, LOWER is better): fraction of {{slot}}-style
    placeholders in the generated text that were left unfilled or don't
    correspond to anything grounded in the source instruction — the model
    inventing/omitting entities instead of copying them from the prompt."""
    gen_slots = extract_slots(generated)
    if not gen_slots:
        return 0.0
    instr_slot_types = {
        re.sub(r"\{\{|\}\}", "", s).strip() for s in extract_slots(instruction)
    }
    ungrounded = sum(
        1 for s in gen_slots
        if re.sub(r"\{\{|\}\}", "", s).strip() not in instr_slot_types
    )
    return ungrounded / max(1, len(gen_slots))


def score_all(generated: str, reference: str, instruction: str,
              gen_vec, ref_vec, instr_vec) -> dict:
    return {
        "correctness": score_correctness(gen_vec, ref_vec),
        "relevance": score_relevance(gen_vec, instr_vec),
        "helpfulness": score_helpfulness(generated),
        "fluency": score_fluency(generated),
        "hallucination_rate": score_hallucination(generated, instruction),
    }


METRIC_COLS = ["correctness", "relevance", "helpfulness", "fluency", "hallucination_rate"]


# ---------------------------------------------------------------------------
# Per-model evaluation on a shared test sample
# ---------------------------------------------------------------------------
def evaluate_model(model_key: str, test_sample: pd.DataFrame,
                    ref_vecs: np.ndarray, instr_vecs: np.ndarray) -> pd.DataFrame:
    config = MODEL_CONFIGS[model_key]
    print(f"\nGenerating {len(test_sample)} responses with '{config['name']}' "
          f"({config['label']}) ...")
    generator = OllamaResponseGenerator(config["name"])
    generated = generator.generate_batch(test_sample["instruction"].tolist())

    print(f"  Embedding generated responses for '{config['name']}' ...")
    gen_vecs = ollama_embed(generated)

    rows = []
    for i, (idx, row) in enumerate(test_sample.iterrows()):
        metrics = score_all(
            generated[i], row["response"], row["instruction"],
            gen_vecs[i], ref_vecs[i], instr_vecs[i],
        )
        rows.append({
            "model": model_key,
            "instruction": row["instruction"],
            "true_intent": row["intent"],
            "predicted_intent": row["predicted_intent"],
            "reference_response": row["response"],
            "generated_response": generated[i],
            **metrics,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main evaluation run: base vs fine-tuned, same test sample
# ---------------------------------------------------------------------------
def run(sample_n: int = 100):
    """sample_n kept moderate by default — this now runs generation TWICE
    per example (once per model), so total time roughly doubles versus a
    single-model run. Time a small sample_n first."""
    os.makedirs(EDA_DIR, exist_ok=True)
    check_ollama_ready()

    df = pd.read_csv(CLEAN_PATH)

    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "model2_svm.pkl"), "rb") as f:
        intent_router = pickle.load(f)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["intent"]
    )
    test_sample = test_df.sample(
        n=min(sample_n, len(test_df)), random_state=42
    ).reset_index(drop=True)

    # Intent routing still uses the classical Model 2 (fast, no LLaMA needed)
    Xte = vectorizer.transform(test_sample["instruction_processed"].astype(str))
    test_sample["predicted_intent"] = intent_router.predict(Xte)

    # Reference/instruction embeddings computed ONCE and reused for both
    # models, so correctness/relevance are directly comparable and we don't
    # pay for redundant embedding calls.
    print("Embedding reference responses and instructions (shared across "
          "both models) ...")
    ref_vecs = ollama_embed(test_sample["response"].tolist())
    instr_vecs = ollama_embed(test_sample["instruction"].tolist())

    all_results = {}
    for model_key in MODEL_CONFIGS:
        results = evaluate_model(model_key, test_sample, ref_vecs, instr_vecs)
        all_results[model_key] = results
        results.to_csv(os.path.join(MODELS_DIR, f"chatbot_eval_results_{model_key}.csv"), index=False)

    # ---- Comparison summary -------------------------------------------
    comparison_rows = []
    for model_key, results in all_results.items():
        row = results[METRIC_COLS].mean().to_dict()
        row["model"] = MODEL_CONFIGS[model_key]["label"]
        row["model_key"] = model_key
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).set_index("model")
    comparison.to_csv(os.path.join(MODELS_DIR, "chatbot_eval_comparison.csv"))

    # ---- Comparison chart -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    comparison[METRIC_COLS].plot(kind="bar", ax=ax)
    ax.set_title("Base vs Fine-tuned LLaMA — No-RAG chatbot evaluation")
    ax.set_ylabel("score (0-1, hallucination_rate lower=better)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "chatbot_model_comparison.png"), dpi=140)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("BASE vs FINE-TUNED — mean scores over sample "
          "(0-1, hallucination_rate lower=better):")
    print(comparison[METRIC_COLS].round(4))
    print("=" * 70)
    return all_results, comparison


if __name__ == "__main__":
    run()