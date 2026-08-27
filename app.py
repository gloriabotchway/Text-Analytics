"""
app.py
======
Streamlit front-end for: "NLP/ML/Data-Viz solution for a LLaMA-based
Customer-Service Chatbot (No-RAG)".

Run with:
    pip install -r requirements.txt
    streamlit run app.py

The app is read-only over the artifacts produced by src/*.py (run
`python src/data_cleaning.py && python src/eda.py && python
src/text_representation.py && python src/train_models.py && python
src/chatbot_eval.py` once beforehand, or use the "Rebuild pipeline" button
in the sidebar).
"""

import json
import os
import pickle
import subprocess
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="LLaMA Customer-Service Chatbot Analytics", layout="wide")

# Resolve paths relative to THIS FILE's location, not the current working
# directory - this app.py lives at the project root, but this still guards
# against being launched with a different working directory (e.g. from an
# IDE run configuration), the same class of bug hit earlier with the other
# scripts in src/.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")
EDA_DIR = os.path.join(PROJECT_ROOT, "eda")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# NOTE: chatbot_eval.py is intentionally NOT in this list. It requires
# Ollama running locally with both the base and fine-tuned models pulled/
# created, which is a heavier prerequisite than the rest of the pipeline
# (pure pandas/sklearn) - it's run separately from the "Chatbot Evaluation"
# page instead, so a missing Ollama setup doesn't block the whole rebuild.
PIPELINE_STAGES = [
    ("Data cleaning", "src/data_cleaning.py"),
    ("Exploratory text analytics", "src/eda.py"),
    ("Text representation (embeddings)", "src/text_representation.py"),
    ("Model training + evaluation", "src/train_models.py"),
]


def artifacts_exist():
    return os.path.exists(DATA) and os.path.exists(f"{MODELS_DIR}/model2_svm.pkl")


@st.cache_resource
def load_artifacts():
    df = pd.read_csv(DATA)
    with open(f"{MODELS_DIR}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{MODELS_DIR}/model1_nb.pkl", "rb") as f:
        model1 = pickle.load(f)
    with open(f"{MODELS_DIR}/model2_svm.pkl", "rb") as f:
        model2 = pickle.load(f)
    comparison = pd.read_csv(f"{MODELS_DIR}/model_comparison.csv", index_col=0)
    with open(f"{MODELS_DIR}/best_model_name.txt") as f:
        best_name = f.read().strip()
    with open(f"{MODELS_DIR}/best_model_interpretation.json") as f:
        interp = json.load(f)
    return dict(
        df=df, vectorizer=vectorizer, model1=model1, model2=model2,
        comparison=comparison, best_name=best_name, interp=interp,
    )


def load_chatbot_eval_artifacts():
    """Loaded separately (not cached at startup) since these come from
    chatbot_eval.py, which requires Ollama to be running with both models
    available - that's a heavier, optional prerequisite compared to the
    rest of the app, so we don't want a missing chatbot-eval run to block
    every other page."""
    comparison_path = f"{MODELS_DIR}/chatbot_eval_comparison.csv"
    if not os.path.exists(comparison_path):
        return None
    comparison = pd.read_csv(comparison_path, index_col=0)
    results = {}
    for key in ["base", "finetuned"]:
        p = f"{MODELS_DIR}/chatbot_eval_results_{key}.csv"
        if os.path.exists(p):
            results[key] = pd.read_csv(p)
    return dict(comparison=comparison, results=results)


def run_pipeline():
    prog = st.progress(0.0, text="Starting pipeline...")
    for i, (label, script) in enumerate(PIPELINE_STAGES):
        prog.progress(i / len(PIPELINE_STAGES), text=f"Running: {label}")
        subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__) or ".", check=True)
    prog.progress(1.0, text="Pipeline complete.")
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline")
st.sidebar.markdown(
    "Data cleaning -> EDA -> Embeddings -> Model 1/2 -> Evaluation -> "
    "LLaMA (No-RAG) chatbot evaluation"
)
if st.sidebar.button("Rebuild pipeline from raw data"):
    with st.spinner("Running full pipeline (this can take a minute)..."):
        run_pipeline()
    st.sidebar.success("Pipeline rebuilt.")

if not artifacts_exist():
    st.warning(
        "No precomputed artifacts found. Click **Rebuild pipeline from raw "
        "data** in the sidebar first (raw dataset must be at "
        "`data/raw_dataset.csv`)."
    )
    st.stop()

A = load_artifacts()

page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Exploratory Text Analytics",
        "Text Representation",
        "Model Comparison",
        "Interpretation",
        "LLaMA Chatbot Demo (No-RAG)",
        "Chatbot Evaluation",
    ],
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Text Analytics for a LLaMA-Based Customer-Service Chatbot (No-RAG)")
    st.markdown(
        """
This app documents an end-to-end, evidence-based NLP/ML/data-visualisation
solution built on the Bitext customer-support instruction/response dataset.

**Pipeline:** Data cleaning (regex) -> Text preprocessing -> Exploratory
text analytics -> Text representation (TF-IDF + SVD embeddings, a
LLaMA-embedding-compatible interface) -> Two intent-classification models
-> Evaluation (precision/recall/F1) -> Model comparison -> Best model ->
Interpretation -> **LLaMA, No-RAG, instruction-tuned chatbot** response
generation and evaluation (correctness, relevance, helpfulness, fluency,
hallucination/error rate).
        """
    )
    df = A["df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Categories", df["category"].nunique())
    c3.metric("Intents", df["intent"].nunique())
    c4.metric("Best routing model", A["best_name"].split(" - ")[0])

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
elif page == "Exploratory Text Analytics":
    st.title("Exploratory Text Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{EDA_DIR}/category_distribution.png", use_container_width=True)
        st.image(f"{EDA_DIR}/text_length_distribution.png", use_container_width=True)
        st.image(f"{EDA_DIR}/slot_usage.png", use_container_width=True)
    with col2:
        st.image(f"{EDA_DIR}/intent_distribution.png", use_container_width=True)
        st.image(f"{EDA_DIR}/top_tokens.png", use_container_width=True)
    with open(f"{EDA_DIR}/eda_summary.txt") as f:
        st.text(f.read())

# ---------------------------------------------------------------------------
# Text representation
# ---------------------------------------------------------------------------
elif page == "Text Representation":
    st.title("Text Representation: Embeddings")
    st.markdown(
        """
Two representations are used:
- **TF-IDF** (sparse, unigram + bigram) - interpretable features feeding
  Model 1 and Model 2.
- **TF-IDF + Truncated SVD (LSA), 128-d** - a dense embedding used as an
  offline, dependency-free stand-in for word2vec / LLaMA token embeddings.
  The interface is designed to be swapped 1:1 for real LLaMA hidden states
  in production (see `src/llama_finetune_recipe.py`).
        """
    )
    import text_representation

    sys.modules["__main__"].TextEmbedder = text_representation.TextEmbedder
    with open(f"{MODELS_DIR}/embedder.pkl", "rb") as f:
        embedder = pickle.load(f)
    word = st.text_input("Find nearest neighbours for a token", value="refund")
    if word:
        sims = embedder.most_similar_words(word.lower(), topn=10)
        if sims:
            st.table(pd.DataFrame(sims, columns=["token", "cosine similarity"]))
        else:
            st.info("Token not in vocabulary - try another (lowercase, stemmed) word.")

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Building & Evaluation")
    st.markdown(
        "**Task:** predict customer `intent` (27 classes) from cleaned "
        "instruction text - this is the router that sits in front of the "
        "LLaMA generator."
    )
    st.dataframe(A["comparison"].round(4), use_container_width=True)
    st.image(f"{EDA_DIR}/model_comparison.png", use_container_width=True)

    st.subheader("Confusion matrices")
    col1, col2 = st.columns(2)
    col1.image(f"{EDA_DIR}/confusion_model1.png", caption="Model 1 - Naive Bayes", use_container_width=True)
    col2.image(f"{EDA_DIR}/confusion_model2.png", caption="Model 2 - Linear SVM", use_container_width=True)

    st.success(f"Best model (highest macro-F1): **{A['best_name']}**")
    with open(f"{MODELS_DIR}/classification_reports.txt") as f:
        st.text(f.read())

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
elif page == "Interpretation":
    st.title("Interpretation of the Best Model")
    st.markdown(
        "Top discriminative tokens per intent for the best model "
        f"(**{A['best_name']}**):"
    )
    intent = st.selectbox("Intent", sorted(A["interp"].keys()))
    st.write(A["interp"][intent])
    st.markdown(
        """
**Reading the model:** the top tokens are largely domain-appropriate and
non-overlapping across intents (e.g. `cancel`/`cancell` for
`cancel_order` vs. `fee`/`penalty` for `check_cancellation_fee`), which
explains the high macro-F1: the intents are lexically well separated. This
also means intent routing is a *low-risk* stage of the pipeline - most of
the residual quality risk in the full system sits downstream, in the LLaMA
generation step itself (see Chatbot Evaluation).
        """
    )

# ---------------------------------------------------------------------------
# Chatbot demo - LIVE Ollama calls, base vs fine-tuned
# ---------------------------------------------------------------------------
elif page == "LLaMA Chatbot Demo (No-RAG)":
    st.title("LLaMA Customer-Service Chatbot - No-RAG Demo")

    try:
        import ollama as ollama_client
        from chatbot_eval import (
            OllamaResponseGenerator, MODEL_CONFIGS, check_ollama_ready,
        )
        OLLAMA_IMPORT_ERROR = None
    except ImportError as e:
        OLLAMA_IMPORT_ERROR = e

    if OLLAMA_IMPORT_ERROR:
        st.error(
            "The `ollama` Python package isn't installed in this "
            "environment (`pip install ollama`), so this page can't reach "
            "your local Ollama server. Everything else in the app still "
            "works without it."
        )
        st.stop()

    st.info(
        "This calls your **local Ollama server** live - no retrieval, no "
        "external documents, just the model responding directly to the "
        "user's message (No-RAG). Requires `ollama serve` running, with "
        "both models available (see `chatbot_eval.py`'s module docstring "
        "for setup)."
    )

    try:
        check_ollama_ready()
        ollama_ready = True
    except RuntimeError as e:
        ollama_ready = False
        st.warning(str(e))

    mode = st.radio(
        "Mode", ["Single model", "Compare base vs fine-tuned"], horizontal=True
    )

    user_msg = st.text_area(
        "Customer message", value="I need help cancelling my order 370795561790"
    )

    # Intent routing (Model 2) always runs - cheap, no Ollama needed, and
    # useful context even when just chatting with the LLM directly.
    vec = A["vectorizer"].transform([user_msg.lower()])
    pred_intent = A["model2"].predict(vec)[0]
    st.caption(f"Predicted intent (Model 2 router): `{pred_intent}`")

    if mode == "Single model":
        model_key = st.selectbox(
            "Model",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda k: MODEL_CONFIGS[k]["label"],
        )
        if st.button("Generate response", disabled=not ollama_ready):
            gen = OllamaResponseGenerator(MODEL_CONFIGS[model_key]["name"])
            with st.spinner(f"Generating with {MODEL_CONFIGS[model_key]['label']}..."):
                response = gen.generate(user_msg)
            st.write("**Generated response:**")
            st.write(response)

    else:  # Compare base vs fine-tuned
        if st.button("Generate from both models", disabled=not ollama_ready):
            col1, col2 = st.columns(2)
            for col, model_key in zip([col1, col2], MODEL_CONFIGS):
                config = MODEL_CONFIGS[model_key]
                with col:
                    st.markdown(f"**{config['label']}** (`{config['name']}`)")
                    gen = OllamaResponseGenerator(config["name"])
                    with st.spinner(f"Generating with {config['label']}..."):
                        response = gen.generate(user_msg)
                    st.write(response)

# ---------------------------------------------------------------------------
# Chatbot evaluation - base vs fine-tuned comparison
# ---------------------------------------------------------------------------
elif page == "Chatbot Evaluation":
    st.title("LLaMA (No-RAG) Response-Quality Evaluation")
    st.markdown(
        "Base model vs. fine-tuned model, evaluated on the **same** "
        "held-out test sample. `correctness` and `relevance` use cosine "
        "similarity between Ollama embeddings (`nomic-embed-text`); "
        "`helpfulness` and `fluency` are rule-based proxies; "
        "`hallucination_rate` counts ungrounded `{{slot}}` placeholders "
        "(lower = better). Produced by running `python src/chatbot_eval.py` "
        "locally (requires Ollama)."
    )

    chat = load_chatbot_eval_artifacts()
    if chat is None:
        st.warning(
            "No chatbot evaluation results found yet. Run "
            "`python src/chatbot_eval.py` in your terminal (with Ollama "
            "serving both `llama3.2` and `llama3.2-customer-service`) to "
            "generate `models/chatbot_eval_comparison.csv`, then reload "
            "this page."
        )
        st.stop()

    st.subheader("Base vs Fine-tuned - mean scores")
    st.dataframe(chat["comparison"].round(4), use_container_width=True)

    chart_path = f"{EDA_DIR}/chatbot_model_comparison.png"
    if os.path.exists(chart_path):
        st.image(chart_path, use_container_width=True)

    st.subheader("Sample generations")
    model_choice = st.selectbox(
        "Model", options=list(chat["results"].keys()),
        format_func=lambda k: {"base": "Base Llama 3.2", "finetuned": "Fine-tuned"}.get(k, k),
    )
    if model_choice in chat["results"]:
        st.dataframe(
            chat["results"][model_choice][
                ["instruction", "true_intent", "predicted_intent",
                 "correctness", "relevance", "helpfulness", "fluency", "hallucination_rate"]
            ].head(50),
            use_container_width=True,
        )
