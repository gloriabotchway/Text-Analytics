# Text Analytics for a LLaMA-Based Customer-Service Chatbot (No-RAG)

**Live app:** https://text-analytics-cs2r7jcnfkfdf7izhnhheq.streamlit.app/

An end-to-end, evidence-based NLP and machine learning pipeline built on the [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — from raw text to a locally fine-tuned LLaMA model, deployed as an interactive Streamlit dashboard.

## What this project does

1. **Cleans and preprocesses** 26,872 customer-support instruction/response pairs using regular expressions and lightweight NLP.
2. **Explores** the corpus: class balance, text length, vocabulary, and slot-placeholder usage.
3. **Represents text** with TF-IDF and a dense TF-IDF+SVD embedding designed as a drop-in stand-in for LLaMA-style hidden-state embeddings.
4. **Builds and compares two intent classifiers** (Naive Bayes vs. Linear SVM) that route a customer's message to 1 of 27 intents.
5. **Fine-tunes a LLaMA model with LoRA** (Low-Rank Adaptation) — entirely locally, via [MLX](https://github.com/ml-explore/mlx), [llama.cpp](https://github.com/ggml-org/llama.cpp), and [Ollama](https://ollama.com) — to build a **No-RAG** chatbot: no retrieval step, no vector store, all domain knowledge baked into the model's weights.
6. **Evaluates the fine-tuned model against the base model** on correctness, relevance, helpfulness, fluency, and hallucination rate.

## Key result

The fine-tuned model is a **1-billion-parameter** LLaMA, compared against the **3-billion-parameter** base model:

| Model | Correctness | Relevance | Helpfulness | Fluency | Hallucination Rate |
|---|---|---|---|---|---|
| Base Llama 3.2 (3B, zero-shot) | 0.894 | 0.858 | 0.277 | 0.857 | **0.010** |
| Fine-tuned (1B, LoRA/SFT) | **0.931** | **0.872** | **0.367** | 0.779 | 0.168 |

The smaller, domain-specialised model beat the larger general-purpose model on correctness, relevance, and helpfulness — but at the cost of a much higher hallucination rate. This is the central finding of the project: **targeted fine-tuning can rival raw model scale, but a No-RAG architecture has no retrieval fallback to catch the hallucinations that trade-off introduces.**

Full methodology, discussion, and limitations are in the accompanying technical report.

## Project structure

```
data/                raw and cleaned datasets
eda/                 exploratory charts (PNG) + chatbot comparison chart
models/              trained classifiers, vectorizer, evaluation results (CSV/JSON/pkl)
src/
  data_cleaning.py           Stage 1-2: regex cleaning + preprocessing
  eda.py                     Stage 3: exploratory text analytics
  text_representation.py     Stage 4: TF-IDF + SVD embeddings
  train_models.py            Stage 5-8: Model 1/2, evaluation, comparison, interpretation
  llama_finetune_recipe.py   Builds the LoRA fine-tuning dataset + prints MLX/Ollama CLI steps
  chatbot_eval.py            Base vs. fine-tuned chatbot evaluation via Ollama
app.py                       Streamlit dashboard (all 7 sections)
requirements.txt
```

## Running it yourself

**Cloud-deployed pages** (Overview, EDA, Text Representation, Model Comparison, Interpretation, Chatbot Evaluation) work out of the box from the committed artifacts — no setup needed beyond `pip install -r requirements.txt` and `streamlit run app.py`.

**The live "LLaMA Chatbot Demo" page** requires [Ollama](https://ollama.com) running locally with:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
plus the fine-tuned model, built via the LoRA recipe in `src/llama_finetune_recipe.py` (requires [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon, or an equivalent fine-tuning setup elsewhere).

To rebuild the full pipeline from the raw dataset:
```bash
python src/data_cleaning.py
python src/eda.py
python src/text_representation.py
python src/train_models.py
python src/chatbot_eval.py   # requires Ollama, see above
streamlit run app.py
```

## Tech stack

Python · pandas · scikit-learn · Streamlit · MLX · llama.cpp · Ollama

## Author

Gloria Botchway, University of Ghana, Department of Mathematics— August 2026
