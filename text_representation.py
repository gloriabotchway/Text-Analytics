"""
text_representation.py
=======================
Stage 4: Text Representation / Word Embeddings.

The target deployment model is LLaMA (a sub-word / BPE-tokenised transformer),
so this module builds two complementary representations that are used
throughout the rest of the pipeline:

1. TF-IDF (sparse, interpretable) - used to train the two classical
   classifiers (Model 1 / Model 2) that sit in front of the LLaMA generator
   as an intent router.

2. Dense embeddings via Truncated SVD on the TF-IDF term-document matrix
   (Latent Semantic Analysis). This is a fully offline, dependency-free
   stand-in for trainable word/sentence embeddings (word2vec / LLaMA hidden
   states) which cannot be produced in this sandbox because it has no
   internet access to download gensim's word2vec or the actual LLaMA
   tokenizer/weights. The interface (`embed(texts)`) is written so that in a
   production environment it can be swapped 1:1 for:
       - gensim Word2Vec / fastText, or
       - the mean-pooled last-hidden-state of the LLaMA tokenizer/model
         (e.g. `AutoModel.from_pretrained("meta-llama/Llama-3.1-8B")`)
   without changing any downstream code.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Resolve paths relative to THIS FILE's location, not the current working
# directory, so this works no matter where it's launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../PythonProject/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     # .../PythonProject

CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBED_DIM = 128


class TextEmbedder:
    """TF-IDF + SVD dense embedder with a word2vec/LLaMA-compatible API."""

    def __init__(self, n_components: int = EMBED_DIM):
        self.tfidf = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, texts):
        tfidf_mat = self.tfidf.fit_transform(texts)
        self.svd.fit(tfidf_mat)
        return self

    def transform_tfidf(self, texts):
        return self.tfidf.transform(texts)

    def embed(self, texts):
        """Return dense (n_samples, EMBED_DIM) semantic vectors."""
        tfidf_mat = self.tfidf.transform(texts)
        return self.svd.transform(tfidf_mat)

    def most_similar_words(self, word, topn=10):
        """Nearest neighbours of a token in the learned embedding space."""
        vocab = self.tfidf.get_feature_names_out()
        if word not in vocab:
            return []
        idx = list(vocab).index(word)
        word_vecs = self.svd.components_.T  # (vocab, n_components)
        target = word_vecs[idx]
        norms = np.linalg.norm(word_vecs, axis=1) * np.linalg.norm(target) + 1e-9
        sims = word_vecs @ target / norms
        top_idx = np.argsort(-sims)[1 : topn + 1]
        return [(vocab[i], float(sims[i])) for i in top_idx]


def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)
    texts = df["instruction_processed"].astype(str).tolist()

    embedder = TextEmbedder(n_components=EMBED_DIM)
    embedder.fit(texts)

    with open(os.path.join(MODELS_DIR, "embedder.pkl"), "wb") as f:
        pickle.dump(embedder, f)

    explained = embedder.svd.explained_variance_ratio_.sum()
    print(f"TF-IDF vocab size      : {len(embedder.tfidf.vocabulary_)}")
    print(f"SVD embedding dim      : {EMBED_DIM}")
    print(f"Variance explained     : {explained:.3f}")

    for w in ["order", "refund", "password", "cancel", "invoice"]:
        sims = embedder.most_similar_words(w, topn=5)
        print(f"nearest to '{w}': {sims}")

    return embedder


if __name__ == "__main__":
    run()