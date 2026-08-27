"""
train_models.py
================
Stage 5 & 6: Building Models + Model Evaluation.

Task: predict the customer's `intent` (27 classes) from the cleaned
instruction text. In the deployed system this classifier sits in front of
the LLaMA generator: {intent -> instruction-tuned prompt template -> LLaMA
response}. Accurate intent routing is what keeps a *No-RAG* LLaMA chatbot
grounded, since the correct response template/slots depend entirely on
correctly identifying what the customer is asking for.

Model 1: Multinomial Naive Bayes over TF-IDF     (fast, strong text baseline)
Model 2: Linear SVM (LinearSVC) over TF-IDF        (typically state-of-the-art
                                                      among linear text models)

Both are evaluated with Accuracy, Precision, Recall, F1 (macro & weighted)
and a confusion matrix, then compared to select the "best model".
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Resolve paths relative to THIS FILE's location, not the current working
# directory, so this works no matter where it's launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../PythonProject/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     # .../PythonProject

CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")
EDA_DIR = os.path.join(PROJECT_ROOT, "eda")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    X = df["instruction_processed"].astype(str)
    y = df["intent"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(name, y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "model": name,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return metrics, report


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run():
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    labels = sorted(y_train.unique())

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    results = []

    # ---- Model 1: Multinomial Naive Bayes -----------------------------------
    m1 = MultinomialNB(alpha=0.3)
    m1.fit(Xtr, y_train)
    pred1 = m1.predict(Xte)
    metrics1, report1 = evaluate("Model 1 - MultinomialNB (TF-IDF)", y_test, pred1, labels)
    plot_confusion(y_test, pred1, labels, "Model 1 (Naive Bayes) - Confusion Matrix",
                   os.path.join(EDA_DIR, "confusion_model1.png"))
    results.append(metrics1)

    # ---- Model 2: Linear SVM --------------------------------------------------
    m2 = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    m2.fit(Xtr, y_train)
    pred2 = m2.predict(Xte)
    metrics2, report2 = evaluate("Model 2 - LinearSVC (TF-IDF)", y_test, pred2, labels)
    plot_confusion(y_test, pred2, labels, "Model 2 (Linear SVM) - Confusion Matrix",
                   os.path.join(EDA_DIR, "confusion_model2.png"))
    results.append(metrics2)

    # ---- Comparison -------------------------------------------------------
    comp_df = pd.DataFrame(results).set_index("model")
    comp_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    comp_df[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(kind="bar", ax=ax)
    ax.set_title("Model comparison - macro-averaged metrics")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "model_comparison.png"), dpi=140)
    plt.close(fig)

    best_name = comp_df["f1_macro"].idxmax()
    best_model = m1 if "Model 1" in best_name else m2

    # ---- Interpretation: most informative tokens per class -------------------
    interp = {}
    feat_names = np.array(vectorizer.get_feature_names_out())
    if best_model is m2:
        coefs = best_model.coef_  # (n_classes, n_features)
        for i, cls in enumerate(best_model.classes_):
            top_idx = np.argsort(coefs[i])[-10:][::-1]
            interp[cls] = feat_names[top_idx].tolist()
    else:
        log_prob = best_model.feature_log_prob_
        for i, cls in enumerate(best_model.classes_):
            top_idx = np.argsort(log_prob[i])[-10:][::-1]
            interp[cls] = feat_names[top_idx].tolist()

    with open(os.path.join(MODELS_DIR, "best_model_interpretation.json"), "w") as f:
        json.dump(interp, f, indent=2)

    # ---- Persist artifacts -----------------------------------------------
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "model1_nb.pkl"), "wb") as f:
        pickle.dump(m1, f)
    with open(os.path.join(MODELS_DIR, "model2_svm.pkl"), "wb") as f:
        pickle.dump(m2, f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    with open(os.path.join(MODELS_DIR, "classification_reports.txt"), "w") as f:
        f.write("MODEL 1 - Multinomial Naive Bayes\n" + "=" * 50 + "\n")
        f.write(report1 + "\n\n")
        f.write("MODEL 2 - Linear SVM\n" + "=" * 50 + "\n")
        f.write(report2 + "\n")

    print(comp_df.round(4))
    print(f"\nBEST MODEL: {best_name}")
    print("\nSample interpretation (top tokens per intent), best model:")
    for k in list(interp.keys())[:5]:
        print(f"  {k}: {interp[k]}")

    return comp_df, best_name


if __name__ == "__main__":
    run()