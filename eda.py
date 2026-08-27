"""
eda.py
======
Stage 3: Exploratory Text Analytics.
Produces summary tables + PNG charts saved to eda/.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# Resolve paths relative to THIS FILE's location, not the current working
# directory, so this works no matter where it's launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../PythonProject/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     # .../PythonProject

CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "eda")


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)


    # ---- 1. Category / intent distribution ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    df["category"].value_counts().plot(kind="bar", ax=ax, color="#3b6ea5")
    ax.set_title("Ticket volume by category")
    ax.set_ylabel("count")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/category_distribution.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    df["intent"].value_counts().plot(kind="barh", ax=ax, color="#4c8c4a")
    ax.set_title("Ticket volume by intent (27 classes)")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/intent_distribution.png", dpi=140)
    plt.close(fig)

    # ---- 2. Text length distributions ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["instr_word_len"], bins=30, color="#3b6ea5")
    axes[0].set_title("Instruction length (words)")
    axes[1].hist(df["resp_word_len"], bins=30, color="#4c8c4a")
    axes[1].set_title("Response length (words)")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/text_length_distribution.png", dpi=140)
    plt.close(fig)

    # ---- 3. Most frequent tokens overall ------------------------------------
    all_tokens = " ".join(df["instruction_processed"].astype(str)).split()
    common = Counter(all_tokens).most_common(25)
    words, counts = zip(*common)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(words[::-1], counts[::-1], color="#a5583b")
    ax.set_title("Top 25 tokens in customer instructions (post-cleaning)")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/top_tokens.png", dpi=140)
    plt.close(fig)

    # ---- 4. Slot / placeholder usage (grounding signal for hallucination) --
    slot_stats = df["num_slots"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    slot_stats.plot(kind="bar", ax=ax, color="#8a3ba5")
    ax.set_title("Number of {{slot}} placeholders per instruction")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/slot_usage.png", dpi=140)
    plt.close(fig)

    # ---- 5. Vocabulary richness per category --------------------------------
    vocab_by_cat = (
        df.groupby("category")["instruction_processed"]
        .apply(lambda s: len(set(" ".join(s).split())))
        .sort_values(ascending=False)
    )

    summary = {
        "n_rows": len(df),
        "n_categories": df["category"].nunique(),
        "n_intents": df["intent"].nunique(),
        "avg_instruction_words": round(df["instr_word_len"].mean(), 2),
        "avg_response_words": round(df["resp_word_len"].mean(), 2),
        "pct_instructions_with_slot": round((df["num_slots"] > 0).mean() * 100, 1),
        "vocab_size_raw": len(set(all_tokens)),
    }

    with open(f"{OUT_DIR}/eda_summary.txt", "w") as f:
        f.write("EXPLORATORY TEXT ANALYTICS SUMMARY\n")
        f.write("=" * 40 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop tokens:\n")
        for w, c in common:
            f.write(f"  {w}: {c}\n")
        f.write("\nVocabulary richness by category (unique tokens):\n")
        f.write(vocab_by_cat.to_string())

    print("EDA complete. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    run()