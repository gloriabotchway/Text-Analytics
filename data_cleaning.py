"""
data_cleaning.py
=================
Stage 1 & 2 of the pipeline: Data Cleaning (Regular Expressions) and Text Preprocessing.

Input : data/raw_dataset.csv   (Bitext customer-support instruction/response corpus)
Output: data/cleaned_dataset.csv
"""

import re
import string
import os
import pandas as pd

# Resolve paths relative to THIS FILE's location, not the current working
# directory — this makes the script work correctly no matter where it's
# launched from (PyCharm's default working dir, a terminal in a different
# folder, double-clicking the file, etc.).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../PythonProject/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     # .../PythonProject

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw_dataset.csv")
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")

# ---------------------------------------------------------------------------
# 1. Regex-based cleaning rules
# ---------------------------------------------------------------------------
# Bitext instructions contain templated slots such as {{Order Number}}.
# These are DELIBERATELY preserved (not stripped) because they are the exact
# grounding slots a "No-RAG" instruction-tuned LLaMA model must learn to
# reproduce verbatim in its response (this is central to the hallucination
# check later in the pipeline). Everything else is normalised.

RE_URL          = re.compile(r"http[s]?://\S+|www\.\S+")
RE_EMAIL        = re.compile(r"\S+@\S+\.\S+")
RE_MULTISPACE   = re.compile(r"\s+")
RE_NON_ALNUM    = re.compile(r"[^a-zA-Z0-9{}\s]")          # keep {{ }} slot braces
RE_REPEATED_CH  = re.compile(r"(.)\1{2,}")                  # soooo -> soo
RE_SLOT         = re.compile(r"\{\{.*?\}\}")                 # {{Order Number}}


def extract_slots(text: str):
    """Return the list of {{Slot}} placeholders found in a string."""
    return RE_SLOT.findall(str(text))


def regex_clean(text: str) -> str:
    """Deterministic, rule-based cleaning (Stage 1)."""
    text = str(text)
    text = RE_URL.sub(" URLTOKEN ", text)
    text = RE_EMAIL.sub(" EMAILTOKEN ", text)
    text = RE_REPEATED_CH.sub(r"\1\1", text)          # collapse elongated chars
    # temporarily protect {{slots}} from the alnum stripper
    slots = RE_SLOT.findall(text)
    for i, s in enumerate(slots):
        text = text.replace(s, f" __SLOT{i}__ ")
    text = RE_NON_ALNUM.sub(" ", text)
    for i, s in enumerate(slots):
        text = text.replace(f"__SLOT{i}__", s)
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Lightweight preprocessing (tokenise / lowercase / stopword removal /
#    light stemming) implemented WITHOUT nltk/spacy because the execution
#    sandbox has no internet access to download corpora. A compact, curated
#    English stopword list and a Porter-style suffix stripper are used
#    instead so the whole pipeline is fully self-contained and reproducible.
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves
""".split())

SUFFIXES = ["ing", "edly", "ed", "ly", "es", "s"]


def simple_stem(word: str) -> str:
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def preprocess(text: str, remove_stopwords: bool = True, stem: bool = True) -> str:
    """Stage 2: lowercase -> tokenise -> stopword removal -> stemming."""
    text = text.lower()
    tokens = text.split()
    out = []
    for tok in tokens:
        if tok.startswith("{{") or "slottoken" in tok:
            out.append(tok)
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(simple_stem(tok) if stem else tok)
    return " ".join(out)


def run():
    df = pd.read_csv(RAW_PATH)
    before_rows = len(df)

    # Drop exact duplicate instruction/response pairs
    df = df.drop_duplicates(subset=["instruction", "response"]).reset_index(drop=True)
    df = df.dropna(subset=["instruction", "response", "intent", "category"])

    df["slots"] = df["instruction"].apply(extract_slots)
    df["instruction_clean_regex"] = df["instruction"].apply(regex_clean)
    df["response_clean_regex"] = df["response"].apply(regex_clean)

    df["instruction_processed"] = df["instruction_clean_regex"].apply(preprocess)
    df["response_processed"] = df["response_clean_regex"].apply(
        lambda t: preprocess(t, remove_stopwords=False, stem=False)
    )  # keep response readable/ungarbled for generation-quality evaluation later

    df["instr_char_len"] = df["instruction"].str.len()
    df["instr_word_len"] = df["instruction"].str.split().apply(len)
    df["resp_char_len"] = df["response"].str.len()
    df["resp_word_len"] = df["response"].str.split().apply(len)
    df["num_slots"] = df["slots"].apply(len)

    df.to_csv(CLEAN_PATH, index=False)

    print(f"Rows before cleaning : {before_rows}")
    print(f"Rows after cleaning  : {len(df)}")
    print(f"Duplicates removed   : {before_rows - len(df)}")
    print(f"Saved -> {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    run()