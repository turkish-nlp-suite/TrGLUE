#!/usr/bin/env python3
import os
import re
import unicodedata
from typing import Iterable, Tuple, Optional, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------------
# Config
# -----------------------------------
#TOKENIZER_NAME = "dbmdz/bert-base-turkish-cased"  # e.g., "dbmdz/bert-base-turkish-cased", "xlm-roberta-base"
TOKENIZER_NAME = "bert-base-cased"  # e.g., "dbmdz/bert-base-turkish-cased", "xlm-roberta-base"
LOWERCASE = False                    # apply lowercasing prior to tokenization
DIGIT_COLLAPSE = False               # replace digits with '0' before tokenization
EXCLUDE_PUNCT = True                 # exclude punctuation tokens from token count
OUTPUT_DIR = "all_stats_en"

# TrGLUE tasks and text fields (aligns with your NER code)
TASKS_AND_FIELDS = [
    ("cola",  ("sentence", None)),
    ("mnli",  ("premise", "hypothesis")),
    ("mrpc",  ("sentence1", "sentence2")),
    ("qnli",  ("question", "sentence")),
    ("qqp",   ("question1", "question2")),
    ("rte",   ("sentence1", "sentence2")),
    ("sst2",  ("sentence", None)),
    ("stsb",  ("sentence1", "sentence2")),  # special split handling: train + validation only
]

# -----------------------------------
# Helpers
# -----------------------------------
WS_SPLIT = re.compile(r"\s+")

def is_punct_str(s: str) -> bool:
    # A token is punctuation if all chars are Unicode punctuation categories
    if not s:
        return True
    for ch in s:
        if not unicodedata.category(ch).startswith("P"):
            return False
    return True

def preprocess_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if LOWERCASE:
        s = s.lower()
    if DIGIT_COLLAPSE:
        s = re.sub(r"\d", "0", s)
    return s

def iter_texts(ds, field1: str, field2: Optional[str], task: str, chunk_size: int = 1000):
    """
    Yields texts in chunks across the appropriate splits.
    For STS-B: only 'train' and 'validation' (exclude 'test').
    For others: 'train', 'validation', 'test' if present.
    For pair tasks, yield each field as a separate text (consistent with your NER iterator).
    """
    if task == "stsb":
        split_order = [s for s in ("train", "test", "validation") if s in ds]
    elif task == "mnli":
        split_order = [s for s in ["train", "validation_matched", "validation_mismatched", "test_matched", "test_mismatched"] if s in ds]
    else:
        # Keep consistent split order; include only if present
        split_order = [s for s in ("train", "validation", "test") if s in ds]

    for split in split_order:
        subset = ds[split]
        n = len(subset)
        for start in range(0, n, chunk_size):
            batch = subset.select(range(start, min(start + chunk_size, n)))
            texts1 = batch[field1] if field1 in batch.column_names else [None] * len(batch)
            if field2 is None:
                for s1 in texts1:
                    if s1:
                        yield preprocess_text(s1)
            else:
                texts2 = batch[field2] if field2 in batch.column_names else [None] * len(batch)
                for s1, s2 in zip(texts1, texts2):
                    if s1:
                        yield preprocess_text(s1)
                    if s2:
                        yield preprocess_text(s2)

def count_subwords_for_text(text: str, tokenizer) -> Tuple[int, int]:
    """
    Returns (num_tokens, num_subwords) for a single text.
    Tokens = whitespace-split tokens with optional punctuation exclusion.
    Subwords = sum of tokenizer.word_ids() groups corresponding to those tokens.

    Implementation detail:
    - We split by whitespace to define "tokens".
    - For each token, we tokenize with is_split_into_words=True to get word_ids mapping.
    - We count how many subword pieces map to that word index.
    """
    if not text:
        return (0, 0)

    # Whitespace tokenization
    raw_tokens = [t for t in WS_SPLIT.split(text) if t != ""]
    # Optionally exclude punctuation tokens
    if EXCLUDE_PUNCT:
        tokens = [t for t in raw_tokens if not is_punct_str(t)]
    else:
        tokens = raw_tokens

    if not tokens:
        return (0, 0)

    # Tokenize as pre-split words
    enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False, return_attention_mask=False)
    word_ids = enc.word_ids()
    # Count subwords per token id
    # word_ids aligns with enc.input_ids length, with each entry being the source word index
    subwords_per_token = [0] * len(tokens)
    for wid in word_ids:
        if wid is not None:
            subwords_per_token[wid] += 1

    num_tokens = len(tokens)
    num_subwords = sum(subwords_per_token)
    return (num_tokens, num_subwords)

def aggregate_dataset(task: str, field1: str, field2: Optional[str], tokenizer) -> Dict[str, float]:
    num_tokens_total = 0
    num_subwords_total = 0

    ds = load_dataset("nyu-mll/GLUE", task)
    for text in tqdm(iter_texts(ds, field1, field2, task), desc=f"Processing {task}", unit="text"):
        t_count, wp_count = count_subwords_for_text(text, tokenizer)
        num_tokens_total += t_count
        num_subwords_total += wp_count

    avg_subwords_per_token = (num_subwords_total / num_tokens_total) if num_tokens_total else 0.0
    return {
        "num_tokens": num_tokens_total,
        "num_subwords": num_subwords_total,
        "avg_subwords_per_token": avg_subwords_per_token,
    }

# -----------------------------------
# Main
# -----------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    overall_tokens = 0
    overall_subwords = 0

    for task, fields in TASKS_AND_FIELDS:
        field1, field2 = fields
        stats = aggregate_dataset(task, field1, field2, tokenizer)

        # Accumulate overall
        overall_tokens += stats["num_tokens"]
        overall_subwords += stats["num_subwords"]

        # Write per-task file
        out_path = os.path.join(OUTPUT_DIR, f"stats_{task}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Total num of tokens {stats['num_tokens']}\n")
            f.write(f"Total num of subwords {stats['num_subwords']}\n")
            f.write(f"Mean number of subwords per token {stats['avg_subwords_per_token']}\n")

    # Overall
    overall_avg = (overall_subwords / overall_tokens) if overall_tokens else 0.0
    finals_path = os.path.join(OUTPUT_DIR, "finals.txt")
    with open(finals_path, "w", encoding="utf-8") as f:
        f.write(f"Total num of tokens {overall_tokens}\n")
        f.write(f"Total num of subwords {overall_subwords}\n")
        f.write(f"Mean number of subwords per token {overall_avg}\n")

    print("Done. Results written to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
