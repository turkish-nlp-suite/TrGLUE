import os
import math
from collections import Counter, defaultdict
from typing import Optional, Dict, Any, Iterable, List

import spacy
from datasets import load_dataset

# =========================
# Config
# =========================
spacy.require_gpu()
nlp = spacy.load("tr_core_news_trf")

# Toggle to use relaxed object set by default (recommended for RTE)
RELAXED_OBJECTS_DEFAULT = True

# =========================
# Linguistic constants
# =========================
SUBJ_DEPS = {"nsubj", "csubj", "nsubj:pass", "csubj:pass"}
OBJ_DEPS_STRICT = {"obj", "iobj", "obl:arg"}  # keep obl:arg even in strict Turkish setting
OBJ_DEPS_RELAX = {"obj", "iobj", "obl:arg", "ccomp", "xcomp"}  # include clausal complements when relaxed
VERB_POS = {"VERB", "AUX"}

# =========================
# Heuristics (improved for RTE)
# =========================
def is_clause_head(tok) -> bool:
    """
    Identify clause heads for Turkish:
    - Finite verbs/aux (exclude clear converbs)
    - Nominal/adjectival predicates with copula (cop child) or AUX child
    - Zero-copula heuristic: nominal/adjectival with tense-bearing AUX child
    """
    # Finite verbs and auxiliaries (avoid converbs)
    if tok.pos_ in VERB_POS:
        vf = tok.morph.get("VerbForm")
        if vf and "Conv" in vf:
            return False
        return True

    # Nominal/adjectival predicates with copula child
    if tok.pos_ in {"NOUN", "ADJ", "PRON"}:
        # copular child
        has_cop = any(c.dep_ == "cop" for c in tok.children)
        has_aux = any(c.pos_ == "AUX" for c in tok.children)
        if has_cop or has_aux:
            return True

    # If token is cop itself, the head is the predicate
    if tok.dep_ == "cop" and tok.head.pos_ in {"NOUN", "ADJ", "PRON"}:
        return True

    return False

def nearest_child(head, labels):
    cand = [c for c in head.children if c.dep_ in labels]
    if not cand:
        return None
    return min(cand, key=lambda x: abs(x.i - head.i))

def recover_subject_for_conj(head):
    """
    If head is a conjunct and lacks a subject, try to inherit from its coordination head.
    """
    if head.dep_ != "conj":
        return None
    parent = head.head
    if parent is None or parent is head:
        return None
    subj = nearest_child(parent, SUBJ_DEPS)
    return subj

def classify_order(subj, obj, head):
    iV = head.i
    iS = subj.i if subj is not None else None
    iO = obj.i if obj is not None else None
    if iS is not None and iO is not None:
        order = sorted([(iS, "S"), (iO, "O"), (iV, "V")], key=lambda x: x[0])
        return "".join(x[1] for x in order)  # SOV/SVO/OSV/OVS/VSO/VOS
    if iS is not None:
        return "SV" if iS < iV else "VS"
    if iO is not None:
        return "OV" if iO < iV else "VO"
    return None

def iter_texts(dataset, key1, key2=None, chunk_size=1_000_000):
    """
    Yields texts in chunks for spaCy.pipe.
    If key2 is provided, yields both key1 and key2 entries as separate docs.
    dataset is a dict-like: split_name -> Dataset
    """
    for split in dataset:
        sliced = dataset[split]
        n = len(sliced)
        for start in range(0, n, chunk_size):
            batch = sliced[start:start + chunk_size]
            if key2 is None:
                for s1 in batch[key1]:
                    if s1:
                        yield s1.strip()
            else:
                for s1, s2 in zip(batch[key1], batch[key2]):
                    if s1:
                        yield s1.strip()
                    if s2:
                        yield s2.strip()

# =========================
# Core counter
# =========================
def count_all(
    dataset,
    key1: str,
    key2: Optional[str] = None,
    pipe_batch_size: int = 1024,
    disable: Iterable[str] = ("ner",),
    punctuation_filter: bool = True,
    keep_per_sentence: bool = False,
    relaxed_objects: bool = RELAXED_OBJECTS_DEFAULT,
    nlp = None,
) -> Dict[str, Any]:
    """
    Stream the dataset through spaCy.pipe on GPU and compute:
      - pooled average dependency distance (excluding punctuation)
      - word-order distributions for clause heads (S/O/V permutations and partials)
      - subject-drop rate among clause heads
    """
    assert nlp is not None, "Please pass a loaded spaCy pipeline."

    obj_deps = OBJ_DEPS_RELAX if relaxed_objects else OBJ_DEPS_STRICT

    total_edges = 0
    total_distance = 0
    per_sentence = [] if keep_per_sentence else None

    # Order counts
    order_counts = Counter()
    full_order_counts = Counter()   # only clauses with both S and O observed
    partial_order_counts = Counter()  # SV/VS/OV/VO

    # Subject drop
    clause_count = 0
    subjectless_clause_count = 0

    texts = iter_texts(dataset, key1, key2)

    for doc in nlp.pipe(texts, batch_size=pipe_batch_size, n_process=1, disable=list(disable)):
        # Sentences
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
        for sent in sents:
            # Dependency distance within sentence
            edges = 0
            dist_sum = 0
            start_i, end_i = sent.start, sent.end
            for tok in sent:
                if punctuation_filter and tok.is_punct:
                    continue
                if tok.head is tok:
                    continue
                hi = tok.head.i
                if hi < start_i or hi >= end_i:
                    continue
                d = abs(tok.i - hi)
                dist_sum += d
                edges += 1
            total_edges += edges
            total_distance += dist_sum
            if keep_per_sentence:
                per_sentence.append({
                    "text": sent.text,
                    "edge_count": edges,
                    "total_distance": dist_sum,
                    "avg_distance": (dist_sum / edges) if edges else 0.0,
                })

            # Clause-level stats
            for head in sent:
                if not is_clause_head(head):
                    continue

                # subject: nearest among children; if conj and none, inherit from parent
                subj = nearest_child(head, SUBJ_DEPS)
                if subj is None and head.dep_ == "conj":
                    subj = recover_subject_for_conj(head)

                # object-like: nearest among chosen deps
                obj = nearest_child(head, obj_deps)

                clause_count += 1
                if subj is None:
                    subjectless_clause_count += 1

                ord_tag = classify_order(subj, obj, head)
                if ord_tag is None:
                    continue
                order_counts[ord_tag] += 1
                if ord_tag in {"SOV", "SVO", "OSV", "OVS", "VSO", "VOS"}:
                    full_order_counts[ord_tag] += 1
                else:
                    partial_order_counts[ord_tag] += 1

    pooled_avg = (total_distance / total_edges) if total_edges else 0.0

    # Non-canonical rate among clauses with both S and O
    total_full = sum(full_order_counts.values())
    non_canonical = (
        full_order_counts.get("OSV", 0)
        + full_order_counts.get("OVS", 0)
        + full_order_counts.get("VSO", 0)
        + full_order_counts.get("VOS", 0)
    )
    non_canonical_rate = (100.0 * non_canonical / total_full) if total_full else 0.0

    # Subject drop rate among clause heads
    subj_drop_rate = (100.0 * subjectless_clause_count / clause_count) if clause_count else 0.0

    result = {
        "pooled_avg_dependency_distance": pooled_avg,
        "totals": {
            "total_edges": total_edges,
            "total_distance": total_distance,
            "exclude_punct": punctuation_filter,
        },
        "orders": dict(order_counts),
        "orders_full_only": dict(full_order_counts),
        "orders_partial": dict(partial_order_counts),
        "non_canonical_rate_percent": non_canonical_rate,  # among full S/O/V
        "finite_clauses": clause_count,  # legacy name retained
        "subjectless_finite_clauses": subjectless_clause_count,
        "subject_drop_rate_percent": subj_drop_rate,
    }
    if keep_per_sentence:
        result["per_sentence"] = per_sentence
    return result

# =========================
# Aggregation helpers
# =========================
def _merge_counts_micro(agg: Dict[str, Any], stats: Dict[str, Any]) -> None:
    agg["total_edges"] += stats["totals"]["total_edges"]
    agg["total_distance"] += stats["totals"]["total_distance"]
    agg["finite_clauses"] += stats["finite_clauses"]
    agg["subjectless_finite_clauses"] += stats["subjectless_finite_clauses"]
    for k, v in stats["orders_full_only"].items():
        agg["orders_full_only"][k] += v
    for k, v in stats["orders_partial"].items():
        agg["orders_partial"][k] += v
    for k, v in stats["orders"].items():
        agg["orders_all"][k] += v

def _finalize_from_totals(agg: Dict[str, Any]) -> None:
    te = agg["total_edges"]
    td = agg["total_distance"]
    fc = agg["finite_clauses"]
    sf = agg["subjectless_finite_clauses"]
    full = sum(agg["orders_full_only"].values())
    noncanon = (
        agg["orders_full_only"].get("OSV", 0)
        + agg["orders_full_only"].get("OVS", 0)
        + agg["orders_full_only"].get("VSO", 0)
        + agg["orders_full_only"].get("VOS", 0)
    )
    agg["pooled_avg_dep_dist"] = (td / te) if te else 0.0
    agg["subject_drop_rate_percent"] = (100.0 * sf / fc) if fc else 0.0
    agg["non_canonical_rate_percent"] = (100.0 * noncanon / full) if full else 0.0

def _kvline(d: dict) -> str:
    if not d:
        return ""
    return ";".join(f"{k}={v}" for k, v in sorted(d.items()))

# =========================
# Runner + writers
# =========================
def run_and_write(
    tasks: List[Dict[str, str]],
    repo: str = "turkish-nlp-suite/TrGLUE",
    out_dir: str = "all_stats",
    splits: Iterable[str] = ("train", "validation", "test"),
    nlp=None,
    pipe_batch_size: int = 1024,
    disable: Iterable[str] = ("ner",),
    punctuation_filter: bool = True,
    relaxed_objects: bool = RELAXED_OBJECTS_DEFAULT,
    exclude_stsb_test: bool = True,
) -> None:
    """
    - For each task: aggregate over splits (micro) and write ONE file all_stats/<task>.txt with totals only.
    - Global summary across all tasks written to finals.txt.
    - Excludes STS-B test if exclude_stsb_test=True.
    """
    assert nlp is not None, "Pass a loaded spaCy pipeline."
    os.makedirs(out_dir, exist_ok=True)

    global_micro = {
        "total_edges": 0,
        "total_distance": 0,
        "finite_clauses": 0,
        "subjectless_finite_clauses": 0,
        "orders_full_only": defaultdict(int),
        "orders_partial": defaultdict(int),
        "orders_all": defaultdict(int),
    }

    for spec in tasks:
        config = spec["config"]
        key1 = spec["key1"]
        key2 = spec.get("key2")

        ds = load_dataset(repo, config)

        # Aggregate per task by summing over included splits
        task_micro = {
            "total_edges": 0,
            "total_distance": 0,
            "finite_clauses": 0,
            "subjectless_finite_clauses": 0,
            "orders_full_only": defaultdict(int),
            "orders_partial": defaultdict(int),
            "orders_all": defaultdict(int),
        }

        for split in splits:
            if split not in ds:
                continue
            # Skip STS-B test if requested
            if exclude_stsb_test and config.lower() in {"stsb", "sts-b", "sts_b"} and split == "test":
                continue

            split_ds = {split: ds[split]}
            stats = count_all(
                dataset=split_ds,
                key1=key1,
                key2=key2,
                pipe_batch_size=pipe_batch_size,
                disable=disable,
                punctuation_filter=punctuation_filter,
                keep_per_sentence=False,
                relaxed_objects=relaxed_objects,
                nlp=nlp,
            )
            _merge_counts_micro(task_micro, stats)
            _merge_counts_micro(global_micro, stats)

        # Finalize task metrics and write to file
        _finalize_from_totals(task_micro)
        out_path = os.path.join(out_dir, f"{config}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            # Write only totals and derived metrics (no per-split blocks)
            lines = [
                f"{task_micro['pooled_avg_dep_dist']}",
                f"{task_micro['non_canonical_rate_percent']}",
                f"{task_micro['subject_drop_rate_percent']}",
                f"{task_micro['total_edges']}",
                f"{task_micro['total_distance']}",
                f"{task_micro['finite_clauses']}",
                f"{task_micro['subjectless_finite_clauses']}",
                _kvline(task_micro["orders_full_only"]),
                _kvline(task_micro["orders_partial"]),
                _kvline(task_micro["orders_all"]),
            ]
            f.write("\n".join(lines).strip() + "\n")
        print(f"Wrote {out_path}")

    # Global finals
    _finalize_from_totals(global_micro)
    finals_path = os.path.join(out_dir, "finals.txt")
    with open(finals_path, "w", encoding="utf-8") as f:
        lines = [
            f"{global_micro['pooled_avg_dep_dist']}",
            f"{global_micro['non_canonical_rate_percent']}",
            f"{global_micro['subject_drop_rate_percent']}",
            f"{global_micro['total_edges']}",
            f"{global_micro['total_distance']}",
            f"{global_micro['finite_clauses']}",
            f"{global_micro['subjectless_finite_clauses']}",
            _kvline(global_micro["orders_full_only"]),
            _kvline(global_micro["orders_partial"]),
            _kvline(global_micro["orders_all"]),
        ]
        f.write("\n".join(lines).strip() + "\n")
    print(f"Wrote {finals_path}")

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Define TrGLUE tasks with correct field names
    tasks = [
        {"config": "cola", "key1": "sentence"},
        {"config": "mnli", "key1": "premise", "key2": "hypothesis"},
        {"config": "mrpc", "key1": "sentence1", "key2": "sentence2"},
        {"config": "qnli", "key1": "question", "key2": "sentence"},
        {"config": "qqp",  "key1": "question1", "key2": "question2"},
        {"config": "rte", "key1": "sentence1", "key2": "sentence2"},
        {"config": "sst2", "key1": "sentence"},
        {"config": "stsb", "key1": "sentence1", "key2": "sentence2"},
    ]

    run_and_write(
        tasks=tasks,
        repo="turkish-nlp-suite/TrGLUE",
        out_dir="all_stats",
        splits=("train", "validation", "test"),
        nlp=nlp,
        pipe_batch_size=1024,
        disable=("ner",),
        punctuation_filter=True,
        relaxed_objects=RELAXED_OBJECTS_DEFAULT,
        exclude_stsb_test=True,
    )
