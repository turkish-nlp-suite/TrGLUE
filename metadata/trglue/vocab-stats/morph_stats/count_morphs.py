import spacy
import numpy as np
from datasets import load_dataset
import math

from zeyrek import MorphAnalyzer
analyzer = MorphAnalyzer()

all_lemmas = set()


def count_sent_morphs(sent):
   morph_counts = []
   result = analyzer.analyze(sent)
   for res in result:
     if res:
       morphs = res[0].morphemes
       pos = res[0].pos
       if pos == "Punc":
           continue
       morphs = morphs if morphs!="Unk" else [res[0].word]
       morph_counts.append(len(morphs))
   return morph_counts


MAX_BIN = 10  # bins 0..9 and 10 as "10+"

def make_hist():
    return [0] * (MAX_BIN + 1)

def update_hist(hist, m):
    b = m if m <= MAX_BIN else MAX_BIN
    hist[b] += 1

def merge_hists(hists):
    merged = [0] * (MAX_BIN + 1)
    for h in hists:
        for i in range(MAX_BIN + 1):
            merged[i] += h[i]
    return merged

def hist_percentile(hist, p):
    # p in [0,100], exact percentile over discrete integer bins
    total = sum(hist)
    if total == 0:
        return 0.0
    # Use nearest-rank (same as numpy's 'linear' approx for discrete with no interpolation)
    # rank index in 0..total-1
    rank = p / 100.0 * (total - 1)
    cum = 0
    for value, count in enumerate(hist):
        prev = cum
        cum += count
        if rank < cum:
            return float(value)
    return float(MAX_BIN)

# Load Turkish transformer pipeline on GPU
spacy.require_gpu()
nlp = spacy.load("tr_core_news_trf")

def iter_texts(taskname, dataset, key1, key2=None, chunk_size=5_000):
    """
    Yields texts in chunks for spaCy.pipe.
    If key2 is provided, yields both key1 and key2 entries as separate docs.
    """
    if taskname == "stsb":
      splits = ["train", "validation"]
    elif taskname == "mnli":
      splits = ["train", "validation_matched", "validation_mismatched", "test_matched", "test_mismatched"]
    else:
      splits = ["train", "test", "validation"]

    print(splits)
    for split in splits:  # e.g., "train", "validation", "test"
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

def count_all(taskname, dataset, key1, key2=None,
              pipe_batch_size=5_000,
              disable=("ner",),
              punctuation_filter=True):
              
    """
    Processes the dataset on GPU using spaCy.pipe.
    Returns:
      - all_count: total tokens (excluding punctuation if punctuation_filter is True)
      - all_vocab: set of unique surface forms
      - all_lemmas: set of unique lemmas
    """
    hist = make_hist()
    num_tokens = 0
    sum_morphs = 0.0
    num_verbs = 0
    num_neg_verbs = 0
    num_evident_verbs =0
    all_bundles=set()

    texts = iter_texts(taskname, dataset, key1, key2)

    for doc in nlp.pipe(texts, batch_size=pipe_batch_size, n_process=1, disable=list(disable)):
        if punctuation_filter:
            toks = [t for t in doc if not t.is_punct]
        else:
            toks = list(doc)

        if not toks:
            continue
        for t in toks:
            pos = t.pos_
            morph_str = str(t.morph)
            bundle = pos + "|" + morph_str
            all_bundles.add(bundle)

            if pos=="VERB":
              num_verbs +=1
              if "Polarity=Neg" in morph_str:
                num_neg_verbs+=1
              if "Evident=Nfh" in morph_str:
                num_evident_verbs +=1

            num_tokens +=1
        morph_counts = count_sent_morphs(doc.text)
        for m in morph_counts:
            sum_morphs += m
            update_hist(hist, m)
    mean_mpt = (sum_morphs / num_tokens) if num_tokens else 0.0
    neg_pct = (100.0 * num_neg_verbs / num_verbs) if num_verbs else 0.0
    evi_pct = (100.0 * num_evident_verbs / num_verbs) if num_verbs else 0.0


    return {
        "hist": hist,
        "distinct_bundles": all_bundles,
        "num_verbal_tokens": num_verbs,
        "num_neg_verbal_tokens": num_neg_verbs,
        "num_evidential_verbal_tokens": num_evident_verbs,
        "mean_morphemes_per_tokens": mean_mpt,
        "num_tokens": num_tokens,
        "sum_morphs": sum_morphs,
        "negation_percent_verbs": neg_pct,
        "evident_percent_verbs": evi_pct,
    }

def merge_hist_summaries(summaries, percentiles=(50, 95, 99)):
    total_bundles = set()
    all_bundles = [summary["distinct_bundles"] for summary in summaries]
    for bundle in all_bundles:
        total_bundles.update(bundle)

    merged_hist = merge_hists([s["hist"] for s in summaries])
    total_tokens = sum(s["num_tokens"] for s in summaries)
    total_morphs = sum(s["sum_morphs"] for s in summaries)
    mean_mpt_overall = (total_morphs / total_tokens) if total_tokens else 0.0

    total_verbs = sum(s["num_verbal_tokens"] for s in summaries)
    total_neg_verbs = sum(s["num_neg_verbal_tokens"] for s in summaries)
    total_evident_verbs = sum(s["num_evidential_verbal_tokens"] for s in summaries)
    neg_pct_overall = (100.0 * total_neg_verbs / total_verbs) if total_verbs else 0.0
    evi_pct_overall = (100.0 * total_evident_verbs / total_verbs) if total_verbs else 0.0

    pct = {f"p{p}_morphemes_per_token": hist_percentile(merged_hist, p) for p in percentiles}

    return {
        "num_tokens": total_tokens,
        "num_bundles": len(total_bundles),
        "mean_morphemes_per_token": mean_mpt_overall,
        "negation_percent_verbs": neg_pct_overall,
        "evident_percent_verbs": evi_pct_overall,
        **pct,
        "hist": merged_hist,  # keep if you want to inspect or compute other percentiles later
    }

# Examples:
summaries=[]

'''
tasks = ["mnli", "stsb", "cola", "mrpc", "qnli", "qqp", "rte", "sst2"]
keys = [
        ("premise", "hypothesis"),
        ("sentence1", "sentence2"),
        "sentence",
        ("sentence1", "sentence2"),
        ("question", "sentence"),
        ("question1", "question2"),
        ("sentence1", "sentence2"),
        "sentence"]
'''

tasks = ["sst2"]
keys = ["sentence"]


for task, key in zip(tasks, keys):
  ds = load_dataset("turkish-nlp-suite/TrGLUE", task)

  if len(key) ==2:
    result = count_all(task, ds, key[0], key[1])
  else:
    result = count_all(task, ds, key)
  summaries.append(result)

  ofile = "all_stats/stats_" + task + ".txt"
  with open(ofile, "w") as ofl:
    hist = result["hist"]
    num_verbal_tokens = result["num_verbal_tokens"]
    num_neg_verbal_tokens = result["num_neg_verbal_tokens"]
    num_evi_verbal_tokens = result["num_evidential_verbal_tokens"]
    mean_morphemes_per_tokens = result["mean_morphemes_per_tokens"]
    negation_percent_verbs = result["negation_percent_verbs"]
    evident_percent_verbs = result["evident_percent_verbs"]
    bundles = result["distinct_bundles"]

    ofl.write("Total num of verbs " +  str(num_verbal_tokens) + "\n")
    ofl.write("Total num of negative verbs " + str(num_neg_verbal_tokens) + "\n")
    ofl.write("Total number of evident verbs " + str(num_evi_verbal_tokens) + "\n")
    ofl.write("Negative verbs percent " + str(negation_percent_verbs) + "\n")
    ofl.write("Evident verbs percent " + str(evident_percent_verbs) + "\n")
    ofl.write("Mean number of morphemes per token " + str(mean_morphemes_per_tokens) + "\n")
    ofl.write("Num of bundles " + str(len(bundles)) + "\n")

    ofl.write("=======\n")
    total = sum(hist) or 1
    ofl.write("Histogram of morphemes per token\n")
    for i, c in enumerate(hist):
      label = f"{i}" if i < len(hist) - 1 else "10+"
      pct = 100.0 * c / total
      ofl.write(f"  {label:>3}: {c:>7}  ({pct:6.2f}%)\n")




overall = merge_hist_summaries(summaries, percentiles=(50, 95, 99))


with open("all_stats/finals.txt", "w") as ofl:
    ofl.write("Num tokens " + str(overall["num_tokens"]) + "\n")
    ofl.write("num bundles " + str(overall["num_bundles"]) + "\n")
    ofl.write("mean morphemes per token " + str(round(overall["mean_morphemes_per_token"], 4)) +"\n")
    ofl.write("p50 morphemes per_token " + str(overall["p50_morphemes_per_token"]) + "\n")
    ofl.write("p95 morphemes per token " + str(overall["p95_morphemes_per_token"]) +"\n")
    ofl.write("p99 morphemes per token " + str(overall["p99_morphemes_per_token"]) +"\n")
    ofl.write("negation percent verbs " + str(round(overall["negation_percent_verbs"], 2)) +"\n")
    ofl.write("evident percent verbs " + str(round(overall["evident_percent_verbs"], 2)) +"\n")
    ofl.write("=======\n")

    hist = overall["hist"]
    total = sum(hist) or 1
    ofl.write("Histogram of morphemes per token\n")
    for i, c in enumerate(hist):
      label = f"{i}" if i < len(hist) - 1 else "10+"
      pct = 100.0 * c / total
      ofl.write(f"  {label:>3}: {c:>7}  ({pct:6.2f}%)\n")
