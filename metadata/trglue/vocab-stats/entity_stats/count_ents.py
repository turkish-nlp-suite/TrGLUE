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

def iter_texts(dataset, key1, key2=None, chunk_size=1_000):
    """
    Yields texts in chunks for spaCy.pipe.
    If key2 is provided, yields both key1 and key2 entries as separate docs.
    """
    if dataset == "stsb":
      splits = ["train", "validation"]
    else:
      splits = ["train", "test", "validation"]
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

def count_all(dataset, key1, key2=None,
              pipe_batch_size=1_000,
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
    num_ents = 0

    texts = iter_texts(dataset, key1, key2)

    for doc in nlp.pipe(texts, batch_size=pipe_batch_size, n_process=1):
        if punctuation_filter:
            toks = [t for t in doc if not t.is_punct]
        else:
            toks = list(doc)

        if not toks:
            continue
        num_tokens += len(toks)
        ent_counts = len(doc.ents)
        num_ents += ent_counts
    mean_mpt = (num_ents / num_tokens) if num_tokens else 0.0


    return {
        "mean_ents_per_token": mean_mpt,
        "num_tokens": num_tokens,
        "num_ents": num_ents,
    }

def merge_hist_summaries(summaries, percentiles=(50, 95, 99)):

    total_tokens = sum(s["num_tokens"] for s in summaries)
    total_ents = sum(s["num_ents"] for s in summaries)
    mean_ents_overall = (total_ents / total_tokens) if total_tokens else 0.0


    return {
        "num_tokens": total_tokens,
        "num_ents": total_ents,
        "mean_ents_per_token": mean_ents_overall,
    }

# Examples:
summaries=[]

tasks = ["stsb" ,"mnli", "cola", "mrpc", "qnli", "qqp", "rte", "sst2"]
keys = [
        ("sentence1", "sentence2"),
        ("premise", "hypothesis"),
        "sentence",
        ("sentence1", "sentence2"),
        ("question", "sentence"),
        ("question1", "question2"),
        ("sentence1", "sentence2"),
        "sentence",]

for task, key in zip(tasks, keys):
  ds = load_dataset("turkish-nlp-suite/TrGLUE", task)

  if len(key) ==2:
    result = count_all(ds, key[0], key[1])
  else:
    result = count_all(ds, key)
  if not summaries:
    pass
  else:
    summaries.append(result)

  ofile = "all_stats/stats_" + task + ".txt"
  with open(ofile, "w") as ofl:
    num_ent_tokens = result["num_ents"]
    num_total_tokens = result["num_tokens"]
    mean_ents_per_token = result["mean_ents_per_token"]

    ofl.write("Total num of tokens " +  str(num_total_tokens) + "\n")
    ofl.write("Total num of ents " +  str(num_ent_tokens) + "\n")
    ofl.write("Mean number of entities per token " + str(mean_ents_per_token) + "\n")




overall = merge_hist_summaries(summaries)


with open("all_stats/finals.txt", "w") as ofl:
    num_ent_tokens = result["num_ents"]
    num_total_tokens = result["num_tokens"]
    mean_ents_per_token = result["mean_ents_per_token"]

    ofl.write("Total num of tokens " +  str(num_tokens) + "\n")
    ofl.write("Total num of ents " +  str(num_ents) + "\n")
    ofl.write("Mean number of entities per token " + str(mean_ents_per_token) + "\n")
