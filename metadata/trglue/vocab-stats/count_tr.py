import spacy
from datasets import load_dataset
nlp = spacy.load('tr_core_news_md')



def count_all(dataset, key1, key2):
  all_count = 0
  all_vocab=set()

  all_lemmas = set()
  for split in dataset:
    sliced = dataset[split]
    for i in range(0, len(sliced), 5):
      instances = sliced[i:i+5]
      #print(instances)
      sentences1 = [sent1.strip() for sent1 in instances[key1]]
      sentences2 = [sent2.strip() for sent2 in instances[key2]]
      sent1 = " ".join(sentences1)
      sent2 = " ".join(sentences2)
      doc1 = nlp(sent1)
      doc2 = nlp(sent2)

      tokens1 = [token for token in doc1 if not token.is_punct]
      tokens2 = [token for token in doc2 if not token.is_punct]
      all_words1 = [token.text for token in tokens1]
      all_words2 = [token.text for token in tokens2]

      all_lemmas1 = [token.lemma_ for token in tokens1]
      all_lemmas2 = [token.lemma_ for token in tokens2]

      all_count += len(all_words1) + len(all_words2)
      all_vocab.update(all_words1)
      all_vocab.update(all_words2)

      all_lemmas.update(all_lemmas1)
      all_lemmas.update(all_lemmas2)
  return all_count, all_vocab, all_lemmas


dataset = load_dataset("turkish-nlp-suite/glue", "mnli")
all_count, all_vocab, all_lemmas = count_all(dataset, "premise", "hypothesis")
print(all_count, len(all_vocab), len(all_lemmas), "mnli")

