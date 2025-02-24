import matplotlib.pyplot as plt
import seaborn as sns
import json

from sentence_transformers import SentenceTransformer, util

# Load the multilingual sentence embedding model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device="cuda:0")

from scipy.stats import ks_2samp, ttest_ind


dataset = []
with open("rte_train.jsonl", "r") as infile:
    for line in infile: 
        data = json.loads(line)
        dataset.append(data)
dataset = dataset[:400]

# Define batch size
batch_size = 200

# Create a DataLoader


# Move computations to GPU if available


# Store all cosine similarity results
all_turkish_similarities = []  # For Turkish sentences
all_english_similarities = []  # For English sentences

# Process each batch
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    sentences1 = [item["sentence1"] for item in batch]
    sentences2 = [item["sentence2"] for item in batch]

    osentences1 = [item["original_sentence1"] for item in batch]
    osentences2 = [item["original_sentence2"] for item in batch]

    turkish1_embeddings = model.encode(sentences1) 
    turkish2_embeddings = model.encode(sentences2) 
    
    english1_embeddings = model.encode(osentences1) 
    english2_embeddings = model.encode(osentences2)
    english_similarities = util.cos_sim(english1_embeddings, english2_embeddings).diagonal()
    
    # Compute cosine similarity for Turkish pairs
    turkish_similarities = util.cos_sim(turkish1_embeddings, turkish2_embeddings).diagonal()
    
    # Compute cosine similarity for Turkish pairs
    all_turkish_similarities.append(turkish_similarities.cpu())  # Move to CPU to store
    
    # Compute cosine similarity for English (original) pairs
    all_english_similarities.append(english_similarities.cpu())  # Move to CPU to store

# Combine all batch results
all_turkish_similarities = torch.cat(all_turkish_similarities, dim=0)
all_english_similarities = torch.cat(all_english_similarities, dim=0)


plt.figure(figsize=(8, 6))
sns.kdeplot(english_similarities, label="English Similarities", fill=True, color='blue')
sns.kdeplot(turkish_similarities, label="Turkish Similarities", fill=True, color='orange')
plt.title("Cosine Similarity Distributions")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()
plt.show()

ks_stat, ks_p_value = ks_2samp(english_similarities, turkish_similarities)
print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")

# t-test
t_stat, t_p_value = ttest_ind(english_similarities, turkish_similarities)
print(f"T-statistic: {t_stat}, p-value: {t_p_value}")
