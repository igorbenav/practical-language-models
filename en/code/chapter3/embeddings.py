# embeddings.py
import numpy as np
from huggingface_hub import hf_hub_download


def load_glove(path):
    words = []
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    vectors = np.array(vectors, dtype=np.float32)
    word_to_index = {word: i for i, word in enumerate(words)}
    return words, vectors, word_to_index


path = hf_hub_download(
    repo_id="igorbenav/glove-6b-50d",
    filename="glove.6B.50d.txt",
    repo_type="dataset",
)
words, vectors, word_to_index = load_glove(path)
print(f"Loaded {len(words)} words, each with {vectors.shape[1]} dimensions")


def get_vector(word):
    if word not in word_to_index:
        print(f"'{word}' not in vocabulary")
        return None
    return vectors[word_to_index[word]]


cat = get_vector("cat")
print(f"\n'cat' vector (first 10 dims): {cat[:10].round(4)}")
print(f"Min: {cat.min():.4f}, Max: {cat.max():.4f}")


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


pairs = [
    ("cat", "dog"),
    ("cat", "kitten"),
    ("cat", "car"),
    ("cat", "democracy"),
    ("happy", "joyful"),
    ("happy", "sad"),
    ("king", "queen"),
    ("king", "banana"),
    ("france", "paris"),
    ("germany", "berlin"),
]

print("\nWord pair similarities:")
for w1, w2 in pairs:
    v1, v2 = get_vector(w1), get_vector(w2)
    if v1 is not None and v2 is not None:
        sim = cosine_similarity(v1, v2)
        print(f"  {w1:12s} - {w2:12s}: {sim:.3f}")


def most_similar(word, n=10):
    vec = get_vector(word)
    if vec is None:
        return []

    # Normalize every word vector to length 1
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    norms[norms == 0] = 1.0
    normalized = vectors / norms[:, np.newaxis]

    # Normalize the query vector
    word_norm = np.sqrt(np.sum(vec ** 2))
    word_normalized = vec / word_norm

    # Dot product of unit vectors = cosine similarity
    similarities = normalized @ word_normalized

    # Sort by similarity (highest first), skip the word itself
    top_indices = np.argsort(similarities)[::-1][1:n+1]
    return [(words[i], similarities[i]) for i in top_indices]


for query in ["cat", "france", "happy", "python"]:
    print(f"\nMost similar to '{query}':")
    for word, sim in most_similar(query):
        print(f"  {word:15s} {sim:.3f}")


def analogy(a, b, c, n=5):
    """a is to b as c is to ???"""
    va, vb, vc = get_vector(a), get_vector(b), get_vector(c)
    if any(v is None for v in [va, vb, vc]):
        return []

    # Compute the target: b - a + c
    target = vb - va + vc

    # Normalize all vectors
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    norms[norms == 0] = 1.0
    normalized = vectors / norms[:, np.newaxis]

    target_norm = np.sqrt(np.sum(target ** 2))
    target_normalized = target / target_norm

    # Find the closest words to the target
    similarities = normalized @ target_normalized

    # Exclude the three input words from results
    for w in [a, b, c]:
        if w in word_to_index:
            similarities[word_to_index[w]] = -1

    top_indices = np.argsort(similarities)[::-1][:n]
    return [(words[i], similarities[i]) for i in top_indices]


print("\n--- Analogies ---")

print("\nman : king :: woman : ???")
for word, sim in analogy("man", "king", "woman"):
    print(f"  {word:15s} {sim:.3f}")

print("\nfrance : paris :: italy : ???")
for word, sim in analogy("france", "paris", "italy"):
    print(f"  {word:15s} {sim:.3f}")

print("\nwalk : walking :: swim : ???")
for word, sim in analogy("walk", "walking", "swim"):
    print(f"  {word:15s} {sim:.3f}")

print("\njapan : sushi :: mexico : ???")
for word, sim in analogy("japan", "sushi", "mexico"):
    print(f"  {word:15s} {sim:.3f}")

print("\n--- Bias in embeddings ---")

print("\nman : doctor :: woman : ???")
for word, sim in analogy("man", "doctor", "woman"):
    print(f"  {word:15s} {sim:.3f}")

print("\nman : programmer :: woman : ???")
for word, sim in analogy("man", "programmer", "woman"):
    print(f"  {word:15s} {sim:.3f}")

print("\nman : brilliant :: woman : ???")
for word, sim in analogy("man", "brilliant", "woman"):
    print(f"  {word:15s} {sim:.3f}")


import matplotlib.pyplot as plt


def pca_2d(vecs):
    """Project vectors to 2 dimensions, preserving as much structure as possible."""
    centered = vecs - vecs.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top2 = eigenvectors[:, -2:][:, ::-1]
    return centered @ top2


word_groups = {
    "animals": ["cat", "dog", "fish", "bird", "horse", "cow", "sheep"],
    "countries": ["france", "germany", "italy", "spain", "japan", "china"],
    "emotions": ["happy", "sad", "angry", "afraid", "surprised"],
}

all_words = []
group_labels = []
for group, wds in word_groups.items():
    all_words.extend(wds)
    group_labels.extend([group] * len(wds))

vecs = np.array([get_vector(w) for w in all_words])
coords = pca_2d(vecs)

colors = {"animals": "steelblue", "countries": "coral", "emotions": "seagreen"}
plt.figure(figsize=(10, 8))
for i, (word, group) in enumerate(zip(all_words, group_labels)):
    plt.scatter(coords[i, 0], coords[i, 1], c=colors[group], s=50)
    plt.annotate(word, (coords[i, 0], coords[i, 1]),
                 fontsize=10, ha="center", va="bottom")
plt.title("Word embeddings projected to 2D (PCA)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
plt.show()

gender_pairs = [
    ("king", "queen"), ("man", "woman"), ("boy", "girl"),
    ("he", "she"), ("his", "her"), ("brother", "sister"),
    ("father", "mother"), ("son", "daughter"),
]

male_words = [p[0] for p in gender_pairs]
female_words = [p[1] for p in gender_pairs]
all_gender = male_words + female_words

vecs = np.array([get_vector(w) for w in all_gender])
coords = pca_2d(vecs)
n = len(gender_pairs)

plt.figure(figsize=(10, 6))
plt.scatter(coords[:n, 0], coords[:n, 1], c="steelblue", s=60, label="male")
plt.scatter(coords[n:, 0], coords[n:, 1], c="coral", s=60, label="female")

for i, (m, f) in enumerate(gender_pairs):
    plt.annotate(m, (coords[i, 0], coords[i, 1]),
                 fontsize=10, ha="center", va="bottom")
    plt.annotate(f, (coords[n+i, 0], coords[n+i, 1]),
                 fontsize=10, ha="center", va="bottom")
    plt.plot([coords[i, 0], coords[n+i, 0]],
             [coords[i, 1], coords[n+i, 1]], "gray", alpha=0.4, linewidth=1)

plt.title("Gender direction in embedding space", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gender_direction.png", dpi=150)
plt.show()

print("\n--- Gender direction consistency ---")
offsets = []
for m, f in gender_pairs:
    offset = get_vector(f) - get_vector(m)
    offsets.append(offset / np.sqrt(np.sum(offset ** 2)))

for i in range(len(offsets)):
    for j in range(i + 1, len(offsets)):
        sim = cosine_similarity(offsets[i], offsets[j])
        print(f"  {gender_pairs[i][0]:8s}\u2192{gender_pairs[i][1]:8s}  vs  "
              f"{gender_pairs[j][0]:8s}\u2192{gender_pairs[j][1]:8s}  : {sim:.3f}")
