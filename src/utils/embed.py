from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from google import genai
from google.genai.types import HttpOptions

def _create_client(path):
    with open(path, "r") as f:
        api_key = f.read().strip()
    return genai.Client(api_key=api_key)

client = _create_client("key.secret")

texts = [
  "LU",
  "TN",
  "MW",
  "MO",
  "MS",
 "Luxembourg",
 "Tunisia",
 "Malawi",
 "Macao",
 "Montserrat",
 "Morocco",
 "Mauritius",
]

# Sort texts: country codes first, then country names
codes = [t for t in texts if len(t) == 2 and t.isupper()]
names = [t for t in texts if t not in codes]
texts = codes + names

# texts = [
#     "CGC",
#     "UAC",
#     "CCC",
#     "UGG",
#     "AUA",
#     "R",
#     "Y",
#     "P",
#     "W",
#     "L",
#     "I",
#     "D"
# ]

result = [
    np.array(e.values) for e in client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
]

# Calculate cosine similarity. Higher scores = greater semantic similarity.

embeddings_matrix = np.array(result)
similarity_matrix = cosine_similarity(embeddings_matrix)


for i, text1 in enumerate(texts):
    for j in range(i + 1, len(texts)):
        text2 = texts[j]
        similarity = similarity_matrix[i, j]
        print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")


response = client.models.count_tokens(
    model="gemini-2.5-flash",
    #contents="LU TN MW MO MS",
    contents="CGCUACCCCUGGAUA",
)
print(response)

# Visualize similarity matrix as a heatmap
plt.figure(figsize=(8, 8))
plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(texts)), texts, rotation=90)
plt.yticks(range(len(texts)), texts)
plt.title('Semantic Similarity Heatmap')
plt.tight_layout()
plt.show()
