import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Load TEXT FAISS
# -----------------------------

text_index = faiss.read_index(
    "vectorstore/text_index.faiss"
)

with open(
    "vectorstore/text_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    text_metadata = json.load(f)


# -----------------------------
# Load IMAGE FAISS
# -----------------------------

image_index = faiss.read_index(
    "vectorstore/image_index.faiss"
)

with open(
    "vectorstore/image_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    image_metadata = json.load(f)


# -----------------------------
# Ask Question
# -----------------------------

question = input("\nEnter your question: ")

print("\nSearching text and images...")


# Create question embedding
query_embedding = model.encode([question])

query_embedding = np.array(
    query_embedding
).astype("float32")


# -----------------------------
# Search TEXT FAISS
# -----------------------------

text_distances, text_indices = text_index.search(
    query_embedding,
    3
)


# -----------------------------
# Search IMAGE FAISS
# -----------------------------

image_distances, image_indices = image_index.search(
    query_embedding,
    1
)


# -----------------------------
# Display TEXT results
# -----------------------------

print("\n========== TEXT RESULTS ==========")

for position, index in enumerate(text_indices[0]):

    if index != -1:

        print("\nText:")
        print(text_metadata[index]["text"])

        print(
            "Distance:",
            text_distances[0][position]
        )


# -----------------------------
# Display IMAGE results
# -----------------------------

print("\n========== IMAGE RESULTS ==========")

for position, index in enumerate(image_indices[0]):

    if index != -1:

        print("\nImage:")
        print(image_metadata[index]["image_path"])

        print("Caption:")
        print(image_metadata[index]["caption"])

        print(
            "Distance:",
            image_distances[0][position]
        )


print("\n====================================")
print("MULTIMODAL RETRIEVAL COMPLETED")
print("====================================")