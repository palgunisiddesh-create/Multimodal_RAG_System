import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from local_llm import generate_answer


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
# User Question
# -----------------------------

question = input("\nEnter your question: ")


print("\nSearching text and images...")


# -----------------------------
# Create question embedding
# -----------------------------

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


# Store retrieved text
text_results = []

for index in text_indices[0]:

    if index != -1:

        text_results.append(
            text_metadata[index]["text"]
        )


# -----------------------------
# Search IMAGE FAISS
# -----------------------------

image_distances, image_indices = image_index.search(
    query_embedding,
    1
)


# Store image captions
image_results = []

for index in image_indices[0]:

    if index != -1:

        image_results.append(
            image_metadata[index]["caption"]
        )


# -----------------------------
# Display retrieved information
# -----------------------------

print("\n========== RETRIEVED TEXT ==========")

for text in text_results:

    print("\n", text)


print("\n========== RETRIEVED IMAGE ==========")

for index in image_indices[0]:

    if index != -1:

        print(
            "\nImage:",
            image_metadata[index]["image_path"]
        )

        print(
            "Caption:",
            image_metadata[index]["caption"]
        )


# -----------------------------
# Generate final answer
# -----------------------------

answer = generate_answer(
    question,
    text_results,
    image_results
)


print("\n====================================")
print("          FINAL RAG ANSWER")
print("====================================")

print(answer)