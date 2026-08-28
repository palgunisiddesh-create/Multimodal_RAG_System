import fitz
import faiss
import numpy as np
import json
import os

from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_path):
    """Extract text from PDF."""

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def create_chunks(text, chunk_size=500):
    """Split text into smaller chunks."""

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_text_faiss(pdf_path):

    # Extract text
    text = extract_text(pdf_path)

    if not text.strip():
        print("No text found in PDF.")
        return

    # Create chunks
    chunks = create_chunks(text)

    print("Total text chunks:", len(chunks))

    # Generate embeddings
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    # Create vectorstore folder
    os.makedirs("vectorstore", exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        "vectorstore/text_index.faiss"
    )

    # Save metadata
    metadata = []

    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": i,
            "text": chunk
        })

    with open(
        "vectorstore/text_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== TEXT FAISS RESULT ==========")
    print("Text FAISS created successfully!")
    print("Index: vectorstore/text_index.faiss")
    print("Metadata: vectorstore/text_metadata.json")
    print("Total chunks:", len(chunks))