import faiss
import numpy as np
import os
import json

from utils.image_embedding import generate_image_embedding


def store_image_in_faiss(image_path, caption):
    """
    Convert image caption into an embedding
    and store it in FAISS.
    """

    # Generate embedding
    embedding = generate_image_embedding(caption)

    # Convert to numpy array
    embedding = np.array([embedding]).astype("float32")

    # Create FAISS index
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embedding
    index.add(embedding)

    # Save FAISS index
    os.makedirs("vectorstore", exist_ok=True)

    index_path = "vectorstore/image_index.faiss"
    faiss.write_index(index, index_path)

    # Store image information
    metadata = {
        "image_path": image_path,
        "caption": caption
    }

    with open("vectorstore/image_metadata.json", "w") as f:
        json.dump([metadata], f, indent=4)

    print("Image embedding stored successfully!")
    print("FAISS index:", index_path)
    print("Image:", image_path)
    print("Caption:", caption)