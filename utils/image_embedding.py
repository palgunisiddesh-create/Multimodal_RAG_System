from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_image_embedding(caption):
    embedding = model.encode(caption)
    return embedding