from utils.image_embedding import generate_image_embedding

caption = "a series of photos showing different landscapes"

embedding = generate_image_embedding(caption)

print("\n========== IMAGE EMBEDDING RESULT ==========")

print("Caption:")
print(caption)

print("\nEmbedding:")
print(embedding)

print("\nEmbedding Dimension:")
print(len(embedding))