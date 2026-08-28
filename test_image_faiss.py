from utils.image_faiss import store_image_in_faiss
import os

image_path = "extracted_images/page_1_image_1.png"

caption = "a series of photos showing different landscapes"

if os.path.exists(image_path):

    store_image_in_faiss(
        image_path,
        caption
    )

    print("\n========== FAISS RESULT ==========")
    print("Image successfully stored in FAISS.")

else:
    print("Image not found!")