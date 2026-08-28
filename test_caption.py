from utils.image_caption import generate_caption
import os

image_path = os.path.join("extracted_images", "page_1_image_1.png")

if os.path.exists(image_path):

    caption = generate_caption(image_path)

    print("\n========== RESULT ==========")
    print("Image :", image_path)
    print("Caption :", caption)

else:
    print("Image not found!")