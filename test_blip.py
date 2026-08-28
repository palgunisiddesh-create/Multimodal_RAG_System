from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

MODEL_NAME = "Salesforce/blip-image-captioning-base"

IMAGE_FOLDER = "extracted_images"

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(MODEL_NAME)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

print("BLIP model loaded successfully.")

# Find images
image_files = []

for filename in os.listdir(IMAGE_FOLDER):

    if filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp")
    ):

        image_files.append(
            os.path.join(IMAGE_FOLDER, filename)
        )

if not image_files:

    print("No images found in extracted_images folder.")
    exit()

print(f"\nFound {len(image_files)} images.")

# Process images
for image_path in image_files:

    print("\n" + "=" * 60)
    print("IMAGE:", image_path)
    print("=" * 60)

    try:

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        output = model.generate(
            **inputs,
            max_new_tokens=50
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        print("Caption:")
        print(caption)

    except Exception as e:

        print("Error processing image:")
        print(e)

print("\n")
print("=" * 60)
print("BLIP IMAGE CAPTIONING COMPLETE")
print("=" * 60)