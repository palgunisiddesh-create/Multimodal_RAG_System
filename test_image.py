from utils.image_extractor import extract_images
import os

# PDF file path
pdf_path = os.path.join("uploads", "sample.pdf")

# Folder to save extracted images
output_folder = "extracted_images"

# Check if PDF exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF not found -> {pdf_path}")
else:
    # Extract images
    image_paths = extract_images(pdf_path, output_folder)

    print("\n========== RESULT ==========")
    print(f"Total Images Extracted: {len(image_paths)}")

    if len(image_paths) == 0:
        print("No images found in the PDF.")
    else:
        print("\nSaved Images:")
        for img in image_paths:
            print(img)