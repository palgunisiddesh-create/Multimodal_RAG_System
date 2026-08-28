import fitz
import os

def extract_images(pdf_path, output_folder):
    """
    Extract all images from a PDF and save them to the output folder.
    Returns a list of image file paths.
    """

    os.makedirs(output_folder, exist_ok=True)

    pdf = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)

        images = page.get_images(full=True)

        for img_index, img in enumerate(images):

            xref = img[0]
            base_image = pdf.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"page_{page_num+1}_image_{img_index+1}.{image_ext}"
            image_path = os.path.join(output_folder, image_name)

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            image_paths.append(image_path)
            print(f"Saved: {image_name}")

    pdf.close()

    return image_paths