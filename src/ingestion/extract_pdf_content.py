import os
import logging
import fitz  # PyMuPDF for PDF processing
from src.ingestion.text_cleaner import clean_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_pdf_content(pdf_path: str, texts_output_dir: str, images_output_dir: str):
    doc = fitz.open(pdf_path)
    base_pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_texts = []

    for page_idx, page in enumerate(doc):
        page_text = page.get_text("text").strip()
        if page_text:
            marker = f"[PAGE {page_idx+1}]\n"
            page_texts.append(f"{marker}{page_text}")

        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list, start=1):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            image_name = f"{base_pdf_name}_page_{page_idx+1}_img_{img_idx}.{ext}"
            image_path = os.path.join(images_output_dir, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(img_bytes)

    if page_texts:
        entire_pdf_text = "\n\n".join(page_texts)
        os.makedirs(texts_output_dir, exist_ok=True)
        text_file_path = os.path.join(texts_output_dir, f"{base_pdf_name}.txt")
        with open(text_file_path, "w", encoding='utf-8') as f:
            f.write(entire_pdf_text)

    doc.close()


def main():
    pdfs_dir = "src/ingestion/data/raw/lectures"
    texts_output_dir = "src/ingestion/data/processed/texts"
    images_output_dir = "src/ingestion/data/raw/images"

    os.makedirs(texts_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning("No PDF files found in the provided directory.")
        return

    logger.info(f"Found {len(pdf_files)} PDF files. Processing...")

    for pdf_fname in pdf_files:
        pdf_path = os.path.join(pdfs_dir, pdf_fname)
        logger.info(f"Extracting from {pdf_fname}...")
        extract_pdf_content(pdf_path, texts_output_dir, images_output_dir)

    logger.info("Extraction complete for all PDFs.")


if __name__ == "__main__":
    main()
