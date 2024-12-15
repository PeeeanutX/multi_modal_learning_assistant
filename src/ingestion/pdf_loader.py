import os
import logging
from typing import List, Optional
from PyPDF2 import PdfReader
from multiprocessing import Pool, cpu_count
from functools import partial


def setup_logger():
    """Set up the logger for the module."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page_number, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logger.warning(f"No text found on page {page_number} of {pdf_path}")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_number} of {pdf_path}: {e}")
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to read {pdf_path}: {e}")
    return None


def process_pdf_file(pdf_file: str, pdf_dir: str, output_dir: str) -> None:
    """Process a single PDF file: extract text and save it as a .txt file."""
    pdf_path = os.path.join(pdf_dir, pdf_file)
    logger.info(f"Processing {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if text:
        output_file = os.path.join(output_dir, pdf_file.replace('.pdf', '.txt'))
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to output_file")
        except Exception as e:
            logger.error(f"Failed to write text to {output_file}: {e}")
    else:
        logger.warning(f"No text extracted from {pdf_path}, skipping.")


def load_pdfs(
        pdf_dir: str,
        output_dir: str,
        recursive: bool = False,
        num_workers: int = None,
        max_files: int = None,
        include_encrypted: bool = False
) -> None:
    """Load PDFs from a directory, extract text, and save as .txt files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory {output_dir}")

    pdf_files = []
    if recursive:
        for root, _, files in os.walk(pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.relpath(os.path.join(root, file), pdf_dir))
    else:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    if not num_workers:
        num_workers = max(1, cpu_count() - 1)

    logger.info(f"Using {num_workers} worker processes.")

    worker_func = partial(process_pdf_file, pdf_dir=pdf_dir, output_dir=output_dir)

    with Pool(num_workers) as pool:
        pool.map(worker_func, pdf_files)


if __name__ == "__main__":
    load_pdfs(
        pdf_dir='data/raw/lectures/',
        output_dir='data/processed/texts/',
        recursive=True,
        num_workers=4,
        max_files=None,
        include_encrypted=False
    )

"""
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            reader = PdfReader(os.path.join(pdf_dir, pdf_file))
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            with open(os.path.join(output_dir, pdf_file.replace('.pdf', '.txt')), 'w', encoding='utf-8') as f:
                f.write(text)
"""
