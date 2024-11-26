import os
import logging
from typing import List, Optional
from PyPDF2 import PdfReader
from multiprocessing import Pool, cpu_count


def setup_logger(log_level=logging.INFO) -> logging.Logger:
    """
    Sets up the logger for the module.

    Args:
        log_level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

logger = setup_logger()


def load_pdfs(pdf_dir, output_dir):
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            reader = PdfReader(os.path.join(pdf_dir, pdf_file))
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            with open(os.path.join(output_dir, pdf_file.replace('.pdf', '.txt')), 'w', encoding='utf-8') as f:
                f.write(text)


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logger.warning(f"No text found on page {page_num} of {pdf_path}")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
        return text
    except Exception as e:
        logger.error(f"Failed to read {pdf_path}: {e}")
        return None
