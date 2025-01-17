#!/usr/bin/env python3
import os
import sys
import logging
import pickle

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.ingestion.extract_pdf_content import extract_pdf_content
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import TextChunker, ChunkerConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.processing.vector_store import VectorStoreFactory, VectorStoreConfig
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(
        pdf_dir: str,
        text_output_dir: str,
        processed_texts_dir: str,
        index_file: str,
        chunks_file: str,
        embeddings_provider: str = 'nvidia',
        embeddings_model: str = 'NV-Embed-QA',
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        processed_image_texts_dir: str = "",  # optional
):
    """
    Prepare data for the LLM-R pipeline using SemanticChunker for chunking:
      1. Extract text from PDFs if needed.
      2. Clean the text.
      3. Split the text semantically.
      4. Embed the chunks & build a FAISS index.
      5. Save chunk strings as a pickle for future retrieval steps.
    """

    # Step 1: Extract PDF text if not done
    if not os.path.exists(text_output_dir):
        logger.info("Extracting text from PDFs (PyMuPDF + [PAGE X]) ...")
        os.makedirs(text_output_dir, exist_ok=True)
        for fname in os.listdir(pdf_dir):
            if fname.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, fname)
                extract_pdf_content(pdf_path, text_output_dir, images_output_dir="src/ingestion/data/raw/images")
    else:
        logger.info("Text files already extracted, skipping PDF extraction.")

    # Step 2: Load & clean text
    text_files = [f for f in os.listdir(processed_texts_dir) if f.lower().endswith('.txt')]
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {processed_texts_dir}. Did extraction run?")

    raw_texts = []
    for txt_file in text_files:
        with open(os.path.join(processed_texts_dir, txt_file), 'r', encoding='utf-8') as f:
            text_data = f.read()
        cleaned = clean_text(text_data)
        if cleaned.strip():
            raw_texts.append(cleaned)
        else:
            logger.warning(f"No usable text after cleaning {txt_file}")

    if processed_image_texts_dir and os.path.isdir(processed_image_texts_dir):
        for fname in os.listdir(processed_image_texts_dir):
            if fname.lower().endswith('.txt'):
                with open(os.path.join(processed_image_texts_dir, fname), 'r', encoding='utf-8') as f:
                    img_caption = f.read().strip()
                if img_caption:
                    raw_texts.append(img_caption)

    logger.info("Chunking text with a RecursiveCharacterTextSplitter approach...")
    chunker_config = ChunkerConfig(
        method='recursive',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunker = TextChunker(chunker_config)

    all_chunk_strs = []
    for text in raw_texts:
        chunks = chunker.chunk_text(text)
        all_chunk_strs.extend(chunks)

    docs = [Document(page_content=c) for c in all_chunk_strs]

    logger.info(f"Initializing embeddings with provider={embeddings_provider}, model={embeddings_model}")
    embeddings_config = EmbeddingsConfig(
        provider=embeddings_provider,
        model_name=embeddings_model
    )
    embed_model = EmbeddingsFactory.get_embeddings_model(embeddings_config)

    vector_store_config = VectorStoreConfig(
        store_type='faiss',
        embedding_model=embed_model,
        faiss_index_path=index_file
    )
    logger.info(f"Creating or loading FAISS index at {index_file} ...")
    vectorstore = VectorStoreFactory.create_vector_store(vector_store_config, docs=docs)
    logger.info("FAISS index ready.")

    # Step 6: Save chunk strings to pickle
    with open(chunks_file, 'wb') as f:
        pickle.dump(all_chunk_strs, f)
    logger.info(f"Saved {len(all_chunk_strs)} semantic chunks to {chunks_file}")


if __name__ == "__main__":
    pdf_dir = 'src/ingestion/data/raw/lectures/'
    text_output_dir = 'src/ingestion/data/processed/texts/'
    processed_texts_dir = text_output_dir
    index_file = 'src/ingestion/data/index.pkl'
    chunks_file = 'src/ingestion/data/chunks.pkl'
    processed_image_texts_dir = 'src/ingestion/data/processed/image_texts'

    prepare_data(
        pdf_dir=pdf_dir,
        text_output_dir=text_output_dir,
        processed_texts_dir=processed_texts_dir,
        index_file=index_file,
        chunks_file=chunks_file,
        embeddings_provider='nvidia',
        embeddings_model='NV-Embed-QA',
        chunk_size=512,
        chunk_overlap=50,
        processed_image_texts_dir=processed_image_texts_dir
    )
