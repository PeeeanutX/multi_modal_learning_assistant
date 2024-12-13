#!/usr/bin/env python3
import os
import sys
import logging
import pickle

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import TextChunker, ChunkerConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.processing.vector_store import VectorStoreFactory, VectorStoreConfig
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables (e.g., for API keys)
load_dotenv()

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
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    Prepare data for the LLM-R pipeline:
    1. Load and extract text from PDFs.
    2. Clean the text.
    3. Chunk the text into manageable segments.
    4. Embed the chunks and build a vector store index.

    Parameters
    ----------
    pdf_dir : str
        Directory containing input PDF files.
    text_output_dir : str
        Directory to save extracted plain text files.
    processed_texts_dir : str
        Directory where processed .txt files are stored.
    index_file : str
        Path to save the FAISS index (.pkl file).
    chunks_file : str
        Path to save the chunks (.pkl file).
    embeddings_provider : str
        Provider for embeddings ('nvidia', 'openai', 'huggingface').
    embeddings_model : str
        Model name for the embeddings.
    chunk_size : int
        Desired chunk size for text splitting.
    chunk_overlap : int
        Overlap size between chunks.
    """

    # Step 1: Extract text from PDFs if not already done
    if not os.path.exists(text_output_dir):
        logger.info("Extracting text from PDFs...")
        load_pdfs(
            pdf_dir=pdf_dir,
            output_dir=text_output_dir,
            recursive=True,
            num_workers=4
        )
    else:
        logger.info("Text files already extracted, skipping PDF loading step.")

    # Step 2: Load and clean text from processed_texts_dir
    text_files = [
        f for f in os.listdir(processed_texts_dir)
        if f.lower().endswith('.txt')
    ]

    if not text_files:
        raise FileNotFoundError(
            f"No .txt files found in {processed_texts_dir}. "
            "Ensure that PDF extraction and text cleaning steps ran successfully."
        )

    all_texts = []
    logger.info("Cleaning extracted texts...")
    for txt_file in text_files:
        txt_path = os.path.join(processed_texts_dir, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        cleaned = clean_text(raw_text)
        if cleaned.strip():
            all_texts.append(cleaned)
        else:
            logger.warning(f"No usable text after cleaning in {txt_file}")

    # Step 3: Chunk the text
    logger.info("Chunking texts into smaller segments...")
    chunker_config = ChunkerConfig(
        method='recursive',  # could also be 'sentence', 'spacy', etc.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunker = TextChunker(chunker_config)

    all_chunks = []
    for text in all_texts:
        chunks = chunker.chunk_text(text)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text chunks were generated. Check your chunking configuration.")

    # Step 4: Convert chunks into Document objects
    docs = [Document(page_content=chunk) for chunk in all_chunks]
    if not docs:
        raise ValueError("No documents created from text chunks.")

    # Step 5: Embeddings and Vector Store Creation
    logger.info(f"Initializing embeddings model '{embeddings_model}' from provider '{embeddings_provider}'...")
    embeddings_config = EmbeddingsConfig(
        provider=embeddings_provider,
        model_name=embeddings_model,
        api_key=None  # Set via environment variable if needed
    )
    embeddings_model = EmbeddingsFactory.get_embeddings_model(embeddings_config)

    vector_store_config = VectorStoreConfig(
        store_type='faiss',
        embedding_model=embeddings_model,
        faiss_index_path=index_file
    )

    logger.info("Creating/loading vector store index...")
    vector_store = VectorStoreFactory.create_vector_store(vector_store_config, docs=docs)

    # Step 6: Save the chunks if needed
    with open(chunks_file, 'wb') as f:
        pickle.dump(all_chunks, f)
    logger.info(f"Saved chunks to {chunks_file}")

    logger.info("Data preparation complete. The vector store is ready and chunks are saved.")


if __name__ == "__main__":
    # Example usage:
    # Adjust these paths to match your project's structure
    pdf_dir = 'src/ingestion/data/raw/lectures/'
    text_output_dir = 'src/ingestion/data/processed/texts/'
    processed_texts_dir = text_output_dir
    index_file = 'src/ingestion/data/index.pkl'
    chunks_file = 'src/ingestion/data/chunks.pkl'

    prepare_data(
        pdf_dir=pdf_dir,
        text_output_dir=text_output_dir,
        processed_texts_dir=processed_texts_dir,
        index_file=index_file,
        chunks_file=chunks_file,
        embeddings_provider='nvidia',  # change as needed
        embeddings_model='NV-Embed-QA',  # change as needed
        chunk_size=500,
        chunk_overlap=50
    )
