import sys
import os

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import chunk_text
from src.processing.embedder import generate_embeddings
from src.processing.vector_store import create_vector_store
from src.retrieval.query_processor import process_query
from src.retrieval.retriever import retrieve_chunks
from src.models.llm_interface import get_response


def main():
    st.title("Learning Assistant Prototype")

    pdf_dir = 'data/raw/lectures/'
    text_output_dir = 'data/processed/texts/'
    processed_texts_dir = 'data/processed/texts/'

    st.info("Loading and processing PDFs...")
    load_pdfs(pdf_dir, text_output_dir)

    st.info("Cleaning text...")
    texts = []
    for text_file in os.listdir(processed_texts_dir):
        if text_file.endswith('.txt'):
            with open(os.path.join(processed_texts_dir, text_file), 'r', encoding='utf-8') as f:
                raw_text = f.read()
                cleaned_text = clean_text(raw_text)
                texts.append(cleaned_text)

    st.info("Splitting text into chunks...")
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    st.info("Generating embeddings...")
    embeddings = generate_embeddings(all_chunks)

    st.info("Creating vector store")
    index = create_vector_store(embeddings)

    query = st.text_input("Ask me anything about the lectures:")
    if query:
        query_embedding = process_query(query)
        retrieved_chunks = retrieve_chunks(query_embedding, index, all_chunks)
        response = get_response(retrieved_chunks, query)
        st.write(response)


if __name__ == "__main__":
    main()