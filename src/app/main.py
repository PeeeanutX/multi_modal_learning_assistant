import sys
import os
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import chunk_text
from src.processing.embedder import get_embeddings_model
from src.processing.vector_store import create_vector_store
from src.retrieval.query_processor import process_query
from src.retrieval.retriever import retrieve_chunks
from src.models.llm_interface import get_nvidia_response


def main():
    st.title("Learning Assistant Prototype")

    pdf_dir = 'src/ingestion/data/raw/lectures/'
    text_output_dir = 'src/ingestion/data/processed/texts/'
    processed_texts_dir = 'src/ingestion/data/processed/texts/'
    index_file = 'src/ingestion/data/index.pkl'
    chunks_file = 'src/ingestion/data/chunks.pkl'

    if os.path.exists(index_file) and os.path.exists(chunks_file):
        st.info("Loading preprocessed data...")
        with open(index_file, 'rb') as f:
            vector_store = pickle.load(f)
        with open(chunks_file, 'rb') as f:
            all_chunks = pickle.load(f)
    else:
        st.info("Processing data... This may take a while.")

    load_pdfs(pdf_dir, text_output_dir)

    texts = []
    for text_file in os.listdir(processed_texts_dir):
        if text_file.endswith('.txt'):
            with open(os.path.join(processed_texts_dir, text_file), 'r', encoding='utf-8') as f:
                raw_text = f.read()
                cleaned_text = clean_text(raw_text)
                texts.append(cleaned_text)

    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    embeddings_model = get_embeddings_model()

    st.info("Creating vector store")
    vector_store = create_vector_store(embeddings_model, all_chunks)

    with open(index_file, 'wb') as f:
        pickle.dump(vector_store, f)
    with open(chunks_file, 'wb') as f:
        pickle.dump(all_chunks, f)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    query = st.text_input("Ask me anything about the lectures:")
    if query:
        """query_embedding = process_query(query)
        docs = retriever.get_relevant_documents(query)"""
        response = get_nvidia_response(retriever, query)
        st.write(response)


if __name__ == "__main__":
    main()
