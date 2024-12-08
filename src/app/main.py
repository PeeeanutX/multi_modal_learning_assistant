import sys
import os
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import TextChunker, ChunkerConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.processing.vector_store import VectorStoreFactory, VectorStoreConfig
from src.retrieval.query_processor import QueryProcessor, QueryProcessorConfig
from src.retrieval.retriever import RetrieverFactory, RetrieverConfig
from src.models.llm_interface import get_nvidia_response
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


def main():
    st.title("Learning Assistant Prototype")

    pdf_dir = 'src/ingestion/data/raw/lectures/'
    text_output_dir = 'src/ingestion/data/processed/texts/'
    processed_texts_dir = 'src/ingestion/data/processed/texts/'
    index_file = 'src/ingestion/data/index.pkl'
    chunks_file = 'src/ingestion/data/chunks.pkl'

    if os.path.exists(index_file) and os.path.exists(chunks_file):
        st.info("Loading preprocessed data...")
        with open(chunks_file, 'rb') as f:
            all_chunks = pickle.load(f)

        if not all_chunks:
            raise ValueError("No chunks found. Ensure the preprocessing pipeline is correct.")

        embeddings_config = EmbeddingsConfig(
            provider='nvidia',
            model_name='NV-Embed-QA',
            api_key=None
        )
        embeddings_model = EmbeddingsFactory.get_embeddings_model(embeddings_config)

        vector_store_config = VectorStoreConfig(
            store_type='faiss',
            embedding_model=embeddings_model,
            faiss_index_path=index_file
        )
        vector_store = VectorStoreFactory.create_vector_store(vector_store_config, docs=[])
    else:
        st.info("Processing data... This may take a while.")
        load_pdfs(pdf_dir, text_output_dir)

        config = ChunkerConfig(
            method='recursive',  # or other methods like 'sentence', 'spacy'
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunker = TextChunker(config)

        texts = []
        for text_file in os.listdir(processed_texts_dir):
            if text_file.endswith('.txt'):
                with open(os.path.join(processed_texts_dir, text_file), 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_text = clean_text(raw_text)
                    texts.append(cleaned_text)

        all_chunks = []
        for text in texts:
            chunks = chunker.chunk_text(text)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No text chunks were generated.")

        docs = [Document(page_content=chunk) for chunk in all_chunks]
        if not docs:
            raise ValueError("No documents were created from the text chunks.")

        embeddings_config = EmbeddingsConfig(
            provider='nvidia',
            model_name='NV-Embed-QA',
            api_key=None
        )
        embeddings_model = EmbeddingsFactory.get_embeddings_model(embeddings_config)

        vector_store_config = VectorStoreConfig(
            store_type='faiss',
            embedding_model=embeddings_model,
            faiss_index_path=index_file
        )
        vector_store = VectorStoreFactory.create_vector_store(vector_store_config, docs=docs)

        with open(chunks_file, 'wb') as f:
            pickle.dump(all_chunks, f)

    retriever_config = RetrieverConfig(
        retriever_type='default',
        search_kwargs={"k": 5}
    )

    retriever = RetrieverFactory.create_retriever(vector_store, retriever_config)

    query_processor_config = QueryProcessorConfig(
        normalize=True
    )
    query_processor = QueryProcessor(query_processor_config)

    query = st.text_input("Ask me anything about the lectures:")
    if query:
        response = get_nvidia_response(retriever, query)
        st.subheader("Assistant Response")
        st.write(response)


if __name__ == "__main__":
    main()
