import sys
import os
import logging
import streamlit as st
from dotenv import load_dotenv

import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import TextChunker, ChunkerConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.processing.vector_store import VectorStoreFactory, VectorStoreConfig
from src.retrieval.query_processor import QueryProcessor, QueryProcessorConfig
from src.retrieval.retriever import DenseRetriever, RetrieverConfig as DRConfig
from src.models.llm_interface import LLMInterface, LLMConfig
from langchain.schema import Document

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    st.title("Learning Assistant Prototype")

    dense_retriever_path = "src/checkpoints/dense_retriever_checkpoint"
    vector_store_path = "src/ingestion/data/index.pkl"

    dr_config = DRConfig(
        dense_retriever_path=dense_retriever_path,
        query_model_name="intfloat/e5-base-v2",
        doc_model_name="intfloat/e5-base-v2",
        vectorstore_path=vector_store_path,
        top_k=5
    )
    retriever = DenseRetriever(dr_config)

    llm_config = LLMConfig(
        provider='nvidia',
        model_name='nvidia/llama-3.1-nemotron-70b-instruct',
        temperature=0.7,
        max_tokens=512
    )

    llm_interface = LLMInterface(config=llm_config, retriever=None)

    query = st.text_input("ENter your query:")
    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving documents..."):
                docs = retriever.retrieve(query, top_k=5)

            st.subheader("Retrieved Documents")
            for i, doc in enumerate(docs, start=1):
                st.write(f"**Document {i}:**")
                st.write(doc.page_content)

            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

            with st.spinner("Generating answer..."):
                response = llm_interface.llm.generate([prompt]).generations[0].text

            st.subheader("Final Answer")
            st.write(response)


if __name__ == "__main__":
    main()
