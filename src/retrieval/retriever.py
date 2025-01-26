import os
import logging
import torch
from dataclasses import dataclass
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel
from src.processing.embedder import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class RetrieverConfig:
    faiss_index_path: str
    top_k: int = 10


class VectorStoreRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
        )

        logger.info(f"Loading FAISS vector store from {self.config.faiss_index_path}")
        self.vectorstore = FAISS.load_local(
            folder_path=self.config.faiss_index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization = True
        )
        logger.info("VectorStoreRetriever initialized successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        if top_k is None:
            top_k = self.config.top_k

        docs = self.vectorstore.similarity_search(query, k=top_k)
        return docs
