import os
import logging
from typing import List, Optional, Union
from dataclasses import dataclass, field
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class VectorStoreConfig:
    store_type: str
    index_name: Optional[str] = None
    embedding_model: Optional[Embeddings] = None
    dimension: Optional[int] = None
    faiss_index_path: Optional[str] = None


class VectorStoreFactory:
    @staticmethod
    def create_vector_store(config: VectorStoreConfig, docs: List[Document]) -> VectorStore:
        """Factory method to create a vector store based on the configuration."""
        logger.info(f"Initializing vector store with type '{config.store_type}'")

        if config.store_type == 'faiss':
            if not config.embedding_model:
                raise ValueError("An embedding model must be provided for FAISS vector store.")

            if config.faiss_index_path and os.path.exists(config.faiss_index_path):
                logger.info(f"Loading FAISS index from '{config.faiss_index_path}'")
                vector_store = FAISS.load_local(
                    config.faiss_index_path,
                    config.embedding_model,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Creating new FAISS index")
                vector_store = FAISS.from_documents(docs, config.embedding_model)

                if config.faiss_index_path:
                    logger.info(f"Saving FAISS index to '{config.faiss_index_path}'")
                    vector_store.save_local(config.faiss_index_path)

            logger.info("FAISS vector store initialized")
            return vector_store
        else:
            raise ValueError(f"Unsupported vector store type: {config.store_type}")
