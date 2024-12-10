import numpy as np
import logging
from typing import List
from dataclasses import dataclass, field

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.retrievers import (
    TimeWeightedVectorStoreRetriever,
    ContextualCompressionRetriever
)
from src.retrieval.reranker import ReRanker
from langchain_core.language_models import BaseLanguageModel

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
    retriever_type: str = 'default'  # Options: 'default', 'mmr', 'time_weighted', 'contextual'
    search_kwargs: dict = field(default_factory=lambda: {"k": 5})
    mmr_kwargs: dict = field(default_factory=lambda: {"k": 5, "fetch_k": 20, "lambda_mult": 0.5})
    time_weighted_kwargs: dict = field(default_factory=lambda: {"k": 5, "decay_rate": 0.1})
    contextual_kwargs: dict = field(default_factory=lambda: {"k": 5})


class RetrieverFactory:
    @staticmethod
    def create_retriever(vector_store: VectorStore, config: RetrieverConfig):
        """Factory method to create a retriever based on the configuration"""
        logger.info(f"Initializing retriever of type '{config.retriever_type}'")

        if config.retriever_type == 'default':
            retriever = vector_store.as_retriever(search_kwargs=config.search_kwargs)
            logger.info("Default VectorStoreRetriever initialized")
        # elif config.retriever_type == 'mmr':
        #   retriever = MMRetriever(
        #       vectorstore=vector_store,
        #        **config.mmr_kwargs
        #   )
        #     logger.info("MMRetriever initialized with Maximum Marginal Relevance")
        elif config.retriever_type == 'time_weighted':
            retriever = TimeWeightedVectorStoreRetriever(
                vectorstore=vector_store,
                **config.time_weighted_kwargs
            )
            logger.info("TimeWeightedVectorStoreRetriever initialized")
        elif config.retriever_type == 'contextual':
            retriever = ContextualCompressionRetriever(
                base_compressor=None,  # TODO: Define compressor
                base_retriever=vector_store.as_retriever(search_kwargs=config.search_kwargs),
                **config.contextual_kwargs
            )
            logger.info("ContextualCompressionRetriever initialized")
        else:
            raise ValueError(f"Unsupported retriever type: {config.retriever_type}")

        return retriever

    @staticmethod
    def retrieve_documents(retriever, query: str, llm: BaseLanguageModel) -> List[Document]:
        """Retrieve relevant documents using the specified retriever and re-rank them."""
        logger.info(f"Retrieving documents for query: '{query}'")
        try:
            documents = retriever.invoke(query)
            logger.info(f"Retrieved {len(documents)} documents")

            re_ranker = ReRanker(llm)
            ranked_documents = re_ranker.re_rank(query, documents)

            logger.info(f"Re-ranked {len(ranked_documents)} documents")
            return ranked_documents
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise


def retrieve_chunks(query_embeddings, index, chunks, top_k=5):
    distances, indices = index.search(np.array([query_embeddings]).astype('float32'), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks
