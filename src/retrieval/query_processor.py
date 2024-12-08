from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
import logging
from typing import Optional, List
from dataclasses import dataclass
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class QueryProcessorConfig:
    """
    Configuration for the QueryProcessor.
    """
    normalize: bool = True


class QueryProcessor:
    def __init__(self, config: QueryProcessorConfig):
        """Initialize the QueryProcessor with a specific configuration."""
        self.config = config
        logger.info("QueryProcessor initialized with configuration:")
        logger.info(f"{self.config}")

    def process_query(self, query: str) -> str:
        """Process and embed the input query based on the configuration."""
        logger.info(f"Processing query: '{query}'")
        try:
            if self.config.normalize:
                query = self._normalize_text(query)
                logger.debug(f"After normalization: '{query}'")

            logger.info("Query embedded successfully")
            return query

        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            raise

    def _normalize_text(self, text: str) -> str:
        """Normalize text by stripping leading/trailing whitespace."""
        return text.strip()

"""       
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
        try:
            embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA")
            query_embedding = embeddings_model.embed_query(query)
            return query_embedding
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
"""
