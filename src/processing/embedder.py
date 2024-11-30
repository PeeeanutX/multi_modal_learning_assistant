import os
import logging
from typing import Optional
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class EmbeddingsConfig:
    """Configuration for embedding models."""
    provider: str = "nvidia "  # Options: 'nvidia, 'openai', 'huggingface'
    model_name: Optional[str] = "NV-Embed-QA"  # e.g., 'NV-Embed-QA' for NVIDIA
    api_key: Optional[str] = "NVIDIA_API_KEY"
    device: Optional[str] = 'cpu'  # Device for HuggingFace embeddings


class EmbeddingsFactory:
    @staticmethod
    def get_embeddings_model(config: EmbeddingsConfig):
        """Factory method to get an embedding model based on the configuration."""
        logger.info(f"Initializing embeddings model with provider '{config.provider}")
        if config.provider == 'nvidia':
            if not config.api_key:
                config.api_key = os.getenv('NVIDIA_API_KEY')
                if not config.api_key:
                    raise ValueError("NVIDIA_API_KEY environment variable not set.")
            model_name = config.model_name or 'NV-Embed-QA'
            embeddings_model = NVIDIAEmbeddings(model=model_name, api_key=config.api_key)
            logger.info(f"NVIDIA embeddings model '{model_name}' initialized")
        elif config.provider == 'openai':
            if not config.api_key:
                config.api_key = os.getenv('OPENAI_API_KEY')
                if not config.api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set.")
            model_name = config.model_name or 'text-embedding-3-large'
            embeddings_model = OpenAIEmbeddings(open_api_key=config.api_key, model=model_name)
            logger.info(f"OpenAI embeddings model '{model_name}' initialized")
        elif config.provider == 'huggingface':
            model_name = config.model_name or 'sentence-transformers/all-mpnet-base-v2'
            embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': config.device})
            logger.info(f"HuggingFace embeddings model '{model_name}' initialized on device '{config.device}'.")
        else:
            raise ValueError(f"Unsupported embeddings provider: {config.provider}")
        return embeddings_model


def oldget_embeddings_model():
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA", api_key=nvidia_api_key)
    return embeddings_model
