from langchain_openai import OpenAIEmbeddings
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


def generate_embeddings(chunks):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA")
    embeddings = embeddings_model.embed_documents(chunks)
    return embeddings


def get_embeddings_model():
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA", api_key=nvidia_api_key)
    return embeddings_model
