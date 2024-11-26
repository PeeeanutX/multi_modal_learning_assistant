from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
import getpass


def process_query(query):
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
