from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA
import os
import getpass


def get_response(retrieved_chunks, query):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", api_key=nvidia_api_key)
    chain = RetrievalQA(llm=llm, retriever=retrieved_chunks)
    response = chain.run(query)
    return response

