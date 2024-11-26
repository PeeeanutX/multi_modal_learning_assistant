from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
import os


def get_nvidia_response(retriever: BaseRetriever, query: str):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")

    llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct", api_key=nvidia_api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    response = qa_chain.run(query)
    return response
