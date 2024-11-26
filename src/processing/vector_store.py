from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def create_vector_store(embeddings_model, chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(docs, embeddings_model)
    return vector_store
