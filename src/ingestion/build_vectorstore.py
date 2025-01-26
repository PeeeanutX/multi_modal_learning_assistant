import os
import re
import logging
import pickle
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from src.processing.embedder import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PAGE_MAGER_REGEX = re.compile(r"\[PAGE\s+(\d+)\]")


def load_pages_as_documents(file_path: str) -> List[Document]:
    """
    For page-level chunking: parse [PAGE X] blocks from the merged text.
    Return a list of Document(page_content=..., metadata={...}).
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    parts = PAGE_MAGER_REGEX.split(full_text)
    docs = []
    for i in range(1, len(parts), 2):
        page_str = parts[i].strip()
        page_text = parts[i + 1].strip()
        if not page_text:
            continue
        page_num = page_str
        doc = Document(
            page_content=page_text,
            metadata={
                "source": base_name,
                "page_num": page_num
            }
        )
        docs.append(doc)
    return docs


def build_vector_store(
        merged_dir="src/ingestion/data/processed/merged",
        faiss_index_path="src/ingestion/data/index"
):
    """
    1) For each merged .txt in merged_dir, parse it into page_level Documents.
    2) Use Jina embeddings to embed them,
    3) Build a FAISS index and save it locally.
    """
    text_files = [f for f in os.listdir(merged_dir) if f.endswith(".txt")]
    all_docs = []
    for fname in text_files:
        path = os.path.join(merged_dir, fname)
        page_docs = load_pages_as_documents(path)
        all_docs.extend(page_docs)

    logger.info(f"Total {len(all_docs)} page-level docs found.")

    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"device": "cuda"}
    )

    logger.info("Creating FAISS vectorstore from page docs...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    vectorstore.save_local(faiss_index_path)
    logger.info(f"FAISS index saved to: {faiss_index_path}")


def main():
    build_vector_store()


if __name__ == "__main__":
    main()
