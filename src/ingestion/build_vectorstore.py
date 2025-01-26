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


def parse_lecture_info(base_name: str):
    """
    A helper function to parse out lecture_number and lecture_name
    from a merged filename.

    For example, if your files look like:
        2024_AIBIS_Lecture_07_AI-Enabled_Insights_&_Decisions_merged.txt
    you might do something like:
        splitted = base_name.split("_")
        # splitted might be ["2024","AIBIS","Lecture","07","AI-Enabled","Insights","&","Decisions","merged"]

    Then pick the right pieces for your metadata.
    This is just an example—adapt to match your actual naming scheme!
    """
    splitted = base_name.split("_")
    lecture_num = "N/A"
    lecture_name = base_name  # fallback

    # Suppose splitted[3] is the lecture number, parted out from '07'
    if len(splitted) > 3 and splitted[3].isdigit():
        lecture_num = splitted[3]
    # Suppose everything after index 4 up until "_merged" is the lecture name
    if "merged" in splitted:
        idx_merged = splitted.index("merged")
        # e.g. splitted[4:idx_merged] => ["AI-Enabled","Insights","&","Decisions"]
        lecture_name = " ".join(splitted[4:idx_merged]).replace("-", " ")
    else:
        # if not 'merged' in filename, just do something else
        lecture_name = " ".join(splitted[4:]).replace("-", " ")

    return lecture_num, lecture_name


def load_pages_as_documents(file_path: str) -> List[Document]:
    """
    For page-level chunking: parse [PAGE X] blocks from the merged text.
    Return a list of Document(page_content=..., metadata={...}).
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    lecture_num, lecture_name = parse_lecture_info(base_name)

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
                "lecture_number": lecture_num,
                "lecture_name": lecture_name,
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
        model_name="BAAI/bge-large-en-v1.5",
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
