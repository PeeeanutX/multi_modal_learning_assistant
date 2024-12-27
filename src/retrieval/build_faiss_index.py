import os
import sys
import torch
import logging
from typing import List
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DENSE_RETRIEVER_PATH = "src/checkpoints/dense_retriever_checkpoint"
FAISS_INDEX_OUTPUT_PATH = "src/ingestion/data/new_index.pkl"

TEXTS_DIR = "src/ingestion/data/processed/texts"
IMAGE_TEXTS_DIR = "src/ingestion/data/processed/image_texts"

tokenizer = AutoTokenizer.from_pretrained(DENSE_RETRIEVER_PATH, use_fast=True)


def extract_filename_metadata(fname: str):
    """
    Example:
      2024_AIBIS_Lecture_06_Human-Centered Design.txt
      2024_AIBIS_Lecture_06_Human-Centered Design_page_17_img_3.txt
    This function will parse out e.g. "06" for lecture_number, etc.
    """
    meta = {}
    # Regex to capture lecture number, name, page, optional image number
    pattern = re.compile(
        r"2024_AIBIS_Lecture_(\d+)_(.+)\.txt$"
    )
    match = pattern.search(fname)
    if match:
        meta["lecture_number"] = match.group(1)  # e.g. "06"
        meta["lecture_name"] = match.group(2)  # e.g. "Human-Centered Design"

    return meta


def chunk_text(text, chunk_token_limit=512) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_token_limit):
        token_chunk = tokens[i : i + chunk_token_limit]
        chunk_text = tokenizer.decode(token_chunk, skip_special_tokens=True)
        chunks.append(chunk_text.strip())
    return chunks


def load_text_and_images() -> List[Document]:
    docs = []
    tokenizer = AutoTokenizer.from_pretrained(DENSE_RETRIEVER_PATH, use_fast=True)

    if os.path.exists(TEXTS_DIR):
        for fname in os.listdir(TEXTS_DIR):
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(TEXTS_DIR, fname)
            with open(full_path, "r", encoding='utf-8') as f:
                entire_text = f.read().strip()

            meta = extract_filename_metadata(fname)
            meta["source"] = fname
            meta["type"] = "pdf_text"

            docs.append(Document(page_content=entire_text, metadata=meta))

    if os.path.exists(IMAGE_TEXTS_DIR):
        for fname in os.listdir(IMAGE_TEXTS_DIR):
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(IMAGE_TEXTS_DIR, fname)
            with open(full_path, "r", encoding='utf-8') as f:
                caption = f.read().strip()

            pattern = re.compile(r"page_(\d+)_img_(\d+)")
            match = pattern.search(fname)
            if match:
                page = match.group(1)
                img_idx = match.group(2)
            else:
                page = "??"
                img_idx = "??"
            meta = {
                "source": fname,
                "type": "image_description",
                "page": page,
                "img_idx": img_idx
            }

            docs.append(Document(page_content=caption, metadata=meta))
    else:
        logger.warning(f"No directory found at {TEXTS_DIR}")

    logger.info(f"Loaded {len(docs)} total document chunks (text + image-derived).")
    return docs


class DenseRetrieverEmbedder:
    """
    A wrapper class to produce document embeddings using the trained dense retriever model.
    It uses the doc encoder of the model to get embeddings.
    """
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        logger.info("Loading dense retriever model for embeddings...")
        self.doc_encoder = AutoModel.from_pretrained(model_path).to(self.device)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, docs: List[Document]) -> List[List[float]]:
        embeddings = []
        for doc in docs:
            text = doc.page_content
            enc = self.doc_tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.doc_encoder(input_ids, attention_mask=attention_mask)
                doc_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
            embeddings.append(doc_emb.squeeze(0).cpu().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # For queries, treat them like single-doc embedding
        doc = Document(page_content=text)
        return self.embed_documents([doc])[0]



def main():
    logger.info("Loading documents...")
    docs = load_text_and_images()
    logger.info(f"Loaded {len(docs)} chunked documents.")

    embedder = DenseRetrieverEmbedder(DENSE_RETRIEVER_PATH, use_gpu=True)
    logger.info("Embedding documents with LLM-R trained dense retriever model...")
    doc_embeddings = embedder.embed_documents(docs)
    logger.info("Building FAISS index from dense retriever embeddings...")

    class PrecomputedEmbeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return doc_embeddings

        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError("Use dense model query encoder directly for query embedding.")

    precomputed_embedder = PrecomputedEmbeddings()

    vectorstore = FAISS.from_documents(docs, precomputed_embedder)

    logger.info(f"Saving FAISS index to {FAISS_INDEX_OUTPUT_PATH}...")
    vectorstore.save_local(FAISS_INDEX_OUTPUT_PATH)
    logger.info("FAISS index built successfully with LLM-R embeddings and integrated image-derived text.")


if __name__ == "__main__":
    main()
