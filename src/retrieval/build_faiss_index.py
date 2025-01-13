import os
import sys
import torch
import logging
import re
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DENSE_RETRIEVER_PATH = "src/checkpoints2/dense_retriever_checkpoint"
FAISS_INDEX_OUTPUT_PATH = "src/ingestion/data/new_index.pkl"

TEXTS_DIR = "src/ingestion/data/processed/texts"
IMAGE_TEXTS_DIR = "src/ingestion/data/processed/image_texts"

PAGE_MARKER_REGEX = re.compile(r"\[PAGE\s*(\d+)\]")

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


def chunk_text(doc: Document, tokenizer, chunk_token_limit=512) -> List[Document]:
    text = doc.page_content
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[Document] = []

    start_idx = 0
    for i in range(0, len(tokens), chunk_token_limit):
        token_chunk = tokens[i : i + chunk_token_limit]
        chunk_text = tokenizer.decode(token_chunk, skip_special_tokens=True).strip()

        new_meta = dict(doc.metadata)
        new_meta["chunk_start"] = i
        chunk_doc = Document(page_content=chunk_text, metadata=new_meta)
        chunks.append(chunk_doc)

    start_idx += len(token_chunk)

    return chunks


def load_texts_as_docs() -> List[Document]:
    docs: List[Document] = []
    for fname in os.listdir(TEXTS_DIR):
        if not fname.endswith(".txt"):
            continue
        full_path = os.path.join(TEXTS_DIR, fname)
        with open(full_path, "r", encoding='utf-8') as f:
            entire_text = f.read()

        # Example metadata handling
        meta = {"source": fname}
        doc = Document(page_content=entire_text, metadata=meta)
        docs.append(doc)
    return docs


def load_entire_text_as_single_doc(full_path: str, base_meta: dict) -> Document:
    """
    Read the entire .txt file (with [PAGE X] markers),
    return a single Document plus store the raw text for page-boundary detection.
    """
    with open(full_path, "r", encoding='utf-8') as f:
        entire_text = f.read()
    doc = Document(page_content=entire_text, metadata=base_meta)
    return doc


def find_page_boundaries(text: str) -> List[Tuple[int, str]]:
    """
    Return a sorted list of (char_offset, page_number) for each [PAGE X] marker found.
    Example: If text has "...[PAGE 2]..." at index 500, return (500, "2").
    """
    boundaries = []
    for match in PAGE_MARKER_REGEX.finditer(text):
        # match.start() => index in text
        # match.group(1) => the number after [PAGE
        offset = match.start()
        pagenum = match.group(1)
        boundaries.append((offset, pagenum))
    return sorted(boundaries, key=lambda x: x[0])


def map_offset_to_page(offset: int, boundaries: List[Tuple[int, str]]) -> str:
    """
    Given an offset in the entire text, find which page that offset belongs to.
    We'll pick the last boundary whose offset <= chunk_offset. If none found, '???'.
    """
    page_label = "???"
    for (b_offset, b_page) in boundaries:
        if b_offset <= offset:
            page_label = b_page
        else:
            break
    return page_label


def load_text_and_images() -> List[Document]:
    """
    1) Reads each big .txt file that may contain multiple [PAGE X] markers.
    2) Splits each file's text by `[PAGE X]` markers -> multiple docs.
    3) Also loads any image .txt files as separate docs.
    """
    docs = []

    if os.path.exists(TEXTS_DIR):
        for fname in os.listdir(TEXTS_DIR):
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(TEXTS_DIR, fname)
            base_meta = extract_filename_metadata(fname)
            base_meta["source"] = fname
            base_meta["type"] = "pdf_text"

            doc = load_entire_text_as_single_doc(full_path, base_meta)

            docs.append(doc)

    if os.path.exists(IMAGE_TEXTS_DIR):
        for fname in os.listdir(IMAGE_TEXTS_DIR):
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(IMAGE_TEXTS_DIR, fname)
            with open(full_path, "r", encoding='utf-8') as f:
                caption = f.read().strip()

            img_pattern = re.compile(r"page_(\d+)_img_(\d+)")
            match = img_pattern.search(fname)
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
    return docs


class SemanticDenseEmbeddings(Embeddings):
    """
    A wrapper class to produce embeddings for the SemanticChunker stage.
    """

    def __init__(self, model_path: str, use_gpu: bool = True):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        logger.info("Loading dense retriever model for embeddings...")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            enc = self.doc_tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                doc_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
            embeddings.append(doc_emb.squeeze(0).cpu().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        enc = self.doc_tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            doc_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return doc_emb.squeeze(0).cpu().tolist()

class DenseRetrieverEmbedder:
        def __init__(self, model_path: str, use_gpu: bool = True):
            self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
            logger.info("Loading LLM-R doc encoder for final doc embeddings...")
            self.doc_encoder = AutoModel.from_pretrained(model_path).to(self.device)
            self.doc_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        def mean_pool(self, last_hidden_state, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        def embed_documents(self, docs: List[Document]) -> List[List[float]]:
            embeddings = []
            for d in docs:
                text = d.page_content
                enc = self.doc_tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                with torch.no_grad():
                    outputs = self.doc_encoder(input_ids, attention_mask=attention_mask)
                    doc_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
                embeddings.append(doc_emb.squeeze(0).cpu().tolist())
            return embeddings


def main():
    logger.info("Loading documents...")
    raw_docs = load_text_and_images()

    tokenizer = AutoTokenizer.from_pretrained(DENSE_RETRIEVER_PATH, use_gpu=True)

    chunk_size = 512

    all_chunked_docs: List[Document] = []
    for doc in raw_docs:
        chunked = chunk_text(doc, tokenizer, chunk_size)
        all_chunked_docs.extend(chunked)

    logger.info(f"Total docs after naive chunking: {len(all_chunked_docs)}")

    embedder = DenseRetrieverEmbedder(DENSE_RETRIEVER_PATH, use_gpu=True)
    doc_embeddings = embedder.embed_documents(all_chunked_docs)

    logger.info("Building FAISS index with these final embeddings ...")

    class PrecomputedEmbeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return doc_embeddings

        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError("Use dense model query encoder directly for query embedding.")

    precomputed_embedder = PrecomputedEmbeddings()

    vectorstore = FAISS.from_documents(all_chunked_docs, precomputed_embedder)

    logger.info(f"Saving FAISS index to {FAISS_INDEX_OUTPUT_PATH}...")
    vectorstore.save_local(FAISS_INDEX_OUTPUT_PATH)
    logger.info("FAISS index built successfully with LLM-R embeddings and integrated image-derived text.")


if __name__ == "__main__":
    main()
