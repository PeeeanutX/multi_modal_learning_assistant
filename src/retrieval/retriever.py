import os
import logging
import numpy as np
from typing import List, Optional
import torch
from dataclasses import dataclass

from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class RetrieverConfig:
    dense_retriever_path: str
    query_model_name: str = "intfloat/e5-base-v2"
    doc_model_name: str = "intfloat/e5-base-v2"
    vectorstore_path: str = "src/ingestion/data/index.pkl"
    top_k: int = 5
    use_gpu: bool = True


class DenseRetriever:
    """
    A dense retriever that uses a dual-encoder (query and doc encoder) to retrieve documents.
    Assumes a FAISS vector store with precomputed doc embeddings, or you can dynamically embed docs.
    """
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu'

        logger.info("Loading dense retriever models...")
        self.query_tokenizer = AutoTokenizer.from_pretrained(self.config.query_model_name, use_fast=True)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(self.config.doc_model_name, use_fast=True)

        self.query_encoder = AutoModel.from_pretrained(self.config.dense_retriever_path).to(self.device)
        self.doc_encoder = AutoModel.from_pretrained(self.config.dense_retriever_path).to(self.device)

        logger.info(f"Loading FAISS vector store from {self.config.vectorstore_path}")
        if not os.path.exists(self.config.vectorstore_path):
            raise FileNotFoundError(f"FAISS index not found at {self.config.vectorstore_path}")
        self.vectorstore = FAISS.load_local(self.config.vectorstore_path, embedding_function=None)
        logger.info("Dense retriever initialized successfully.")

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_query(self, query: str) -> torch.Tensor:
        enc = self.query_tokenizer(query, truncation=True, max_length=self.config.top_k*32, return_tensors='pt')
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.query_encoder(input_ids, attention_mask=attention_mask)
            query_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return query_emb.squeeze(0)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        if top_k is None:
            top_k = self.config.top_k
        query_emb = self.embed_query(query).cpu().numpy()
        docs = self.vectorstore.similarity_search_by_vector(query_emb, k=top_k)

        return docs
