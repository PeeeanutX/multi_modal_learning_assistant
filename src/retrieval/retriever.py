import os
import logging
import torch
from dataclasses import dataclass
from typing import List, Optional
from langchain_community.vectorstores import FAISS
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
    faiss_index_path: str
    top_k: int = 10
    use_gpu: bool = True


class DenseRetrieverEmbedder:
    """
    A dense retriever that uses the LLM-R embeddings stored in FAISS.
    """
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info("Loading dense retriever model for embeddings...")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            enc = self.tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                doc_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
            embeddings.append(doc_emb.squeeze(0).cpu().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        enc = self.tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            query_emb = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return query_emb.squeeze(0).cpu().tolist()


class DenseRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.embedder = DenseRetrieverEmbedder(self.config.dense_retriever_path, use_gpu=self.config.use_gpu)

        logger.info(f"Loading FAISS vector store from {self.config.faiss_index_path}")
        self.vectorstore = FAISS.load_local(
            self.config.faiss_index_path,
            embeddings = self.embedder,
            allow_dangerous_deserialization = True
        )
        logger.info("Dense retriever initialized successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        if top_k is None:
            top_k = self.config.top_k

        query_emb = self.embedder.embed_query(query)
        docs = self.vectorstore.similarity_search_by_vector(query_emb, k=top_k)
        return docs
