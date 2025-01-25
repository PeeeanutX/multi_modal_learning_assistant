import os
import sys
import json
import logging
import argparse
import gzip
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import toml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processing.vector_store import VectorStoreFactory,VectorStoreConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.retrieval.retriever import RetrieverConfig
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CandidateRetrievalConfig:
    queries_path: str
    output_path: str
    index_file: str
    embeddings_provider: str = 'huggingface'
    embeddings_model: str = 'jinaai/jina-embeddings-v3'
    top_k: int = 10
    doccache_file: str = ''


def load_queries(queries_path: str) -> List[Dict]:
    """
    Load queries from a JSONL file. Each line should be a JSON object containing
    at least 'query' and 'query_id' fields, and optionally 'answers'.
    Example:
    {
      "query_id": "q1",
      "query": "What is photosynthesis?",
      "answers": ["The process by which plants make food using sunlight."]
    }
    """
    queries = []
    open_func = gzip.open if queries_path.endswith('.gz') else open
    with open_func(queries_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if 'query' not in data or 'query_id' not in data:
                logger.error("Each query must contain at least 'query' and 'query_id'. Skipping line.")
                continue
            queries.append(data)
    logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


def load_documents(chunks_file: str) -> List[Document]:
    """
    Loads documents (chunks)  from a pickle file produced by data_preparation.py.
    Each entry is expected to be a chunk of text.
    """
    import pickle
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    docs = [Document(page_content=c) for c in chunks]
    logger.info(f"Loaded {len(docs)} documents (chunks) from {chunks_file}")
    return docs


def retrieve_candidates(cfg: CandidateRetrievalConfig):
    queries = load_queries(cfg.queries_path)
    if not queries:
        logger.error("No queries found. Exiting.")
        sys.exit(1)

    logger.info(f"Loading FAISS index from {cfg.index_file} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True}
    )

    vector_store_config = VectorStoreConfig(
        store_type='faiss',
        embedding_model=embeddings,
        faiss_index_path=cfg.index_file
    )
    vectorstore = VectorStoreFactory.create_vector_store(vector_store_config, docs=[])

    results = []
    for q in queries:
        query_text = q['query']
        query_emb = embeddings.embed_query(query_text)
        docs = vectorstore.similarity_search_by_vector(query_emb, k=cfg.top_k)

        candidates = []
        for i, d in enumerate(docs):
            candidates.append({
                "doc_id": i,
                "contents": d.page_content
            })

        results.append({
            "query_id": q['query_id'],
            "query": q['query'],
            "answers": q.get('answers', []),
            "candidates": candidates
        })

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    with open(cfg.output_path, 'w', encoding='utf-8') as out_f:
        for entry in results:
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Saved retrieval results to {cfg.output_path}")


def save_results(results: List[Dict], output_path: str):
    """
    Save the retrieval results as a JSONL file.
    Each line: {"query_id": ..., "query": ..., "answers": [...], "candidates": [...]}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode= 'wt'
    if output_path.endswith('.gz'):
        f = gzip.open(output_path, mode, encoding='utf-8')
    else:
        f = open(output_path, mode, encoding='utf-8')

    with f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(results)} retrieval results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Initial Candidate Retrieval for LLM-R pipeline")
    parser.add_argument('--queries-path', default='src/ingestion/data/queries.jsonl', help='Path to input queries file (JSONL)')
    parser.add_argument('--output-path', default='src/ingestion/data/initial_candidates.jsonl', help='Path to output file (JSONL or JSONL.GZ)')
    parser.add_argument('--index-file', default='src/ingestion/data/index.pkl', help='Path to FAISS index file')
    parser.add_argument('--embeddings-provider', default='huggingface', help='Embeddings provider: nvidia, openai, huggingface')
    parser.add_argument('--embeddings-model', default='jinaai/jina-embeddings-v3', help='Embeddings model name')
    parser.add_argument('--top-k', type=int, default=10, help='Number of candidates to retrieve')
    args = parser.parse_args()

    cfg = CandidateRetrievalConfig(
        queries_path=args.queries_path,
        output_path=args.output_path,
        index_file=args.index_file,
        embeddings_provider=args.embeddings_provider,
        embeddings_model=args.embeddings_model,
        top_k=args.top_k
    )
    retrieve_candidates(cfg)


if __name__ == "__main__":
    main()