import os
import sys
import json
import logging
import argparse
import gzip
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import toml
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processing.vector_store import VectorStoreFactory,VectorStoreConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.retrieval.retriever import RetrieverFactory, RetrieverConfig
from src.retrieval.retriever import BM25Okapi
from langchain.schema import Document

load_dotenv()

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
    chunks_file: str
    embeddings_provider: str = 'nvidia'
    embeddings_model: str = 'NV-Embed-QA'
    top_k: int = 20
    use_bm25_fallback: bool = False

    bm25_top_k: int = 50
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
    if queries_path.endswith('.gz'):
        f = gzip.open(queries_path, 'rt', encoding='utf-8')
    else:
        f = open(queries_path, 'r', encoding='utf-8')

    with f:
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


def build_retriever(index_file: str,
                    embeddings_provider: str,
                    embeddings_model: str) -> Tuple:
    """
    Build a vector-based retriever using a stored FAISS index and the chosen embeddings.
    Returns (retriever, vector_store) for usage.
    """
    embeddings_config= EmbeddingsConfig(
        provider=embeddings_provider,
        model_name=embeddings_model,
        api_key=None
    )
    embeddings_model_obj = EmbeddingsFactory.get_embeddings_model(embeddings_config)

    vector_store_config = VectorStoreConfig(
        store_type='faiss',
        embedding_model=embeddings_model_obj,
        faiss_index_path=index_file
    )
    vector_store = VectorStoreFactory.create_vector_store(vector_store_config, docs=[])
    retriever_config = RetrieverConfig(retriever_type='default', search_kwargs={"k": 20})
    retriever = RetrieverFactory.create_retriever(vector_store, retriever_config)

    return retriever, vector_store


def retrieve_candidates(queries: List[Dict],
                        retriever,
                        top_k: int,
                        use_bm25_fallback: bool,
                        docs: List[Document],
                        bm25_top_k: int) -> List[Dict]:
    """
    Retrieve top-k candidates for each query using the vector-based retriever.
    If use_bm25_fallback is True and the vector retriever returns insufficient results,
    use BM25 as a fallback.
    """
    results = []
    if use_bm25_fallback:
        doc_texts = [doc.page_content for doc in docs]
        tokenized_docs = [t.lower().split() for t in doc_texts]
        bm25 = BM25Okapi(tokenized_docs)

    for q in queries:
        query_text = q['query']
        retrieved_docs = retriever.invoke(query_text)
        if len(retrieved_docs) < top_k and use_bm25_fallback:
            tokenized_query = query_text.lower().split()
            scores = bm25.get_scores(tokenized_query)
            doc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_top_k]
            fallback_docs = [docs[i] for i in doc_indices]
            retrieved_docs.extend(fallback_docs[:top_k - len(retrieved_docs)])

        retrieved_docs = retrieved_docs[:top_k]

        candidates = []
        for doc_idx, d in enumerate(retrieved_docs):
            candidates.append({
                "doc_id": doc_idx,
                "contents": d.page_content,
            })

        results.append({
            "query_id": q['query_id'],
            "query": q['query'],
            "answers": q.get('answers', []),
            "candidates": candidates
        })
    return results


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
    parser.add_argument('--chunks-file', default='src/ingestion/data/chunks.pkl', help='Path to chunks.pkl file')
    parser.add_argument('--embeddings-provider', default='nvidia', help='Embeddings provider: nvidia, openai, huggingface')
    parser.add_argument('--embeddings-model', default='NV-Embed-QA', help='Embeddings model name')
    parser.add_argument('--top-k', type=int, default=20, help='Number of candidates to retrieve')
    parser.add_argument('--use-bm25-fallback', action='store_true', help='Use BM25 as a fallback retriever if needed')
    parser.add_argument('--bm25-top-k', type=int, default=50, help='Number of BM25 candidates if fallback is used')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.index_file):
        logger.error(f"Index file not found: {args.index_file}")
        sys.exit(1)
    if not os.path.exists(args.chunks_file):
        logger.error(f"Chunks file not found: {args.chunks_file}")
        sys.exit(1)

    queries = load_queries(args.queries_path)
    if not queries:
        logger.error("No queries loaded. Aborting.")
        sys.exit(1)

    docs = load_documents(args.chunks_file)

    retriever, vector_store = build_retriever(
        args.index_file,
        args.embeddings_provider,
        args.embeddings_model
    )

    results = retrieve_candidates(
        queries=queries,
        retriever=retriever,
        top_k=args.top_k,
        use_bm25_fallback=args.use_bm25_fallback,
        docs=docs,
        bm25_top_k=args.bm25_top_k
    )

    save_results(results, args.output_path)


if __name__ == "__main__":
    main()