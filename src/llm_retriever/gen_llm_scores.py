import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import argparse
import gzip
from typing import List, Dict, Any

from src.models.llm_interface import LLMInterface, LLMConfig
from langchain.schema import Document


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_candidates(input_path: str) -> List[Dict[str, Any]]:
    """
    Load initial candidates from JSONL (JSONL.GZ) file.
    Each line should be a JSON object with "query", "answers", and "candidates".
    E.g.
    {
      "query_id": "q1",
      "query": "...",
      "answers": ["correct answer..."],
      "candidates": [{"contents": "...", "doc_id": 0}, ...]
    }
    """
    data = []
    if input_path.endswith('.gz'):
        f = gzip.open(input_path, 'rt', encoding='utf-8')
    else:
        f = open(input_path, 'r', encoding='utf-8')

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'query' not in obj or 'answers' not in obj or 'candidates' not in obj:
                logger.warning(f"Missing required fields in line: {line}")
                continue
            data.append(obj)

    logger.info(f"Loaded {len(data)} entries from {input_path}")
    return data


def batch_score_with_llm(llm: LLMInterface,
                         queries: List[str],
                         answers: List[str],
                         candidates: List[str],
                         delimiter: str = '\n') -> List[float]:
    """
    Score candidate documents using the LLM by evaluating the log probability of the correct answer.
    The approach:
    For each candidate, we form a prompt like:
        <candidate_contents>
        <query>

    and measure the LLM log-likelihood of the answer.
    """
    input_texts = []
    output_texts = []

    for query, ans, cand in zip(queries, answers, candidates):
        input_texts.append(f"{cand.strip()}{delimiter}{query.strip()}")
        output_texts.append(ans.strip())

    scores = llm.batch_score(input_texts, output_texts)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Generate LLM scores for retrieved candidates")
    parser.add_argument('--input-path', default='src/ingestion/data/initial_candidates.jsonl',
                        help='Path to initial candidates file')
    parser.add_argument('--output-path', default='src/ingestion/data/llm_scored_candidates.jsonl',
                        help='Path to save scored candidates file')
    parser.add_argument('--llm-provider', default='huggingface', help='LLM provider (e.g., nvidia, openai, huggingface)')
    parser.add_argument('--llm-model-name', default='tiiuae/falcon-7b', help='LLM model name')
    parser.add_argument('--llm-temperature', type=float, default=0.3, help='LLM generation temperature')
    parser.add_argument('--llm-max-tokens', type=int, default=512, help='Max tokens for LLM responses')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for scoring')
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Load model in 4-bit quantized mode (bitsandbytes required).')
    parser.add_argument('--load-in-8bit', action='store_true',
                        help='Load model in 8-bit quantized mode (bitsandbytes required).')
    parser.add_argument('--device-map', default='auto',
                        help='Device map for model sharding. e.g. "auto", "balanced", "sequential" or a dict.')
    parser.add_argument('--max-memory-per-gpu', default=None,
                        help='Maximum memory per GPU, e.g. "20GiB". Used for model sharding.')
    args = parser.parse_args()

    data = load_candidates(args.input_path)
    if not data:
        logger.error("No data loaded from input file.")
        sys.exit(1)

    # Initialize LLM interface
    llm_config = LLMConfig(
        provider=args.llm_provider,
        model_name=args.llm_model_name,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens
    )
    llm_interface = LLMInterface(config=llm_config, retriever=None)

    # Prepare output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.output_path.endswith('.gz'):
        out_f = gzip.open(args.output_path, 'wt', encoding='utf-8')
    else:
        out_f = open(args.output_path, 'w', encoding='utf-8')

    with out_f:
        # We'll process queries in a loop. If dataset is large, consider batching.
        # For each entry, we have one query and multiple candidates.
        for entry in data:
            query = entry['query']
            # For scoring, we'll use the first answer in 'answers'
            # If multiple answers exist, you might average scores or pick the first.
            ans = entry['answers'][0] if entry['answers'] else ""
            candidates = entry['candidates']

            if not candidates:
                # No candidates to score
                entry['doc_scores'] = []
                out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                continue

            cand_texts = [c['contents'] for c in candidates]
            queries_batch = [query] * len(cand_texts)
            answers_batch = [ans] * len(cand_texts)

            # Score candidates in batches if necessary
            scores = []
            for start in range(0, len(cand_texts), args.batch_size):
                end = start + args.batch_size
                batch_queries = queries_batch[start:end]
                batch_answers = answers_batch[start:end]
                batch_cands = cand_texts[start:end]

                batch_scores = batch_score_with_llm(llm=llm_interface,
                                                    queries=batch_queries,
                                                    answers=batch_answers,
                                                    candidates=batch_cands)
                scores.extend(batch_scores)

            # Attach the scores to the candidates
            for i, s in enumerate(scores):
                # s is a float log-likelihood or similar scoring metric
                entry['candidates'][i]['llm_score'] = s

            entry['doc_scores'] = scores
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"LLM scores generated and saved to {args.output_path}")


if __name__ == "__main__":
    main()