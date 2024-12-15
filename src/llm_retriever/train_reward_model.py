import os
import sys
import json
import logging
import argparse
import random
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from typing import List, Dict, Any
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RewardTrainingConfig:
    input_path: str = "llm_scored_candidates.jsonl"
    output_dir: str = "./reward_model_checkpoint"
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    negative_samples: int = 3
    top_k_as_positive: int = 1
    bottom_k_as_negative: int = -1


class RewardDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = self.tokenizer(
            ex["query"] + " " + ex["candidate"],
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        inputs["labels"] = ex["label"]
        return inputs


def prepare_training_data(
        input_path: str,
        top_k_as_positive: int = 1,
        bottom_k_as_negative: int = -1,
        negative_samples: int = 3
) -> List[Dict[str, Any]]:
    """
    Prepare pairwise training data from LLM scored candidates.
    For each query, pick top_k_as_positive candidates as positives and a subset of negatives.
    Create pairwise examples: (query, positive), label=1 and (query, negative), label=0.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = obj.get("query", "")
            answers = obj.get("answers", [])
            candidates = obj.get("candidates", [])
            scores = [c.get("llm_score", 0.0) for c in candidates]

            if not candidates or not scores:
                continue

            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            positives = sorted_indices[:top_k_as_positive]
            if bottom_k_as_negative <= 0:
                negatives = sorted_indices[top_k_as_positive:]
            else:
                negatives = sorted_indices[-bottom_k_as_negative:]

            if len(negatives) == 0 or len(positives) == 0:
                continue

            if 0 < negative_samples < len(negatives):
                negatives = random.sample(negatives, negative_samples)

            for pos_idx in positives:
                pos_candidate = candidates[pos_idx]["contents"]
                data.append({
                    "query": query,
                    "candidate": pos_candidate,
                    "label": 1
                })
                for neg_idx in negatives:
                    neg_candidate = candidates[neg_idx]["contents"]
                    data.append({
                        "query": query,
                        "candidate": neg_candidate,
                        "label": 0
                    })

    random.shuffle(data)
    return data


def main():
    parser = argparse.ArgumentParser(description="Train a reward model from LLM scored candidates")
    parser.add_argument('--input-path', default='src/ingestion/data/llm_scored_candidates.jsonl', help='Path to LLM scored candidates')
    parser.add_argument('--output-dir', default='./reward_model_checkpoint', help='Where to save the trained model')
    parser.add_argument('--model-name', default='bert-base-uncased', help='Base model name for reward model')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--learning-rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--num-train-epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--per-device-train-batch-size', type=int, default=16, help='Train batch size')
    parser.add_argument('--per-device-eval-batch-size', type=int, default=16, help='Eval batch size')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--top-k-as-positive', type=int, default=1, help='Top k candidates as positive examples')
    parser.add_argument('--bottom-k-as-negative', type=int, default=-1,
                        help='How many from bottom as negatives, -1=all')
    parser.add_argument('--negative-samples', type=int, default=3, help='Number of negative samples per positive')
    parser.add_argument('--eval-ratio', type=float, default=0.1, help='Ratio of data for evaluation')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Preparing training data from {args.input_path}")
    examples = prepare_training_data(
        input_path=args.input_path,
        top_k_as_positive=args.top_k_as_positive,
        bottom_k_as_negative=args.bottom_k_as_negative,
        negative_samples=args.negative_samples
    )

    if not examples:
        logger.error("No training examples prepared. Check input data and parameters.")
        sys.exit(1)

    eval_size = int(len(examples) * args.eval_ratio)
    train_data = examples[eval_size:]
    eval_data = examples[:eval_size]

    logger.info(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = RewardDataset(train_data, tokenizer, args.max_length)
    eval_dataset = RewardDataset(eval_data, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete. Reward model saved at {}".format(args.output_dir))


if __name__ == "__main__":
    main()
