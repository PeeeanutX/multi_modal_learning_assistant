"""
train_cross_encoder.py

This script trains a reward model (a cross-encoder) using LLM-derived rankings.
The reward model learns to distinguish higher-quality in-context examples from lower quality ones
based on the preference signals extracted from the LLm scores.

**Assumptions:**
- Input data is a JSONL or JSONL.GZ file containing entries with:
  {
    "query": str,
    "pos_context": str,
    "neg_contexts": [list_of_strs],
    ...
  }

  Where 'pos_context' is a high-quality candidate context chosen by the LLM,
  and 'neg_contexts' are one or more lower-quality candidate contexts.

- The model will be trained in a pairwise manner:
  For each query, we have (query, pos_context) vs (query, neg_context).
  The reward model should give a high score to the positive pair.

**Usage:**
python train_cross_encoder.py \
  --data-file /path/to_training_Data.jsonl.gz \
  --model-name google/electra-base-discriminator \
  --output-dir ./reward_model_output \
  --per-device-train-batch-size 16 \
  --learning-rate 1e-5

"""

import os
import json
import gzip
import logging
import argparse
from typing import Dict, List
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Add custom arguments if needed."""
    data_file: str = field(
        default=None,
        metadata={"help": "Path to the training data file (JSONL or JSONL.GZ)."}
    )


class RewardModelDataset(Dataset):
    """
    A dataset for training the reward model on pairwise comparisons.
    Each example consists of a single query, one positive context, and one or more negative contexts.
    We'll turn these into multiple training examples for each negative context.
    """

    def __init__(self, data_file: str, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_file)

    def _load_data(self, path: str) -> List[Dict]:
        logger.info(f"Loading data from {path}")
        open_func = gzip.open if path.endswith('.gz') else open
        examples = []
        with open_func(path, 'rt', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line.strip())
                query = ex.get("query", "")
                pos_context = ex.get("pos_context", "")
                neg_contexts = ex.get("neg_contexts", [])
                # Create pairwise training instances
                for neg in neg_contexts:
                    examples.append({
                        "query": query,
                        "pos_context": pos_context,
                        "neg_context": neg
                    })
        logger.info(f"Loaded {len(examples)} examples.")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        ex = self.examples[idx]

        # We'll create two sets of inputs:
        # (query, pos_context) -> label: 1
        # (query, neg_context) -> label: 0
        return {
            "query_pos": ex["query"],
            "context_pos": ex["pos_context"],
            "query_neg": ex["query"],
            "context_neg": ex["neg_context"]
        }


class RewardModelCollator(DataCollatorWithPadding):
    """
    Custom collator that takes the pairwise examples and
    produces model inputs suitable for cross-encoder training.

    We'll encode the positive pair and negative pair separately, then
    stack them. We'll rely on the model producing logits for both pairs
    and use a cross-entropy loss to differentiate them.
    """

    def __call__(self, features):
        batch_query_pos = [f["query_pos"] for f in features]
        batch_context_pos = [f["context_pos"] for f in features]
        batch_query_neg = [f["query_neg"] for f in features]
        batch_context_neg = [f["context_neg"] for f in features]

        # Encode positive pairs
        pos_encodings = self.tokenizer(
            batch_query_pos, batch_context_pos,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=False
        )

        # Encode negative pairs
        neg_encodings = self.tokenizer(
            batch_query_neg, batch_context_neg,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=False
        )

        # Stack positive and negative encodings so batch size doubles
        # We'll create a combined batch:
        # First half: positive pairs, label = 1
        # Second half: negative pairs, label = 0
        combined_input_ids = pos_encodings["input_ids"] + neg_encodings["input_ids"]
        combined_attention_mask = pos_encodings["attention_mask"] + neg_encodings["attention_mask"]

        # Now pad
        batch_dict = self.tokenizer.pad(
            {"input_ids": combined_input_ids, "attention_mask": combined_attention_mask},
            return_tensors="pt"
        )

        # Labels: first half are positive (label=1), second half negative (label=0)
        labels = torch.tensor([1]*len(features) + [0]*len(features), dtype=torch.long)

        batch_dict["labels"] = labels
        return batch_dict


def main():
    parser = argparse.ArgumentParser(description="Train a cross-encoder reward model.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to training data file (JSONL/JSONL.GZ).")
    parser.add_argument("--model-name", type=str, default="google/electra-base-discriminator", help="Pretrained model name or path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for model checkpoints.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Create dataset and data collator
    train_dataset = RewardModelDataset(args.data_file, tokenizer=tokenizer, max_length=256)
    data_collator = RewardModelCollator(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="no",  # Could be "steps" if you have a dev set
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU is available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training of cross-encoder reward model...")
    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy loading later
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()