import os
import sys
import json
import logging
import argparse
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from torch.nn import KLDivLoss
from transformers.trainer_callback import TrainerCallback

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DenseRetrieverTrainingConfig:
    input_path: str = "src/ingestion/data/llm_scored_candidates.jsonl"
    reward_model_path: str = "src/checkpoints/reward_model_checkpoint"
    output_dir: str = "src/checkpoints/dense_retriever_checkpoint"
    query_model_name:  str = "microsoft/deberta-v3-large"
    doc_model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 128
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_ratio: float = 0.1
    seed: int = 42
    temperature: float = 1.0
    kd_cont_loss_weight: float = 0.2
    kd_loss_weight: float = 1.0


class DenseRetrieverDataset(Dataset):
    """
    This dataset creates pairs of (query, multiple candidates) with associated reward scores.
    We'll produce one example per query: query input, list of candidates, and their normalized scores.
    The trainer will need a custom collator or logic in the training step to compute the distribution.
    """
    def __init__(self, examples: List[Dict], query_tokenizer, doc_tokenizer, max_length: int):
        self.examples = examples
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query_text = ex["query"]
        doc_texts = [c["contents"] for c in ex["candidates"]]
        scores = ex["doc_scores"]

        scores_tensor = torch.tensor(scores, dtype=torch.float)
        dist = torch.softmax(scores_tensor, dim=0)

        query_enc = self.query_tokenizer(
            query_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        doc_enc = self.doc_tokenizer(
            doc_texts,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "doc_input_ids": doc_enc["input_ids"],
            "doc_attention_mask": doc_enc["attention_mask"],
            "scores": dist,
        }


def prepare_training_data(input_path: str, eval_ratio: float = 0.1):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'query' not in obj or 'candidates' not in obj or 'doc_scores' not in obj:
                continue
            if not obj['candidates'] or not obj['doc_scores']:
                continue
            data.append(obj)

    random.shuffle(data)
    eval_size = int(len(data) * eval_ratio)
    eval_data = data[:eval_size]
    train_data = data[eval_size:]
    return train_data, eval_data


class DenseRetrieverModel(torch.nn.Module):
    """
    A dual-encoder model: separate encoders for queries and docs.
    We'll produce embeddings and compute scores via dot product or cosine similarity.
    We'll train by minimizing KL divergense between predicted distribution and reward model model distribution.
    """
    def __init__(self, query_model_name: str, doc_model_name: str):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.doc_encoder = AutoModel.from_pretrained(doc_model_name)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask, **kwargs):
        B, M, L = doc_input_ids.shape

        doc_input_ids = doc_input_ids.view(B*M, L)
        doc_attention_mask = doc_attention_mask.view(B*M, L)

        query_outputs = self.query_encoder(query_input_ids, attention_mask=query_attention_mask)
        doc_outputs = self.doc_encoder(doc_input_ids, attention_mask=doc_attention_mask)

        query_emb = self.mean_pool(query_outputs.last_hidden_state, query_attention_mask)
        doc_emb = self.mean_pool(doc_outputs.last_hidden_state, doc_attention_mask)
        doc_emb = doc_emb.view(B, M, -1)

        return query_emb, doc_emb

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


def collate_fn(batch, device):
    query_input_ids = torch.stack([ex['query_input_ids'] for ex in batch], dim=0)
    query_attention_mask = torch.stack([ex['query_attention_mask'] for ex in batch], dim=0)

    max_docs = max([ex['doc_input_ids'].size(0) for ex in batch])
    doc_input_ids_list = []
    doc_attention_mask_list = []
    scores_list = []
    for ex in batch:
        pad_docs = max_docs - ex['doc_input_ids'].size(0)
        if pad_docs > 0:
            pad_ids = torch.full((pad_docs, ex['doc_input_ids'].size(1)), fill_value=0, dtype=torch.long)
            pad_mask = torch.zeros((pad_docs, ex['doc_attention_mask'].size(1)), dtype=torch.long)
            pad_scores = torch.zeros(pad_docs)
            doc_input_ids_list.append(torch.cat([ex['doc_input_ids'], pad_ids], dim=0))
            doc_attention_mask_list.append(torch.cat([ex['doc_attention_mask'], pad_mask], dim=0))
            scores_list.append(torch.cat([ex['scores'], pad_scores], dim=0))
        else:
            doc_input_ids_list.append(ex['doc_input_ids'])
            doc_attention_mask_list.append(ex['doc_attention_mask'])
            scores_list.append(ex['scores'])
    doc_input_ids = torch.stack(doc_input_ids_list, dim=0)
    doc_attention_mask = torch.stack(doc_attention_mask_list, dim=0)
    scores = torch.stack(scores_list, dim=0)

    return {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "doc_input_ids": doc_input_ids,
        "doc_attention_mask": doc_attention_mask,
        "scores": scores
    }


class DenseRetrieverTrainerCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training is complete!")


def main():
    parser = argparse.ArgumentParser(description="Train a dense retriever via knowledge distillation from reward model")
    parser.add_argument('--input-path', default='src/ingestion/data/llm_scored_candidates.jsonl', help='Path to LLM scored candidates')
    parser.add_argument('--reward-model-path', default='/content/multi_modal_learning_assistant/reward_model_checkpoint', help='Path to reward model')
    parser.add_argument('--output-dir', default='src/checkpoints/dense_retriever_checkpoint', help='Output directory')
    parser.add_argument('--query-model-name', default='microsoft/deberta-v3-large', help='Query encoder model name')
    parser.add_argument('--doc-model-name', default='microsoft/deberta-v3-large', help='Document encoder model name')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--num-train-epochs', type=int, default=3)
    parser.add_argument('--per-device-train-batch-size', type=int, default=4)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=4)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-ratio', type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Preparing training data from {args.input_path}")
    train_data, eval_data = prepare_training_data(args.input_path, eval_ratio=args.eval_ratio)
    logger.info(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizers
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model_name, use_fast=True)
    doc_tokenizer = AutoTokenizer.from_pretrained(args.doc_model_name, use_fast=True)

    train_dataset = DenseRetrieverDataset(train_data, query_tokenizer, doc_tokenizer, args.max_length)
    eval_dataset = DenseRetrieverDataset(eval_data, query_tokenizer, doc_tokenizer, args.max_length)

    model = DenseRetrieverModel(args.query_model_name, args.doc_model_name).to(device)

    # We'll implement a custom compute_loss function via a subclass of Trainer:
    class DenseRetrieverTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            query_input_ids = inputs["query_input_ids"]
            query_attention_mask = inputs["query_attention_mask"]
            doc_input_ids = inputs["doc_input_ids"]
            doc_attention_mask = inputs["doc_attention_mask"]
            target_dist = inputs["scores"]  # distribution from LLM/reward model

            query_emb, doc_emb = model(query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask)

            # Compute similarity scores: using dot product or cosine similarity
            # shape: (batch, num_docs, hidden_dim)
            # query_emb: (batch, hidden_dim)
            # doc_emb: (batch, num_docs, hidden_dim)
            # Expand query_emb: (batch, 1, hidden_dim)
            query_emb = query_emb.unsqueeze(1)
            # Dot product:
            logits = torch.sum(query_emb * doc_emb, dim=-1)  # (batch, num_docs)

            # Convert logits to log probabilities:
            retriever_log_probs = F.log_softmax(logits, dim=-1)

            # KL Divergence between target_dist and retriever_log_probs
            # target_dist: (batch, num_docs), retriever_log_probs: (batch, num_docs)
            kl_loss = F.kl_div(retriever_log_probs, target_dist, reduction="batchmean", log_target=False)

            loss = kl_loss  # can add more terms if needed

            return (loss, (query_emb, doc_emb, logits)) if return_outputs else loss

    data_collator = lambda batch: collate_fn(batch, device)

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
        label_names=["scores"],
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = DenseRetrieverTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[DenseRetrieverTrainerCallback()]
    )

    logger.info("Starting training dense retriever...")
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"Dense retriever trained and saved at {args.output_dir}")
    query_tokenizer.save_pretrained(args.output_dir)
    doc_tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()
