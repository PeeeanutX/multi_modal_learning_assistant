import os
import torch
import faiss
import numpy as np
from typing import List
from transformers import AutoModel, AutoTokenizer

##############################################
# 1) Configuration
##############################################
DOC_ENCODER_PATH = "src/checkpoints/dense_retriever_checkpoint"
# ^ Adjust if your doc encoder is in a subfolder like "checkpoint-183"

# Where you want to save the FAISS index
FAISS_INDEX_PATH = "src/ingestion/data/index/index.faiss"

# Where your processed .txt files live
DOCS_DIR = "src/ingestion/data/processed/merged"

# For demonstration, we’ll just read them as a list of raw strings.
# If you want page-level chunking, you can replicate your original chunking logic.


##############################################
# 2) Load the doc model & tokenizer
##############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading doc encoder from {DOC_ENCODER_PATH}...")
doc_tokenizer = AutoTokenizer.from_pretrained(DOC_ENCODER_PATH)
doc_model = AutoModel.from_pretrained(DOC_ENCODER_PATH).to(device)
doc_model.eval()  # set to eval mode


##############################################
# 3) Define a mean-pooling helper
##############################################
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool the encoder output using the attention mask.
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


##############################################
# 4) Function to embed a list of texts
##############################################
def embed_texts_with_doc_encoder(texts: List[str]) -> np.ndarray:
    """
    Tokenizes a list of texts, runs them through the doc encoder,
    returns a NumPy array of shape (len(texts), hidden_dim).
    """
    all_embeddings = []
    batch_size = 8  # or 16, etc.

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = doc_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = doc_model(**inputs)
            # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            # embeddings shape: (batch_size, hidden_dim)
        all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all batches
    return np.concatenate(all_embeddings, axis=0)


##############################################
# 5) Read your documents
##############################################
def load_all_documents() -> List[str]:
    """
    Read all .txt files in DOCS_DIR into a single list of texts.
    If you need special chunking logic (page by page), adapt here.
    """
    texts = []
    for fname in os.listdir(DOCS_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DOCS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()
                # For simplicity, treat entire file as one chunk
                # or split it by pages, paragraphs, etc.
                texts.append(raw_text)
    return texts


##############################################
# 6) Main: embed and build FAISS
##############################################
def main():
    print("Loading documents...")
    doc_texts = load_all_documents()
    print(f"Loaded {len(doc_texts)} documents. Embedding...")

    doc_embeddings = embed_texts_with_doc_encoder(doc_texts)
    print(f"Embeddings shape: {doc_embeddings.shape}")

    # Build FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # simple L2 index

    print("Adding embeddings to FAISS index...")
    index.add(doc_embeddings)

    print(f"Writing FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Done.")


if __name__ == "__main__":
    main()