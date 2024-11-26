import numpy as np


def retrieve_chunks(query_embeddings, index, chunks, top_k=5):
    distances, indices = index.search(np.array([query_embeddings]).astype('float32'), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks
