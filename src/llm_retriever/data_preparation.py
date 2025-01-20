import os
import sys
import logging
import pickle
import re
from typing import List, Tuple
from transformers import AutoTokenizer
from langchain.schema import Document

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain.schema import Document

from src.ingestion.extract_pdf_content import extract_pdf_content
from src.ingestion.text_cleaner import clean_text
from src.processing.chunker import TextChunker, ChunkerConfig
from src.processing.embedder import EmbeddingsFactory, EmbeddingsConfig
from src.processing.vector_store import VectorStoreFactory, VectorStoreConfig

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

PAGE_MARKER_REGEX = re.compile(r"\[PAGE\s*(\d+)\]")


def find_page_boundaries(text: str) -> List[Tuple[int, str]]:
    boundaries = []
    for match in PAGE_MARKER_REGEX.finditer(text):
        offset = match.start()
        pagenum = match.group(1)
        boundaries.append((offset, pagenum))
    return sorted(boundaries, key=lambda x: x[0])


def map_offset_to_page(char_offset: int, boundaries: List[Tuple[int, str]]) -> str:
    """
    Given a character offset in the entire text, find which page that offset belongs to.
    We'll pick the last boundary whose offset <= char_offset. If none found, '???'.
    """
    page_label = "???"
    for (b_offset, b_page) in boundaries:
        if b_offset <= char_offset:
            page_label = b_page
        else:
            break
    return page_label


def token_chunk_text_with_pages(
        text: str,
        tokenizer,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        base_metadata: dict = None
) -> list:
    """
    Token-based chunking of 'text' using a Hugging Face tokenizer.
    Also detects [PAGE X] markers and stores 'page' metadata for each chunk.
    """
    if base_metadata is None:
        base_metadata = {}

    page_boundaries = find_page_boundaries(text)

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    input_ids = enc["input_ids"]
    offsets = enc["offsets_mapping"] if "offsets_mapping" in enc else enc["offset_mapping"]
    total_tokens = len(input_ids)

    docs = []
    current_start = 0

    while current_start < total_tokens:
        current_end = current_start + chunk_size
        token_slice = input_ids[current_start:current_end]
        offset_slice = offsets[current_start:current_end]

        chunk_text = tokenizer.decode(token_slice, skip_special_tokens=True).strip()

        if offset_slice:
            chunk_char_start = offset_slice[0][0]
        else:
            chunk_char_start = 0

        page_label = map_offset_to_page(chunk_char_start, page_boundaries)

        chunk_meta = dict(base_metadata)
        chunk_meta["page"] = page_label
        chunk_meta["token_start"] = current_start
        chunk_meta["token_end"] = min(current_end, total_tokens)
        chunk_meta["char_start"] = chunk_char_start

        doc = Document(page_content=chunk_text, metadata=chunk_meta)
        docs.append(doc)

        move_step = chunk_size - chunk_overlap
        if move_step <= 0:
            logger.warning("chunk_overlap is too large; overlap can't exceed chunk_size.Stopping.")
            break
        current_start += move_step

    return docs


def prepare_data(
        pdf_dir: str,
        text_output_dir: str,
        processed_texts_dir: str,
        index_file: str,
        chunks_file: str,
        embeddings_provider: str = 'huggingface',
        embeddings_model: str = 'src/checkpoints/dense_retriever_checkpoint',
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        processed_image_texts_dir: str = "",  #
):
    """
    Prepare data for the LLM-R pipeline using SemanticChunker for chunking:
      1. Extract text from PDFs if needed.
      2. Clean the text.
      3. Split the text semantically.
      4. Embed the chunks & build a FAISS index.
      5. Save chunk strings as a pickle for future retrieval steps.
    """

    # Step 1: Extract PDF text if not done
    if not os.path.exists(text_output_dir):
        logger.info("Extracting text from PDFs (PyMuPDF + [PAGE X]) ...")
        os.makedirs(text_output_dir, exist_ok=True)
        for fname in os.listdir(pdf_dir):
            if fname.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, fname)
                extract_pdf_content(pdf_path, text_output_dir, images_output_dir="src/ingestion/data/raw/images")
    else:
        logger.info("Text files already extracted, skipping PDF extraction.")

    # Step 2: Load & clean text
    text_files = [f for f in os.listdir(processed_texts_dir) if f.lower().endswith('.txt')]
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {processed_texts_dir}. Did extraction run?")

    raw_texts = []
    for txt_file in text_files:
        with open(os.path.join(processed_texts_dir, txt_file), 'r', encoding='utf-8') as f:
            text_data = f.read()
        cleaned = clean_text(text_data)
        if cleaned.strip():
            raw_texts.append((txt_file, cleaned))
        else:
            logger.warning(f"No usable text after cleaning {txt_file}")

    if processed_image_texts_dir and os.path.isdir(processed_image_texts_dir):
        for fname in os.listdir(processed_image_texts_dir):
            if fname.lower().endswith('.txt'):
                with open(os.path.join(processed_image_texts_dir, fname), 'r', encoding='utf-8') as f:
                    img_caption = f.read().strip()
                if img_caption:
                    raw_texts.append((fname, img_caption))

    logger.info(f"Loading tokenizer from {embeddings_model} for token-based chunking...")
    tokenizer = AutoTokenizer.from_pretrained(embeddings_model, use_fast=True)

    all_chunk_strs = []
    docs = []
    logger.info(f"Token chunking each text with chunk_size={chunk_size}, overlap={chunk_overlap}...")

    for (fname, text_data) in raw_texts:
        base_meta = {"source": fname}
        chunked_docs = token_chunk_text_with_pages(
            text_data,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_metadata=base_meta
        )
        docs.extend(chunked_docs)
        for d in chunked_docs:
            all_chunk_strs.append(d.page_content)

    logger.info(f"Total docs after token-based chunking: {len(docs)}")

    # Step 3: Build embeddings & index
    logger.info(f"Initializing embeddings with provider={embeddings_provider}, model={embeddings_model}")
    embeddings_config = EmbeddingsConfig(
        provider=embeddings_provider,
        model_name=embeddings_model
    )
    embed_model = EmbeddingsFactory.get_embeddings_model(embeddings_config)

    vector_store_config = VectorStoreConfig(
        store_type='faiss',
        embedding_model=embed_model,
        faiss_index_path=index_file
    )
    logger.info(f"Creating or loading FAISS index at {index_file} ...")
    vectorstore = VectorStoreFactory.create_vector_store(vector_store_config, docs=docs)
    logger.info("FAISS index ready.")

    # Step 6: Save chunk strings to pickle
    with open(chunks_file, 'wb') as f:
        pickle.dump(all_chunk_strs, f)
    logger.info(f"Saved {len(all_chunk_strs)} semantic chunks to {chunks_file}")


if __name__ == "__main__":
    pdf_dir = 'src/ingestion/data/raw/lectures/'
    text_output_dir = 'src/ingestion/data/processed/texts/'
    processed_texts_dir = text_output_dir
    index_file = 'src/ingestion/data/index.pkl'
    chunks_file = 'src/ingestion/data/chunks.pkl'
    processed_image_texts_dir = 'src/ingestion/data/processed/image_texts'

    prepare_data(
        pdf_dir=pdf_dir,
        text_output_dir=text_output_dir,
        processed_texts_dir=processed_texts_dir,
        index_file=index_file,
        chunks_file=chunks_file,
        embeddings_provider='huggingface',
        embeddings_model='src/checkpoints/dense_retriever_checkpoint',
        chunk_size=512,
        chunk_overlap=0,
        processed_image_texts_dir=processed_image_texts_dir
    )
