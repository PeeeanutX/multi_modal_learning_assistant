import logging
from typing import List, Optional
from dataclasses import dataclass, field

import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
)

from langchain.schema import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

nltk.download('punkt_tab', quiet=True)


@dataclass
class ChunkerConfig:
    method: str = 'recursive'  # Options: 'recursive', 'character', 'sentence', 'spacy', 'nltk'
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    length_function: Optional[callable] = len
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    language: str = 'end_core_web_sm'  # For SpacyTextSplitter


class TextChunker:
    def __init__(self, config: ChunkerConfig):
        """Initializes the TextChunker with a specified configuration,"""
        if config.length_function is not None and not callable(config.length_function):
            raise ValueError("The `length_function` must be callable (e.g., `len`) or `None`.")
        self.config = config
        self.text_splitter = self._initialize_text_splitter()
        logger.info(f"Initialized TextChunker with method '{self.config.method}'")

    def _initialize_text_splitter(self):
        """Initialize the appropriate text splitter based on the method."""
        if self.config.method == 'recursive':
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.config.length_function,
                separators=self.config.separators
            )
        elif self.config.method == 'character':
            return CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.config.length_function
            )
        elif self.config.method == 'sentence':
            return SentenceTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.method == 'spacy':
            return SpacyTextSplitter(
                pipeline=self.config.language,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.method == 'nltk':
            return NLTKTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported chunking method: {self.config.method}")

    def chunk_text(self, text: str) -> List[str]:
        """Split the text into chunks"""
        logger.info("Starting text chunking")
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Text chunked into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error during text chunking: {e}")
            raise


class SentenceTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the SentenceTextSplitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on sentences."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += ' ' + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                overlap = ' '.join(chunks[max(0, i - 1):i + 1])
                overlapped_chunks.append(overlap)
            chunks = overlapped_chunks

        return chunks
