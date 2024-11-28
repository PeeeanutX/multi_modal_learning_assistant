import logging
from typing import List, Optional
from dataclasses import dataclass, field

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


@dataclass
class ChunkerConfig:
    method: str = 'recursive'  # Options: 'recursive', 'character', 'sentence', 'spacy', 'nltk'
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    length_function: Optional[callable] = None
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    language: str = 'end_core_web_sm'  # For SpacyTextSplitter


class TextChunker:
    def __init__(self, config: ChunkerConfig):
        """Initializes the TextChunker with a specified configuration,"""
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
        """Splits the input text into chunks based on the selected method."""
        if self.split_method == "recursive":
            return self._split_with_recursive(text)
        elif self.split_method == "sentence":
            return self._split_by_sentence(text)
        elif self.split_method == "custom" and self.custom_delimiter:
            return self._split_by_custom_delimiter(text)
        else:
            logger.warning("Invalid split method or missing custom delimiter. Falling back to recursive")
            return self._split_with_recurisve(text)

    def _split_with_recursive(self, text: str) -> List[str]:
        """Splits text using LangChain's RecursiveCharacterTextSplitter."""
        logger.info("Splitting text using recursive character splitter.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)
        logger.info(f"Split into {len(chunks)} chunks using recursive method.")
        return chunks

    def _split_by_sentence(self, text: str) -> List[str]:
        """Splits text into chunks by sentences."""
        logger.info("Splitting text by sentences.")
        sentences = sent_tokenize(text, language=self.language)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
            logger.info(f"Split into {len(chunks)} chunks using sentence method.")

    def _split_by_custom_delimiter(self, text: str) -> List[str]:
        """Splits text using a custom delimiter."""
        logger.info(f"Splitting text using custom delimiter: '{self.custom_delimiter}'")
        segments = text.split(self.custom_delimiter)
        chunks = []
        current_chunk = ""
        for segment in segments:
            if len(current_chunk) + len(segment) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = segment
            else:
                current_chunk += self.custom_delimiter + segment
        if current_chunk:
            chunks.append(current_chunk.strip())
        logger.info(f"Split into {len(chunks)} chunks using custom delimiter.")
        return chunks

    def split_to_documents(self, text: str) -> List[Document]:
        """Splits the text into langChain 'Document' objects"""
        logger.info("Splitting text into Document objects.")
        chunks = self.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        logger.info(f"Created {len(documents)} Document objects.")
        return documents

    """
    def chunk_text(text, chunk_size=500,  chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks
    """


if __name__ == "__main__":
    raw_text = (
        "This is an example paragraph. It contains multiple sentences. "
        "Here is another sentence to make it longer. Let's add more content "
        "to test the splitting functionality. We want to see how this performs."
    )

    chunker = AdvancedTextChunker(chunk_size=100, chunk_overlap=10, split_method="recursive")
    chunks = chunker.split_text(raw_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")
