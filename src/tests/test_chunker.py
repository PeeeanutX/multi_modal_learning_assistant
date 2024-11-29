from src.processing.chunker import TextChunker, ChunkerConfig

# Initialize chunker configuration
config = ChunkerConfig(
    method='spacy',
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# Create a TextChunker instance
chunker = TextChunker(config)

# Sample text
text = "Your long text goes here..."

# Chunk the text
chunks = chunker.chunk_text(text)

# Output the number of chunks
print(f"Number of chunks: {len(chunks)}")

# Print the first chunk as an example
print("First chunk:")
print(chunks[0])
