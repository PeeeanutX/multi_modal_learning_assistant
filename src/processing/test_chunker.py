from src.processing.chunker import AdvancedTextChunker

text = "This is a long text that needs to be split into smaller chunks. Each chunk should be manageable."

chunker = AdvancedTextChunker(chunk_size=50, chunk_overlap=10, split_method="recursive")
chunks = chunker.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
