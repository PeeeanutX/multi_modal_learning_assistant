from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=500,  chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks