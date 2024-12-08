def test_retriever():
    from src.retrieval.retriever import RetrieverFactory, RetrieverConfig
    from langchain.schema import Document

    retriever_config = RetrieverConfig(
        retriever_type='default',
        search_kwargs={"k": 5}
    )
    retriever = RetrieverFactory.create_retriever(vector_store, retriever_config)
    query = "Explain Basic Design Process of AI-Based BIS."
    documents = RetrieverFactory.retrieve_documents(retriever, query)
    assert len(documents) > 0, "No documents retrieved."
    print(f"Retrieved {len(documents)} documents.")
    for doc in documents:
        print(doc.page_content)


test_retriever()