import logging
from typing import List
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class ReRanker:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.prompt_template = """
        Given the following query and documents, rank the documents based on their relevance to the query.
        The most relevant document should be ranked first.

        Query: {query}

        Documents:
        {documents}

        Ranked Documents:
        """
        self.prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template=self.prompt_template
        )

    def re_rank(self, query: str, documents: List[Document]) -> List[Document]:
        logger.info(f"Re-ranking documents for query: '{query}'")
        try:
            documents_text = "\n".join([doc.page_content for doc in documents])
            input_data = self.prompt.format(query=query, documents=documents_text)

            messages = [HumanMessage(content=input_data)]

            result = self.llm.invoke(messages)

            if isinstance(result, AIMessage):
                ranked_documents_text = result.content.split("\n")
                ranked_documents = [Document(page_content=doc) for doc in ranked_documents_text if doc.strip()]
            else:
                raise ValueError("Unexpected response type from LLM")

            logger.info(f"Re-ranked {len(ranked_documents)} documents")
            return ranked_documents
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            raise
