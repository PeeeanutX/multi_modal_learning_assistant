import os
import logging
from typing import Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.language_models import BaseLanguageModel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class LLMConfig:
    """
    Configuration for the Language Model Interface.
    """
    provider: str = 'nvidia'  # Options: 'nvidia, 'openai', 'huggingface'
    model_name: Optional[str] = 'nvidia/llama-3.1-nemotron-70b-instruct'
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512


class LLMInterface:
    def __init__(self, config: LLMConfig, retriever: BaseRetriever):
        """Initialize the LLM Interface with a specific configuration."""
        self.config = config
        self.retriever = retriever
        self.llm = self._initialize_llm()
        self.chain = self._initialize_chain()
        logger.info(
            f"LLMInterface initialized with provider '{self.config.provider}' and model '{self.config.model_name}'"
        )

    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize the language model based on the configuration."""
        provider = self.config.provider.lower()
        if provider == 'nvidia':
            if not self.config.api_key:
                self.config.api_key = os.getenv('NVIDIA_API_KEY')
                if not self.config.api_key:
                    raise ValueError("NVIDIA_API_KEY environment variable not set.")
            model_name = self.config.model_name or 'nvidia/llama-3.1-nemotron-70b-instruct'
            llm = ChatNVIDIA(
                model=model_name,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"NVIDIA LLM '{model_name}' initialized")
            return llm
        elif provider == 'openai':
            if not self.config.api_key:
                self.config.api_key = os.getenv('OPENAI_API_KEY')
                if not self.config.api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set.")
            model_name = self.config.model_name or 'text-davinci-003'
            llm = ChatOpenAI(
                model_name=model_name,
                openai_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"OpenAI LLM '{model_name}' initialized")
            return llm
        elif provider == 'huggingface':
            model_name = self.config.model_name or 'gpt2'
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                model_kwargs={
                    'temperature': self.config.temperature,
                    'max_length': self.config.max_tokens
                }
            )
            logger.info(f"HuggingFace LLM '{model_name}' initialized")
            return llm
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _initialize_chain(self):
        """
        Initialize the chain using create_retrieval_chain.
        This chain will handle retrieving documents and generating answers.
        """
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Do not make up an answer.
        
        {context}
        
        Question: {input}
        Answer:
        """
        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=prompt_template
        )

        try:
            combine_docs_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt
            )
            logger.info("Combine Docs Chain initialized successfully")

            chain = create_retrieval_chain(
                retriever=self.retriever,
                combine_docs_chain=combine_docs_chain
            )
            logger.info("Retrieval chain initialized successfully")
            return chain
        except TypeError as te:
            logger.error(f"Error initializing retrieval chain: {te}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing retrieval chain: {e}")

    def generate_response(self, query: str) -> str:
        """Generate a response to the query using the language model and the retriever."""
        try:
            logger.info(f"Generating response for query: '{query}'")
            result = self.chain.invoke({"input": query})
            answer = result.get("answer", "No answer found.")
            logger.info("Response generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            raise


def get_nvidia_response(retriever: BaseRetriever, query: str) -> str:
    """Generate a response using NVIDIA's LLM, given a retriever and a query"""
    try:
        llm_config = LLMConfig(provider='nvidia')
        llm_interface = LLMInterface(llm_config, retriever)
        response = llm_interface.generate_response(query)
        return response
    except Exception as e:
        logger.error(f"Error in get_nvidia_response: {e}")
        raise


"""
def get_nvidia_response(retriever: BaseRetriever, query: str):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")

    llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct", api_key=nvidia_api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    response = qa_chain.run(query)
    return response
"""