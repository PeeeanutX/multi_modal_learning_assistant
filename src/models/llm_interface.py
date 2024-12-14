import os
import logging
from typing import Optional, List
from dataclasses import dataclass

import torch.cuda
from torch.nn import CrossEntropyLoss

from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.language_models import BaseLanguageModel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from transformers import AutoTokenizer, AutoModelForCausalLM

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

        if self.config.provider.lower() == 'huggingface':
            self._initialize_hf_model()

        if self.retriever is not None:
            self.chain = self._initialize_chain()
        else:
            self.chain = None

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
            model_name = self.config.model_name or 'mistralai/Pixtral-Large-Instruct-2411'
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=self.config.temperature,  # moved out of model_kwargs
                max_length=self.config.max_tokens,  # moved out of model_kwargs
                model_kwargs={}  # or remove model_kwargs entirely if empty
            )
            logger.info(f"HuggingFace LLM '{model_name}' initialized")
            return llm
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _initialize_hf_model(self):
        """
        Initialize a Hugging Face model and tokenizer for batch scoring.
        This is only needed if provider='huggingface'.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = self.config.model_name or 'tiiuae/falcon-7b'
        logger.info(f"Loading Hugging Face model and tokenizer for scoring: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
        if self.chain is None:
            raise ValueError("No chain initialized. This LLMInterface may be for scoring only.")
        try:
            logger.info(f"Generating response for query: '{query}'")
            result = self.chain.invoke({"input": query})
            answer = result.get("answer", "No answer found.")
            logger.info("Response generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            raise

    def batch_score(self, input_texts: List[str], output_texts: List[str], delimiter='\n\n') -> list:
        """
        Compute a score (e.g., average log probability) for each (input_text, output_text) pair.
        This method only works if provider='huggingface' and we have a model & tokenizer loaded.
        """
        if self.config.provider.lower() != 'huggingface':
            raise NotImplementedError("batch_score is currently only implemented for the huggingface provider.")

        assert len(input_texts) == len(output_texts), "input_texts and output_texts must have the same length"
        self.model.eval()

        batch_size = 8
        scores = []

        for start in range(0, len(input_texts), batch_size):
            end = start + batch_size
            batch_inp = input_texts[start:end]
            batch_out = output_texts[start:end]

            combined_texts = [f"{i.strip()}{delimiter}{o.strip()}" for i, o in zip(batch_inp, batch_out)]

            # Tokenize combined input+output
            tokenized = self.tokenizer(combined_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            # Tokenize just the output_texts to find output length
            out_tokenized = self.tokenizer(batch_out, return_tensors='pt', padding=True, truncation=True)
            out_ids = out_tokenized['input_ids'].to(self.device)
            out_attention_mask = out_tokenized['attention_mask'].to(self.device)

            # Tokenize input+delimiter alone to know where output starts
            inp_delim_texts = [f"{i.strip()}{delimiter}" for i in batch_inp]
            inp_delim_tokenized = self.tokenizer(inp_delim_texts, return_tensors='pt', padding=True, truncation=True)
            inp_delim_length = inp_delim_tokenized['input_ids'].size(1)
            out_length = out_ids.size(1)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            labels_for_loss = shift_labels.clone()
            labels_for_loss[:] = -100

            seq_len = shift_labels.size(1)
            for bi in range(shift_labels.size(0)):
                start_idx = inp_delim_length - 1  # output starts after input+delimiter
                end_idx = start_idx + out_length
                end_idx = min(end_idx, seq_len)
                labels_for_loss[bi, start_idx:end_idx] = shift_labels[bi, start_idx:end_idx]

            loss_fct = CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels_for_loss.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            valid_token_mask = (labels_for_loss != -100)
            sum_loss = (per_token_loss * valid_token_mask).sum(dim=1)
            num_valid_tokens = valid_token_mask.sum(dim=1)
            avg_loss = sum_loss / num_valid_tokens.float().clamp_min(1.0)

            avg_log_prob = -avg_loss
            scores.extend(avg_log_prob.cpu().tolist())

        return scores


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
