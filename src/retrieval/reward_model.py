import logging
from typing import List
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class RewardModel:
    def __init__(self, model_name: str = "google/electra-base-discriminator"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def score(self, query: str, documents: List[Document]) -> List[float]:
        logger.info(f"Scoring documents for query: '{query}'")
        try:
            inputs = self.tokenizer([query + " " + doc.page_content for doc in documents], return_tensors="pt",
                                    padding=True, truncation=True)
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[:, 1]
            logger.info(f"Scored {len(scores)} documents")
            return scores.tolist()
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            raise
