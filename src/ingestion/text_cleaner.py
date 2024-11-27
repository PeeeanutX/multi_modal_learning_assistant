import re
import unicodedata
import logging
from typing import Optional, List
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import download
import contractions
from spellchecker import SpellChecker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def clean_text(
        text: str,
        lowercase: bool = True,
        remove_numbers: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        perform_stemming: bool = True,
        perform_lemmatization: bool = True,
        correct_spelling: bool = False,
        expand_contractions: bool = True
) -> str:
    logger.info("Starting text cleaning")

    text = unicodedata.normalize('NFKD', text)
    logger.debug(f"After Unicode normalization: {text}")

    text = re.sub(r'<[^>]+>', ' ', text)
    logger.debug(f"After removing HTML tags: {text}")

    if expand_contractions:
        text = contractions.fix(text)
        logger.debug(f"After expanding contractions: {text}")

    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    logger.debug(f"After removing URLs and emails: {text}")

    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
        logger.debug(f"After removing numbers: {text}")

    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
        logger.debug(f"After removing punctuation: {text}")

    if lowercase:
        text = text.lower()
        logger.debug(f"After converting to lowercase: {text}")

    tokens = word_tokenize(text)
    logger.debug(f"After tokenization: {text}")

    if remove_stopwords:
        tokens  = [word for word in tokens if word not in stop_words]
        logger.debug(f"After removing stop words: {tokens}")

    if correct_spelling:
        tokens = [spell.correction(word) for word in tokens]
        logger.debug(f"After spelling correction: {tokens}")

    if perform_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
        logger.debug(f"After stemming: {tokens}")
    elif perform_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        logger.debug(f"After lemmatization: {tokens}")

    cleaned_text = ' '.join(tokens)
    logger.info("Finished text cleaning")
    return cleaned_text

