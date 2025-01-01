import logging
import random
from typing import Any, Dict, List
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from src.app.analytics_manager import init_analytics_in_state
import time

logger = logging.getLogger(__name__)


def init_session_state():
    """
    Initialize top-level session_state variables if they don't already exist.
    This function should be called once near the start of the Streamlit app.
    """
    if 'uak' not in st.session_state:
        st.session_state.uak = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'docs_history' not in st.session_state:
        st.session_state.docs_history = []

    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = None

    if 'loaded_convo' not in st.session_state:
        st.session_state.loaded_convo = False

    if 'queries_since_quiz' not in st.session_state:
        st.session_state.queries_since_quiz = 0

    if 'quiz_history' not in st.session_state:
        st.session_state.quiz_history = []

    if 'learning_goal' not in st.session_state:
        st.session_state.learning_goal = "No specific goal yet"

    if 'skill_level' not in st.session_state:
        st.session_state.skill_level = "Intermediate"

    init_analytics_in_state()