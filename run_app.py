import streamlit as st
from streamlit import Page, navigation
import sys
import os
from functools import partial

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app.main import assistant_page
from src.pages.peer_qa_page import peer_qa_page
from src.pages.scenario_practice_page import scenario_practice_page
from src.pages.analytics_page import analytics_page
from src.models.llm_interface import LLMInterface, LLMConfig
from src.retrieval.retriever import DenseRetriever, RetrieverConfig

def run_app():
    llm_config = LLMConfig(
        provider='nvidia',
        model_name='nvidia/llama-3.1-nemotron-70b-instruct',
        temperature=0.2,
        max_tokens=512
    )
    retriever_cfg = RetrieverConfig(
        dense_retriever_path="src/checkpoints/dense_retriever_checkpoint",
        faiss_index_path="src/ingestion/data/index.pkl",
        top_k=10,
        use_gpu=True
    )
    retriever = DenseRetriever(retriever_cfg)
    llm_interface = LLMInterface(config=llm_config, retriever=None)

    scenario_page_with_llm = partial(scenario_practice_page, llm_interface=llm_interface)
    peer_qa_with_llm = partial(peer_qa_page)

    pages = [
        Page(
            page=assistant_page,
            title="Learning Assistant",
            icon=":material/android:",
            url_path="assistant",
            default=True
        ),
        Page(
            page=peer_qa_with_llm,
            title="Peer Q&A",
            icon=":material/chat_bubble:",
            url_path="peer-qa"
        ),
        Page(
            page=scenario_page_with_llm,
            title="Real Case Practice",
            icon="📝"
        ),
        Page(
            page=analytics_page,
            title="Analytics", icon="📊",
        )
    ]

    current_page = st.navigation(
        pages=pages,
        position="sidebar",
        expanded=False
    )

    current_page.run()


if __name__ == "__main__":
    run_app()