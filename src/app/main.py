import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import streamlit as st
from dotenv import load_dotenv

import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval.retriever import DenseRetriever, RetrieverConfig as DRConfig
from src.models.llm_interface import LLMInterface, LLMConfig
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def refine_text(llm_interface: LLMInterface, text: str, mode: str = "refine") -> str:
    """
    Take the existing text, optionally refine or simplify using the same LLM or
    a specialized rewriting LLM. We'll do a direct LLM call with a short prompt
    """
    if mode == "refine":
        prompt = (f"Please refine the following text for clarity and correctness:\n\n"
                  f"{text}\n\nRefined version:")
    else:
        prompt = (f"Please simplify the following text so it's easier to understand:\n\n"
                  f"{text}\n\nSimplified version:")

    result = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
    refined_output = result.generations[0][0].message.content
    return refined_output


def show_conversation():
    """Display the conversation from session_state['message'], plus auto-scroll"""
    st.subheader("Conversation History")
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state['messages']:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            st.write(f"**{role}:** {msg.content}")

        st.write('<div id="chat-end"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <script>
        window.onload = function() {
            var chatEnd = document.getElementById("chat-end"):
            if (chatEnd){
                chatEnd.scrollIntoView({behavior: 'smooth'});
            }
        }
        </script>
        """,
        unsafe_allow_html=True
    )


def main():
    st.title("Multimodal Learning Assistant Prototype")

    if st.button("Reset Chat"):
        for key in ['messages', 'docs_history', 'last_answer']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'docs_history' not in st.session_state:
        st.session_state['docs_history'] = []

    dense_retriever_path = "src/checkpoints/dense_retriever_checkpoint"
    faiss_index_path = "src/ingestion/data/new_index.pkl"

    dr_config = DRConfig(
        dense_retriever_path=dense_retriever_path,
        faiss_index_path=faiss_index_path,
        top_k=5,
        use_gpu=True
    )
    retriever = DenseRetriever(dr_config)

    llm_config = LLMConfig(
        provider='nvidia',
        model_name='nvidia/llama-3.1-nemotron-70b-instruct',
        temperature=0.7,
        max_tokens=512
    )
    llm_interface = LLMInterface(config=llm_config, retriever=None)

    show_conversation()

    with st.form("query_form", clear_on_submit=False):
        query = st.text_input("Enter your query:", value="", key="query_input")
        submit_button = st.form_submit_button("Search")

    if submit_button:
        query = query.strip()
        if not query:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving documents..."):
                docs = retriever.retrieve(query, top_k=5)
                st.session_state['docs_history'].append(docs)

            context = "\n\n".join([d.page_content for d in docs])
            prompt = (
                f"Use the following context to answer the query.\n\nContext:\n{context}\n"
                f"Query: {query}\nAnswer:"
            )
            with st.spinner("Generating answer..."):
                result = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
                final_answer = result.generations[0][0].message.content

            st.session_state['messages'].append(HumanMessage(content=query))
            st.session_state['messages'].append(AIMessage(content=final_answer))
            st.session_state['last_answer'] = final_answer

            st.rerun()

    if st.session_state['docs_history']:
        show_docs = st.checkbox("Show Retrieved Documents for Last Query")
        if show_docs:
            st.subheader("Retrieved Documents for Last Query")
            last_docs = st.session_state['docs_history'][-1]
            for i, doc in enumerate(last_docs, start=1):
                lecture_num = doc.metadata.get("lecture_number", "N/A")
                lecture_name = doc.metadata.get("lecture_name", "N/A")
                page = doc.metadata.get("page", "N/A")
                img_number = doc.metadata.get("img", None)

                content_preview = doc.page_content[:1000]
                if len(doc.page_content) > 1000:
                    content_preview += "..."

                st.markdown(f"**Document {i}:**")
                st.write(f"- **Lecture #:** {lecture_num}")
                st.write(f"- **Lecture Name**: {lecture_name}")
                st.write(f"- **Page:** {page}")
                if img_number:
                    st.write(f"- **Image #:** {img_number}")

                st.write(f"\n**Content Preview**:\n\n{content_preview}")

    if 'last_answer' in st.session_state:
        st.subheader("Enhance or Simplify the Answer")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Refine Answer"):
                with st.spinner("Refining..."):
                    refined = refine_text(llm_interface, st.session_state['last_answer'], mode="refine")
                st.session_state['messages'].append(AIMessage(content=f"(Refined) {refined}"))
                st.session_state['last_answer'] = refined
                st.rerun()

        with col2:
            if st.button("Simplify Explanation"):
                with st.spinner("Simplifying..."):
                    simplified = refine_text(llm_interface, st.session_state['last_answer'], mode="simplify")
                st.session_state['messages'].append(AIMessage(content=f"(Simplified) {simplified}"))
                st.session_state['last_answer'] = simplified
                st.rerun()


if __name__ == "__main__":
    main()
