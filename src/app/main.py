import sys
import os
import random
import logging

import streamlit as st
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app.state_handlers import init_session_state
from src.app.user_profile import load_user_data, save_user_data, handle_access_key
from src.app.ui_components import (
    build_header_section,
    show_conversation,
    refine_text,
    show_refine_simplify_ui,
    show_goal_ui,
    show_short_term_goal_ui
)
from src.app.quiz_manager import (
    should_offer_quiz,
    generate_quiz_with_llm,
    show_micro_assessment_dialog
)
from src.app.recommendation_manager import show_adaptive_recommendation
from src.retrieval.retriever import DenseRetriever, RetrieverConfig
from src.models.llm_interface import LLMInterface, LLMConfig
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

st.set_page_config(
    page_title="Multimodal Learning Assistant",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def assistant_page():
    st.title(":robot_face: Multimodal Learning Assistant Prototype")
    st.caption("Built with Streamlit, PyMuPDF, Transformers, and more!")

    init_session_state()
    build_header_section()
    handle_access_key()

    dense_retriever_path = "src/checkpoints2/dense_retriever_checkpoint"
    faiss_index_path = "src/ingestion/data/new_index.pkl"

    dr_config = RetrieverConfig(
        dense_retriever_path=dense_retriever_path,
        faiss_index_path=faiss_index_path,
        top_k=15,
        use_gpu=True
    )
    retriever = DenseRetriever(dr_config)

    llm_config = LLMConfig(
        provider='nvidia',
        model_name='nvidia/llama-3.1-nemotron-70b-instruct',
        temperature=0.2,
        max_tokens=512
    )
    llm_interface = LLMInterface(config=llm_config, retriever=None)

    st.subheader("Skill Level & Persona")
    col1, col2 = st.columns([1.7, 1.3])

    with col1:
        skill_levels = ["Beginner", "Intermediate", "Advanced"]
        current_skill = st.radio(
            "Skill Level (Affects Quiz Difficulty & Tone):",
            skill_levels,
            index=1,
            horizontal=True
        )
        st.session_state['skill_level'] = current_skill

    with col2:
        persona_options = ["Default Chatbot", "Mentor Mode (Expert)"]
        selected_persona = st.selectbox(
            "Select Persona:",
            persona_options,
            index=0,
            help="Choose the style or persona of the AI. 'Mentor Mode' offers advanced, high-level discussions."
        )
        st.session_state["persona"] = selected_persona

    show_goal_ui()
    show_short_term_goal_ui()
    show_conversation()

    with st.form("query_form", clear_on_submit=False):
        st.markdown("### Ask a Question")
        query = st.text_input("Enter your query:", value="", key="query_input")
        submit_button = st.form_submit_button("Search", type="primary")

    if submit_button:
        query = query.strip()
        if not query:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving documents..."):
                docs = retriever.retrieve(query, top_k=10)
                st.session_state['docs_history'].append(docs)

            st.session_state.queries_since_quiz += 1

            if current_skill == "Beginner":
                style_prefix = "Use very simple language and analogies for novices."
                summary_instructions = (
                    "Provide a short explanation as a **paragraph** with illustrative examples, "
                    "focusing on clarity for a novice audience.\n"
                )
            elif current_skill == "Advanced":
                style_prefix = "Use advanced technical terms and in-depth analysis."
                summary_instructions = (
                    "Also provide a **concise bullet-point summary** at the end, enumerating "
                    "key insights susccintly.\n"
                )
            else:
                style_prefix = "Use moderately technical language."
                summary_instructions = (
                    "Include a **short summary** in 2-3 bullet points at the end, highlighting key ideas\n"
                )

            persona_prefix = ""
            if selected_persona == "Mentor Mode (Expert)":
                persona_prefix = (
                    "You are a highly experienced mentor, capable of providing thorough, high-level insights. "
                    "You speak with authority and clarity, referencing deeper knowledge as needed.\n"
                )

            learning_goal = st.session_state.get("learning_goal", "No specific goal")
            goal_prefix = f"\nThe user's learning goal is: {learning_goal}\nPlease address it if relevant.\n"

            short_term = st.session_state.get("short_term_goal", None)
            short_term_prefix = ""
            if short_term:
                short_term_prefix = (
                    f"\nThe user has a short-term study goal: '{short_term}'. "
                    "Adapt your explanation accordingly.\n"
                )

            full_context = "\n\n".join([d.page_content for d in docs])
            prompt = (
                f"{persona_prefix}"
                f"Use the following context to answer the query.\n\n"
                f"Context:\n{full_context}\n"
                f"Query: {query}\n"
                f"Answer:\n"
                f"{style_prefix}\n"
                f"{goal_prefix}\n"
                f"{short_term_prefix}\n"
                f"{summary_instructions}\n"
                "If relevant, keep the answer aligned with the persona, skill level, short-term and long-term goal "
                "and summary instructions."
            )

            with st.spinner("Generating answer..."):
                result = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
                final_answer = result.generations[0][0].message.content

            st.session_state['messages'].append(HumanMessage(content=query))
            st.session_state['messages'].append(AIMessage(content=final_answer))
            st.session_state['last_answer'] = final_answer

            if st.session_state.uak:
                data = load_user_data()
                user_profile = data.get(st.session_state.uak, {})
                minimal_conv = []
                for msg in st.session_state['messages']:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    minimal_conv.append({"role": role, "content": msg.content})
                user_profile["conversation"] = minimal_conv
                user_profile["skill_level"] = current_skill
                user_profile["learning_goal"] = st.session_state["learning_goal"]
                user_profile["persona"] = selected_persona

                if "analytics" in st.session_state:
                    user_profile["analytics"] = st.session_state["analytics"]

                data[st.session_state.uak] = user_profile
                save_user_data(data)

            st.rerun()


    if st.session_state['docs_history']:
        with st.expander("Show Retrieved Documents for Last Query", expanded=False):
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
                st.write("---")

    if should_offer_quiz():
        conv_text = ""
        for msg in st.session_state['messages']:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conv_text += f"{role}: {msg.content}\n"

        quiz_data = generate_quiz_with_llm(llm_interface, conv_text, current_skill)
        st.session_state['quiz_data'] = quiz_data
        show_micro_assessment_dialog(llm_interface, current_skill)

    if st.session_state['last_answer']:
        show_refine_simplify_ui(llm_interface)

    show_adaptive_recommendation()
