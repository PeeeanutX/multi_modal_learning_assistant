import sys
import os
import random
import logging

import streamlit as st
from dotenv import load_dotenv

NVIDIA_API_KEY=""

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
    show_short_term_goal_ui,
    reset_button
)
from src.app.quiz_manager import (
    should_offer_quiz,
    generate_quiz_with_llm,
    show_micro_assessment_dialog
)
from src.app.recommendation_manager import show_adaptive_recommendation
from src.retrieval.retriever import VectorStoreRetriever, RetrieverConfig
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


@st.cache_resource
def get_retriever():
    faiss_index_path = "src/ingestion/data/index"
    r_config = RetrieverConfig(
        faiss_index_path=faiss_index_path,
        top_k=10
    )
    retriever = VectorStoreRetriever(r_config)
    return retriever


@st.cache_resource
def get_llm_interface():
    llm_config = LLMConfig(
        provider='openai',
        model_name='gpt-4o',
        temperature=0.3,
        max_tokens=512,
	api_key=NVIDIA_API_KEY
    )
    return LLMInterface(config=llm_config, retriever=None)


import streamlit as st
import uuid


def config_step():
    st.title("Configuration Step")
    st.markdown(
        """
        Configure your session settings below before jumping into the AI chat. 
        You'll be able to revisit these settings later if needed.
        """
    )

    # A small expander for the Access Key settings
    with st.expander("🔑 Access Key (UAK) Settings", expanded=True):
        # Example auto-assign or user override logic
        if not st.session_state.get("uak"):
            st.session_state.uak = str(uuid.uuid4())[:8]

        st.write("The Access Key uniquely identifies your session.")
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            entered_uak = st.text_input(
                "User Access Key (UAK)",
                value=st.session_state.uak,
                label_visibility="collapsed"
            )
        with col_btn:
            if st.button("Set UAK"):
                if entered_uak.strip():
                    st.session_state.uak = entered_uak.strip()
                    st.success(f"UAK set to: {st.session_state.uak}")
                else:
                    st.warning("Please enter a valid key or keep the auto-generated one.")

        st.info(f"**Current UAK**: {st.session_state.uak}")

    # Skill level section
    st.subheader("1) Select Your Skill Level")
    st.write(
        "This will determine how the assistant tailors explanations, "
        "examples, and quizzes. Beginners get simpler language, advanced users get deeper analysis."
    )
    skill_levels = ["Beginner", "Intermediate", "Advanced"]
    selected_skill = st.radio(
        "Skill Level (Affects Quiz Difficulty & Tone):",
        skill_levels,
        index=skill_levels.index(st.session_state.get('skill_level', "Intermediate")),
        horizontal=True
    )

    # Persona
    st.subheader("2) Choose the Assistantâ€™s Persona")
    st.write(
        "A persona changes how the AI communicates with you. "
        "You can pick the default style or an expert mentor style."
    )
    persona_options = ["Default Chatbot", "Mentor Mode (Expert)"]
    selected_persona = st.selectbox(
        "Persona:",
        persona_options,
        index=persona_options.index(st.session_state.get('persona', "Default Chatbot")),
        help="Pick a style or persona for the AI."
    )

    # Short-term goal
    st.subheader("3) Short-Term Goal")
    st.write(
        "Let the assistant know your immediate objective (e.g., 'I have 10 minutes for a quick refresher'). "
        "The AI can adjust its explanations or focus accordingly."
    )
    st.session_state.short_term_goal = st.text_input(
        "Enter your short-term study goal:",
        value=st.session_state.get("short_term_goal", "")
    )

    # Save the userâ€™s new skill_level and persona into session_state
    if st.button("Save & Go to Chat", type="primary"):
        st.session_state['skill_level'] = selected_skill
        st.session_state['persona'] = selected_persona
        st.session_state.current_page = "chat"
        st.rerun()


def chat_step():
    st.title(":robot_face: Multimodal Learning Assistant")
    st.caption("AIBIS!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### User Access Key (UAK)")
        st.write(st.session_state.get('uak', 'N/A'))

        st.markdown("#### Skill Level")
        st.write(st.session_state.get('skill_level', 'N/A'))

    with col2:
        st.markdown("#### Persona")
        st.write(st.session_state.get('persona', 'N/A'))

        st.markdown("#### Short-Term Goal")
        st.write(st.session_state.get('short_term_goal', 'N/A'))

    init_session_state()
    retriever = get_retriever()
    llm_interface = get_llm_interface()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Ask a Question")
        with st.form("query_form"):
            query_input = st.text_area(
                "Enter your query:",
                value="",
                key="query_input_left",
                height=100,  # Initial height for multi-line input
                help="Press Shift+Enter for new lines"
            )
            submitted = st.form_submit_button("Search")

        if submitted:
            query = query_input.strip()
            if not query:
                st.warning("Please enter a query before searching")
            else:
                with st.spinner("Retrieving documents..."):
                    docs = retriever.retrieve(query, top_k=10)
                    st.session_state['docs_history'].append(docs)

                st.session_state.queries_since_quiz += 1

                selected_skill = st.session_state.get('skill_level', 'Intermediate')
                selected_persona = st.session_state.get('persona', 'Default Chatbot')

                if selected_skill == "Beginner":
                    style_prefix = "Use very simple language and analogies for novices."
                    summary_instructions = (
                        "Provide a short explanation as a **paragraph** with illustrative examples, "
                        "focusing on clarity for a novice audience.\n"
                    )
                elif selected_skill == "Advanced":
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
                    f"{short_term_prefix}\n"
                    f"{summary_instructions}\n"
                    "If relevant, keep the answer aligned with the persona, skill level, short-term goal "
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
                    user_profile["skill_level"] = selected_skill
                    user_profile["short_term_goal"] = st.session_state["short_term_goal"]
                    user_profile["persona"] = selected_persona

                    if "analytics" in st.session_state:
                        user_profile["analytics"] = st.session_state["analytics"]

                    data[st.session_state.uak] = user_profile
                    save_user_data(data)

            show_conversation()

        if should_offer_quiz():
            st.info("A quiz is no available! Click below whenever you're ready to check your knowledge.")
            if st.button("Open Knowledge Check"):
                conv_text = ""
                for msg in st.session_state['messages']:
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    conv_text += f"{role}: {msg.content}\n"

                quiz_data = generate_quiz_with_llm(llm_interface, conv_text, st.session_state['skill_level'])
                st.session_state['quiz_data'] = quiz_data
                show_micro_assessment_dialog(llm_interface, st.session_state['skill_level'])

        show_adaptive_recommendation()

        if st.button("Back to Config"):
            st.session_state.current_page = "config"
            st.rerun()

    with col_right:
        st.markdown("### Retrieved Documents")
        if st.session_state['docs_history']:
            last_docs = st.session_state['docs_history'][-1]
            for i, doc in enumerate(last_docs, start=1):
                lecture_num = doc.metadata.get("lecture_number", "N/A")
                lecture_name = doc.metadata.get("lecture_name", "N/A")
                page = doc.metadata.get("page_num", "N/A")

                content_preview = doc.page_content[:1000]
                if len(doc.page_content) > 1000:
                    content_preview += "..."

                st.markdown(f"**Document {i}:**")
                st.write(f"- **Lecture #:** {lecture_num}")
                st.write(f"- **Lecture Name**: {lecture_name}")
                st.write(f"- **Page:** {page}")

                snippet = doc.page_content[:200] + "..."
                st.write(f"`Snippet:` {snippet}")
                st.write("---")


def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "config"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.current_page == "config":
        config_step()
    else:
        chat_step()


if __name__ == "__main__":
    main()
