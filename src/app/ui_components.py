import uuid

import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from src.models.llm_interface import LLMInterface
from typing import Optional


def build_header_section():
    """UI for user to input or generate an Access Key (UAK)."""
    st.header("User Profile & Preferences")
    st.write("Enter an Access Key (UAK) or leave blank to generate a new one:")

    col_key, col_button = st.columns([3, 1])
    with col_key:
        entered_uak = st.text_input("User Access Key (Optional)", label_visibility="collapsed")
    with col_button:
        if st.button("Generate or Load Key", type="primary"):
            if entered_uak.strip():
                st.session_state.uak = entered_uak.strip()
                st.toast(f"Loaded/Created user profile for UAK: {st.session_state.uak}")
            else:
                new_uak = str(uuid.uuid4())[:8]
                st.session_state.uak = new_uak
                st.toast(f"Generated new UAK: {new_uak}")

    if st.session_state.uak:
        st.info(f"Current UAK: {st.session_state.uak}")
    else:
        st.warning("No UAK set. Without an UAK, your conversation won't be persisted across sessions.")

    reset_col1, reset_col2 = st.columns([2, 1])
    with reset_col1:
        st.write(" ")
    with reset_col2:
        if st.button("Reset Chat", type="secondary"):
            for key in ['messages', 'docs_history', 'last_answer', 'queries_since_quiz']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def show_conversation():
    st.subheader("Conversation History", help="All messages exchanged so far")
    with st.container():
        for msg in st.session_state['messages']:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            st.write(f"**{role}** {msg.content}")

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


def refine_text(llm_interface, text: str, mode: str = "refine") -> str:
    """Refine or simplify a pice of text using the LLM."""
    if mode == "refine":
        prompt = (
            "Please refine the following text to ensure clarity and correctness.\n\n"
            f"{text}\n\nRefined version:"
        )
    else:
        prompt = (
            "Please simplify the following text so it's easier to understand.\n\n"
            f"{text}\n\nSimplified version:"
        )
    result = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
    return result.generations[0][0].message.content


def show_refine_simplify_ui(llm_interface):
    """UI to refine or simplify the last answer. Uses st.popover or just a container."""
    st.subheader("Enhance or Simplify the Answer")
    last_answer = st.session_state.get('last_answer', "")
    feedback = st.feedback(options="thumbs", key="feedback_widget", disabled=False)
    if feedback is not None:
        if feedback == 0:
            st.toast("Thanks for your feedback: Thumbs Down.")
        else:
            st.toast("Thanks for your feedback: Thumbs Up!")

    with st.popover("Improve Answer", icon=":material/auto_fix_high:"):
        st.write("Refine or Simplify the last LLM answer.")
        if st.button("Refine", type="secondary", help="Refine answer for clarity."):
            with st.spinner("Refining..."):
                refined = refine_text(llm_interface, last_answer, mode="refine")
            st.session_state['messages'].append(AIMessage(content=f"(Refined) {refined}"))
            st.session_state['last_answer'] = refined
            st.rerun()

        if st.button("Simplify", type="secondary", help="Simplify explanation for easier reading."):
            with st.spinner("Simplifying..."):
                simplified = refine_text(llm_interface, last_answer, mode="simplify")
            st.session_state['messages'].append(AIMessage(content=f"(Simplified) {simplified}"))
            st.session_state['last_answer'] = simplified
            st.rerun()
