import uuid

import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from src.models.llm_interface import LLMInterface
from typing import Optional


def build_header_section():
    """UI for user to either use an auto-generated UAK or enter an existing one."""
    if not st.session_state.get("uak"):
        new_uak = str(uuid.uuid4())[:8]
        st.session_state.uak = new_uak

    st.header("User Profile & Preferences")
    st.write("Below is your autoâ€generated Access Key. You can overwrite it with a known key to restore a session.")

    col_key, col_button = st.columns([4, 1])
    with col_key:
        entered_uak = st.text_input(
            "User Access Key (UAK)",
            value=st.session_state.uak,
            label_visibility="collapsed"
        )
    with col_button:
        if st.button("Set Key", type="primary"):
            if entered_uak.strip():
                st.session_state.uak = entered_uak.strip()
                st.toast(f"Using UAK: {st.session_state.uak}")
            else:
                st.warning("Please enter a valid key or keep the auto-assigned one.")

    st.info(f"Current UAK: {st.session_state.uak}")

def reset_button():
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
    """
    Display the conversation with latest messages on top
    """
    st.subheader("Conversation History", help="All messages exchanged so far")

    messages = st.session_state['messages']

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            role_name = "user"
        else:
            role_name = "assistant"

        with st.chat_message(name=role_name):
            st.write(msg.content)


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


def show_goal_ui():
    """
    Let the user specify or update their learning goal.
    This can be a text input or a selectbox, depending on the needs
    """
    st.markdown("#### Learning Goal")
    existing_goal = st.session_state.get('learning_goal', "")
    new_goal = st.text_input(
        "What is your current learning goal?",
        value=existing_goal,
        help="For example: pass an exam, master advanced terminology, or general understanding."
    )
    if new_goal != existing_goal:
        st.session_state.learning_goal = new_goal


def show_short_term_goal_ui():
    """
    UI letting the user specify a short-term objective,
    like '10 minutes to study' or 'Need a quick refresher'.
    """
    st.subheader("Short-Term Study Goal")
    predefined_goals = [
        "Quick 5-minute refresher",
        "15-minute targeted practice",
        "Deep dive if time allows"
    ]
    goal_mode = st.radio(
        "Choose a quick objective:",
        ["Choose a preset", "Custom goal"]
    )

    if goal_mode == "Choose a preset":
        selected = st.selectbox("Preset Options", predefined_goals)
        st.session_state["short_term_goal"] = selected
    else:
        custom_goal = st.text_input(
            "Enter your short-term goal (e.g., 'I have 10 minutes')",
            key="custom_short_term_goal"
        )
        if custom_goal.strip():
            st.session_state["short_term_goal"] = custom_goal.strip()

        if "short_term_goal" in st.session_state and st.session_state["short_term_goal"]:
            st.info(f"Your current short-term goal: {st.session_state['short_term_goal']}")
        else:
            st.warning("No short-term goal set yet.")
