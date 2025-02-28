import streamlit as st
import uuid

def run():
    st.title("User Configuration")

    if not st.session_state.get("uak"):
        new_uak = str(uuid.uuid4())[:8]
        st.session_state.uak = new_uak

    st.header("User Profile & Preferences")
    st.write("Below is your auto‐generated Access Key. You can overwrite it with a known key to restore a session.")

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

    skill_levels = ["Beginner", "Intermediate", "Advanced"]
    st.session_state.skill_level = st.radio(
        "Select skill level:",
        skill_levels,
        index=skill_levels.index(st.session_state.get("skill_level", "Intermediate"))
    )

    persona_options = ["Default Chatbot", "Mentor Mode (Expert)"]
    st.session_state.persona = st.selectbox(
        "Select Persona",
        persona_options,
        index=persona_options.index(st.session_state.get('persona', "Default Chatbot"))
    )

    st.session_state.short_term_goal = st.text_input(
        "Short-Term Goal?",
        value=st.session_state.get("short_term_goal", "")
    )

    st.write("Configuration saved to `st.session_state`.")
