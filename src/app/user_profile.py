import os
import json
import uuid
import logging
import streamlit as st

USER_DATA_FILE = "src/app/user_data.json"


def load_user_data() -> dict:
    """Load all user data from local JSON. Returns empty dictionary if none found."""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_data(data: dict):
    """Save the entire user-data dictionary to local JSON."""
    os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def handle_access_key():
    """
    Render the UI for user to enter or generate a new UAK; load from JSON if found.
    """
    if st.session_state.loaded_convo:
        return

    user_data = load_user_data()

    if st.session_state.uak and st.session_state.uak in user_data:
        st.session_state.loaded_convo = True
        user_profile = user_data[st.session_state.uak]
        if "conversation" in user_profile:
            from langchain.schema import HumanMessage, AIMessage
            messages = []
            for m in user_profile["conversation"]:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))
            st.session_state['messages'] = messages

        if "skill-level" in user_profile:
            st.session_state['skill_level'] = user_profile["skill_level"]

        if "short_term_goal" in user_profile:
            st.session_state['short_term_goal'] = user_profile["short_term_goal"]

        if "analytics" in user_profile:
            from src.app.analytics_manager import init_analytics_in_state
            init_analytics_in_state()
            saved_analytics = user_profile["analytics"]
            for key, val in saved_analytics.items():
                st.session_state["analytics"][key] = val
