import streamlit as st
from src.app.analytics_manager import show_user_analytics_dashboard


def analytics_page():
    st.title("Your Study Analytics")
    show_user_analytics_dashboard()