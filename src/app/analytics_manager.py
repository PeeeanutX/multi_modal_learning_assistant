import time
import math
import streamlit as st

def init_analytics_in_state():
    """Initialize analytics dict if not present in session_state."""
    if 'analytics' not in st.session_state:
        st.session_state['analytics'] = {
            "quiz_correct_count": 0,
            "quiz_total_count": 0,
            "reading_time_seconds": 0.0,
            "exercise_time_seconds": 0.0,
            "last_activity_timestamp": time.time(),
        }


def record_quiz_outcome(is_correct: bool):
    """Record quiz results to analytics (mastery)."""
    analytics = st.session_state['analytics']
    analytics["quiz_total_count"] += 1
    if is_correct:
        analytics["quiz_correct_count"] += 1


def record_time_spent(mode: str):
    """
    mode: 'reading' or 'exercise'
    """
    analytics = st.session_state['analytics']
    now = time.time()
    elapsed = now - analytics["last_activity_timestamp"]

    if mode == 'reading':
        analytics["reading_time_seconds"] += elapsed
    elif mode == 'exercise':
        analytics["exercise_time_seconds"] += elapsed

    analytics["last_activity_timestamp"] = now


def show_user_analytics_dashboard():
    """Show a mini-dash of user's mastery, reading vs. exercise time."""
    if 'analytics' not in st.session_state:
        st.warning("No analytics data yet.")
        return

    analytics = st.session_state['analytics']

    total_quizzes = analytics["quiz_total_count"]
    correct_quizzes = analytics["quiz_correct_count"]
    mastery_percent = (correct_quizzes / total_quizzes * 100) if total_quizzes > 0 else 0.0

    reading_sec = analytics["reading_time_seconds"]
    exercise_sec = analytics["exercise_time_seconds"]
    total_sec = reading_sec + exercise_sec
    reading_pct = reading_sec / total_sec * 100 if total_sec > 0 else 0
    exercise_pct = exercise_sec / total_sec * 100 if total_sec > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("Mastery Level (%)", f"{mastery_percent:.1f}%")
    col2.metric("Reading vs. Exercises", f"{reading_pct:.1f}% / {exercise_pct:.1f}%")

    with st.expander("Analytics Details"):
        st.write(f"**Quizzes** {correct_quizzes} / {total_quizzes} correct.")
        st.write(f"**Reading Time** ~{math.ceil(reading_sec)}s,  "
                 f"**Exercise Time** ~{math.ceil(exercise_sec)}s")

    st.write("---")
    st.caption("Analytics are stored in session state. "
               "Persist them in user_profile if you want them saved across reloads.")
