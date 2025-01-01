import streamlit as st
from typing import List, Dict, Any


def compute_adaptive_skill_level(
        performance_history: List[Dict[str, Any]],
        current_skill: str
) -> (str, str):
    """
    Analyze user performance history and the current skill level
    to suggest a recommended new skill level (or remain the same),
    but only if at least 4 quizzes have been completed.

    Returns a tuple (recommended_skill, reason_text).

    reason_text is a short textual explanation ofm the logic or thresholds used.
    """

    if len(performance_history) < 4:
        return current_skill, (
            "We only adapt skill after 4 or more quizzes. Keep practicing!"
        )

    total_quizzes = len(performance_history)
    correct_answers = sum(1 for q in performance_history if q.get("correct"))
    accuracy = correct_answers / total_quizzes

    base_reason = (
        f"Out of your last {total_quizzes} quizzes, you answered "
        f"{correct_answers} correctly, giving you an accuracy of {accuracy:.0%}."
    )

    if current_skill == "Advanced":
        if accuracy < 0.5:
            return (
                "Intermediate",
                base_reason + (
                    " We noticed your accuracy dropped below 50% at the Advanced level. "
                    "We suggest stepping down to Intermediate for more solid practice."
                )
            )
        else:
            return (
                "Advanced",
                base_reason + " You're performing adequately at Advanced level-no changes needed."
            )

    elif current_skill == "Beginner":
        if accuracy >= 0.75:
            return (
                "Intermediate",
                base_reason + (
                    " We noticed you breezed through the last few questions (≥75% accuracy). "
                    "We're introducing more complex tasks by moving you up to Intermediate."
                )
            )
        else:
            return (
                "Beginner",
                base_reason + " Keep practicing at Beginner level-no changes needed yet."
            )

    else:
        if accuracy >= 0.85:
            return (
                "Advanced",
                base_reason + (
                    " You have excelled at Intermediate (≥85% accuracy). "
                    "We recommend moving you up to Advanced."
                )
            )
        elif accuracy < 0.3:
            return (
                "Beginner",
                base_reason + (
                    " The accuracy is under 30% at Intermediate. "
                    "We recommend moving you down to Beginner for more foundational practice."
                )
            )
        else:
            return (
                "Intermediate",
                base_reason + " Your performance suggests staying at Intermediate."
            )


def show_adaptive_recommendation():
    """
    Display a recommendation for a new skill level
    ONLY if 4 or more quizzes have been completed.

    Provide an explicit explanation of *why* the system is recommending a change.
    Also allow the user to accept or override the recommendation if it differs.
    """
    current_skill = st.session_state.get("skill_level", "Intermediate")
    quiz_history = st.session_state.get("quiz_history", [])

    num_quizzes = len(quiz_history)
    st.markdown("### Skill Level Recommendation")

    if num_quizzes < 4:
        st.write(
            f"You have completed **{num_quizzes}** quiz{'es' if num_quizzes!=1 else ''}. "
            f"We recommend taking at least 4 quizzes before adjusting your skill level."
        )
        st.divider()
        return

    recommended_skill, reason_text = compute_adaptive_skill_level(quiz_history, current_skill)

    st.write(reason_text)

    if recommended_skill == current_skill:
        st.info("No change recommended. You're good to continue at your current skill level.")
        st.divider()
        return

    st.warning(
        f"Recommendation: Switch from **{current_skill}** to **{recommended_skill}**."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept Recommendation"):
            st.session_state["skill_level"] = recommended_skill
            st.success(f"Skill level changed to {recommended_skill}")
    with col2:
        if st.button("Stay at Current"):
            st.info("You have chosen to remain at your current skill level, overriding the recommendation.")

    st.divider()
