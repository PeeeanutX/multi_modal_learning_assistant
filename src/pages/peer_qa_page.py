import streamlit as st
from src.app.peer_qa_manager import (
    post_question,
    post_answer,
    upvote_question,
    upvote_answer,
    search_questions,
)


def user_has_upvoted_question(q, user_uak):
    """Return True if user_uak is in q['voters']. Otherwise False."""
    if not user_uak:
        return False
    return user_uak in q.get("voters", [])


def user_has_upvoted_answer(ans, user_uak):
    """Return True if user_uak is in ans['voters']. Otherwise False."""
    if not user_uak:
        return False
    return user_uak in ans.get("voters", [])


def show_question_item(q, user_uak):
    """Helper to render a single question + answers."""
    st.markdown(f"**Q:** {q['question_text']}")
    st.write(f"Skill Tag: {q['skill_tag']}")
    st.write(f"Upvotes: {q['upvotes']}")

    already_voted_q = user_has_upvoted_question(q, user_uak)

    upvote_btn_label = "Upvote"
    if st.button(upvote_btn_label, key=f"upvote_q_{q['id']}", disabled=already_voted_q):
        success = upvote_question(q['id'], user_uak)
        if success:
            st.rerun()
        else:
            st.error("Failed to upvote question or you already voted.")

    for ans in q["answers"]:
        with st.container():
            st.markdown(f"- **Answer**: {ans['answer_text']}")
            st.write(f"  Upvotes: {ans['upvotes']}")
            already_voted_a = user_has_upvoted_answer(ans, user_uak)

            if st.button(upvote_btn_label, key=f"upvote_a_{ans['answer_text']}", disabled=already_voted_a):
                success = upvote_answer(q['id'], ans['answer_text'], user_uak)
                if success:
                    st.rerun()
                else:
                    st.error("Failed to upvote answer.")

    with st.expander("Answer this question"):
        new_ans = st.text_area("Your Answer", key=f"ans_for_{q['id']}")
        if st.button(f"Post Answer to {q['id']}"):
            if post_answer(q['id'], new_ans, user_uak):
                st.success("Posted answer!")
                st.rerun()
            else:
                st.error("Could not post answer. Possibly empty or question not found.")
    st.markdown("---")


def peer_qa_page():
    st.title("📖 Peer Q&A")
    st.write("Ask or answer questions from other learners.")

    user_uak = st.session_state.uak
    search_col, filter_col = st.columns([3,1])
    with search_col:
        search_term = st.text_input("Search questions:", value="")
    with filter_col:
        skill_filter = st.selectbox("Skill Filter", ["All", "Beginner", "Intermediate", "Advanced"])

    results = search_questions(search_term, skill_filter)
    results_sorted = sorted(results, key=lambda x: x["upvotes"], reverse=True)

    st.subheader(f"Found {len(results_sorted)} question(s)")
    for q in results_sorted:
        show_question_item(q, user_uak)

    with st.expander("Ask a new question"):
        st.write("Post your question below.")
        new_q_text = st.text_area("Question:")
        new_q_skill = st.selectbox("Skill Tag for question", ["Beginner", "Intermediate", "Advanced"])
        if st.button("Submit Question"):
            if post_question(new_q_text, new_q_skill):
                st.success("Question posted successfully!")
                st.rerun()
            else:
                st.error("Question was empty or there was a saving error.")
