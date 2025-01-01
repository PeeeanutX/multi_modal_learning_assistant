import json
import random
import streamlit as st
from typing import Dict, Any, Optional


from langchain.schema import HumanMessage
from src.models.llm_interface import LLMInterface


def should_offer_quiz() -> bool:
    """
    Decide whether to offer a quiz to the user:
      - After 4 user queries, definitely.
      - After that threshold, there's a random chance each time.
    """
    queries_since = st.session_state.get('queries_since_quiz', 0)
    if queries_since < 2:
        return False
    if queries_since == 2:
        return True
    else:
        return random.random() < 0.3


def generate_quiz_with_llm(
        llm_interface: LLMInterface,
        conversation_text: str,
        skill_level: str
) -> dict:
    """
    Use the LLM to produce a short multiple-choice question based on the conversation text.
    The question should be JSON-encoded with keys: question, choices, correct_idx.
    """
    prompt_str = f"""
    Below is a conversation summary and context:

    {conversation_text}

    Task:
    1. Identify a key concept or idea from this conversation.
    2. Create exactly ONE short multiple-choice question that tests knowledge about it.
    3. Provide 3 or 4 answer choices, and specify which choice index is correct.

    Difficulty: {skill_level}

    Output Format:
    Return STRICT JSON, like this example (no extra commentary):

    {{
      "question": "Which best describes the difference between X and Y?",
      "choices": [
        "Answer A",
        "Answer B",
        "Answer C",
        "Answer D"
      ],
      "correct_idx": 2
    }}

    No disclaimers, no extra text beyond that JSON structure.
    """

    response = llm_interface.llm.generate([[HumanMessage(content=prompt_str)]])
    raw_output = response.generations[0][0].message.content.strip()

    cleaned_output = raw_output.replace("```json", "").replace("```", "").replace("`", "")

    st.write("**DEBUG** Model’s raw quiz output:", raw_output)

    try:
        parsed = json.loads(cleaned_output)
        if all(k in parsed for k in ("question", "choices", "correct_idx")):
            return parsed
        else:
            st.warning("Quiz data missing keys. Fallback to default.")
            return {
                "question": "Which statement is correct about neural networks?",
                "choices": ["...", "...", "..."],
                "correct_idx": 0
            }
    except Exception as e:
        st.error(f"Couldn’t parse LLM output as JSON. Error: {e}\nRaw: {raw_output}")
        return {
            "question": "What is an example fallback question?",
            "choices": ["Option A", "Option B", "Option C"],
            "correct_idx": 1
        }

@st.dialog("Quick Knowledge Check!", width="large")
def show_micro_assessment_dialog(llm_interface, skill_level: str):
    """
    Display a modal (dialog) with a short quiz.
    The quiz_data dict typically has the structure:
      {
        "question": "string",
        "choices": ["Choice A", "Choice B", "Choice C", ...],
        "correct_idx": 1
      }

    on_submit_callback is an optional callback that can be invoked after a user
    answers the quiz. E.g., for logging results or updating session state outside
    the dialog context.
    """

    st.markdown("### Micro-Assessment")
    quiz_data = st.session_state.get('quiz_data', {})
    if not quiz_data:
        st.warning("No quiz data found!")
        return

    question = quiz_data.get("question", "No question available.")
    choices = quiz_data.get("choices", [])
    correct_idx = quiz_data.get("correct_idx", 0)

    st.write(question)

    user_choice = st.radio("Your Answer:", choices, index=0)

    quiz_submitted = st.button("Submit Answer")

    if quiz_submitted:
        # Evaluate the user’s answer
        if user_choice == choices[correct_idx]:
            st.success("Correct! Great job.")
            st.session_state.quiz_history.append({"question": question, "correct": True})
        else:
            st.error("That's not correct. Keep learning, you'll get it next time.")
            st.session_state.quiz_history.append({"question": question, "correct": False})

        st.session_state['queries_since_quiz'] = 0
        st.write("Thank you for participating!")
        st.stop()
