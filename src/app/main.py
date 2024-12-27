import random
import sys
import os
import json
import uuid

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import streamlit as st
from dotenv import load_dotenv

import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval.retriever import DenseRetriever, RetrieverConfig as DRConfig
from src.models.llm_interface import LLMInterface, LLMConfig
from langchain.schema import HumanMessage, AIMessage


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Multimodal Learning Assistant",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

USER_DATA_FILE = "src/app/user_data.json"

def load_user_data() -> dict:
    """Load all user data from local JSON. Returns empty dict if none found."""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load user_data.json properly. Returning empty.")
            return {}
    return {}


def save_user_data(data: dict):
    """Save the entire user-data dictionary to local JSON."""
    os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def refine_text(llm_interface: LLMInterface, text: str, mode: str = "refine") -> str:
    """
    Take the existing text, optionally refine or simplify using the same LLM or
    a specialized rewriting LLM. We'll do a direct LLM call with a short prompt
    """
    if mode == "refine":
        prompt = (
            f"Please refine the following text for clarity and correctness:\n\n"
            f"{text}\n\nRefined version:"
        )
    else:
        prompt = (
            f"Please simplify the following text so it's easier to understand:\n\n"
            f"{text}\n\nSimplified version:"
        )

    result = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
    refined_output = result.generations[0][0].message.content
    return refined_output


def show_conversation():
    st.subheader("Conversation History")
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state['messages']:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            st.write(f"**{role}:** {msg.content}")

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


def should_offer_quiz() -> bool:
    """
    Decide whether to offer a quiz:
      - After 4 user queries (guaranteed).
      - Also, once we've passed 4 queries, there's a random chance each time.
    """
    queries_since_last_quiz = st.session_state.get('queries_since_quiz', 0)
    if queries_since_last_quiz < 2:
        return False

    if queries_since_last_quiz == 2:
        return True
    else:
        if random.random() < 0.3:
            return True
    return False


def generate_quiz_with_llm(llm_interface: LLMInterface, conversation_text: str, skill_level: str) -> dict:
    """
    Use the LLM to produce a short multiple-choice question about the conversation.

    Returns a dict like:
    {
      "question": "...",
      "choices": ["...","...","..."],
      "correct_idx": 1
    }

    You can expand the prompt to control how many choices you want, etc.
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

    No disclaimers, no additional text beyond that JSON structure.
    """

    response = llm_interface.llm.generate([[HumanMessage(content=prompt_str)]])
    raw_output = response.generations[0][0].message.content.strip()

    raw_output = raw_output.replace("```json", "")
    raw_output = raw_output.replace("```", "")

    raw_output = raw_output.replace("`", "")

    st.write("**DEBUG** Model’s raw quiz output:", raw_output)

    try:
        quiz_data = json.loads(raw_output)
        if all(k in quiz_data for k in ("question", "choices", "correct_idx")):
            return quiz_data
        else:
            st.warning("The LLM JSON is missing required keys. Fallback to default question.")
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
def show_micro_assessment_dialog(llm_interface: LLMInterface, skill_level: str):
    """
    Display a modal (dialog) with a short quiz.
    Quiz is generated by the LLm from conversation history.
    """
    st.write("**Micro-Assessment**")

    quiz_data = st.session_state.get('quiz_data', {})
    if not quiz_data:
        st.warning("No quiz data found!")
        return

    question = quiz_data["question"]
    choices = quiz_data["choices"]
    correct_idx = quiz_data["correct_idx"]

    st.write(question)
    user_choice = st.radio("Your Answer:", choices, index=0)
    quiz_submitted = st.button("Submit Answer")

    if quiz_submitted:
        if user_choice == choices[correct_idx]:
            st.success("Correct! Great job.")
            st.session_state['quiz_history'].append({"question": question, "correct": True})
        else:
            st.error("That's not correct. Keep learning, you'll get it next time.")
            st.session_state['quiz_history'].append({"question": question, "correct": False})

    st.session_state['queries_since_quiz'] = 0
    st.write("Thanks fore participating!")
    st.stop()


def main():
    st.title(":robot_face: Multimodal Learning Assistant Prototype")
    st.caption("Built with Streamlit, PyMuPDF, Transformers, and more!")

    if 'uak' not in st.session_state:
        st.session_state.uak = None
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'docs_history' not in st.session_state:
        st.session_state['docs_history'] = []
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'loaded_convo' not in st.session_state:
        st.session_state.loaded_convo = False
    if 'queries_since_quiz' not in st.session_state:
        st.session_state.queries_since_quiz = 0
    if 'quiz_history' not in st.session_state:
        st.session_state.quiz_history = []

    with st.container(border=True):
        st.header("User Profile & Preferences")
        st.write("Enter an Access Key (UAK) or leave blank to generate a new one:")
        col_key, col_button = st.columns([3,1])
        with col_key:
            entered_uak = st.text_input("User Access Key (Optional)", label_visibility="collapsed")
        with col_button:
            generate_key_button = st.button("Generate or Load Key", type="primary")

        if generate_key_button:
            user_data = load_user_data()
            if entered_uak.strip():
                st.session_state.uak = entered_uak.strip()
                if st.session_state.uak not in user_data:
                    user_data[st.session_state.uak] = {
                        "preferences": {},
                        "conversation": [],
                        "skill_level": "Intermediate"
                    }
                    save_user_data(user_data)
                st.toast(f"Loaded/Created user profile for UAK: {st.session_state.uak}")
            else:
                new_uak = str(uuid.uuid4())[:8]
                st.session_state.uak = new_uak
                user_data[new_uak] = {
                    "preferences": {},
                    "conversation": [],
                    "skill_level": "Intermediate"
                }
                save_user_data(user_data)
                st.toast(f"Generated new UAK: {new_uak}")

            st.rerun()

        if st.session_state.uak:
            st.info(f"Current UAK: {st.session_state.uak}")
        else:
            st.warning("No UAK set. Without an UAK, your conversation won't be persisted across sessions.")

    if st.session_state.uak and not st.session_state.loaded_convo:
        all_data = load_user_data()
        if st.session_state.uak in all_data:
            user_profile = all_data[st.session_state.uak]
            if "conversation" in user_profile and user_profile["conversation"]:
                st.session_state['messages'] = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in user_profile["conversation"]
                ]
        st.session_state.loaded_convo = True

    reset_col1, reset_col2 = st.columns([2,1])
    with reset_col1:
        st.write(" ")
    with reset_col2:
        if st.button("Reset Chat", type="secondary"):
            for key in ['messages', 'docs_history', 'last_answer', 'queries_since_quiz']:
                if key in st.session_state:
                    del st.session_state[key]
            if st.session_state.uak:
                data = load_user_data()
                if st.session_state.uak in data:
                    data[st.session_state.uak]["conversation"] = []
                    save_user_data(data)
            st.rerun()

    user_data_all = load_user_data()
    skill_level_default = "Intermediate"
    if st.session_state.uak in user_data_all:
        skill_level_default = user_data_all[st.session_state.uak].get("skill_level", "Intermediate")

    skill_levels = ["Beginner", "Intermediate", "Advanced"]
    chosen_skill_level = st.radio(
        "Skill Level (Affects Quiz Difficulty & Tone):",
        skill_levels,
        index=skill_levels.index(skill_level_default),
        horizontal=True
    )

    style_choice = st.segmented_control(
        label="Conversation Style",
        options=["Concise", "Detailed", "Friendly", "Technical"],
        key="style_segmented",
    )

    show_conversation()

    dense_retriever_path = "src/checkpoints/dense_retriever_checkpoint"
    faiss_index_path = "src/ingestion/data/new_index.pkl"

    dr_config = DRConfig(
        dense_retriever_path=dense_retriever_path,
        faiss_index_path=faiss_index_path,
        top_k=20,
        use_gpu=True
    )
    retriever = DenseRetriever(dr_config)

    llm_config = LLMConfig(
        provider='nvidia',
        model_name='nvidia/llama-3.1-nemotron-70b-instruct',
        temperature=0.7,
        max_tokens=512
    )
    llm_interface = LLMInterface(config=llm_config, retriever=None)

    with st.form("query_form", clear_on_submit=False):
        st.markdown("### Ask a Question")
        query = st.text_input("Enter your query:", value="", key="query_input")
        submit_button = st.form_submit_button("Search", type="primary")

    if submit_button:
        query = query.strip()
        if not query:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving documents..."):
                docs = retriever.retrieve(query, top_k=10)
                st.session_state['docs_history'].append(docs)

            st.session_state.queries_since_quiz += 1

            context = "\n\n".join([d.page_content for d in docs])
            if chosen_skill_level == "Beginner":
                tone_prefix = "Use very simple language and analogies for novices."
            elif chosen_skill_level == "Advanced":
                tone_prefix = "Use advanced technical terms and in-depth analysis."
            else:
                tone_prefix = "Use moderately technical explanations."

            style_prefix = f"(Respond in a {style_choice} style)" if style_choice else ""
            prompt = (
                f"Use the following context to answer the query.\n\n"
                f"Context:\n{context}\n"
                f"Query: {query}\nAnswer:\n"
                f"{tone_prefix}\n{style_prefix}"
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
                for m in st.session_state['messages']:
                    if isinstance(m, HumanMessage):
                        minimal_conv.append({"role": "user", "content": m.content})
                    else:
                        minimal_conv.append({"role": "assistant", "content": m.content})
                user_profile["conversation"] = minimal_conv
                user_profile["skill_level"] = chosen_skill_level
                data[st.session_state.uak] = user_profile
                save_user_data(data)

            st.rerun()

    if st.session_state['docs_history']:
        with st.expander("Show Retrieved Documents for Last Query", expanded=False):
            last_docs = st.session_state['docs_history'][-1]
            for i, doc in enumerate(last_docs, start=1):
                lecture_num = doc.metadata.get("lecture_number", "N/A")
                lecture_name = doc.metadata.get("lecture_name", "N/A")
                page = doc.metadata.get("page", "N/A")
                img_number = doc.metadata.get("img", None)

                content_preview = doc.page_content[:1000]
                if len(doc.page_content) > 1000:
                    content_preview += "..."

                st.markdown(f"**Document {i}:**")
                st.write(f"- **Lecture #:** {lecture_num}")
                st.write(f"- **Lecture Name**: {lecture_name}")
                st.write(f"- **Page:** {page}")
                if img_number:
                    st.write(f"- **Image #:** {img_number}")
                st.write(f"\n**Content Preview**:\n\n{content_preview}")
                st.divider()

    if should_offer_quiz():
        conversation_text = ""
        for msg in st.session_state['messages']:
            if isinstance(msg, HumanMessage):
                conversation_text += f"\nUser: {msg.content}"
            else:
                conversation_text += f"\nAssistant: {msg.content}"

        quiz_data = generate_quiz_with_llm(llm_interface, conversation_text, chosen_skill_level)
        st.session_state['quiz_data'] = quiz_data

        show_micro_assessment_dialog(llm_interface, chosen_skill_level)

    if st.session_state['last_answer']:
        st.subheader("Enhance or Simplify the Answer")

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
                    refined = refine_text(llm_interface, st.session_state['last_answer'], mode="refine")
                st.session_state['messages'].append(AIMessage(content=f"(Refined) {refined}"))
                st.session_state['last_answer'] = refined

                if st.session_state.uak:
                    data = load_user_data()
                    user_profile = data.get(st.session_state.uak, {})
                    minimal_conv = []
                    for m in st.session_state['messages']:
                        role = "user" if isinstance(m, HumanMessage) else "assistant"
                        minimal_conv.append({"role": role, "content": m.content})
                    user_profile["conversation"] = minimal_conv
                    data[st.session_state.uak] = user_profile
                    save_user_data(data)
                st.rerun()

            if st.button("Simplify", type="secondary", help="Simplify explanation for easier reading."):
                with st.spinner("Simplifying..."):
                    simplified = refine_text(llm_interface, st.session_state['last_answer'], mode="simplify")
                st.session_state['messages'].append(AIMessage(content=f"(Simplified) {simplified}"))
                st.session_state['last_answer'] = simplified

                if st.session_state.uak:
                    data = load_user_data()
                    user_profile = data.get(st.session_state.uak, {})
                    minimal_conv = []
                    for m in st.session_state['messages']:
                        role = "user" if isinstance(m, HumanMessage) else "assistant"
                        minimal_conv.append({"role": role, "content": m.content})
                    user_profile["conversation"] = minimal_conv
                    data[st.session_state.uak] = user_profile
                    save_user_data(data)
                st.rerun()


if __name__ == "__main__":
    main()
