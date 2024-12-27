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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set a page title, icon, and wide layout
st.set_page_config(
    page_title="Multimodal Learning Assistant",
    page_icon=":books:",
    layout="wide"
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
    """Perform a 'refine' or 'simplify' LLM call to transform text."""
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

def message_box(role: str, content: str):
    """
    A helper to display messages with a bit of styling.
    'role' is either 'User' or 'Assistant'. We'll color them differently.
    """
    if role == "User":
        box_color = "#D7ECFF"  # Light-blue
    else:
        box_color = "#E8E8E8"  # Light-gray

    st.markdown(
        f"""
        <div style="background-color: {box_color}; padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 0.5rem;">
            <strong>{role}:</strong> {content}
        </div>
        """,
        unsafe_allow_html=True
    )

def show_conversation():
    """Display the conversation from session_state['messages'], plus auto-scroll."""
    st.subheader("Conversation History")
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state['messages']:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            message_box(role, msg.content)

        st.write('<div id="chat-end"></div>', unsafe_allow_html=True)

    # Attempt to auto-scroll to bottom
    st.markdown(
        """
        <script>
        window.onload = function() {
            var chatEnd = document.getElementById("chat-end");
            if (chatEnd){
                chatEnd.scrollIntoView({behavior: 'smooth'});
            }
        }
        </script>
        """,
        unsafe_allow_html=True
    )

def main():
    # ------------------------------------
    # Sidebar - Preferences & Info
    # ------------------------------------
    with st.sidebar:
        st.title(":gear: Preferences")

        st.markdown("**Appearance**")
        # Add toggles/sliders for future expansions, e.g. text size or theme
        # This is just a placeholder
        # text_size = st.slider("Chat Font Size", 12, 24, 16)

        st.markdown("---")
        st.markdown("**Session Management**")

        # (We'll show the user input box for the UAK in the sidebar for a cleaner UI)
        st.write("Enter an Access Key (UAK) or leave blank to generate a new one:")
        entered_uak = st.text_input("User Access Key (Optional)", key="sidebar_uak")
        generate_key_button = st.button("Generate/Load Key")

        st.markdown("---")
        st.markdown(
            """
            **What is this?**  
            This is a prototype chatbot that retrieves documents from a custom 
            FAISS index and uses an LLM to answer queries.  
            **Developer**: You  
            **Built with**: Streamlit, LangChain, ...
            """
        )

    st.title("Multimodal Learning Assistant Prototype")
    st.write(
        "<p style='color: grey; font-size:0.9rem;'>Welcome! Ask questions, refine answers, and explore documents.</p>",
        unsafe_allow_html=True
    )

    if 'uak' not in st.session_state:
        st.session_state.uak = None

    if generate_key_button:
        user_data = load_user_data()
        if entered_uak.strip():
            st.session_state.uak = entered_uak.strip()
            if st.session_state.uak not in user_data:
                user_data[st.session_state.uak] = {
                    "preferences": {},
                    "conversation": [],
                }
                save_user_data(user_data)
            st.success(f"Loaded/Created user profile for UAK: {st.session_state.uak}")
        else:
            new_uak = str(uuid.uuid4())[:8]
            st.session_state.uak = new_uak
            user_data[new_uak] = {
                "preferences": {},
                "conversation": [],
            }
            save_user_data(user_data)
            st.success(f"Generated new UAK: {new_uak}")
        st.experimental_rerun()

    # Show current UAK status
    if st.session_state.uak:
        st.info(f"**Current UAK**: {st.session_state.uak}")
    else:
        st.warning("No UAK set. Without a UAK, your conversation won't persist across sessions.")

    # session_state initialization
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'docs_history' not in st.session_state:
        st.session_state['docs_history'] = []
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'loaded_convo' not in st.session_state:
        st.session_state.loaded_convo = False

    # Attempt to load conversation from user_data once
    if st.session_state.uak and not st.session_state.loaded_convo:
        all_data = load_user_data()
        if st.session_state.uak in all_data:
            user_profile = all_data[st.session_state.uak]
            if "conversation" in user_profile and user_profile["conversation"]:
                st.session_state['messages'] = [
                    HumanMessage(content=m["content"]) if m["role"]=="user"
                    else AIMessage(content=m["content"])
                    for m in user_profile["conversation"]
                ]
        st.session_state.loaded_convo = True

    # Reset Chat
    if st.button("Reset Chat"):
        for key in ['messages', 'docs_history', 'last_answer']:
            if key in st.session_state:
                del st.session_state[key]
        # Also remove from user_data if we want a fresh start
        if st.session_state.uak:
            data = load_user_data()
            if st.session_state.uak in data:
                data[st.session_state.uak]["conversation"] = []
                save_user_data(data)
        st.experimental_rerun()

    # Show conversation
    show_conversation()

    # Setup retrieval and LLM
    dense_retriever_path = "src/checkpoints/dense_retriever_checkpoint"
    faiss_index_path = "src/ingestion/data/new_index.pkl"

    dr_config = DRConfig(
        dense_retriever_path=dense_retriever_path,
        faiss_index_path=faiss_index_path,
        top_k=5,
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

    # Query Form
    with st.form("query_form", clear_on_submit=False):
        query = st.text_input("Enter your query:", value="", key="query_input")
        submit_button = st.form_submit_button("Search")

    # On search
    if submit_button:
        query = query.strip()
        if not query:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving documents..."):
                docs = retriever.retrieve(query, top_k=5)
                st.session_state['docs_history'].append(docs)

            context = "\n\n".join([d.page_content for d in docs])
            prompt = (
                f"Use the following context to answer the query.\n\nContext:\n{context}\n"
                f"Query: {query}\nAnswer:"
            )
            with st.spinner("Generating answer..."):
                result = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
                final_answer = result.generations[0][0].message.content

            st.session_state['messages'].append(HumanMessage(content=query))
            st.session_state['messages'].append(AIMessage(content=final_answer))
            st.session_state['last_answer'] = final_answer

            # Persist conversation if we have UAK
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
                data[st.session_state.uak] = user_profile
                save_user_data(data)

            st.experimental_rerun()

    # Show retrieved docs
    if st.session_state['docs_history']:
        show_docs = st.checkbox("Show Retrieved Documents for Last Query")
        if show_docs:
            st.subheader("Retrieved Documents for Last Query")
            last_docs = st.session_state['docs_history'][-1]
            for i, doc in enumerate(last_docs, start=1):
                lecture_num = doc.metadata.get("lecture_number", "N/A")
                lecture_name = doc.metadata.get("lecture_name", "N/A")
                page = doc.metadata.get("page", "N/A")
                img_number = doc.metadata.get("img", None)

                # Additional styling for doc info
                st.markdown(f"**Document {i}:**")
                st.write(
                    f"- **Lecture #:** {lecture_num}\n"
                    f"- **Lecture Name:** {lecture_name}\n"
                    f"- **Page:** {page}"
                )
                if img_number:
                    st.write(f"- **Image #:** {img_number}")

                content_preview = doc.page_content[:1000]
                if len(doc.page_content) > 1000:
                    content_preview += "..."
                st.write(f"**Content Preview**:\n\n{content_preview}")

    # Enhance or simplify last answer
    if st.session_state['last_answer']:
        st.subheader("Enhance or Simplify the Answer")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Refine Answer"):
                with st.spinner("Refining..."):
                    refined = refine_text(llm_interface, st.session_state['last_answer'], mode="refine")
                st.session_state['messages'].append(AIMessage(content=f"(Refined) {refined}"))
                st.session_state['last_answer'] = refined

                # Persist conversation
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
                    data[st.session_state.uak] = user_profile
                    save_user_data(data)

                st.experimental_rerun()

        with col2:
            if st.button("Simplify Explanation"):
                with st.spinner("Simplifying..."):
                    simplified = refine_text(llm_interface, st.session_state['last_answer'], mode="simplify")
                st.session_state['messages'].append(AIMessage(content=f"(Simplified) {simplified}"))
                st.session_state['last_answer'] = simplified

                # Persist conversation
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
                    data[st.session_state.uak] = user_profile
                    save_user_data(data)

                st.experimental_rerun()


if __name__ == "__main__":
    main()
