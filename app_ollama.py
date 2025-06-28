# app_ollama.py
import streamlit as st
from utils_ollama import get_answer
import os

def main():
    """
    The main function that runs the Streamlit application for the local Ollama RAG.
    """
    st.set_page_config(page_title="Local Chat with PDFs (Ollama)", layout="wide")
    st.header("Local RAG Application: Chat with your PDFs using Ollama 💬")

    # Check for FAISS index
    if not os.path.exists("faiss_index_ollama"):
        st.error("FAISS index for Ollama not found. Please run the `ingest_ollama.py` script first.")
        st.info(
            "Instructions:\n"
            "1. Make sure Ollama is running.\n"
            "2. Ensure you have pulled the necessary models: `ollama pull llama3` and `ollama pull nomic-embed-text`.\n"
            "3. Create a folder named 'data' and place your PDFs inside.\n"
            "4. Run the command: `python ingest_ollama.py`"
        )
        st.stop()

    # --- Main Content Area for Q&A ---
    st.subheader("Ask a Question")
    user_question = st.text_input(
        "Enter your question about the content of the PDFs:",
        key="user_question",
        help="Type your question here and press Enter."
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if user_question:
        with st.spinner("Finding the answer locally with Ollama..."):
            answer = get_answer(user_question)
            # Add user query and response to session state
            st.session_state.conversation.insert(0, (user_question, answer))

    st.markdown("---")

    # Display conversation history
    if st.session_state.conversation:
        st.subheader("Conversation History")
        for i, (question, response) in enumerate(st.session_state.conversation):
             with st.expander(f"Q: {question}", expanded=(i==0)):
                st.write(f"**Answer:** {response}")


if __name__ == "__main__":
    main()
