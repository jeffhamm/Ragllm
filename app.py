# app.py
import streamlit as st
from utils import get_answer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="Chat with Pre-processed PDFs", layout="wide")
    st.header("RAG Application: Chat with your PDFs 💬")

    # Check for API key and FAISS index
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key not found. Please create a .env file and add your GOOGLE_API_KEY.")
        st.stop()

    if not os.path.exists("faiss_index_store"):
        st.error("FAISS index not found. Please run the `ingest.py` script first to process your PDF documents.")
        st.info("Instructions: \n1. Create a folder named 'data'. \n2. Place your PDF files inside the 'data' folder. \n3. Run the command: `python ingest.py`")
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
        with st.spinner("Finding the answer..."):
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
