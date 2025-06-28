# utils_ollama.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index_ollama" # Path to the Ollama-specific index
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"

def get_conversational_chain():
    """
    Creates and configures a conversational QA chain using a local Ollama model.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not in the provided context, just say, "The answer is not available in the context."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatOllama(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_answer(user_question):
    """
    Handles user input, performs similarity search with Ollama embeddings, and gets the answer.
    """
    if not user_question:
        return "Please enter a question."

    if not os.path.exists(FAISS_INDEX_PATH):
        return f"FAISS index not found at '{FAISS_INDEX_PATH}'. Please run the ingest_ollama.py script first."

    try:
        # Load embeddings and vector store
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        # Perform similarity search
        docs = vector_store.similarity_search(user_question)

        # Get the answer from the conversational chain
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        error_message = str(e)
        print(f"An error occurred: {error_message}")
        if "connection refused" in error_message.lower():
            return "Connection to Ollama failed. Please make sure the Ollama application is running."
        return f"An error occurred: {e}"
