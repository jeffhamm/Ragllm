# chatbot_core.py: This module defines the core logic of the PDFchatbot.
# It builds a RAG pipeline using LangChain.

from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader,PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from rebuild_db import delete_old_faiss_index
from rebuild_db import rebuild_faiss_database
from rebuild_db import process_documents


# --- Configuration ---
DB_PATH = "faiss_index_store" # Directory where your FAISS index will be saved
DOCUMENTS_DIR = "pdfs" # Directory containing your PDFs
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Or your chosen model (e.g., from HuggingFace, local)


def build_qa_chain():

    loaded_db = FAISS.load_local(
        DB_PATH,
        HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME),
        allow_dangerous_deserialization=True # Required for loading FAISS indexes from disk
    )

        
    # Creates a retriever from the FAISS database to find relevant chunks based on a question
    retriever = loaded_db.as_retriever() # Create a retriever to find relevant chunks based on a question

    llm = ChatOllama(model="gemma3:4b") # Combines the retriever with mistral
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain # The function 'qa_chain()' returns a ready-to-use question-answering chain