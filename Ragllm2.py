import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Constants
OLLAMA_API_ENDPOINT = "http://localhost:11434"
PDF_FOLDER = r"C:\Users\JHAMMOND\Desktop\Work folder\temp files\SSNIT DOCUMENTS"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "llama2"

# Step 1: Load and process PDFs
def load_and_process_pdfs(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    return texts

# Step 2: Create Vector Database with Chroma
def create_vector_db(texts, model_name, api_endpoint, persist_directory):
    embeddings = OllamaEmbeddings(model=model_name, base_url=api_endpoint)
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    return vector_db

# Step 3: Set Up the RAG Pipeline
def setup_rag_pipeline(vector_db, model_name, api_endpoint):
    llm = Ollama(model=model_name, base_url=api_endpoint)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
    return qa_chain

# Step 4: Query the RAG Pipeline
def query_rag_pipeline(qa_chain, query):
    return qa_chain.run(query)

if __name__ == "__main__":
    texts = load_and_process_pdfs(PDF_FOLDER)
    print(f"Loaded {len(texts)} text chunks from PDFs.")

    vector_db = create_vector_db(texts, MODEL_NAME, OLLAMA_API_ENDPOINT, PERSIST_DIRECTORY)
    print("Vector database created and persisted to './chroma_db'!")

    qa_chain = setup_rag_pipeline(vector_db, MODEL_NAME, OLLAMA_API_ENDPOINT)
    print("RAG pipeline is ready!")
