import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings # Or your preferred embedding model
from langchain_community.vectorstores import FAISS # Assuming you're using LangChain's FAISS integration
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
DB_PATH = "faiss_index_store" # Directory where your FAISS index will be saved
DOCUMENTS_DIR = "pdfs" # Directory containing your PDFs
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Or your chosen model (e.g., from HuggingFace, local)


# --- Step 1: Delete the old FAISS index (if it exists) ---
def delete_old_faiss_index(db_path):
    if os.path.exists(db_path):
        print(f"Deleting old FAISS index directory: {db_path}")
        # FAISS.save_local saves it as a directory with index.faiss and index.pkl
        for filename in os.listdir(db_path):
            file_path = os.path.join(db_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        os.rmdir(db_path) # Remove the directory itself
    else:
        print(f"No old FAISS index found at {db_path}.")

# --- Step 2 & 3: Load, Chunk, and Embed All Current Documents ---
def process_documents(documents_directory, embedding_model_name):
    # print(f"Loading documents from {documents_directory}...")

    
    loader = DirectoryLoader(
        documents_directory, 
        glob="**/*.pdf", 
        loader_cls=UnstructuredPDFLoader,
        loader_kwargs={"mode": "elements", "strategy": "hi_res" # "hi_res" is good for tables/charts
    }
        )
    documents = loader.load()
    # print(f"Loaded {len(documents)} raw documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        # is_separator_regex=False,
        add_start_index=True    # Add metadata for start index of chun
    )
    chunks = text_splitter.split_documents(documents)
    # print(f"Split into {len(chunks)} chunks.")

    embeddings_model = HuggingFaceEmbeddings(model=embedding_model_name) # Or your specific embedding model
    print("Generating embeddings for chunks...")
    # LangChain's FAISS.from_documents handles embedding for you
    return chunks, embeddings_model

# --- Main Rebuild Function ---
def rebuild_faiss_database(documents_directory, vector_db_path, embedding_model_name):
    # 1. Delete the old index
    delete_old_faiss_index(vector_db_path)

    # 2. & 3. Load, Chunk, and Embed all documents
    chunks, embeddings_model = process_documents(documents_directory, embedding_model_name)

    # 4. Create a brand new FAISS index
    #vprint("Creating new FAISS vector store...")
    # This automatically trains and adds if necessary for the chosen index type
    new_db = FAISS.from_documents(chunks, embeddings_model)
    # print(f"New FAISS index created with {new_db.index.ntotal} vectors.")

    # 5. Save the new index
    print(f"Saving new FAiss index to {vector_db_path}...")
    new_db.save_local(vector_db_path)
    print("FAISS index rebuilt successfully!\n")
    return new_db

# --- How to use it ---
if __name__ == "__main__":
    # Ensure your documents directory exists and contains your PDFs
    # For example, create a 'documents' folder and put your new_document.pdf and old_document.pdf inside
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Created directory: {DOCUMENTS_DIR}. Please place your PDF documents inside.")
        # You might want to exit or add a placeholder PDF for testing
        exit()

    # Rebuild the database
    faiss_db = rebuild_faiss_database(DOCUMENTS_DIR, DB_PATH, EMBEDDING_MODEL_NAME)

    # Now you can load and use the new database in your RAG application
    # print(f"Loading the newly built FAISS index from {DB_PATH} for RAG operations...")
    # # Make sure to load with the same embedding model
    # loaded_db = FAISS.load_local(
    #     DB_PATH,
    #     HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME),
    #     allow_dangerous_deserialization=True # Required for loading FAISS indexes from disk
    # )

    