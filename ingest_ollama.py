# ingest_ollama.py
import os
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# --- Configuration ---
PDF_DIRECTORY = "pdfs"
FAISS_INDEX_PATH = "faiss_index_ollama" # Use a separate index for Ollama
EMBEDDING_MODEL = "nomic-embed-text" # Model for creating embeddings

def get_pdf_elements_with_unstructured(pdf_directory):
    """
    Extracts elements (text, tables) from all PDF files in a given directory
    using the 'unstructured' library.
    """
    full_text = ""
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' not found.")
        return full_text

    print("Using 'unstructured' to process PDFs...")
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing {pdf_path}...")
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_images_in_pdf=False
                )
                for element in elements:
                    if "unstructured.documents.elements.Table" in str(type(element)):
                        full_text += "\n\n--- TABLE START ---\n"
                        full_text += element.metadata.text_as_html
                        full_text += "\n--- TABLE END ---\n\n"
                    else:
                        full_text += element.text + "\n"
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename} with unstructured: {e}")
    return full_text

def get_text_chunks(text):
    """Splits a long string of text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_and_save_vector_store(text_chunks, index_path):
    """Creates and saves a FAISS vector store from text chunks using Ollama embeddings."""
    if not text_chunks:
        print("No text chunks to process. Vector store not created.")
        return

    try:
        print(f"Initializing Ollama embeddings with model: {EMBEDDING_MODEL}")
        # Make sure Ollama application is running
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        print("Creating vector store... This may take some time depending on document size.")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(index_path)
        print(f"Vector store created and saved successfully at '{index_path}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Please ensure the Ollama application is running and the model '{EMBEDDING_MODEL}' is downloaded ('ollama pull {EMBEDDING_MODEL}').")

def main():
    """Main function to process PDFs and create the vector store."""
    print("Starting local PDF processing for Ollama...")
    raw_text = get_pdf_elements_with_unstructured(PDF_DIRECTORY)

    if not raw_text:
        print("No text extracted from PDFs. Exiting.")
        return

    print("Splitting text into chunks...")
    text_chunks = get_text_chunks(raw_text)

    if text_chunks:
        print(f"Created {len(text_chunks)} text chunks.")
        create_and_save_vector_store(text_chunks, FAISS_INDEX_PATH)
    else:
        print("Could not create text chunks.")

if __name__ == "__main__":
    main()
