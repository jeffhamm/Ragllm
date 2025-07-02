# ingest.py
import os
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")

# --- Configuration ---
PDF_DIRECTORY = "pdfs"
FAISS_INDEX_PATH = "faiss_index_store"

def get_pdf_elements_with_unstructured(pdf_directory):
    """
    Extracts elements (text, tables) from all PDF files in a given directory
    using the 'unstructured' library. Includes a fallback to an OCR-only
    strategy for problematic files.
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
                # First, try the high-resolution strategy
                print("--> Trying 'hi_res' strategy...")
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_images_in_pdf=False
                )
            except Exception as e:
                # If hi_res fails, print a warning and fall back to OCR-only
                print(f"--> WARNING: 'hi_res' strategy failed for {filename} with error: {e}")
                print("--> Retrying with 'ocr_only' strategy...")
                try:
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy="ocr_only",
                        infer_table_structure=True,
                        extract_images_in_pdf=False
                    )
                except Exception as e_ocr:
                    # If ocr_only also fails, print an error and skip the file
                    print(f"--> ERROR: 'ocr_only' strategy also failed for {filename}. Skipping file. Error: {e_ocr}")
                    continue # Move to the next file

            # Process the extracted elements
            for element in elements:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    full_text += "\n\n--- TABLE START ---\n"
                    full_text += element.metadata.text_as_html
                    full_text += "\n--- TABLE END ---\n\n"
                else:
                    full_text += element.text + "\n"
            print(f"Successfully processed {filename}")

    return full_text

def get_text_chunks(text):
    """
    Splits a long string of text into smaller chunks.

    Args:
        text (str): The input text.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_and_save_vector_store(text_chunks, index_path):
    """
    Creates and saves a FAISS vector store from text chunks.

    Args:
        text_chunks (list): A list of text chunks.
        index_path (str): The path to save the FAISS index.
    """
    if not text_chunks:
        print("No text chunks to process. Vector store not created.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(index_path)
        print(f"Vector store created and saved successfully at '{index_path}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def main():
    """
    Main function to process PDFs and create the vector store.
    """
    print("Starting robust PDF processing...")
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
