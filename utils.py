# utils.py
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index_store"

def get_conversational_chain():
    """
    Creates and configures a conversational QA chain.

    Returns:
        A configured question-answering chain.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not directly provided in the context but you can use the material to deduce an answer, you can do so, else just say, "The answer is not available in the context."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=api_key
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_answer(user_question):
    """
    Handles user input, performs similarity search, and gets the answer.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        str: The answer from the model, or an error message.
    """
    if not user_question:
        return "Please enter a question."

    if not os.path.exists(FAISS_INDEX_PATH):
        return "FAISS index not found. Please run the ingest.py script first."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

        # The allow_dangerous_deserialization is set to True as we trust the source of the index.
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]

    except Exception as e:
        return f"An error occurred: {e}"
