from langchain import LLMChain, PromptTemplate
from langchain.llms import Llama2
from langchain.vectorstores import Chroma
from langchain.embeddings import OolamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the Llama2 model
llm = Llama2(model_name="llama2")

# Initialize the Oolama embeddings
embeddings = OolamaEmbeddings()

# Initialize the ChromaDB vector store

#folder path
pdf_folder = r"C:\Users\JHAMMOND\Desktop\Work folder\temp files\SSNIT DOCUMENTS" 

# Load PDF documents from a folder
loader = PyMuPDFLoader(pdf_folder)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create the Chroma vector store
vector_store = Chroma.from_documents(docs, embedding_function=embeddings)
vector_store = Chroma(embedding_function=embeddings)

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the following context: {context}\nAnswer the question: {question}"
)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt_template=prompt_template)

# Create the RetrievalQA chain
qa_chain = RetrievalQA(llm_chain=llm_chain, retriever=vector_store.as_retriever())

# Example usage
context = "Your context here"
question = "Your question here"
answer = qa_chain.run({"context": context, "question": question})

print(answer)
