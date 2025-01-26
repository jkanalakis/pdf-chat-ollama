import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Template for generating concise answers using retrieved context
PROMPT_TEMPLATE = """
You are an AI assistant specialized in answering questions based on retrieved text from PDF documents.
Your goal is to produce a concise answer using only the provided context. 
If the context does not contain enough information, respond with "I don't know."

Instruction Guidelines:
• Limit your response to three sentences maximum.
• Be precise and factual.
• Do not provide information not supported by the context.

Question: {question}
Context: {context}

Answer:
"""

# Directory where uploaded PDF files will be stored
PDFS_DIRECTORY = 'pdf_files/'

# Initialize embeddings, vector store, and LLM model
ollama_embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
vector_store = InMemoryVectorStore(ollama_embeddings)
ollama_model = OllamaLLM(model="deepseek-r1:14b")


def save_uploaded_pdf(uploaded_file):
    """
    Save the uploaded PDF file to the PDFs directory.

    Args:
        uploaded_file (UploadedFile): A file-like object representing the uploaded PDF.
    """
    with open(PDFS_DIRECTORY + uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())


def load_pdf_content(file_path):
    """
    Load the PDF content from the specified file path using PDFPlumberLoader.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of Document objects containing the PDF's text.
    """
    file_loader = PDFPlumberLoader(file_path)
    documents = file_loader.load()
    return documents


def split_documents_into_chunks(documents):
    """
    Split documents into smaller chunks to improve vector search and retrieval performance.

    Args:
        documents (list): A list of Document objects.

    Returns:
        list: A list of Document objects, each containing a chunk of text.
    """
    # The chunk size is set to 1000 tokens/characters, with an overlap of 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_document_chunks(document_chunks):
    """
    Index the chunked documents in the vector store for later retrieval.

    Args:
        document_chunks (list): A list of chunked Document objects.
    """
    vector_store.add_documents(document_chunks)


def search_documents(query):
    """
    Retrieve the most relevant documents from the vector store given a query.

    Args:
        query (str): The user's query.

    Returns:
        list: A list of the most relevant Document objects.
    """
    return vector_store.similarity_search(query)


def generate_answer(question, relevant_docs):
    """
    Generate an answer to the user's question using the provided relevant documents.

    Args:
        question (str): The question asked by the user.
        relevant_docs (list): A list of Document objects deemed relevant to the question.

    Returns:
        str: The generated answer text.
    """
    # Concatenate the content from all relevant documents
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create a prompt and chain it with the language model
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    question_answer_chain = prompt | ollama_model

    return question_answer_chain.invoke({"question": question, "context": context_text})


# Streamlit UI setup
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    # Save and process the uploaded PDF
    save_uploaded_pdf(uploaded_file)
    documents = load_pdf_content(PDFS_DIRECTORY + uploaded_file.name)
    chunked_documents = split_documents_into_chunks(documents)
    index_document_chunks(chunked_documents)

    # Chat interface for question answering
    user_question = st.chat_input()
    if user_question:
        st.chat_message("user").write(user_question)

        # Retrieve relevant documents and generate an answer
        found_documents = search_documents(user_question)
        answer = generate_answer(user_question, found_documents)

        st.chat_message("assistant").write(answer)