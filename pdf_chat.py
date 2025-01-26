import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROMPT_TEMPLATE = """
"""

PDFS_DIRECTORY = 'chat-with-pdf/pdf_files/'

ollama_embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
vector_store = InMemoryVectorStore(ollama_embeddings)
ollama_model = OllamaLLM(model="deepseek-r1:14b")


def save_uploaded_pdf(uploaded_file):
    with open(PDFS_DIRECTORY + uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())


def load_pdf_content(file_path):
    file_loader = PDFPlumberLoader(file_path)
    documents = file_loader.load()
    return documents


def split_documents_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_document_chunks(document_chunks):
    vector_store.add_documents(document_chunks)


def search_documents(query):
    return vector_store.similarity_search(query)


def generate_answer(question, relevant_docs):
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    question_answer_chain = prompt | ollama_model

    return question_answer_chain.invoke({"question": question, "context": context_text})


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