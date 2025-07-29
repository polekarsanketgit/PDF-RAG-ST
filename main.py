import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import torch
import os
import tempfile

@st.cache_resource
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

@st.cache_resource
def split_documents(_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(_documents)
    return chunks

@st.cache_resource
def create_embeddings_and_vectorstore(_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(_chunks, embeddings)
    return vectorstore

def get_conversational_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input():
    print("DEBUG: handle_user_input called.")
    user_question = st.session_state.question_input
    if user_question and st.session_state.conversation:
        print(f"User question: {user_question}")
        response = st.session_state.conversation({'question': user_question})
        print(f"LLM Response: {response}")
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response['answer'])
        st.session_state.question_input = ""  # Clear the input box

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon=":books:")
    st.header("RAG Chatbot :books:")

    # Initialize session state for conversation history and chain
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # Sidebar for API Key and PDF upload
    with st.sidebar:
        st.subheader("Your API Key")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

        st.subheader("Upload Your PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if openai_api_key and uploaded_file:
            # Check file size
            if uploaded_file.size > 300 * 1024:  # 300 KB limit
                st.warning("File size exceeds 300KB. Please upload a smaller PDF.")
                st.session_state.conversation = None
                st.session_state.pdf_processed = False
            else:
                st.success("API Key and PDF loaded!")
                # Load PDF and create vectorstore only once
                if st.session_state.conversation is None:
                    with st.spinner("Processing PDF..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            documents = load_pdf(tmp_file_path)
                            chunks = split_documents(documents)
                            vectorstore = create_embeddings_and_vectorstore(chunks)
                            st.session_state.conversation = get_conversational_chain(vectorstore, openai_api_key)
                            st.session_state.pdf_processed = True
                            os.remove(tmp_file_path)  # Clean up the temporary file
                            print("DEBUG: PDF processed and chatbot chain initialized.")
                            st.success("PDF processed and chatbot ready!")
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
                            st.session_state.conversation = None
                            st.session_state.pdf_processed = False
        elif not openai_api_key:
            st.warning("Please enter your OpenAI API Key to start the chatbot.")
        elif not uploaded_file:
            st.warning("Please upload a PDF file to start the chatbot.")

    # Chat interface - enable if PDF is processed and conversation chain exists
    if st.session_state.conversation and st.session_state.pdf_processed:
        # Display chat history first
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**You:** {message}")
            else:
                st.write(f"**Bot:** {message}")
        
        # Then show the input box
        st.text_input("Ask a question about your document:", key="question_input", on_change=handle_user_input)
        
    elif not openai_api_key or not uploaded_file:
        st.info("Please enter your API key and upload a PDF to begin.")

if __name__ == '__main__':
    main()