import streamlit as st
import time
import os
from logging_utlity import setup_logging
from doc_loader import load_documents
from vector_manager import create_vector_store
from rag_pipeline import (
    create_rag_chain, 
    get_history_aware_retriever, 
    get_conversational_rag_chain
)
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_groq import ChatGroq
DATA_DIR = "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192" # Groq model

logger = setup_logging()

st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", 
    type="password",
    help="Get your key from https://console.groq.com/keys"
)
if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to start.")
    st.stop()
@st.cache_resource(show_spinner="Initializing RAG pipeline...")
def initialize_pipeline(_groq_api_key):
    try:
        logger.info("Loading documents...")
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            st.error(f"Data directory '{DATA_DIR}' is empty or missing.")
            st.stop()
        
        docs = load_documents(DATA_DIR)
        if not docs:
            st.error("Failed to load any documents. Check logs.")
            st.stop()
        logger.info(f"Loaded {len(docs)} document chunks.")
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Creating vector store...")
        retriever = create_vector_store(docs, embeddings)
        logger.info("Vector store created successfully.")
        logger.info(f"Setting up Groq LLM with model: {LLM_MODEL}")
        llm = ChatGroq(
            api_key=_groq_api_key, 
            model=LLM_MODEL
        )
        logger.info("Creating RAG chain...")
        history_aware_retriever = get_history_aware_retriever(llm, retriever)
        question_answer_chain = create_rag_chain(llm)
        rag_chain = get_conversational_rag_chain(
            history_aware_retriever, question_answer_chain
        )
        
        logger.info("Application initialized successfully.")
        
        return rag_chain

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        st.error(f"An error occurred during setup: {e}")
        st.stop()
st.set_page_config(page_title="LangChain RAG Prototype", page_icon="🤖")
st.title("🤖 LangChain RAG Prototype")
st.caption("A demo for the Pythrust Generative AI Engineer Intern assignment.")
if "messages" not in st.session_state:
    st.session_state.messages = []
rag_chain = initialize_pipeline(groq_api_key)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                chat_history = [
                    (msg["role"], msg["content"]) 
                    for msg in st.session_state.messages[:-1]
                ]
                
                logger.info(f"Invoking RAG chain for: {prompt}")
                response = rag_chain.invoke({
                    "chat_history": chat_history,
                    "input": prompt
                })
                
                full_response = response.get("answer", "Sorry, I couldn't generate a response.")
                message_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                logger.info("Response generated.")
            
            except Exception as e:
                logger.error(f"Error during response generation: {e}", exc_info=True)
                st.error("An error occurred while generating the response. Please check your API key and terminal logs.")