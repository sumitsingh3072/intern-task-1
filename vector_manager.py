from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from logging_utlity import setup_logging

logger = setup_logging()

def create_vector_store(chunks: list, embeddings: Embeddings) -> VectorStoreRetriever:
    if not chunks:
        logger.error("No chunks provided to create vector store.")
        raise ValueError("Cannot create vector store from empty document chunks.")       
    try:
        logger.info("Creating FAISS vector store from document chunks...")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        logger.info("FAISS vector store created successfully.")
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        logger.error(f"Failed to create vector store: {e}", exc_info=True)
        raise