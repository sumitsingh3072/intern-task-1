import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logging_utlity import setup_logging

logger = setup_logging()

def load_documents(data_dir: str) -> list:
    try:
        txt_loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True
        )
        logger.info(f"Loading .txt documents from {data_dir}...")
        txt_docs = txt_loader.load()
        logger.info(f"Found {len(txt_docs)} .txt documents.")
        md_loader = DirectoryLoader(
            data_dir,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True
        )
        logger.info(f"Loading .md documents from {data_dir}...")
        md_docs = md_loader.load()
        logger.info(f"Found {len(md_docs)} .md documents.")
        pdf_loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader, #type: ignore
            show_progress=True,
            use_multithreading=True
        )
        logger.info(f"Loading .pdf documents from {data_dir}...")
        pdf_docs = pdf_loader.load()
        logger.info(f"Found {len(pdf_docs)} .pdf documents.")
        documents = txt_docs + md_docs + pdf_docs
        
        if not documents:
            logger.warning(f"No documents (.txt, .md, or .pdf) found in {data_dir}.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} total documents into {len(chunks)} chunks.")
        
        return chunks

    except Exception as e:
        logger.error(f"Error loading documents: {e}", exc_info=True)
        return []