# LangChain RAG Prototype (Pythrust Assignment)

A modular Retrieval-Augmented Generation (RAG) prototype built with LangChain and Streamlit. The project demonstrates a local RAG workflow using Hugging Face embeddings and an in-memory FAISS vector store. The pipeline can use a real LLM via Groq (example: llama3-8b-8192) to generate answers based on retrieved document context.

## Features

- Modular code: separate modules for the UI, data loading, vector store, and RAG pipeline.
- Supports text, markdown and PDF inputs (.txt, .md, .pdf).
- Uses Hugging Face embeddings (e.g. `all-MiniLM-L6-v2`) to embed document chunks.
- In-memory FAISS vector store for fast retrieval.
- History-aware retriever for follow-up conversational queries.
- Optionally uses Groq (llama3) as the LLM for generation.
- Streamlit chat UI for quick prototyping and testing.

## Project structure

```
.
├── data/                   # Put your .txt, .md, .pdf documents here
├── app.py                  # Main Streamlit application
├── doc_loader.py          # Document loading & splitting
├── rag_pipeline.py         # LangChain chain definitions
├── vector_store.py         # FAISS vector store creation
├── logging_utlity.py                # Logging utilities
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Pre-requisites

- Python 3.10+ recommended
- PowerShell (Windows) or CMD
- Internet connection for downloading embedding models on first run

## Getting a Groq API key

1. Visit https://console.groq.com/ and sign up (or sign in).
2. Create a new API key and copy it. You'll paste it into the app sidebar at runtime.

Note: If you prefer not to use Groq or don't have an API key, you can replace the LLM in `rag_pipeline.py` with a mock LLM (e.g., LangChain's `FakeListLLM`) or another provider.

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1    
```

If you're using CMD:

```cmd
venv\Scripts\activate.bat
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Prepare the `data/` directory

Place your `.txt`, `.md`, or `.pdf` files in the `data/` directory.

## Running the app

Start Streamlit from your activated virtual environment:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501) in your browser.

When the web UI opens, paste your Groq API key into the sidebar input (if required by your `rag_pipeline.py` implementation). The app will initialize the vector store and (on first run) download embedding model weights. After initialization you can ask questions about your documents.

## Notes & recommendations

- The first run may take time to download the embedding model; subsequent runs will be faster.
- FAISS is currently used in-memory. If you need persistence across runs, update `vector_store.py` to persist the index to disk.
- To swap the LLM (e.g., use OpenAI, local Llama, or a mock LLM) update `rag_pipeline.py` where the LLM is instantiated.

## Troubleshooting

- Missing packages: ensure the venv is activated and re-run `pip install -r requirements.txt`.
- Embedding model download fails: check network connectivity and retry.
- Streamlit opens a different port: check the terminal output for the served URL.

## License

This repository is provided as a small prototype for an intern assignment. Adapt and reuse as needed.
