# LangChain RAG Prototype (Pythrust Assignment)

A small, modular Retrieval-Augmented Generation (RAG) prototype built with LangChain and Streamlit. This project demonstrates a local RAG workflow using Hugging Face embeddings and a FAISS in-memory vector store. For LLM behavior the project uses LangChain's `FakeListLLM` as a mock LLM so the app runs without a paid API.

## Features

- Modular code split into focused modules (UI, data loading, vector store, RAG pipeline).
- Uses local/open-source embeddings (Hugging Face `all-MiniLM-L6-v2`).
- In-memory FAISS vector store for retrieval.
- Conversational, history-aware retrieval for follow-up questions.
- Streamlit-based chat UI for quick interaction and testing.

## Project structure

```
.
├── data/
│   ├── 01_python_basics.txt
│   ├── 02_python_functions.txt
│   └── 03_python_loops.txt
├── app.py                  # Main Streamlit application
├── data_loader.py          # Document loading & splitting
├── rag_pipeline.py         # LangChain chain definitions
├── vector_store.py         # FAISS vector store creation
├── utils.py                # Logging utilities
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1    # PowerShell
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

Place the provided `.txt` files into the `data/` directory. The repo includes example files about Python so the app can run out-of-the-box.

## Run the app

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501) in your browser.

On first run the embedding model may be downloaded, which can take a short while.

## Notes & assumptions

- This prototype uses a mock LLM (`FakeListLLM`) so it can run without external LLM API keys. Replace the LLM in `rag_pipeline.py` to connect to a real model.
- Embeddings are created with Hugging Face models; the first run will download model weights.
- FAISS is used in-memory for simplicity. For persistence, adapt `vector_store.py` to persist to disk.

## Troubleshooting

- If you see errors about missing packages, re-run `pip install -r requirements.txt` in the activated venv.
- If Streamlit serves on a different port, check the terminal output for the exact URL.

## License

This project is provided as a simple assignment prototype. Modify and adapt as needed.

---

If you'd like, I can also:

- add a short usage example in `app.py`, or
- add a `requirements.txt` pin list if you'd like reproducible installs.
