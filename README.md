# PDF RAG Project

A Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LangChain**, **Mistral AI**, and **ChromaDB**.

The application allows users to upload PDF documents, process and chunk the content, generate embeddings using Mistral AI, store the vectors in ChromaDB, and retrieve relevant information for question answering.

---

## Features

- Upload PDF documents using FastAPI
- Extract text from PDFs using `PyPDFLoader`
- Multiple chunking strategies:
  - Recursive Character Chunking
  - Character Chunking
  - Token Chunking
  - Semantic Chunking
- Generate embeddings using Mistral AI
- Store document embeddings in ChromaDB
- Perform similarity search
- REST APIs using FastAPI
- Environment variable support using `.env`
- Docker/cloud deployment can be added later

---

## Project Structure

```text
Rag_document_Project/
│
├── venv/
│
├── src/
│   ├── .env
│   ├── main.py
│   ├── api.py
│   ├── ingestions.py
│   │
│   ├── test.py
│   │
│   └── vectorstores/
│       ├── chromadb.py
│       └── pinecone.py
│
├── chroma_db/
│
└── README.md