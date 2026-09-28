# PDF RAG API

A FastAPI-based **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions based only on the uploaded document.

The system uses **Mistral embeddings, ChromaDB, hybrid retrieval, BM25 keyword search, CrossEncoder reranking, and Mistral LLM generation with streaming responses**.

---

## Features

- Upload PDF documents
- Extract text from PDFs
- Multiple document chunking strategies
  - Recursive Character Chunking
  - Character Chunking
  - Token Chunking
  - Semantic Chunking
- Generate embeddings using Mistral
- Store document chunks in ChromaDB
- Document-specific retrieval using `document_id`
- Hybrid retrieval
  - Vector similarity search
  - BM25 keyword search
- CrossEncoder reranking
- Prompt assembly using retrieved context
- LLM-based answer generation
- Answers restricted to uploaded document context
- Page-based citations
- Guardrails to reduce hallucinations
- Streaming responses
- FastAPI Swagger documentation

---

## Architecture

```text
                    ┌─────────────────────┐
                    │      PDF Upload     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PDF Extraction   │
                    │    PyPDFLoader      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Chunking Layer    │
                    │                     │
                    │ Recursive           │
                    │ Character           │
                    │ Token               │
                    │ Semantic            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Mistral Embeddings  │
                    │    mistral-embed    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ChromaDB       │
                    │ Vector Storage      │
                    └─────────────────────┘


Question
   │
   ▼
┌─────────────────────┐
│   Query Transform   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Hybrid Retrieval  │
│                     │
│ Vector Search       │
│ BM25 Keyword Search │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CrossEncoder      │
│     Reranking       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Prompt Assembly    │
│                     │
│ Context + Question  │
│ + Page Information  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Mistral LLM       │
│     Generation      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Citations &         │
│ Guardrails          │
└──────────┬──────────┘
           │
           ▼
     Streaming Response
```

---

## Project Structure

```text
Rag_document_Project/
│
├── src/
│   ├── main.py
│   ├── api.py
│   ├── ingestions.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── prompts.py
│   │
│   └── vectorstores/
│       └── chromadb.py
│
├── chroma_db/
│
├── temp/
│
├── .env
├── requirements.txt
└── README.md
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| FastAPI | REST API |
| PyPDFLoader | PDF text extraction |
| LangChain | RAG pipeline |
| Mistral AI | Embeddings and LLM |
| ChromaDB | Vector database |
| BM25 | Keyword-based retrieval |
| Sentence Transformers | CrossEncoder reranking |
| Pydantic | Request validation |
| Uvicorn | ASGI server |

---

## Installation

### 1. Clone the project

```bash
git clone <your-repository-url>
cd Rag_document_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file inside the `src` folder:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

The application uses this key for:

- Mistral embeddings
- Mistral LLM generation

---

## Run the Application

From the `src` directory:

```bash
python main.py
```

The API will start at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# API Endpoints

## 1. Upload PDF

### Endpoint

```http
POST /upload
```

### Request

Use `multipart/form-data`.

Form-data:

```text
file: your_document.pdf
```

### Example Response

```json
{
    "message": "PDF uploaded successfully",
    "filename": "example.pdf",
    "document_id": "a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "file_size": 245678,
    "result": {
        "document_id": "a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        "pages": 5,
        "recursive_chunks": 20,
        "character_chunks": 18,
        "token_chunks": 22,
        "semantic_chunks": 12,
        "chroma": {
            "message": "Documents stored in ChromaDB",
            "chunks": 12
        }
    }
}
```

Save the returned `document_id`.

It is required when asking questions about that document.

---

# 2. Ask Question

### Endpoint

```http
POST /ask
```

### Request Body

```json
{
    "question": "What is this document about?",
    "document_id": "YOUR_DOCUMENT_ID"
}
```

The API always returns the generated answer as a **streaming response**.

There is no `stream: false` parameter because streaming is the default behavior of the `/ask` endpoint.

---

## RAG Retrieval Flow

When a question is received, the following steps are performed:

### 1. Query Transformation

The user question is cleaned and prepared for retrieval.

### 2. Vector Search

The question is converted into an embedding using Mistral embeddings.

ChromaDB searches for semantically similar document chunks.

The search is filtered using:

```text
document_id
```

This ensures that the question retrieves information only from the selected uploaded document.

### 3. Keyword Search

BM25 is used to identify chunks containing relevant keywords.

### 4. Hybrid Search

Vector search and keyword search results are combined.

This helps handle both:

- Semantic questions
- Exact keywords or terms

### 5. CrossEncoder Reranking

The retrieved candidates are passed to a CrossEncoder:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The CrossEncoder scores the relevance between:

```text
Question ↔ Document Chunk
```

The highest-relevance chunks are selected for generation.

---

# Prompt Assembly

The selected document chunks are converted into context for the LLM.

Each chunk includes its page information:

```text
[Page 1]
Document content...

[Page 2]
Document content...
```

The prompt also contains instructions that the model should answer only from the supplied document context.

---

# LLM Generation

The application uses:

```text
Mistral Small
```

for answer generation.

The temperature is configured as:

```text
0
```

This makes the generation more deterministic.

The LLM receives:

```text
Document Context
+
User Question
+
RAG Instructions
```

---

# Citations

The prompt instructs the LLM to include page citations.

Example:

```text
The document explains the importance of data security.

[Page 2]
```

If information comes from multiple pages:

```text
The system uses authentication and authorization mechanisms.

[Page 2]
[Page 4]
```

---

# Guardrails

The RAG prompt contains rules to reduce hallucinations.

The model is instructed to:

- Use only the retrieved document context
- Not use outside knowledge
- Not make assumptions
- Not invent information
- Return a fallback message when the answer is not available

Fallback response:

```text
Information not found in the uploaded document.
```

---

# Streaming

The `/ask` endpoint uses FastAPI `StreamingResponse`.

Instead of waiting for the complete LLM response, generated content can be sent progressively to the client.

Conceptually:

```text
Question
   ↓
Retrieve Context
   ↓
Rerank Documents
   ↓
Build Prompt
   ↓
LLM
   ↓
Token/Chunk
   ↓
Client
   ↓
Token/Chunk
   ↓
Client
```

This improves the perceived response time for longer answers.

---

# Error Handling

The application handles common errors such as:

### Invalid file type

```text
Only PDF files are allowed
```

### Empty PDF

```text
Uploaded PDF is empty
```

### PDF processing failure

```text
Could not read/process PDF
```

### LLM service failure

The API returns a user-friendly error instead of exposing the complete internal exception.

### Mistral rate limit

If the Mistral API returns HTTP `429`, the generation layer retries with exponential backoff.

After the retry limit is reached, the API returns a temporary rate-limit message.

---

# Testing with Postman

## Upload PDF

Select:

```text
POST http://localhost:8000/upload
```

Go to:

```text
Body → form-data
```

Add:

```text
Key: file
Type: File
Value: your PDF
```

Send the request.

Copy the returned:

```text
document_id
```

---

## Ask Question

Select:

```text
POST http://localhost:8000/ask
```

Go to:

```text
Body → raw → JSON
```

Use:

```json
{
    "question": "What is this document about?",
    "document_id": "YOUR_DOCUMENT_ID"
}
```

Send the request.

The response is generated using the retrieved document context and streamed to the client.

---

# Example Questions

After uploading a PDF, you can ask:

```text
What is this document about?
```

```text
What are the main points discussed in the document?
```

```text
What methodology is used?
```

```text
What are the key findings?
```

```text
What does the document say about security?
```

If the requested information is not present in the document:

```text
Information not found in the uploaded document.
```

---

# Data Flow

### Document Ingestion

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Extraction
 ↓
Chunking
 ↓
Mistral Embeddings
 ↓
ChromaDB
```

### Question Answering

```text
User Question
 ↓
Query Transformation
 ↓
Vector Search
 ↓
BM25 Search
 ↓
Hybrid Results
 ↓
CrossEncoder Reranking
 ↓
Top Documents
 ↓
Prompt Assembly
 ↓
Mistral LLM
 ↓
Citations & Guardrails
 ↓
Streaming Response
```

---

# Important Notes

- Each uploaded document receives a unique `document_id`.
- The `document_id` is used to isolate retrieval to the selected document.
- Semantic chunking is currently stored in ChromaDB.
- The other chunking strategies are currently generated for comparison/measurement.
- ChromaDB data is persisted in the `chroma_db` directory.
- The CrossEncoder model may be downloaded the first time it is used.
- A valid Mistral API key is required for embeddings and LLM generation.
- API rate limits can affect both embedding and generation requests.

---

# Future Improvements

Possible future enhancements:

- Conversation/session memory
- Multiple-document RAG
- Metadata-based filtering
- Authentication and authorization
- Redis caching
- PostgreSQL for document/user metadata
- Better hybrid retrieval using score fusion
- Async/background document processing
- SSE/WebSocket