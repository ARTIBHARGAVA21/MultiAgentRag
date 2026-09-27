import os
import shutil

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from ingestions import process_document
from vectorstores.chromadb import search_chroma


router = APIRouter()


# Temporary PDF directory
UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload API
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF files are allowed"
        }
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )
    result = process_document(file_path)
    return {
        "filename": file.filename,
        "message": "PDF processed successfully",
        "result": result
    }


# Ask Request
class AskRequest(BaseModel):
    question: str
    top_k: int = 4

# Ask API
@router.post("/ask")
async def ask_question(request: AskRequest):
    documents = search_chroma(
        request.question,
        k=request.top_k
    )

    if not documents:

        return {
            "answer": "Not found in document",
            "sources": []
        }
    sources = []
    for doc in documents:
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    # For now return retrieved chunks.
    # LLM generation can be added next.
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )
    return {
        "question": request.question,
        "context": context,
        "sources": sources
    }