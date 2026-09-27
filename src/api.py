import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

from ingestions import process_document
from retrieval import retrieve_context


router = APIRouter()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class AskRequest(BaseModel):
    question: str
    document_id: str


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # Check file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    document_id = str(uuid.uuid4())

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{document_id}_{file.filename}"
    )

    # Save PDF
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save PDF: {str(e)}"
        )

    # Check file size
    file_size = os.path.getsize(file_path)

    if file_size == 0:
        os.remove(file_path)

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )

    # Process PDF
    try:
        result = process_document(
            file_path=file_path,
            document_id=document_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read/process PDF: {str(e)}"
        )

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "document_id": document_id,
        "file_size": file_size,
        "result": result
    }


@router.post("/ask")
async def ask_question(
    request: AskRequest
):

    result = retrieve_context(
        query=request.question,
        document_id=request.document_id,
        top_k=4
    )

    documents = result["documents"]

    if not documents:
        return {
            "question": request.question,
            "answer": "Information not found in the uploaded document.",
            "context": []
        }

    context = []

    for document in documents:
        context.append({
            "content": document.page_content,
            "metadata": document.metadata
        })

    return {
        "question": request.question,
        "transformed_query": result["transformed_query"],
        "context": context
    }
