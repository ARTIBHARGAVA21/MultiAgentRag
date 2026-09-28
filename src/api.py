import os
import shutil
import uuid
from fastapi import (APIRouter,UploadFile,File,HTTPException)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ingestions import process_document
from retrieval import retrieve_context
from generation import (generate_answer,stream_answer)


router = APIRouter()



# UPLOAD DIRECTORY
UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR,exist_ok=True)



# REQUEST MODEL
class AskRequest(BaseModel):
    question: str
    document_id: str
    stream: bool = False



# UPLOAD PDF
@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    # CHECK FILE
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing"
        )
    # CHECK EXTENSION
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    # CREATE DOCUMENT ID
    document_id = str(
        uuid.uuid4()
    )
    file_path = os.path.join(
        UPLOAD_DIR,
        f"{document_id}_{file.filename}"
    )
    # SAVE PDF
    try:
        with open(file_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save PDF: {str(e)}"
        )

    # CHECK FILE SIZE
    try:
        file_size = os.path.getsize(
            file_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not check file size: {str(e)}"
        )
    if file_size == 0:

        os.remove(
            file_path
        )
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )
    # PROCESS PDF
    try:

        result = process_document(
            file_path=file_path,
            document_id=document_id
        )

    except Exception as e:
        # Delete failed upload
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=f"Could not read/process PDF: {str(e)}"
        )
    # RESPONSE
    return {
        "message":"PDF uploaded successfully",
        "filename":file.filename,
        "document_id":document_id,
        "file_size":file_size,
        "result":result
    }


# ASK QUESTION
@router.post("/ask")
async def ask_question(
    request: AskRequest
):
    # VALIDATE QUESTION
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    # RETRIEVAL
    try:

        result = retrieve_context(
            query=request.question,
            document_id=request.document_id,
            top_k=4
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )
    documents = result["documents"]
    # NO DOCUMENT FOUND
    if not documents:
        return {

            "question":
                request.question,

            "transformed_query":
                result["transformed_query"],

            "answer":
                "Information not found in the uploaded document.",

            "context":
                []
        }
    # STREAMING RESPONSE
    if request.stream:

        return StreamingResponse(

            stream_answer(
                question=request.question,
                documents=documents
            ),

            media_type="text/plain",

            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    # LLM GENERATION
    answer = await generate_answer(
        question=request.question,
        documents=documents
    )
    # BUILD CONTEXT RESPONSE
    context = []
    for document in documents:

        page_number = document.metadata.get(
            "page",
            "unknown"
        )

        if isinstance(page_number, int):

            page_number += 1
        context.append({

            "content":document.page_content,
            "page":page_number,
            "metadata":document.metadata
        })

    # FINAL RESPONSE
    return {
        "question":request.question,
        "transformed_query": result["transformed_query"],
        "answer":answer,
        "context":context
    }
