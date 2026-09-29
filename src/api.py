
import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from ingestions import process_document
from loaders import (
    supported_extensions,
    UnsupportedFileTypeError
)
from retrieval import retrieve_context
from generation import stream_answer


router = APIRouter()


UPLOAD_DIR = "temp"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


class AskRequest(BaseModel):

    question: str

    document_id: str


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    extension = (
        file.filename.rsplit(".", 1)[-1].lower()
        if file.filename and "." in file.filename
        else ""
    )

    allowed = supported_extensions() + ["xlsx"]

    if extension not in allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: .{extension or 'unknown'}. "
                f"Allowed types are: {', '.join(sorted(allowed))}"
            )
        )

    document_id = str(
        uuid.uuid4()
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{document_id}_{file.filename}"
    )

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save document: {str(e)}"
        )

    file_size = os.path.getsize(
        file_path
    )

    if file_size == 0:

        os.remove(
            file_path
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded document is empty"
        )

    try:

        result = process_document(
            file_path=file_path,
            document_id=document_id
        )

    except UnsupportedFileTypeError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read/process document: {str(e)}"
            )
        )

    return {
        "message": "Document uploaded successfully",
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

        async def not_found():

            yield (
                "Information not found in "
                "the uploaded document."
            )

        return StreamingResponse(
            not_found(),
            media_type="text/plain"
        )

    return StreamingResponse(
        stream_answer(
            question=request.question,
            documents=documents
        ),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )