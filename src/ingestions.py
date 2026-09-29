
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

from langchain_experimental.text_splitter import SemanticChunker

from loaders import load_document
from vectorstores.chromadb import (
    embedding_model,
    store_in_chroma,
)


def process_document(file_path: str, document_id: str):
    # 1. LOAD DOCUMENT (PDF, DOCX, TXT, MD, CSV, JSON, HTML, XLSX)
    documents = load_document(file_path)
    # 2. ADD DOCUMENT ID
    for document in documents:
        document.metadata["document_id"] = document_id
    # 3. RECURSIVE CHUNKING
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    recursive_chunks = recursive_splitter.split_documents(
        documents
    )
    for chunk in recursive_chunks:
        chunk.metadata["chunking_strategy"] = "recursive"
    # 4. CHARACTER CHUNKING
    character_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    character_chunks = character_splitter.split_documents(
        documents
    )
    for chunk in character_chunks:
        chunk.metadata["chunking_strategy"] = "character"
    # 5. TOKEN CHUNKING
    token_splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    token_chunks = token_splitter.split_documents(
        documents
    )
    for chunk in token_chunks:
        chunk.metadata["chunking_strategy"] = "token"
    # 6. SEMANTIC CHUNKING
    #
    # Only meaningful for prose. Non-prose formats
    # (CSV, JSON, XLSX) are already split into rows,
    # so they fall back to recursive chunking.
    semantic_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    try:
        semantic_chunks = semantic_splitter.split_documents(
            documents
        )
        if not semantic_chunks:
            semantic_chunks = recursive_chunks
    except Exception:
        semantic_chunks = recursive_chunks
    for chunk in semantic_chunks:
        chunk.metadata["chunking_strategy"] = "semantic"
    # 7. STORE SEMANTIC CHUNKS
    chroma_result = store_in_chroma(
        semantic_chunks
    )
    return {
        "document_id": document_id,
        "pages": len(documents),
        "recursive_chunks": len(recursive_chunks),
        "character_chunks": len(character_chunks),
        "token_chunks": len(token_chunks),
        "semantic_chunks": len(semantic_chunks),
        "chroma": chroma_result
    }
