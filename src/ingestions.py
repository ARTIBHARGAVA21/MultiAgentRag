from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

from langchain_experimental.text_splitter import SemanticChunker

from vectorstores.chromadb import embedding_model, store_in_chroma


def process_document(file_path: str):

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Recursive Chunking
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    recursive_chunks = recursive_splitter.split_documents(documents)

    for chunk in recursive_chunks:
        chunk.metadata["chunking_strategy"] = "recursive"

    # 3. Character Chunking
    character_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )

    character_chunks = character_splitter.split_documents(documents)

    for chunk in character_chunks:
        chunk.metadata["chunking_strategy"] = "character"

    # 4. Token Chunking
    token_splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    token_chunks = token_splitter.split_documents(documents)

    for chunk in token_chunks:
        chunk.metadata["chunking_strategy"] = "token"

    # 5. Semantic Chunking
    semantic_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    semantic_chunks = semantic_splitter.split_documents(documents)

    for chunk in semantic_chunks:
        chunk.metadata["chunking_strategy"] = "semantic"

    # 6. Store semantic chunks in ChromaDB
    chroma_result = store_in_chroma(semantic_chunks)

    return {
        "pages": len(documents),
        "recursive_chunks": len(recursive_chunks),
        "character_chunks": len(character_chunks),
        "token_chunks": len(token_chunks),
        "semantic_chunks": len(semantic_chunks),
        "chroma": chroma_result
    }