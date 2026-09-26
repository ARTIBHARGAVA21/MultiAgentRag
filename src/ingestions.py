from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)


def process_document(file_path):

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    recursive_chunks = recursive_splitter.split_documents(documents)
    character_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )

    character_chunks = character_splitter.split_documents(documents)
    token_splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    token_chunks = token_splitter.split_documents(documents)

    return {
        "pages": len(documents),
        "recursive_chunks": len(recursive_chunks),
        "character_chunks": len(character_chunks),
        "token_chunks": len(token_chunks),
    }
