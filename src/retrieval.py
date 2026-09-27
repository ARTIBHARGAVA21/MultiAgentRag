
from typing import List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from vectorstores.chromadb import search_chroma



# 1. QUERY TRANSFORMATION
def transform_query(query: str) -> str:
    """
    Basic query transformation.

    For now we clean the user's question.

    Later this can be replaced with an LLM-based
    query rewriting approach.
    """
    query = query.strip()
    return query



# 2. VECTOR SIMILARITY SEARCH
def vector_search(
    query: str,
    document_id: str,
    k: int = 10
) -> List[Document]:
    """
    Search ChromaDB using vector similarity.
    Only chunks belonging to the requested
    client document are retrieved.
    """
    results = search_chroma(
        query=query,
        k=k,
        filter={
            "document_id": document_id
        }
    )
    return results



# 3. BM25 KEYWORD SEARCH
def keyword_search(
    query: str,
    documents: List[Document],
    k: int = 10
) -> List[Document]:

    """
    BM25 searches exact/keyword matches.
    Example:
    Query:
        JWT authentication
    It can find chunks containing:
        JWT
        authentication
    """
    if not documents:
        return []
    tokenized_documents = [
        document.page_content.lower().split()
        for document in documents
    ]
    bm25 = BM25Okapi(
        tokenized_documents
    )
    query_tokens = query.lower().split()
    scores = bm25.get_scores(
        query_tokens
    )
    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )
    return [
        document
        for document, score in ranked_documents[:k]
    ]


# 4. HYBRID SEARCH
def hybrid_search(
    query: str,
    document_id: str,
    k: int = 10
) -> List[Document]:

    """
    Hybrid search combines:

        Vector Search
              +
        Keyword Search

    """

    # -----------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------

    vector_results = vector_search(
        query=query,
        document_id=document_id,
        k=k
    )


    # -----------------------------------------
    # KEYWORD SEARCH
    #
    # We use the vector candidates as the
    # candidate pool for BM25.
    # -----------------------------------------

    keyword_results = keyword_search(
        query=query,
        documents=vector_results,
        k=k
    )


    # -----------------------------------------
    # COMBINE RESULTS
    # -----------------------------------------

    combined = {}

    for document in vector_results:

        key = (
            document.metadata.get("document_id"),
            document.metadata.get("page"),
            document.page_content
        )

        combined[key] = document


    for document in keyword_results:

        key = (
            document.metadata.get("document_id"),
            document.metadata.get("page"),
            document.page_content
        )

        combined[key] = document


    return list(combined.values())


# =========================================================
# 5. RERANKING
# =========================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 4
) -> List[Document]:

    """
    CrossEncoder reranks the retrieved documents.

    Example:

        10 retrieved chunks
                 ↓
             Reranker
                 ↓
           Best 4 chunks
    """

    if not documents:
        return []


    pairs = [
        [
            query,
            document.page_content
        ]
        for document in documents
    ]


    scores = reranker.predict(
        pairs
    )


    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )


    return [
        document
        for document, score
        in ranked_documents[:top_k]
    ]


# =========================================================
# 6. COMPLETE RETRIEVAL PIPELINE
# =========================================================

def retrieve_context(
    query: str,
    document_id: str,
    top_k: int = 4
):

    # -----------------------------------------
    # STEP 1
    # QUERY TRANSFORMATION
    # -----------------------------------------

    transformed_query = transform_query(
        query
    )


    # -----------------------------------------
    # STEP 2 + 3
    # HYBRID SEARCH
    # -----------------------------------------

    candidates = hybrid_search(
        query=transformed_query,
        document_id=document_id,
        k=10
    )


    # -----------------------------------------
    # STEP 4
    # RERANKING
    # -----------------------------------------

    final_documents = rerank_documents(
        query=transformed_query,
        documents=candidates,
        top_k=top_k
    )


    # -----------------------------------------
    # STEP 5
    # RETURN CONTEXT
    # -----------------------------------------

    return {
        "original_query": query,
        "transformed_query": transformed_query,
        "documents": final_documents
    }
