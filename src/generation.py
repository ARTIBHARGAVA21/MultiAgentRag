from typing import AsyncGenerator, List
import os

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from prompts import RAG_PROMPT


# ---------------------------------------------------------
# LLM CONFIGURATION
#
# Generation runs on the HuggingFace Inference API (cloud).
# The embedding model runs locally, so no token is needed
# for retrieval - only the LLM requires this key.
# ---------------------------------------------------------

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found. "
        "Add it to src/.env "
        "(create one at https://huggingface.co/settings/tokens)."
    )

HF_LLM_REPO = os.getenv(
    "HF_LLM_REPO",
    "meta-llama/Llama-3.1-8B-Instruct"
)


def _build_llm() -> ChatHuggingFace:

    endpoint = HuggingFaceEndpoint(
        repo_id=HF_LLM_REPO,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        streaming=True
    )

    return ChatHuggingFace(llm=endpoint)


llm = _build_llm()


# ---------------------------------------------------------
# 1. PROMPT ASSEMBLY
# ---------------------------------------------------------

def build_context(
    documents: List[Document]
) -> str:

    context_parts = []

    for document in documents:

        page_number = document.metadata.get(
            "page",
            "unknown"
        )

        if isinstance(page_number, int):
            page_number += 1

        content = document.page_content.strip()

        context_parts.append(
            f"[Page {page_number}]\n{content}"
        )

    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------
# 2. RAG CHAIN
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    RAG_PROMPT
)


rag_chain = (
    {
        "context": lambda x: x["context"],
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------
# 3. STREAMING RESPONSE
# ---------------------------------------------------------

def _is_rate_limit_error(
    error: Exception
) -> bool:

    error_message = str(error).lower()

    return (
        "429" in error_message
        or "rate limit" in error_message
        or "rate_limit" in error_message
        or "too many requests" in error_message
    )


async def stream_answer(
    question: str,
    documents: List[Document]
) -> AsyncGenerator[str, None]:

    context = build_context(
        documents
    )

    try:

        async for chunk in rag_chain.astream(
            {
                "context": context,
                "question": question
            }
        ):

            if chunk:

                yield chunk

    except Exception as e:

        if _is_rate_limit_error(e):

            yield (
                "\n\nThe AI service is temporarily "
                "rate-limited. Please try again later."
            )

        else:

            print(
                f"Generation error: {e}"
            )

            yield (
                "\n\nSorry, I could not generate "
                "the answer because the AI service "
                "encountered an error."
            )