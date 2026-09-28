import asyncio
from typing import AsyncGenerator, List
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from prompts import RAG_PROMPT


# LLM CONFIGURATION
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    max_retries=0
)


# 1. PROMPT ASSEMBLY
def build_prompt(
    question: str,
    documents: List[Document]
) -> str:

    context_parts = []

    for document in documents:

        page_number = document.metadata.get(
            "page",
            "unknown"
        )

        # PyPDFLoader page numbers start from 0
        if isinstance(page_number, int):
            page_number += 1

        content = document.page_content.strip()

        context_parts.append(
            f"[Page {page_number}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    return prompt



# 2. HELPER - CHECK RATE LIMIT
def is_rate_limit_error(error: Exception) -> bool:
    error_message = str(error).lower()
    return (
        "429" in error_message
        or "rate limit" in error_message
        or "rate_limit" in error_message
        or "too many requests" in error_message
    )



# 3. LLM INFERENCE
async def generate_answer(
    question: str,
    documents: List[Document]
) -> str:

    prompt = build_prompt(
        question=question,
        documents=documents
    )
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = await llm.ainvoke(
                prompt
            )
            return response.content
        except Exception as e:
            # RATE LIMIT
            if is_rate_limit_error(e):
                if attempt == max_attempts - 1:
                    return (
                        "The AI service is temporarily "
                        "rate-limited. Please try again "
                        "after some time."
                    )
                # Exponential backoff:
                # 2 seconds
                # 4 seconds
                # 8 seconds
                wait_time = 2 ** (attempt + 1)
                await asyncio.sleep(
                    wait_time
                )
            else:
                # OTHER ERROR
                return (
                    "Sorry, I could not generate the "
                    "answer because the AI service "
                    "encountered an error."
                )
    return (
        "Sorry, I could not generate the answer."
    )
# 4. STREAMING RESPONSE

async def stream_answer(
    question: str,
    documents: List[Document]
) -> AsyncGenerator[str, None]:

    prompt = build_prompt(
        question=question,
        documents=documents
    )
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            async for chunk in llm.astream(
                prompt
            ):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            # RATE LIMIT
            if is_rate_limit_error(e):
                if attempt == max_attempts - 1:
                    yield (
                        "\n\nThe AI service is "
                        "temporarily rate-limited. "
                        "Please try again later."
                    )
                    return
                wait_time = 2 ** (attempt + 1)
                await asyncio.sleep(
                    wait_time
                )
            else:
                yield (
                    "\n\nSorry, I could not generate "
                    "the answer because the AI service "
                    "encountered an error."
                )
                return
