
import dspy


class DocumentQA(dspy.Signature):
    """
    You are an AI document analysis assistant.

    Your day-to-day responsibilities are:
    - Answer questions using only the retrieved document context.
    - Summarize relevant information from documents.
    - Extract important information when requested.
    - Compare information when multiple relevant sections are available.
    - Include page citations when the page information is available.
    - Never use outside knowledge.
    - Never invent information.
    - If the answer is not available in the context,
      clearly say that it was not found.
    """

    context = dspy.InputField(
        desc="Relevant content retrieved from the uploaded document."
    )

    question = dspy.InputField(
        desc="User question about the uploaded document."
    )

    answer = dspy.OutputField(
        desc=(
            "Answer the question using only the provided context. "
            "Include relevant page citations."
        )
    )


class DocumentRAG(dspy.Module):

    def __init__(self):
        super().__init__()

        self.answer_question = dspy.Predict(
            DocumentQA
        )

    def forward(self, context, question):

        return self.answer_question(
            context=context,
            question=question
        )