
import dspy

from dspy_config import lm, HF_TOKEN


# Configure main LM
dspy.configure(lm=lm)


def rag_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None
):
    if not hasattr(prediction, "answer"):
        return dspy.Prediction(
            score=0.0,
            feedback="The program did not return an answer."
        )

    expected = example.answer.lower().strip()
    actual = prediction.answer.lower().strip()

    if expected in actual:
        return dspy.Prediction(
            score=1.0,
            feedback=(
                "The answer contains the expected information "
                "and is grounded in the provided context."
            )
        )

    return dspy.Prediction(
        score=0.0,
        feedback=(
            "The answer does not contain the expected information. "
            "Use only the provided context and include the "
            "relevant page citation."
        )
    )


# Small training set for GEPA
trainset = [
    dspy.Example(
        context="[Page 1] The company was established in 2010.",
        question="When was the company established?",
        answer="The company was established in 2010. [Page 1]"
    ).with_inputs("context", "question"),

    dspy.Example(
        context="[Page 2] The backend uses Python and FastAPI.",
        question="Which technologies are used for backend development?",
        answer="Python and FastAPI are used for backend development. [Page 2]"
    ).with_inputs("context", "question"),

    dspy.Example(
        context="[Page 3] The project focuses on document security.",
        question="What is the main focus of the project?",
        answer="The project focuses on document security. [Page 3]"
    ).with_inputs("context", "question"),
]


# Separate validation set
valset = [
    dspy.Example(
        context="[Page 4] The application stores documents securely.",
        question="How are documents stored?",
        answer="The application stores documents securely. [Page 4]"
    ).with_inputs("context", "question"),
]


# Reflection model used by GEPA
reflection_lm = dspy.LM(
    "huggingface/meta-llama/Llama-3.1-8B-Instruct",
    api_key=HF_TOKEN,
    temperature=0.7,
    max_tokens=1000
)


student = DocumentRAG()


# Keep GEPA small to avoid hundreds of API calls
optimizer = dspy.GEPA(
    metric=rag_metric,
    reflection_lm=reflection_lm,
    auto="light",
    num_threads=1,
    seed=0
)


print("Starting GEPA optimization...")

optimized_program = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset
)


optimized_program.save("optimized_rag.json")

print()
print("GEPA optimization completed.")
print("Saved: optimized_rag.json")