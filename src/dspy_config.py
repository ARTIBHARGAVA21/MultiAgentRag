
import os

import dspy
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(
    ENV_FILE,
    override=True
)


HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found. "
        "Add it to src/.env "
        "(create one at https://huggingface.co/settings/tokens)."
    )


lm = dspy.LM(
    "huggingface/meta-llama/Llama-3.1-8B-Instruct",
    api_key=HF_TOKEN,
    temperature=0
)


dspy.configure(
    lm=lm
)