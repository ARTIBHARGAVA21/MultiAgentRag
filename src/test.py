import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(".env")
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    provider="novita",  # or "auto"
    token=api_key,
)

try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=128,
    )
    print("API SUCCESS")
    print(response.choices[0].message.content)
except Exception as e:
    print("API FAILED")
    print(e)