import os

from dotenv import load_dotenv
from mistralai.client import Mistral


load_dotenv(".env", override=True)

api_key = os.getenv("MISTRAL_API_KEY")

print("Key exists:", bool(api_key))
print("Key length:", len(api_key) if api_key else 0)


client = Mistral(
    api_key=api_key
)

response = client.embeddings.create(
    model="mistral-embed",
    inputs=["Hello world"]
)

print("SUCCESS")
print("Embedding length:", len(response.data[0].embedding))