import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Environment variables load karo (.env se)
load_dotenv()

# 2. Embedding ke liye local Ollama ka URL aur dummy key use karo
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
api_key = os.getenv("OLLAMA_API_KEY", "ollama")
model = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

client = OpenAI(base_url=base_url, api_key=api_key)

# 3. Text ko mathematical vectors (Embeddings) me convert karo
res = client.embeddings.create(
    model=model,
    input="i am a good boy"
)

# 4. Result print karo
embedding_vector = res.data[0].embedding
print("✅ Embedding generated successfully!")
print(f"Model used: {model}")
print(f"Total Dimensions (Vector length): {len(embedding_vector)}")
print(f"Sample values (first 5 numbers): {embedding_vector[:5]}")
