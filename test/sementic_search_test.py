import numpy
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
def get_embeding(text):
    res = client.embeddings.create(
        model=model,
        input=text
    )
    return res.data[0].embedding

text1 = get_embeding("i love coding in python")
text2 = get_embeding("i love coding")

dot_product = numpy.dot(text1 , text2)
lenA = numpy.linalg.norm(text1)
lenB = numpy.linalg.norm(text2)

cosine_similarity = dot_product / (lenA * lenB)
print(cosine_similarity)

