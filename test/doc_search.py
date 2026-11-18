import os
from dotenv import load_dotenv
from openai import OpenAI


# 1. Environment variables load karo (.env se)
load_dotenv()

# 2. Embedding ke liye local Ollama ka URL aur dummy key use karo
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
api_key = os.getenv("OLLAMA_API_KEY", "ollama")

client = OpenAI(base_url=base_url, api_key=api_key)

KB_FOLDER=os.getenv("KB_FOLDER", "knowledge_base")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

def load_document(folder=KB_FOLDER):
    documents = []
    
    for docs in sorted(os.listdir(folder)):
        if not docs.endswith(".txt"):
            continue
        
        path = os.path.join(folder , docs)
        with open(path , "r" , encoding="utf-8") as f:
            documents.append(
                {
                    "source" : docs,
                    "text":f.read()
                }
            )
    return documents

# add embeddings to the documents
def add_embeddings(documents, model=EMBED_MODEL):
    for doc in documents:
        response = client.embeddings.create(
            model=model,
            input=doc["text"]
        )
        doc["embedding"] = response.data[0].embedding
    return documents 

def get_embedding(text, model=EMBED_MODEL):
    res = client.embeddings.create(
        model=model,
        input=text
    )
    return res.data[0].embedding

# 4. Cosine similarity function
import numpy as np
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

 # search for the most relevant document based on cosine similarity
def search_documents(query, documents, top_k=3):
    query_vector = get_embedding(query)
    similarities = []
    for doc in documents:
        similarity = cosine_similarity(query_vector, doc["embedding"])
        similarities.append({
            "score" : float(similarity),
            "source" : doc["source"],
            "text" : doc["text"],
            "size" : len(doc["text"])
        })
    
    # Sort by similarity in descending order and return top_k
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return  similarities[:top_k]

#testing method

def test(user_query , documents , top_k=3):
    results = search_documents(user_query, documents, top_k)
    print(f"Top {top_k} results for query: '{user_query}'")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Size: {result['size']} characters")
        print(f"Text: {result['text'][:200]}...")  # Print first 200 characters of the text

# let's start testing
documents = load_document()
documents_with_embeddings = add_embeddings(documents)

print(f"Loaded {len(documents_with_embeddings)} documents with embeddings.")

user_query = "how many days i will get for maternity leave ?"
test(user_query, documents_with_embeddings, top_k=3)