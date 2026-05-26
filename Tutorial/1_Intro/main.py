from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import ollama

app = FastAPI(title="Local AI Knowledge Base API")

# Initialize persistent local database folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 🔥 FIX: Route ChromaDB's vectorizer directly to your host machine's Ollama engine.
# This completely bypasses the broken inside-the-container ONNX download step.
# ollama_embedding = embedding_functions.OllamaEmbeddingFunction(
#     url="http://host.docker.internal:11434/api/embeddings",
#     model_name="llama3.2:3b"
# )

collection = chroma_client.get_or_create_collection(
    name="company_policy",
    # embedding_function=ollama_embedding
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        print("Step 1: Commencing ChromaDB Vector Search")
        
        # ChromaDB will now safely talk to host.docker.internal to embed the question
        search_results = collection.query(
            query_texts=[request.question],
            n_results=1
        )
        print("Search results raw payload:", search_results)
        
        print("Step 2: Checking Data Validation")
        if (not search_results or 
            'documents' not in search_results or 
            not isinstance(search_results['documents'], list) or 
            len(search_results['documents']) == 0 or 
            not search_results['documents'][0]):
            
            return {"answer": "I looked through the database, but I couldn't find any context matching your question."}
        
        retrieved_context = search_results['documents'][0][0]
        print("Successfully retrieved context:", retrieved_context)

        # Step 3: Construct the grounded context prompt
        system_prompt = "Answer the user's question based on the provided context ONLY. Keep it short."
        augmented_prompt = f"Question: {request.question}\n\nContext:\n{retrieved_context}"
        
        print("Step 4: Sending prompt to local Llama text generator")
        ollama.client.host = "http://host.docker.internal:11434"

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_prompt}
            ]
        )

        return {"answer": response['message']['content']}
    
    except Exception as e:
        print("🛑 SYSTEM EXCEPTION OCCURRED:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def health_check():
    return {"status": "ok", "model": "llama3.2:3b", "vector_store": "ChromaDB"}