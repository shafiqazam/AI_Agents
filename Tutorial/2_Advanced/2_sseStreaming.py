import os

# 🛑 FIX 1: Nuke Telemetry AT THE OS LEVEL before importing ChromaDB
os.environ["CHROMA_SERVER_NO_TELEMETRY"] = "True"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
import asyncio
import httpx
import json

app = FastAPI(title="Local AI Knowledge Base API")

# Initialize Chroma cleanly (No need for Settings injection anymore)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_policy")

class QueryRequest(BaseModel):
    question: str

async def llm_token_streamer(prompt_text: str, context: str):
    # Simulate streaming tokens from the LLM via Ollama API
    url = "http://host.docker.internal:11434/api/chat"

    system_instruction = (
        "You are an internal corporate policy assistant. "
        "Answer questions concisely based on the provided context. "
    )

    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt_text}"}
        ],
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="LLM API request failed")
            
            async for line in response.aiter_lines():
                if line: 
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")

                        if token:
                            # Push Server-Sent Event (SSE)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    
                    except json.JSONDecodeError:
                        continue

                    # Yield control back to Uvicorn's event loop
                    await asyncio.sleep(0.01)

@app.post("/query")
async def running_agent_loop(request: QueryRequest):
    try:
        print(f"\n📥 Async Request Received: '{request.question}'")
        
        retrieved_context = "No specific policy text found."
        
        # 🛑 FIX 2: Prevent the Core Agent Crash by checking DB size first!
        db_count = collection.count()
        if db_count > 0:
            print(f"Step 1: Inspecting Vector Database (Found {db_count} total records)...")
            search_results = collection.query(query_texts=[request.question], n_results=1)
            
            # Safely extract text if a match is found
            if (search_results and 'documents' in search_results and 
                search_results['documents'] and search_results['documents'][0]):
                retrieved_context = search_results['documents'][0][0]
                print(f"-> Context Injected: '{retrieved_context}'")
        else:
            print("⚠️ WARNING: Vector Database is completely empty (0 records). Skipping search.")
        
        print("🚀 Opening Server-Sent Events (SSE) stream back to user client...")
        
        return StreamingResponse(
            llm_token_streamer(request.question, retrieved_context),
            media_type="text/event-stream"
        )

    except Exception as e:
        print("🛑 CORE AGENT CRASH:", str(e))
        raise HTTPException(status_code=500, detail=str(e))