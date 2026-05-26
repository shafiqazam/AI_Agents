import os
import json
import asyncio
import httpx
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 🚀 UNIVERSAL OPENTELEMETRY INITIALIZATION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Read the collector endpoint injected by our docker-compose.yml file
# Defaulting to localhost for local debugging flexibility
collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
# Set up the OpenTelemetry Tracer Provider
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

# Configure the OTLP exporter to stream traces to the Phoenix container over HTTP
span_exporter = OTLPSpanExporter(endpoint=f"{collector_endpoint}/v1/traces")
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

# Create a manual tracer instance for our custom LLM streaming blocks
tracer = trace.get_tracer(__name__)

app = FastAPI(title="Local AI Knowledge Base API")

FastAPIInstrumentor.instrument_app(app)

# Initialize Chroma cleanly (No need for Settings injection anymore)
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
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

    with tracer.start_as_current_span("ollama_llm_generation") as span:
        span.set_attribute("llm.model_name", "llama3.2:3b")
        span.set_attribute("input.prompt", prompt_text)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        span.set_status(trace.StatusCode.ERROR, "Ollama connection failed")
                        yield f"data: {json.dumps({'error': 'LLM API request failed'})}\n\n"
                    full_response_tokens = []
                    async for line in response.aiter_lines():
                        if line: 
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("message", {}).get("content", "")

                                if token:
                                    full_response_tokens.append(token)
                                    # Push Server-Sent Event (SSE)
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                            
                            except json.JSONDecodeError:
                                continue

                            # Yield control back to Uvicorn's event loop
                            await asyncio.sleep(0.01)
                    span.set_attribute("output.response", "".join(full_response_tokens))

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            yield f"data: {json.dumps({'error': f'Network failure: {str(e)}'})}\n\n"

@app.post("/query")
async def running_agent_loop(request: QueryRequest):

    current_span = trace.get_current_span()
    try:
        print(f"\n📥 Async Request Received: '{request.question}'")

        with tracer.start_as_current_span("input_guardrail_check") as span:
            span.set_attribute("user.question", request.question)
            
            if "joke" in request.question.lower() or "hack" in request.question.lower():
                span.set_attribute("guardrail.passed", False)
                print("🛡️ Guardrail triggered! Blocking request.")
                # 🔥 We must explicitly return an error stream here to stop the agent!
                async def blocked_stream():
                    yield f"data: {json.dumps({'token': 'I cannot answer that.'})}\n\n"
                return StreamingResponse(blocked_stream(), media_type="text/event-stream")
            
            span.set_attribute("guardrail.passed", True)
                
        retrieved_context = "No specific policy text found."
        
        # Open an explicit sub-span (child node) for the Vector DB operations
        with tracer.start_as_current_span("vector_db_search") as db_span:
            db_span.set_attribute("db.system", "chromadb")
            db_span.set_attribute("db.query", request.question)
            db_span.set_attribute("retrieval.quality_grade", "POOR") 
            
            try:
                db_count = collection.count()
                db_span.set_attribute("db.collection_size", db_count)
                
                if db_count > 0:
                    search_results = collection.query(query_texts=[request.question], n_results=1)
                    if search_results and 'documents' in search_results and search_results['documents'][0]:
                        retrieved_context = search_results['documents'][0][0]
                        db_span.set_attribute("db.match_found", True)
                        db_span.set_attribute("db.retrieved_document", retrieved_context)
                        db_span.set_attribute("retrieval.quality_grade", "HIGH")
                        print(f"✅ Vector Match Found!")
                else:
                    db_span.set_attribute("db.match_found", False)
                    db_span.set_attribute("retrieval.quality_grade", "POOR")
                    db_span.set_attribute("routing.action", "switched_to_general_knowledge")
                    print("⚠️ ChromaDB is empty inside this container. Skipping vector search.")
            except Exception as chroma_error:
                db_span.record_exception(chroma_error)
                print(f"⚠️ ChromaDB Error Bypassed: {chroma_error}")
        
        print("🚀 Opening Server-Sent Events (SSE) stream back to user client...")
        
        return StreamingResponse(
            llm_token_streamer(request.question, retrieved_context),
            media_type="text/event-stream"
        )

    except Exception as e:
        current_span.record_exception(e)
        print("🛑 CORE AGENT CRASH:", str(e))
        raise HTTPException(status_code=500, detail=str(e))