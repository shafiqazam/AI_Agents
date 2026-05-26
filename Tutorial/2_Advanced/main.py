import os
import json
import asyncio
import httpx
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List

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

# Memory management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        # Including user, system instructions, and assistant responses in the history
        self.MAX_HISTORY = 10

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        # Keep only the last N messages to manage memory
        if len(self.sessions[session_id]) > self.MAX_HISTORY:
            self.sessions[session_id] = self.sessions[session_id][-self.MAX_HISTORY:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

memory = SessionManager()

# Initialize Chroma cleanly (No need for Settings injection anymore)
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="company_policy")

class QueryRequest(BaseModel):
    question: str
    session_id: str

async def llm_token_streamer(session_id: str, prompt_text: str, context: str):
    # Simulate streaming tokens from the LLM via Ollama API
    url = "http://host.docker.internal:11434/api/chat"

    system_instruction = (
        "You are an internal corporate policy assistant. "
        "Answer questions concisely based on the provided context. "
    )

    messages = memory.get_history(session_id)
    
    final_messages = [ {"role": "system", "content": system_instruction} ] + messages
    
    final_messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt_text}"})

    payload = {
        "model": "llama3.2:3b",
        "messages": final_messages,
        "stream": True,
    }

    print(final_messages)

    with tracer.start_as_current_span("ollama_llm_generation") as span:
        span.set_attribute("llm.model_name", "llama3.2:3b")
        span.set_attribute("input.prompt", prompt_text)
        # span.set_attribute("memoryAgent", final_messages)
        
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

                    memory.add_message(session_id, "assistant", "".join(full_response_tokens))

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
        # Open an explicit sub-span for the entire hybrid retrieval process
        with tracer.start_as_current_span("hybrid_retrieval_and_rerank") as retrieve_span:
            retrieve_span.set_attribute("db.system", "chromadb")
            retrieve_span.set_attribute("db.query", request.question)
            
            retrieved_context = "No specific policy text found."
            candidates = []

            try:
                db_count = collection.count()
                if db_count > 0:
                    # -------------------------------------------------------------
                    # PATH 1: Semantic Vector Search (Pull top 5 raw matches)
                    # -------------------------------------------------------------
                    vector_results = collection.query(query_texts=[request.question], n_results=5)
                    if vector_results and 'documents' in vector_results and vector_results['documents'][0]:
                        for doc in vector_results['documents'][0]:
                            if doc not in candidates:
                                candidates.append(doc)

                    # -------------------------------------------------------------
                    # PATH 2: Lexical Keyword Matcher (Fallback/Complementary)
                    # -------------------------------------------------------------
                    # Fetching all records to look for direct substring keyword hits
                    all_docs = collection.get()
                    if all_docs and 'documents' in all_docs:
                        # Extract keywords from the user question (simple split, could be enhanced with NLP techniques)
                        keywords = [word.lower() for word in request.question.split() if len(word) > 3]
                        for doc in all_docs['documents']:
                            # If a document contains exact keyword matches, prioritize it as a candidate
                            if any(kw in doc.lower() for kw in keywords) and doc not in candidates:
                                candidates.append(doc)

                    retrieve_span.set_attribute("retrieval.total_candidates", len(candidates))

                    # -------------------------------------------------------------
                    # LAYER 3: The Re-ranking Filter
                    # -------------------------------------------------------------
                    if candidates:
                        print(f"🔎 Funneling {len(candidates)} candidate documents into the Re-ranker...")
                        
                        # Open a specific nested child span for the re-ranker calculation
                        with tracer.start_as_current_span("cross_encoder_reranker") as rerank_span:
                            ranked_docs = []
                            query_words = set(request.question.lower().split())

                            for doc in candidates:
                                # Programmatic Cross-Encoder Simulation: Calculate a contextual overlap score
                                doc_words = set(doc.lower().split())
                                intersection = query_words.intersection(doc_words)
                                
                                # Pure intersection count represents basic lexical alignment
                                # Division by total length provides structural density scaling
                                relevance_score = len(intersection) / (len(query_words) + 0.1)
                                ranked_docs.append((relevance_score, doc))

                            # Sort candidates by relevance score in descending order
                            ranked_docs.sort(key=lambda x: x[0], reverse=True)
                            
                            # Select the absolute highest scoring context chunk
                            top_score, best_doc = ranked_docs[0]
                            
                            rerank_span.set_attribute("rerank.highest_score", top_score)
                            
                            if top_score > 0.1:
                                retrieved_context = best_doc
                                retrieve_span.set_attribute("retrieval.quality_grade", "HIGH")
                                retrieve_span.set_attribute("db.match_found", True)
                                retrieve_span.set_attribute("db.retrieved_document", retrieved_context)
                                print(f"✅ Re-ranker Selected Best Match (Score: {top_score:.2f})")
                            else:
                                retrieve_span.set_attribute("retrieval.quality_grade", "POOR")
                                retrieve_span.set_attribute("db.match_found", False)
                                # Fallback operation triggered if scores are too low
                                with tracer.start_as_current_span("fallback_llm_routing") as fallback_span:
                                    fallback_span.set_attribute("routing.action", "switched_to_general_knowledge")
                                    retrieved_context = "Use your general knowledge. No internal company documents match."
                                print("⚠️ Low re-ranker confidence. Activating fallback context.")
                else:
                    retrieve_span.set_attribute("retrieval.quality_grade", "POOR")
                    print("⚠️ ChromaDB is completely empty inside this container.")

            except Exception as retrieval_error:
                retrieve_span.record_exception(retrieval_error)
                retrieve_span.set_status(trace.StatusCode.ERROR, str(retrieval_error))
                print(f"🛑 Retrieval Pipeline Error: {retrieval_error}")
        
        print("🚀 Opening Server-Sent Events (SSE) stream back to user client...")
        
        return StreamingResponse(
            llm_token_streamer(request.session_id, request.question, retrieved_context),
            media_type="text/event-stream"
        )

    except Exception as e:
        current_span.record_exception(e)
        print("🛑 CORE AGENT CRASH:", str(e))
        raise HTTPException(status_code=500, detail=str(e))