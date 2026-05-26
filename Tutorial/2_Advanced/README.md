# Enterprise Local RAG Agent with Stateful Memory & Hybrid Retrieval

An production-patterned, containerized AI Knowledge Base API built with **FastAPI**, **ChromaDB**, and **Ollama (Llama 3.2:3b)**. This system implements production-grade architecture patterns, moving beyond basic tutorials to tackle real-world challenges like multi-turn conversation memory, hybrid retrieval leakage, and sub-span telemetry tracing.

---

## 🏗️ System Architecture

The following pipeline illustrates how incoming client requests flow statelesssly through our gateway, query memory history, extract data through a dual-path retrieval funnel, and undergo deterministic re-ranking filtering before being streamed back to the user.

```mermaid
graph TD
    User([📱 User Client]) -->|1. POST /query {question, session_id}| API[⚡ FastAPI Gateway]
    
    subgraph State Management Layer
        API -->|2. Fetch History| Mem[🧠 Session Memory Store]
        Mem -->|3. Return Message History| API
    end

    subgraph Dual-Path Retrieval Funnel
        API -->|4. Parallel Query| Chroma[(🗄️ ChromaDB Vector Storage)]
        Chroma -->|Path 1| Sem[🧬 Semantic Embedding Search]
        Chroma -->|Path 2| Lex[🔎 Lexical Substring Scan]
        Sem & Lex -->|5. Candidate Extraction| Pool[🔀 Merged Candidate Pool]
    end

    subgraph High-Fidelity Filtering
        Pool -->|6. Top Chunks| Reranker[🎯 Content Density Re-ranker]
        Reranker -->|7. Evaluates Context Score| RankEval{Score > 0.1 Threshold?}
        RankEval -->|Yes: Inject Best Chunk| LLM[🤖 Ollama / Llama 3.2]
        RankEval -->|No: Trigger Guardrail| Fallback[⚠️ Fallback System Instructions]
    end

    LLM -->|8. SSE Streamed Tokens| API
    Fallback -->|8. SSE Streamed Tokens| API
    API -->|9. Archive Context & Answer| Mem
    API -.->|Telemetry Pipeline| Phoenix[(📊 Arize Phoenix / OpenTelemetry)]
    API -->|10. Server-Sent Events Stream| User