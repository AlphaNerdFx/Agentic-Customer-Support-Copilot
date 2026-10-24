# SPEC: FDE Support Copilot — Architecture & Technical Specification

> **Version:** 1.0 | **Last Updated:** 2026-10-19 | **Status:** Draft
> **References:** [PRD.md](./PRD.md) | [AGENTS.md](./AGENTS.md)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Streamlit)                     │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Chat UI  │  │ Citation   │  │ Approval     │  │ Metrics   │  │
│  │          │  │ Cards      │  │ Gate UI      │  │ Dashboard │  │
│  └────┬─────┘  └────────────┘  └──────────────┘  └───────────┘  │
│       │              REST API (HTTP)                            │
└───────┼─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  API Layer (Routers)                    │    │ 
│  │  POST /chat  │  POST /approve  │  POST /reject          │    │
│  │  GET /metrics │  GET /health                            │    │
│  └──────┬──────────────────────────────────────────────────┘    │
│         │                                                       │
│  ┌──────▼──────────────────────────────────────────────────┐    │
│  │              ORCHESTRATION (LangGraph)                  │    │
│  │                                                         │    │
│  │  ┌──────────┐   ┌───────────┐   ┌──────────────┐        │    │
│  │  │ Classify  │──▶│ Retrieve  │──▶│ Generate     │      │    │
│  │  │ Intent    │   │ Context   │   │ Draft Reply  │        │   │
│  │  └──────────┘   └───────────┘   └──────┬───────┘        │    │
│  │                                         │                 │     │
│  │                              ┌──────────▼───────────┐    │     │
│  │                              │ Tool Router           │    │     │
│  │                              │ (conditional edge)    │    │     │
│  │                              └──────┬───────┬───────┘    │     │
│  │                                     │       │            │     │
│  │                          ┌──────────▼┐  ┌──▼──────────┐ │     │
│  │                          │ No Tool   │  │ Propose Tool │ │     │
│  │                          │ → Respond │  │ → interrupt()│ │     │
│  │                          └───────────┘  └─────────────┘ │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │ RAG Service  │  │ Tool Service  │  │ Metrics Service     │     │
│  │ (retrieval + │  │ (escalate,   │  │ (logging, latency,  │     │
│  │  citation)   │  │  refund,     │  │  token tracking)    │     │
│  │              │  │  email)      │  │                     │     │
│  └──────┬───────┘  └──────────────┘  └─────────────────────┘     │
│         │                                                         │
│  ┌──────▼──────────────────────────────────────────────────┐     │
│  │              STORAGE LAYER                                │     │
│  │  ┌────────────┐  ┌────────────────┐  ┌────────────────┐ │     │
│  │  │ Chroma     │  │ SQLite         │  │ LangGraph      │ │     │
│  │  │ (vectors)  │  │ (sessions,     │  │ Checkpointer   │ │     │
│  │  │            │  │  metrics, logs)│  │ (state)        │ │     │
│  │  └────────────┘  └────────────────┘  └────────────────┘ │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │              EXTERNAL SERVICES                            │     │
│  │  OpenAI API (embeddings + GPT-4o)  │  LangSmith (traces) │     │
│  └──────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Decisions

### Why Each Choice Was Made (Architecture Decision Records)

**ADR-001: LangGraph over raw LangChain chains**

- **Decision:** Use LangGraph for agent orchestration
- **Context:** The copilot requires mid-conversation interruption for human approval. Basic LangChain chains are linear and cannot pause/resume.
- **Rationale:** LangGraph provides native `interrupt()` that dehydrates graph state, waits for human input, then resumes. This is the exact pattern needed for HITL approval gates.
- **Trade-off:** Slightly steeper learning curve than basic chains, but avoids building custom state management.

**ADR-002: Chroma for MVP vector store**

- **Decision:** Chroma for V1, with abstraction layer for future swap
- **Context:** Need zero-config vector store for rapid development
- **Rationale:** `pip install chromadb` — no Docker, no cloud account, no API key. Abstract behind a `VectorStoreService` interface so swapping to Qdrant/pgvector is a config change.
- **Trade-off:** Chroma struggles at scale and with concurrent writes. Acceptable for demo/evaluation workloads.

**ADR-003: Streamlit for V1 frontend**

- **Decision:** Streamlit as thin UI layer calling FastAPI backend
- **Context:** 1-week timeline, solo developer, Python-primary skillset
- **Rationale:** Ship a working demo fast. FastAPI backend exposes clean REST API — frontend is swappable. The backend IS the product; the frontend is the demo layer.
- **Trade-off:** Limited styling, full-script reruns on interaction. Mitigated by keeping all state server-side.

**ADR-004: OpenAI text-embedding-3-small for embeddings**

- **Decision:** Use OpenAI's smallest embedding model
- **Context:** Need cost-effective, high-quality embeddings for the deduplicated help center corpus across 6 operational domains derived from the Bitext dataset (~18K relevant rows consolidated with query-enrichment)
- **Rationale:** $0.02/1M tokens. 8K token context handles long articles. Matryoshka Representation Learning allows dimension reduction for storage optimization. Zero infrastructure.
- **Trade-off:** Vendor dependency on OpenAI. Mitigated by abstracting embedding calls behind an interface.

**ADR-005: GPT-4o for generation and tool-calling**

- **Decision:** GPT-4o as primary LLM
- **Context:** Need reliable structured output, tool-calling, and instruction following for citation-grounded responses
- **Rationale:** Best-in-class tool-calling accuracy. Native JSON mode / structured outputs. Reasonable cost for a portfolio project.
- **Trade-off:** ~$2.50/1M input tokens, $10/1M output tokens. Acceptable for demo workloads.

---

## 3. Directory Structure

```
fde-copilot/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Lint + test + Docker build
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app factory
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py               # POST /chat, POST /approve, POST /reject
│   │   │   ├── metrics.py            # GET /metrics
│   │   │   └── health.py             # GET /health
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py               # Request/Response Pydantic models
│   │   │   ├── tools.py              # Tool call schemas
│   │   │   └── metrics.py            # Metrics response models
│   │   └── dependencies.py           # FastAPI dependency injection
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 # Settings (pydantic-settings, env vars)
│   │   └── exceptions.py             # Custom exception classes
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                  # LangGraph state machine definition
│   │   ├── state.py                  # TypedDict state schema
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── classify.py           # Intent classification node
│   │   │   ├── retrieve.py           # RAG retrieval node
│   │   │   ├── generate.py           # Citation-grounded generation node
│   │   │   └── tools.py              # Tool proposal + interrupt node
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── escalate.py           # escalate_ticket tool
│   │   │   ├── refund.py             # initiate_refund tool
│   │   │   └── email.py              # send_followup_email tool
│   │   └── prompts/
│   │       ├── system.py             # System prompt with citation instructions
│   │       └── templates.py          # Prompt templates
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py                 # Dataset → knowledge articles → chunks → embeddings
│   │   ├── chunking.py               # Chunking strategies (semantic, parent-child)
│   │   ├── retriever.py              # Vector search + citation metadata extraction
│   │   └── store.py                  # VectorStoreService interface + Chroma implementation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── session.py                # Conversation session management
│   │   └── metrics.py                # Metrics collection and aggregation
│   └── evaluation/
│       ├── __init__.py
│       ├── dataset.py                # Golden test dataset loading
│       ├── evaluator.py              # Citation accuracy, task completion, LLM-judge
│       └── report.py                 # Evaluation report generation
├── frontend/
│   ├── app.py                        # Streamlit main entry point
│   ├── components/
│   │   ├── chat.py                   # Chat message rendering
│   │   ├── citations.py              # Citation card components
│   │   ├── approval.py               # Approval gate UI
│   │   └── dashboard.py              # Metrics dashboard
│   └── utils.py                      # API client helpers
├── data/
│   ├── knowledge_base/               # Generated KB articles (from ingestion)
│   └── eval/
│       └── golden_test_set.json      # 50+ annotated test cases
├── scripts/
│   ├── ingest.py                     # CLI script to run ingestion pipeline
│   └── evaluate.py                   # CLI script to run evaluation suite
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_retriever.py
│   │   ├── test_tools.py
│   │   └── test_citation_parsing.py
│   ├── integration/
│   │   ├── test_graph.py             # LangGraph end-to-end
│   │   └── test_api.py              # FastAPI endpoint tests
│   └── evaluation/
│       └── test_eval_pipeline.py
├── docker-compose.yml                # One-command local setup
├── Dockerfile                        # Multi-stage Python build
├── .env.example                      # Required environment variables
├── pyproject.toml                    # Dependencies (uv/poetry)
├── README.md                         # Architecture diagram, setup, ADRs
├── PRD.md                            # Product Requirements Document
├── SPEC.md                           # This file
└── AGENTS.md                         # Agent behavioral rules + methodology
```

---

## 4. API Specification

### POST /chat

**Request:**

```json
{
  "session_id": "uuid-or-null",
  "message": "How do I get a refund for a damaged item?",
  "stream": false
}
```

**Response:**

```json
{
  "session_id": "abc-123",
  "response": {
    "content": "Based on our refund policy, you can request a refund for damaged items within 30 days of delivery [1]. To initiate the process, you'll need to provide your order number and photos of the damage [2].",
    "citations": [
      {
        "index": 1,
        "source_id": "kb-article-042",
        "title": "Refund Policy for Damaged Items",
        "excerpt": "Customers may request a full refund for items that arrive damaged...",
        "relevance_score": 0.94
      },
      {
        "index": 2,
        "source_id": "kb-article-108",
        "title": "How to Submit a Refund Request",
        "excerpt": "To process your refund, please provide your order number...",
        "relevance_score": 0.89
      }
    ],
    "tool_proposal": {
      "tool_name": "initiate_refund",
      "parameters": {
        "reason": "damaged_item",
        "requires_evidence": true
      },
      "rationale": "Customer is requesting a refund for damaged item. This requires your approval before processing.",
      "status": "awaiting_approval"
    }
  },
  "metrics": {
    "latency_ms": 2340,
    "tokens_used": 847,
    "chunks_retrieved": 5,
    "chunks_cited": 2
  }
}
```

### POST /approve

Resumes an interrupted LangGraph session by submitting the human approval decision via `Command(resume=...)` using `session_id` as the checkpoint `thread_id`.

**Request:**

```json
{
  "session_id": "abc-123",
  "tool_call_id": "tc-456",
  "approved": true,
  "modification": null
}
```

**Response:**

```json
{
  "session_id": "abc-123",
  "tool_result": {
    "tool_name": "initiate_refund",
    "status": "executed",
    "result": "Refund request RFN-789 created successfully. Customer will be notified via email.",
    "executed_at": "2026-10-19T14:30:00Z"
  }
}
```

### POST /reject

Resumes an interrupted LangGraph session with a rejection decision and explanation, routing the state machine away from tool execution back to conversation completion.

**Request:**

```json
{
  "session_id": "abc-123",
  "tool_call_id": "tc-456",
  "reason": "Need to verify order details first"
}
```

**Response:**

```json
{
  "session_id": "abc-123",
  "tool_result": {
    "tool_name": "initiate_refund",
    "status": "rejected",
    "reason": "Need to verify order details first",
    "executed_at": "2026-10-19T14:30:05Z"
  }
}
```

### GET /metrics

**Response:**

```json
{
  "total_sessions": 142,
  "total_queries": 487,
  "avg_response_time_ms": 2100,
  "citation_rate": 0.91,
  "tool_proposals": 67,
  "tool_approvals": 54,
  "tool_rejections": 13,
  "avg_citations_per_response": 2.3
}
```

### GET /health

**Response:**

```json
{
  "status": "healthy",
  "vector_store": "connected",
  "llm_provider": "connected",
  "uptime_seconds": 3600
}
```

---

## 5. LangGraph State Machine

### State Schema

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class CopilotState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]
    session_id: str

    # RAG
    retrieved_chunks: list[dict]       # [{source_id, title, content, score}]
    citations: list[dict]              # [{index, source_id, title, excerpt, score}]

    # Intent Classification
    intent: str                        # billing, technical, account, order, general
    requires_tool: bool

    # Tool Calling
    proposed_tool: dict | None         # {tool_name, parameters, rationale}
    tool_status: Literal["none", "proposed", "approved", "rejected", "executed"]
    tool_result: str | None

    # Metrics
    token_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
```

### Graph Definition

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_graph():
    graph = StateGraph(CopilotState)

    # Nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("propose_tool", propose_tool_node)       # Uses interrupt()
    graph.add_node("execute_tool", execute_tool_node)

    # Edges
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")

    # Conditional: does response warrant a tool call?
    graph.add_conditional_edges(
        "generate_response",
        route_tool_decision,
        {"no_tool": END, "needs_tool": "propose_tool"}
    )

    # After tool proposal (interrupt resumes here)
    graph.add_conditional_edges(
        "propose_tool",
        route_approval_decision,
        {"approved": "execute_tool", "rejected": END}
    )
    graph.add_edge("execute_tool", END)

    # Compile with checkpointer for state persistence
    memory = SqliteSaver.from_conn_string("data/checkpoints.db")
    return graph.compile(checkpointer=memory)
```

### HITL Interrupt & Resume Pattern

```python
from langgraph.types import interrupt, Command

def propose_tool_node(state: CopilotState) -> dict:
    """Proposes a tool action and pauses for human approval via interrupt()."""
    tool_proposal = state["proposed_tool"]

    # This pauses graph execution, serializes state to the checkpointer,
    # and returns the interrupt payload to the API layer.
    human_decision = interrupt({
        "tool_name": tool_proposal["tool_name"],
        "parameters": tool_proposal["parameters"],
        "rationale": tool_proposal["rationale"],
        "prompt": "Do you approve this action?"
    })

    # Execution resumes here when graph.invoke(Command(resume=...)) is executed
    return {
        "tool_status": "approved" if human_decision.get("approved") else "rejected",
        "tool_result": None
    }

# FastAPI Router Resume Handler Pattern (in src/api/routers/chat.py):
# result = graph.invoke(
#     Command(resume={"approved": body.approved, "reason": getattr(body, "reason", None)}),
#     config={"configurable": {"thread_id": body.session_id}}
# )
```

---

## 6. RAG Pipeline Detail

### Ingestion Flow (Query-Enriched Deduplication)

```
Bitext Dataset (27K rows)
    │
    ▼
Filter to 6 Target Domains (Drop FEEDBACK & NEWSLETTER)
    │
    ▼
Category Normalization:
  - PAYMENT + INVOICE → BILLING
  - DELIVERY + SHIPPING_ADDRESS → SHIPPING
  - CANCELLATION_FEE + ORDER (cancel_order) → CANCELLATION
  - ORDER (place_order, change_order) → ORDER
  - ACCOUNT → ACCOUNT
  - REFUND → REFUND
    │
    ▼
Group by (Category, Intent, Canonical Response)
    │
    ▼
Deduplicate & Aggregate Query Variations:
  - Extract canonical policy response as article content
  - Collect unique user queries into sample_queries list
    │
    ▼
Format Query-Enriched Chunks:
  Title: {intent_title}
  Category: {category} | Intent: {intent}
  Typical Customer Questions:
  - {sample_query_1}
  - {sample_query_2}
  Support Policy / Resolution:
  {response}
    │
    ▼
Embed Chunks (text-embedding-3-small, 1536 dims)
    │
    ▼
Store in Chroma with metadata: {source_id, category, intent, sample_count, sample_queries}
```

### Retrieval Flow

```
User Query
    │
    ▼
Embed query (text-embedding-3-small)
    │
    ▼
Vector similarity search (Chroma, top-k=5)
    │
    ▼
Re-rank by relevance score (optional: Cohere Rerank)
    │
    ▼
Format as numbered context blocks with source metadata
    │
    ▼
Pass to LLM with citation instructions
```

### Citation Prompt Template

```python
CITATION_SYSTEM_PROMPT = """You are a customer support copilot. Answer the user's 
question using ONLY the provided context. Follow these rules strictly:

1. Every factual claim MUST include an inline citation [N] referencing the source.
2. If the context does not contain enough information, say: "I don't have enough 
   information in our knowledge base to answer this. Consider checking with a 
   senior agent."
3. Do NOT extrapolate or infer beyond what the sources explicitly state.
4. If the user's issue suggests an actionable step (refund, escalation, email), 
   propose the appropriate tool action.

Context:
{context}

Each source is labeled [N] with its title. Use these labels in your citations."""
```

---

## 7. Testing Strategy

### Unit Tests (Deterministic)

- `test_chunking.py` — Verify chunk sizes, overlap, metadata preservation
- `test_retriever.py` — Mock Chroma, verify query → chunk mapping
- `test_tools.py` — Verify tool schemas, parameter validation
- `test_citation_parsing.py` — Verify citation tag extraction from LLM output

### Integration Tests

- `test_graph.py` — End-to-end LangGraph flow with mocked LLM
- `test_api.py` — FastAPI endpoint tests (httpx TestClient)

### Evaluation Pipeline (Non-Deterministic)

- **Golden dataset:** 50+ queries with expected source articles
- **Citation accuracy:** % where cited source matches expected source
- **Task completion:** % where copilot produces usable draft
- **HITL coverage:** 100% of irreversible intents trigger approval (deterministic)
- **LLM-as-judge:** GPT-4 scores on relevance, groundedness, completeness (1-5)

### CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ruff
      - run: ruff check src/ tests/
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v
  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t fde-copilot .
```

---

## 8. Deployment

### Docker Compose (Local Development)

```yaml
# docker-compose.yml
version: "3.8"
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://backend:8000
    depends_on:
      - backend
```

### Environment Variables

```bash
# .env.example
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./data/chroma
LANGSMITH_API_KEY=ls-...          # Optional: observability
LANGSMITH_PROJECT=fde-copilot
LOG_LEVEL=INFO
```

---

## 9. Risks & Mitigations

| Risk                               | Likelihood | Impact | Mitigation                                                     |
| :--------------------------------- | :--------- | :----- | :------------------------------------------------------------- |
| OpenAI API rate limits during demo | Medium     | High   | Implement retry with exponential backoff; cache common queries |
| LLM hallucinating citations        | High       | High   | Post-processing validation; constrained output via Pydantic    |
| Chroma data loss (in-memory mode)  | Medium     | Medium | Enable persistent storage; document backup process             |
| 1-week timeline overrun            | Medium     | High   | P0 features only; cut metrics dashboard before cutting HITL    |
| Bitext dataset quality issues      | Low        | Medium | Manual review of 20 sample articles during ingestion           |
