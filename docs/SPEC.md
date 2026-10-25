# SPEC: FDE Support Copilot — Architecture & Technical Specification

> **Version:** 2.1 | **Last Updated:** 2026-10-25 | **Status:** Production Verified (`118/118` Zero-Mock Tests Passing, `ADR-006` Tier B TTL + SQLite WAL Shipped)
> **References:** [PRD.md](./PRD.md) | [ADR.md](./ADR.md) | [DEVELOPMENT.md](./DEVELOPMENT.md) | [CHANGELOG.md](../CHANGELOG.md)

---

## 1. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Streamlit :8501)                    │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ Chat UI  │  │ Citation   │  │ Approval Gate UI │  │ Metrics &  │  │
│  │          │  │ Cards [N]  │  │ + Soft/Hard TTL  │  │ Audit Log  │  │
│  └────┬─────┘  └────────────┘  └────────┬─────────┘  └────────────┘  │
│       └─────────────────────────────────┼────────────────────────────┘
│                            REST API (HTTP JSON)
┌─────────────────────────────────────────▼────────────────────────────┐
│                         BACKEND (FastAPI :8000)                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  POST /chat │ POST /approve │ POST /reject │ GET /sessions/{id}│  │
│  │  GET /metrics │ GET /health                                    │  │
│  └──────────────────────────────┬─────────────────────────────────┘  │
│                                 │                                    │
│  ┌──────────────────────────────▼─────────────────────────────────┐  │
│  │                 ORCHESTRATION (LangGraph)                      │  │
│  │  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐  │  │
│  │  │ classify_intent │──▶│ retrieve_context │──▶│  generate   │  │  │
│  │  │ + PII / Inject  │   │ (Cat->Global FB) │   │  _response  │  │  │
│  │  └─────────────────┘   └──────────────────┘   └──────┬──────┘  │  │
│  │                                                      │         │  │
│  │                                       ┌──────────────▼──────┐  │  │
│  │                                       │ route_tool_decision │  │  │
│  │                                       └───┬─────────────┬───┘  │  │
│  │                                 no_tool   │             │      │  │
│  │                                ┌──────────▼┐   ┌────────▼────┐ │  │
│  │                                │    END    │   │propose_tool │ │  │
│  │                                └───────────┘   │ interrupt() │ │  │
│  │                                                └──┬───────┬──┘ │  │
│  │                                          rejected │       │    │  │
│  │                                           ┌───────▼┐ ┌────▼──┐ │  │
│  │                                           │  END   │ │execute│ │  │
│  │                                           └────────┘ │ _tool │ │  │
│  │                                                      └───────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │
│  │ ChromaDB Store   │  │ SQLite (WAL Mode)  │  │ LangGraph Saver  │  │
│  │ 20 KB Articles   │  │ Sessions, Metrics, │  │ Thread State     │  │
│  │ Cosine Similarity│  │ CAS TTL Audit Log  │  │ Checkpoints      │  │
│  └──────────────────┘  └────────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
fde-copilot/
├── .github/workflows/ci.yml          # Ruff lint + 118 pytest + Golden Eval + Docker build
├── docs/
│   ├── PRD.md                        # Product Requirements & SLA targets
│   ├── SPEC.md                       # System architecture & API specification
│   ├── ADR.md                        # Architecture Decision Records (ADR-001 to ADR-006)
│   └── DEVELOPMENT.md                # Local setup, testing, evaluation & FinOps guide
├── src/
│   ├── api/
│   │   ├── main.py                   # FastAPI app factory & lifespan initialization
│   │   ├── dependencies.py           # Singleton DI for Graph, Chroma, Session & Metrics
│   │   ├── routers/
│   │   │   ├── chat.py               # POST /chat, /approve, /reject, GET /sessions/{id}
│   │   │   ├── metrics.py            # GET /metrics
│   │   │   └── health.py             # GET /health
│   │   └── schemas/                  # Strict Pydantic v2 HTTP request/response contracts
│   ├── core/
│   │   ├── config.py                 # Settings (defaults to gpt-4o-mini & text-embedding-3-small)
│   │   └── exceptions.py             # Custom domain exceptions (ApprovalGateError, etc.)
│   ├── agent/
│   │   ├── graph.py                  # LangGraph state machine & invoke/resume helpers
│   │   ├── state.py                  # CopilotState TypedDict schema
│   │   ├── nodes/                    # classify, retrieve, generate, tools (propose/execute)
│   │   ├── tools/                    # escalate_ticket, initiate_refund, send_followup_email
│   │   └── prompts/                  # System prompts, PII redaction, injection detection
│   ├── rag/
│   │   ├── chunking.py               # Bitext 6-category normalization & query-enriched dedup
│   │   ├── ingest.py                 # Seed dataset -> canonical KB articles -> ChromaDB
│   │   ├── retriever.py              # Category pre-pass, global fallback & citation sanitizer
│   │   └── store.py                  # ChromaVectorStore + OpenAI & Deterministic 384-d embeddings
│   ├── services/
│   │   ├── session.py                # SQLite WAL session store + Two-Tier TTL + CAS expiry
│   │   └── metrics.py                # SQLite WAL telemetry & tool_expired counter
│   └── evaluation/
│       ├── dataset.py                # Golden dataset loader (59 scenarios)
│       ├── evaluator.py              # Zero-mock RAG + HITL + LLM-judge evaluator
│       └── report.py                 # JSON & Markdown report formatter
├── frontend/
│   ├── app.py                        # Streamlit operator workspace
│   ├── utils.py                      # Backend HTTP client
│   └── components/                   # chat.py, citations.py, approval.py, dashboard.py
├── data/
│   ├── knowledge_base/articles.json  # 20 consolidated query-enriched KB articles
│   └── eval/                         # golden_test_set.json & evaluation_report.json
├── scripts/
│   ├── ingest.py                     # CLI to ingest KB into ChromaDB
│   ├── evaluate.py                   # CLI to run golden evaluation (--limit supported)
│   └── benchmark_finops.py           # CLI for FinOps & latency benchmarking
├── tests/                            # 118 zero-mock unit, integration, TTL, & eval tests
├── CHANGELOG.md                      # Version history (v0.1.0 -> v1.1.0)
├── Dockerfile                        # Multi-stage backend image
├── Dockerfile.frontend               # Streamlit frontend image
├── docker-compose.yml                # Full-stack local orchestration
└── pyproject.toml                    # Dependencies & tool configuration
```

---

## 3. REST API Specification

### `POST /chat`
Processes a customer message through the LangGraph pipeline (`classify_intent` $\rightarrow$ `retrieve_context` $\rightarrow$ `generate_response` $\rightarrow$ conditional `propose_tool`).
- **Active Gate Guard:** If the session has an active (`< Hard TTL`) pending tool gate, returns `HTTP 409 Conflict`.
- **Lazy Hard-TTL Auto-Unblock (`ADR-006`):** If a pending tool gate has exceeded its Hard TTL (`1200s` for `initiate_refund`, `3600s` for `escalate_ticket` / `send_followup_email`), `/chat` atomically transitions the gate to `expired` in SQLite, increments `tool_expired`, resumes the paused LangGraph checkpoint with an expiry rejection, and processes the new chat message seamlessly.

### `POST /approve`
Resumes an interrupted LangGraph session (`Command(resume={"approved": True, "modification": ...})`).
- Supports optional supervisor parameter modifications validated against the tool's Pydantic v2 schema (`HTTP 422` on invalid parameters without double-counting telemetry).
- If the proposal has exceeded its Hard TTL, transitions the gate to `expired`, clears the checkpoint, and returns `HTTP 400 Bad Request` (`Tool proposal ... has expired`).

### `POST /reject`
Resumes an interrupted LangGraph session (`Command(resume={"approved": False, "reason": ...})`) and logs the supervisor's rejection reason in `tool_audit_log`. Returns `HTTP 400 Bad Request` if the gate has already expired.

### `GET /sessions/{session_id}`
Hydrates full conversation history, inline citations, and any pending `ToolProposalPayload` (including computed `age_seconds`, `ttl_Remaining_seconds`, `is_stale`, and `is_expired` fields). Lazily expires overdue proposals on read.

### `GET /metrics`
Returns real-time system telemetry (`SystemMetricsResponse`):
- `total_sessions`, `total_queries`, `avg_response_time_ms`
- `citation_rate`, `avg_citations_per_response`
- `tool_proposals`, `tool_approvals`, `tool_rejections`, `tool_expired`

### `GET /health`
Returns service readiness (`status`, `vector_store`, `llm_provider`, `uptime_seconds`).

---

## 4. Core Engineering Mechanisms

### 4.1 Two-Stage Category Retrieval & Deterministic Citation Sanitization
1. **Stage 1 (Category Pre-Pass):** Queries ChromaDB with `where={"category": intent}` and `min_score=0.22` (`relevance_score = max(0.0, 1.0 - cosine_distance / 2.0)`).
2. **Stage 2 (Global Fallback):** If Stage 1 yields zero chunks (e.g., cross-category phrasing), retries across all categories with `min_score=0.15`.
3. **Citation Sanitizer (`validate_and_clean_citations`):** Regex-scans `[N]` tags in the draft reply, retains only valid `1 <= N <= len(retrieved_chunks)` indices, strips hallucinated `[0]`/`[99]` tags, repairs orphaned punctuation, and returns the exact subset of cited articles.

### 4.2 Guarded Tool-Calling & Two-Tier Empirical TTL (`ADR-006` Tier B)
| Tool Name | Action Risk Profile | Soft TTL (Stale Warning) | Hard TTL (Auto-Expire) | Audit Prefix |
|:----------|:--------------------|:-------------------------|:-----------------------|:-------------|
| `initiate_refund` | Immediate financial payout (`<= $10,000`) | `600s` (10 min) | `1200s` (20 min) | `RFN-*` |
| `escalate_ticket` | Internal Tier-2/Tier-3 queue transfer | `600s` (10 min) | `3600s` (60 min) | `ESC-*` |
| `send_followup_email` | Asynchronous external customer dispatch | `600s` (10 min) | `3600s` (60 min) | `EML-*` |

### 4.3 SQLite WAL Concurrency & Parameterized DB-API Protection
Both `SessionService` and `MetricsService` initialize SQLite connections via a context manager that enforces:
- `PRAGMA journal_mode=WAL;` (allows concurrent readers alongside a writer)
- `PRAGMA busy_timeout=5000;` (prevents immediate `sqlite3.OperationalError: database is locked` under concurrent requests)
- `PRAGMA synchronous=NORMAL;`
- Strict parameterized `?` placeholders via Python's `sqlite3.Cursor.execute(sql, params)` (C-level single-statement execution preventing stacked queries and SQL injection).
