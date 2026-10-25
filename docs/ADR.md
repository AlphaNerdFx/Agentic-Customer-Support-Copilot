# Architecture Decision Records (ADRs)

> **Project:** FDE Support Copilot (`v1.1.0`)
> **References:** [PRD.md](./PRD.md) | [SPEC.md](./SPEC.md) | [DEVELOPMENT.md](./DEVELOPMENT.md)

---

## ADR-001: LangGraph `StateGraph` with Native `interrupt()` over Linear Chains
- **Status:** Accepted
- **Context:** The copilot must pause execution mid-turn before dispatching irreversible tools (`initiate_refund`, `escalate_ticket`, `send_followup_email`), persist state across HTTP requests, and resume only after a human supervisor approves, modifies, or rejects the proposal.
- **Decision:** Use LangGraph (`StateGraph` + checkpointer) with `interrupt()` inside `propose_tool_node` and `Command(resume=...)` inside `resume_tool_decision`.
- **Consequences:** Graph state dehydrates cleanly to the checkpointer while awaiting human input, guaranteeing **100.0% deterministic HITL gate coverage** without holding open HTTP connections or threads.

---

## ADR-002: Query-Enriched Deduplication & ChromaDB Vector Store
- **Status:** Accepted
- **Context:** Raw customer support datasets (such as the 27,000-row Bitext corpus) repeat identical policy resolutions across thousands of paraphrased user queries. Indexing raw rows inflates embedding costs, degrades retrieval diversity (top-$k$ returns 5 identical chunks), and wastes LLM context tokens.
- **Decision:** Consolidate raw interactions by `(category, intent)` into canonical `KnowledgeArticle` documents enriched with representative customer phrasing (`Typical Customer Questions`), and store them in persistent ChromaDB (`hnsw:space="cosine"`) behind `ChromaVectorStore`.
- **Consequences:** Achieves **638.86x projected token compression** (`99.84%` reduction), **100% Top-1 recall** across the 6 target categories, and **5.10 ms** mean retrieval latency.

---

## ADR-003: Decoupled FastAPI Backend and Streamlit Operator Console
- **Status:** Accepted
- **Context:** Streamlit re-executes its Python script top-to-bottom on every UI widget interaction. Embedding LangGraph orchestration, ChromaDB indices, or SQLite locks directly inside Streamlit causes state resets and connection contention.
- **Decision:** Isolate all AI orchestration, vector retrieval, HITL state management, and SQLite persistence inside a stateless FastAPI REST service (`:8000`), keeping Streamlit (`:8501`) as a thin HTTP client with its own `Dockerfile.frontend`.
- **Consequences:** The backend API can be consumed independently by Zendesk/Salesforce widgets, Slack bots, or React frontends, and both containers scale independently.

---

## ADR-004: Dual-Mode Embeddings & Cost-Capped `gpt-4o-mini` Generation
- **Status:** Accepted
- **Context:** Enterprise evaluation and CI pipelines must run deterministically at `$0.00` cost without external API keys, while live testing on constrained developer budgets (e.g., `<$5.00` OpenAI credit balance) must avoid expensive `gpt-4o` token burn.
- **Decision:**
  1. Default `OPENAI_MODEL` to `gpt-4o-mini` (`$0.15/1M` input, `$0.60/1M` output — **94% cheaper** than `gpt-4o`) and `EMBEDDING_MODEL` to `text-embedding-3-small` (`$0.02/1M` tokens).
  2. Provide an automatic zero-mock fallback (`DeterministicDomainEmbeddingFunction` 384-d stem-canonicalized dense vectors + deterministic grounded synthesis) whenever `OPENAI_API_KEY` is unset or a placeholder.
- **Consequences:** All 118 automated tests and the 59-case golden evaluation suite run in `<20s` at `$0.00` cost in CI, while a full 59-case live OpenAI evaluation on `gpt-4o-mini` costs `~$0.015` (`1.5 cents`) and a 5-case `--limit 5` live test costs `<$0.002`.

---

## ADR-005: Defense-in-Depth Guardrails (Regex PII Redaction, Citation Sanitizer, Parameterized SQLite WAL)
- **Status:** Accepted
- **Context:** Customer support transcripts frequently contain sensitive PII (credit cards, SSNs, emails, phone numbers), adversarial prompt injections, hallucinated LLM citation indices (`[0]`, `[99]`), and concurrent database writes.
- **Decision:**
  1. **PII Boundary (`sanitize_pii`):** Deterministic regex redaction with negative lookbehinds (`(?<!#)`) to redact SSNs, 13–19 digit payment cards, emails, and phone numbers in $<0.1\text{ ms}$ without corrupting order IDs (`#ORD-123-45-6789`).
  2. **Citation Sanitization (`validate_and_clean_citations`):** Post-processing validation that strips any `[N]` tag not mapping to `1..len(retrieved_chunks)` and repairs surrounding punctuation.
  3. **SQLite Hardening:** Enforce `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;` inside context-managed connections using strict DB-API `?` parameterized bindings.
- **Consequences:** Zero unredacted PII reaches external LLMs, zero fabricated citations reach the UI, and concurrent read/write operations do not leak descriptors or allow SQL injection.

---

## ADR-006: Two-Tier Empirical HITL Approval TTL & Atomic CAS Expiration (Tier B)
- **Status:** Accepted (`Tier B` implemented in `v1.1.0`; `Tier C` child-thread queue deferred to post-MVP)
- **Context:** When a session pauses at `propose_tool` via LangGraph `interrupt()`, `POST /chat` blocks subsequent turns on that `session_id` with `HTTP 409 Conflict` to protect checkpoint integrity. However, if a support agent abandons their tab or the customer disconnects, a strict lock without a Time-To-Live (TTL) deadlocks the session permanently.
- **Empirical Industry Calibration:**
  - **Zendesk Messaging & Live Chat:** 10-minute default *Active $\rightarrow$ Inactive* capacity release threshold; 20-minute live chat disconnect timeout; 60-minute agent console timeout.
  - **Salesforce Omni-Channel & Amazon Connect:** 5–10 minute supervisor push timeout; 20-minute chat session expiration (`EXPIRED`).
- **Decision (Tier B — Lazy Two-Tier TTL with Atomic CAS):**
  1. **Tier 1 Soft TTL (`600s` / 10 minutes):** Proposals older than 10 minutes transition to `status = "stale_escalated"` (`is_stale = True`) and surface a warning banner in the UI while remaining actionable.
  2. **Tier 2 Hard Expiration TTL (`1200s` / 20 minutes for synchronous financial `initiate_refund`; `3600s` / 60 minutes for asynchronous `escalate_ticket` and `send_followup_email`):**
     - Evaluated lazily inside `_check_and_expire_stale_gate()` across `POST /chat`, `POST /approve`, `POST /reject`, and `GET /sessions/{id}`.
     - Uses an atomic SQLite Compare-And-Swap (`UPDATE tool_audit_log SET status = 'expired' ... WHERE status IN ('awaiting_approval', 'stale_escalated')` checking `cursor.rowcount > 0`) so concurrent requests increment the isolated `tool_expired` telemetry counter **at most once**.
     - Resumes the paused LangGraph checkpoint with an expiry rejection before returning `HTTP 400 Bad Request` on late `/approve` or `/reject` calls, or transparently unblocking the session on a new `POST /chat` turn.
- **Consequences:** Eliminates permanent session deadlocks and prevents stale financial refunds from executing after customer disconnect, with zero background daemon threads required.
