# PRD: FDE Support Copilot

> **Version:** 2.1 | **Last Updated:** 2026-10-25 | **Status:** Shipped & Verified (`v1.1.0` — 94.2% Citation Accuracy, 96.6% Task Completion, 100% HITL Coverage, Tier B Empirical TTL)
> **References:** [SPEC.md](./SPEC.md) | [ADR.md](./ADR.md) | [DEVELOPMENT.md](./DEVELOPMENT.md) | [CHANGELOG.md](../CHANGELOG.md)

---

## Problem Statement

Support agents spend 30–40% of their time searching knowledge bases for answers before drafting replies. When they locate an article, there is no automated verification that the drafted reply is grounded in the authoritative source — leading to inconsistent quality and eroded customer trust. Furthermore, irreversible actions (refunds, ticket escalations, external customer emails) occur without deterministic approval gates or audit trails, creating financial and compliance risk.

**This copilot solves three problems simultaneously:**
1. Slow first-response time caused by manual knowledge base search (`14.0 min` baseline $\rightarrow$ `<4.0 min` with copilot).
2. Ungrounded replies that cannot be traced back to authoritative policy sources.
3. Unguarded irreversible actions executing without human-in-the-loop (HITL) supervision and immutable audit logging.

### Success Metrics & Verified Results

| Metric | Target | Verified Result (`v1.1.0`) | Measurement Method |
|:-------|:-------|:---------------------------|:-------------------|
| Citation Accuracy | $\ge 85.0\%$ | **94.2%** (`56/59`) | 59 annotated ground-truth query-source pairs (`data/eval/golden_test_set.json`) |
| Task Completion Rate | $\ge 70.0\%$ | **96.6%** (`57/59`) | 59 realistic support & adversarial scenarios |
| HITL Gate Coverage | $100.0\%$ | **100.0%** (`18/18`) | Deterministic verification — every irreversible action triggers `interrupt()` |
| False-Positive Interrupts | $\le 5.0\%$ | **0.0%** (`0/41`) | Informational queries pass uninterrupted |
| Time-to-First-Token / Turn | $< 3000\text{ ms}$ (p90) | **16.2 ms** (local) / **74.4 ms** (HTTP p90) | Request-level telemetry across 360+ timed turns |
| Response Quality | $\ge 3.50 / 5.0$ | **4.97 / 5.0** | Automated 4-dimension rubric + LLM-as-judge (`gpt-4o-mini`) |

---

## User Stories

### US-1: Citation-Grounded Draft Reply
**As a** support agent, **I want to** paste a customer query and receive a draft reply with inline citations, **so that** I can verify every factual claim against its source before sending.

**Acceptance Criteria:**
- Draft reply appears within 3 seconds of query submission.
- Every factual claim includes a 1-indexed citation tag `[N]` linking to the corresponding retrieved source article.
- Expanding a citation card reveals the source article ID, title, category, content excerpt, and cosine similarity score.
- Hallucinated out-of-bounds citation indices (e.g., `[0]`, `[99]`) are deterministically stripped before rendering.
- If no relevant source meets the minimum relevance threshold (`0.15`), the copilot explicitly abstains ("I don't have enough information in our knowledge base to answer this...") rather than fabricating policy details.

### US-2: Guarded Tool-Calling with Human Approval & Empirical TTL
**As a** support agent, **I want** the copilot to propose irreversible actions (`initiate_refund`, `escalate_ticket`, `send_followup_email`) **and pause for my explicit approval** before executing, **so that** I maintain control over customer-impacting decisions without deadlocking abandoned sessions.

**Acceptance Criteria:**
- When an actionable intent is detected, LangGraph pauses execution via `interrupt()` and renders an interactive approval gate card showing tool name, JSON parameters, rationale, and live TTL status.
- Supervisor can **Approve** (as-is), **Modify Parameters** (validated via strict Pydantic v2 schemas), or **Reject** (with mandatory audit reason).
- **Two-Tier Empirical TTL (`ADR-006` Tier B):**
  - **Soft TTL (`600s` / 10 min):** Marks pending proposal as `is_stale=True` (`status = "stale_escalated"`) and displays a supervisor escalation warning banner.
  - **Hard TTL (`1200s` / 20 min for synchronous `initiate_refund`; `3600s` / 60 min for `escalate_ticket` & `send_followup_email`):** Lazily expires abandoned gates on the next `POST /chat` or `GET /sessions/{id}`, resumes the paused LangGraph checkpoint with an expiry rejection, increments the isolated `tool_expired` counter via atomic SQLite Compare-And-Swap (`cursor.rowcount > 0`), and returns `HTTP 400 Bad Request` on late `/approve` or `/reject` attempts.

### US-3: Multi-Turn Conversation Context & Session Persistence
**As a** support agent, **I want** the copilot to maintain conversation context across multiple turns and browser refreshes, **so that** follow-up questions retain state.

**Acceptance Criteria:**
- Copilot retains conversation history within SQLite (`PRAGMA journal_mode=WAL`) and LangGraph checkpoints.
- Prompt assembly preserves a 10-turn verbatim sliding window (`MAX_VERBATIM_MESSAGES = 10`) and summarizes older turns to bound input token costs.
- Input sanitization strips credit card numbers, SSNs, emails, and phone numbers (`[REDACTED_*]`) prior to LLM invocation and blocks adversarial prompt injections (`OUT_OF_SCOPE` short-circuit).

### US-4: Adoption & FinOps Telemetry Dashboard
**As a** team lead, **I want to** monitor live adoption, quality, and HITL gate metrics, **so that** I can quantify operational ROI and audit tool decisions.

**Acceptance Criteria:**
- Dashboard tracks: total sessions, total queries, average latency, citation grounding rate, average citations per reply, and a 4-way HITL breakdown (`Proposed`, `Approved`, `Rejected`, `TTL Expired`).

---

## Scope

### Knowledge Base Grounding & Dataset Scope
The knowledge base is derived from the **Bitext Customer Support LLM Chatbot Training Dataset**, filtered to **6 core operational categories** and consolidated into 20 canonical articles (`data/knowledge_base/articles.json`) via query-enriched deduplication (**638.86x** projected token compression over 27,000 raw rows):
- **`BILLING`** $\leftarrow$ Bitext `PAYMENT` + `INVOICE`
- **`SHIPPING`** $\leftarrow$ Bitext `DELIVERY` + `SHIPPING_ADDRESS`
- **`CANCELLATION`** $\leftarrow$ Bitext `CANCELLATION_FEE` + `ORDER` (`cancel_order`)
- **`ORDER`** $\leftarrow$ Bitext `ORDER` (`place_order`, `change_order`, `track_order`)
- **`ACCOUNT`** $\leftarrow$ Bitext `ACCOUNT`
- **`REFUND`** $\leftarrow$ Bitext `REFUND`

### V1.1 Shipped Capabilities
- Query-enriched RAG ingestion & ChromaDB vector retrieval with category-filtered pre-pass (`0.22`) and global fallback (`0.15`).
- Dual-mode execution: live OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) when `OPENAI_API_KEY` is set, or zero-cost deterministic 384-d stem-canonicalized embeddings + grounded synthesis when offline.
- Guarded tool schemas (`escalate_ticket`, `initiate_refund`, `send_followup_email`) with LangGraph `interrupt()` / `Command(resume=...)` and Tier B Two-Tier Empirical TTL (`ADR-006`).
- FastAPI REST backend + Streamlit operator console + Docker Compose + GitHub Actions CI.
