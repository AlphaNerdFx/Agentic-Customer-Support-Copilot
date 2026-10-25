# Changelog

All notable service, architectural, and operational changes to the **FDE Support Copilot** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-10-25

### Added
- **Two-Tier Empirical HITL Approval TTL (`ADR-006` Tier B):**
  - Added `SOFT_TTL_SECONDS = 600` (10-minute supervisor stale warning threshold, calibrated to Zendesk Messaging capacity release) and `HARD_TTL_BY_TOOL` (`1200s` / 20 minutes for synchronous financial `initiate_refund`; `3600s` / 60 minutes for asynchronous `escalate_ticket` and `send_followup_email`).
  - Added lazy expiration evaluation (`_check_and_expire_stale_gate`) across `POST /chat`, `POST /approve`, `POST /reject`, and `GET /sessions/{session_id}` to automatically unblock abandoned LangGraph `interrupt()` checkpoints without background daemon threads.
  - Added atomic Compare-And-Swap (`cursor.rowcount > 0`) in `SessionService.expire_tool_proposal()` and an isolated `tool_expired` telemetry counter in `MetricsService` and `SystemMetricsResponse`.
  - Added live TTL status banners, remaining countdowns, and expired gate locking in the Streamlit approval UI (`frontend/components/approval.py`) and a dedicated `TTL Expired` KPI metric in `frontend/components/dashboard.py`.
  - Added 15 integration tests in `tests/integration/test_ttl_expiration.py` verifying Soft TTL warnings, Hard TTL auto-unblocking on `POST /chat`, `400 Bad Request` guards on late `/approve` and `/reject`, per-tool differential timeouts, and single-increment CAS idempotency (`118/118` total tests passing).
- **Small-Batch Live Evaluation Flag (`--limit`):**
  - Added `--limit N` CLI argument to `scripts/evaluate.py` to support low-cost (`<$0.01`) live OpenAI test runs over a subset of the golden dataset.

### Changed
- **SQLite Concurrency & Descriptor Hardening (`src/services/session.py`, `src/services/metrics.py`):**
  - Replaced raw `with sqlite3.connect(...)` blocks (which only commit/rollback transactions without closing file descriptors) with `@contextlib.contextmanager` connection wrappers that explicitly `conn.close()` in `finally:` blocks.
  - Enabled `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL;` across all SQLite connections.
- **Cost-Protected Default Model (`src/core/config.py`, `.env.example`, `.env`):**
  - Updated default `OPENAI_MODEL` from `gpt-4o` to `gpt-4o-mini` (`$0.15/1M` input, `$0.60/1M` output — 94% cost reduction) to prevent accidental credit exhaustion during live API testing.
- **Documentation Consolidation (`docs/`):**
  - Consolidated project documentation into `docs/PRD.md`, `docs/SPEC.md`, `docs/ADR.md`, and `docs/DEVELOPMENT.md`, and removed redundant scratch/teaching files.

---

## [1.0.0] - 2026-10-25

### Added
- **Core RAG Pipeline (`src/rag/`, `scripts/ingest.py`):**
  - Implemented Bitext 6-category normalization (`BILLING`, `SHIPPING`, `CANCELLATION`, `ORDER`, `ACCOUNT`, `REFUND`) and query-enriched deduplication (`src/rag/chunking.py`) achieving `638.86x` projected token compression (`20` canonical articles in `data/knowledge_base/articles.json`).
  - Implemented `ChromaVectorStore` (`src/rag/store.py`) with dual-mode `OpenAIEmbeddingAdapter` (`text-embedding-3-small`) and 384-d stem-canonicalized `DeterministicDomainEmbeddingFunction`.
  - Implemented `RAGRetriever` (`src/rag/retriever.py`) with two-stage category-filtered pre-pass (`0.22`) and global fallback (`0.15`), plus deterministic `validate_and_clean_citations` stripping hallucinated `[0]`/`[99]` tags (`100%` Top-1 recall, `5.10 ms` mean latency).
- **Guarded Agent & LangGraph HITL State Machine (`src/agent/`):**
  - Implemented strict Pydantic v2 tool schemas (`initiate_refund`, `escalate_ticket`, `send_followup_email`) with deterministic audit references (`RFN-*`, `ESC-*`, `EML-*`).
  - Implemented regex PII redaction (`sanitize_pii` with negative lookbehinds `(?<!#)`) and adversarial prompt injection detection (`detect_prompt_injection`) in `src/agent/prompts/templates.py`.
  - Implemented 5-node LangGraph `StateGraph` (`classify_intent` $\rightarrow$ `retrieve_context` $\rightarrow$ `generate_response` $\rightarrow$ `propose_tool` with `interrupt()` $\rightarrow$ `execute_tool`) and `Command(resume=...)` with supervisor parameter modification support.
- **FastAPI Backend & Streamlit Operator Console (`src/api/`, `frontend/`):**
  - Implemented REST endpoints (`POST /chat`, `POST /approve`, `POST /reject`, `GET /sessions/{session_id}`, `GET /metrics`, `GET /health`) with `HTTP 409 Conflict` active gate protection and `HTTP 422` parameter validation.
  - Built Streamlit interface with inline citation expanders, interactive Approve/Modify/Reject cards, and telemetry dashboard.
- **Evaluation Engine, FinOps Benchmarks & DevOps (`src/evaluation/`, `scripts/`, `Dockerfile`, `.github/workflows/ci.yml`):**
  - Curated 59-scenario golden evaluation dataset (`data/eval/golden_test_set.json`) and zero-mock evaluator achieving **94.2%** citation accuracy, **96.6%** task completion, **100.0%** HITL gate coverage, and **4.97/5.0** quality score.
  - Added multi-stage `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, and GitHub Actions CI workflow.

---

## [0.1.0] - 2026-10-24

### Added
- Initial repository scaffold, package layout, and locked dependency graph (`pyproject.toml`, `uv.lock`).
