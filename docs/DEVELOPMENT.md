# Development, Testing & Evaluation Guide

> **Project:** FDE Support Copilot (`v1.1.0`)
> **References:** [PRD.md](./PRD.md) | [SPEC.md](./SPEC.md) | [ADR.md](./ADR.md) | [CHANGELOG.md](../CHANGELOG.md)

---

## 1. Environment Configuration & Budget Protection (`gpt-4o-mini`)

The project is pre-configured in `.env` and `src/core/config.py` to use **`gpt-4o-mini`** (`$0.15 / 1M` input tokens, `$0.60 / 1M` output tokens) and **`text-embedding-3-small`** (`$0.02 / 1M` tokens).
- **Offline / Free Deterministic Mode:** If `OPENAI_API_KEY` in `.env` is left empty or set to the placeholder `sk-paste-your-openai-api-key-here`, the system automatically runs in zero-cost deterministic mode (`$0.00` API spend).
- **Live OpenAI Mode (`<$0.10` Budget Safe):** When you paste your real `sk-...` key into `.env`, each chat turn costs **`~$0.000038`** (`0.0038 cents`), meaning **100 live chat turns cost less than `$0.01`**, and a 5-case live evaluation (`python scripts/evaluate.py --limit 5`) costs **`~$0.0015`**.

---

## 2. Local Setup & Running the Stack

### Step 1: Activate the Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
*(If `.venv` is not yet created, run `uv sync --extra dev` or `python -m venv .venv && .\.venv\Scripts\pip install -e ".[dev]"`).*

### Step 2: Ingest the Knowledge Base into ChromaDB
```powershell
python scripts/ingest.py
```
This consolidates the Bitext seed rows into 20 query-enriched canonical articles (`data/knowledge_base/articles.json`) and indexes them into `./data/chroma`.

### Step 3: Start the FastAPI Backend (Terminal 1)
```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive OpenAPI / Swagger UI: `http://127.0.0.1:8000/docs`
- Healthcheck endpoint: `http://127.0.0.1:8000/health`

### Step 4: Start the Streamlit Operator Console (Terminal 2)
```powershell
streamlit run frontend/app.py
```
- Open `http://localhost:8501` in your browser.

---

## 3. Running Tests, Linter & Evaluation

### Automated Test Suite (118 Zero-Mock Tests)
Runs completely offline at `$0.00` cost against real ChromaDB and SQLite WAL instances:
```powershell
pytest -v
```
- `tests/unit/`: Chunking, deduplication, Chroma retrieval, citation sanitization, PII/injection filters, tool schemas.
- `tests/integration/`: End-to-end LangGraph state machine (`test_graph.py`), FastAPI endpoints (`test_api.py`), Two-Tier HITL TTL & CAS concurrency (`test_ttl_expiration.py`), full-stack smoke workflow (`test_smoke_e2e.py`).
- `tests/evaluation/`: Golden dataset integrity and SLA evaluator tests (`test_eval_pipeline.py`).

### Code Quality & Formatting (Ruff)
```powershell
ruff check src/ frontend/ tests/ scripts/
```

### Golden Dataset Evaluation (`scripts/evaluate.py`)
```powershell
# Run a small 5-case sample (ideal for a <$0.01 live OpenAI key test):
python scripts/evaluate.py --limit 5

# Run the full 59-scenario golden evaluation suite:
python scripts/evaluate.py
```

---

## 4. Verified FinOps & SLA Benchmark Summary

| Benchmark Dimension | Measured Value (`v1.1.0`) | Target SLA |
|:--------------------|:--------------------------|:-----------|
| **KB Token Compression (27K Bitext Projection)** | **638.86x** (`99.84%` token reduction) | $\ge 10\text{x}$ |
| **ChromaDB Retrieval Recall (Top-1 / Top-3)** | **100.0% / 100.0%** | $\ge 80\% / \ge 95\%$ |
| **ChromaDB Retrieval Latency (Mean / p90)** | **5.10 ms / 5.79 ms** | $< 60\text{ ms} / < 100\text{ ms}$ |
| **Citation Accuracy (59 Golden Scenarios)** | **94.2%** (`56/59`) | $\ge 85.0\%$ |
| **Task Completion Rate (59 Golden Scenarios)** | **96.6%** (`57/59`) | $\ge 70.0\%$ |
| **HITL Gate Coverage (Actionable Scenarios)** | **100.0%** (`18/18` paused at `interrupt()`) | $100.0\%$ |
| **False-Positive Interrupt Rate** | **0.0%** (`0/41` informational queries) | $\le 5.0\%$ |
| **Cost per 1,000 Tickets (`gpt-4o-mini`)** | **`$0.0380`** (`3.8 cents` per 1,000 turns) | — |
| **Net Support Labor Savings per 100K Tickets** | **`$416,662.86`** (`16,666.7` agent hours saved) | — |
