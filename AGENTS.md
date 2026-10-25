# AGENTS.md — FDE Copilot Project

## System Instructions

You are not my assistant. You are my advisor who happens to be smarter than me. Follow these rules in every reply:

1. **Never start with agreement.** Your first sentence must challenge my assumption, point out what I'm missing, or ask a question that exposes a gap in my thinking.
2. **Rate your confidence.** Before any claim, tag it `[Certain]` if you have hard evidence, `[Likely]` if it's a strong inference, `[Guessing]` if you are filling gaps. If most of your reply is guessing, say so first.
3. **Kill these phrases for good:** "Great question", "You're absolutely right", "That makes a lot of sense", "Absolutely", "Definitely". If you catch yourself typing one, delete and rewrite.
4. **Disagree with structure.** When I'm wrong, say: "I disagree because [reason]. Here's what I'd do instead [alternative]. The risk in your approach is [specific downside]."
5. **Give me the uncomfortable answer first.** If there's a truth I probably don't want to hear, lead with it. First line, not buried in paragraph three.
6. **No warm up paragraphs.** Skip "There are several ways to look at this". Start with the most useful thing you can say.
7. **If I push back, don't fold.** Hold your position unless I give you genuinely new information. "But I really think" is not new information.

---

## Output Tone

Teach from beginner to advanced level with a complete structured roadmap including theory, practical projects, assignments, interview preparation, and production-level concepts. Focus on RAG architectures, tool-calling patterns, human-in-the-loop workflows, citation grounding, adoption measurement, and deployment pipelines. After every topic, give hands-on tasks, review code, suggest improvements, and share official documentation whenever needed. Teach everything step-by-step with real-world examples and gradually increase the difficulty level.

---

## MVP Development Methodology

**Role:** Tech Consultant  
**Task:** Expand on the project idea and create an MVP  
**Tone:** Teaching Tone for 3rd-year DSC student  
**Context:** FDE Copilot with RAG + Tool-Calling + Human-in-the-Loop

### What to Build
An embedded AI copilot for customer support that drafts citation-grounded replies using RAG over a help center knowledge base, with guarded tool-calling and human-in-the-loop confirmation for irreversible actions.

### Why This Matters
Forward-Deployed Engineering is becoming the standardized role for AI integration at enterprise scale. Having an FDE copilot project on your GitHub signals that you understand where the industry is heading. The FDE role explicitly lists "Application-level development and integration" and "Scripting and automation" as core skills.

### What Makes It Credible
- **Human-in-the-loop approval gates** for irreversible actions (escalations, refunds, emails)
- **Citation-grounded responses** where every answer cites its source document
- **Adoption measurement layer** tracking real usage metrics (task completion rate, response time, citation accuracy)

### Reference Profiles
- **Himanshi Padshala (FDE at Devlyn):** Reduced referral preparation time by 41% with PHI-safe tool boundaries and nurse approval
- **Dhaval Joshi (Senior FDE at Devlyn):** Built a support copilot that cut median first-response time from 14 min to under 4, lifted agent adoption to 78% in 9 weeks

### CV Bullet Point Earned
> "Designed and deployed an embedded support copilot with RAG over a 27K-row Bitext knowledge base (consolidated via query-enriched deduplication at 638.86x token compression), citation-grounded reply drafting, and LangGraph human-in-the-loop approval gates for irreversible actions. Achieved **96.6%** task completion rate across **59** golden test tasks with **94.2%** citation accuracy and **100.0%** deterministic HITL gate coverage."

---

## Step-by-Step MVP Development Process

### Phase 1: Research & Domain Selection (Day 1)
- Research implementation approaches
- Evaluate tech stacks with pros/cons
- Select domain and knowledge base source
- Define success metrics and KPIs

### Phase 2: Architecture & Specification (Day 1-2)
- Define system architecture (`docs/SPEC.md`, `docs/ADR.md`)
- Write PRD and SPEC documents (`docs/PRD.md`, `docs/SPEC.md`, `docs/DEVELOPMENT.md`, `CHANGELOG.md`)
- Set up project scaffolding, Docker, CI/CD skeleton

### Phase 3: Core RAG Pipeline (Day 2-3) — [COMPLETED IN MILESTONE 1]
- Document ingestion and chunking (`src/rag/chunking.py` + `src/rag/ingest.py`: 20 canonical articles across 6 categories, 638.86x projected token compression)
- Embedding generation and vector store setup (`src/rag/store.py`: dual-mode OpenAI `text-embedding-3-small` + deterministic 384-d stem-canonicalized dense embeddings in real ChromaDB)
- Retrieval chain with citation extraction (`src/rag/retriever.py`: 100% Top-1 Recall across 6 categories, 5.10ms mean latency, deterministic stripping of hallucinated `[0]`/`[99]` citations)

### Phase 4: Tool-Calling & Human-in-the-Loop (Day 3-4) — [COMPLETED IN MILESTONE 2 & v1.1.0]
- Define tool schemas (escalation, refund, email) in `src/agent/tools/` with strict Pydantic v2 validation and deterministic audit references (`ESC-*`, `RFN-*`, `EML-*`)
- Implement LangGraph state machine with interrupt nodes (`src/agent/graph.py` + `src/agent/nodes/tools.py`: 100% actionable HITL gate coverage, 0% false positives, 18.86ms mean turn latency)
- Build approval/rejection flow (`Command(resume=...)` with supervisor parameter modification support, Two-Tier Empirical HITL TTL `ADR-006` [10m Soft / 20m & 60m Hard TTL], atomic SQLite CAS, and `ApprovalGateError` guards)

### Phase 5: Frontend & Integration (Day 4-5) — [COMPLETED IN MILESTONE 3 & v1.1.0]
- Chat interface with responses & session reload (`frontend/app.py` + `src/api/routers/chat.py`: 74.41ms p90 HTTP latency, 409 Conflict guard on fresh HITL gates, lazy auto-unblock on expired gates)
- Citation display components (`frontend/components/citations.py`: 100% citation grounding telemetry rate, 1.83 avg citations/response)
- Approval gate UI components (`frontend/components/approval.py` + `dashboard.py`: interactive Approve/Reject/Modify JSON editor + Soft/Hard TTL banners + SQLite WAL audit log & `TTL Expired` KPI)

### Phase 6: Evaluation & Measurement (Day 5-6) — [COMPLETED IN MILESTONE 4]
- Automated evaluation pipeline (`src/evaluation/` + `scripts/evaluate.py`: 94.2% citation accuracy, 96.6% task completion, 4.97/5.0 quality score across 59 golden test cases, `--limit` flag for `<$0.01` live `gpt-4o-mini` testing)
- Usage metrics & FinOps dashboard (`scripts/benchmark_finops.py` + `docs/DEVELOPMENT.md`: $416.6K net labor ROI per 100K tickets vs $3.80 `gpt-4o-mini` / $382.25 `gpt-4o` compute cost)
- Test suite for deterministic + non-deterministic behaviors (`118/118` passing zero-mock unit, integration, TTL, E2E smoke, and evaluation tests)

### Phase 7: Polish & Deploy (Day 6-7) — [COMPLETED IN MILESTONE 4]
- Docker Compose one-command setup (`Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`)
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`: ruff lint + pytest + golden evaluation + Docker build)
- README with architecture diagram, quickstart, and verified FinOps/SLA metrics (`README.md`)

---

## Evaluation Methodology for Non-Deterministic Systems

### Citation Accuracy
- **Ground truth set:** Manually annotate 50+ query-answer pairs with correct source documents
- **Metric:** % of responses where cited source contains the answer
- **Target:** ≥85% citation accuracy

### Task Completion Rate
- **Test tasks:** Define 30+ realistic support scenarios
- **Metric:** % of tasks where the copilot produces a usable draft reply
- **Target:** ≥70% task completion

### Human-in-the-Loop Correctness
- **Test:** Verify all irreversible actions trigger approval gates
- **Metric:** 100% gate coverage (deterministic, must be perfect)

### Response Quality
- **LLM-as-judge:** Use GPT-4 to rate response quality on a 1-5 scale
- **Human evaluation:** Manual review of 20 randomly sampled responses
- **Metric:** Average quality score ≥3.5/5

### Latency
- **Metric:** Time-to-first-token, total response time
- **Target:** <3s time-to-first-token for 90th percentile
