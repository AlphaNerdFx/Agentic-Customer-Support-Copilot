# PRD: FDE Support Copilot

> **Version:** 1.0 | **Last Updated:** 2026-10-19 | **Status:** Draft

---

## Problem Statement

Support agents spend 30-40% of their time searching knowledge bases for answers before they can even begin drafting replies. When they find the right article, there's no way to verify the reply is grounded in the actual source — leading to inconsistent quality and eroded customer trust. Irreversible actions (refunds, escalations, account changes) happen without systematic approval gates, creating compliance risk.

**This copilot solves three problems simultaneously:**
1. Slow first-response time caused by manual knowledge base search
2. Ungrounded replies that can't be traced back to authoritative sources
3. Unguarded irreversible actions with no audit trail

### Success Metrics

| Metric | Target | Measurement Method |
|:-------|:-------|:-------------------|
| Citation Accuracy | ≥85% | 50+ annotated ground truth query-source pairs |
| Task Completion Rate | ≥70% | 30+ realistic support scenarios |
| HITL Gate Coverage | 100% | Deterministic test — every irreversible action triggers approval |
| Time-to-First-Token | <3s (p90) | Request-level latency logging |
| Response Quality (LLM-judge) | ≥3.5/5 | GPT-4 evaluation on relevance, groundedness, completeness, tone |

---

## User Stories

### US-1: Citation-Grounded Draft Reply
**As a** support agent, **I want to** paste a customer query and receive a draft reply with inline citations, **so that** I can verify every claim against its source before sending.

**Acceptance Criteria:**
- Draft reply appears within 3 seconds of query submission
- Every factual claim includes a citation tag `[N]` linking to the corresponding source article
- Clicking a citation reveals the source article title, content excerpt, and relevance score
- If no relevant source is found, the copilot states "I don't have enough information" instead of hallucinating

### US-2: Tool-Calling with Human Approval
**As a** support agent, **I want** the copilot to propose irreversible actions (escalation, refund, email) **and wait for my explicit approval** before executing, **so that** I maintain control over customer-impacting decisions.

**Acceptance Criteria:**
- When the copilot identifies an actionable intent (refund, escalation, email), it renders an approval card with: action type, parameters, rationale
- Agent can approve, reject, or modify the action
- Rejected actions log the reason and return to conversation
- Approved actions execute and log the approval with timestamp + agent ID

### US-3: Conversation Context
**As a** support agent, **I want** the copilot to maintain conversation context across multiple turns, **so that** follow-up questions don't require re-explaining the issue.

**Acceptance Criteria:**
- Copilot retains full conversation history within a session
- Follow-up queries reference previous context without repetition
- Session state persists across page refreshes (within same session)

### US-4: Metrics Dashboard
**As a** team lead, **I want to** view adoption and quality metrics, **so that** I can assess whether the copilot is actually improving support operations.

**Acceptance Criteria:**
- Dashboard shows: total queries, citation accuracy rate, average response time, tool usage breakdown, approval/rejection ratio
- Metrics update in near real-time (within 1 minute)

---

## Scope

### Knowledge Base Grounding & Dataset Scope
The knowledge base is built from the **Bitext Customer Support LLM Chatbot Training Dataset**, filtered down to the **6 core operational categories** and mapped into structured domain articles using query-enriched deduplication:
- **`BILLING`** $\leftarrow$ Bitext `PAYMENT` + `INVOICE`
- **`SHIPPING`** $\leftarrow$ Bitext `DELIVERY` + `SHIPPING_ADDRESS`
- **`CANCELLATION`** $\leftarrow$ Bitext `CANCELLATION_FEE` + Bitext `ORDER` (`cancel_order` intent)
- **`ORDER`** $\leftarrow$ Bitext `ORDER` (`place_order`, `change_order`)
- **`ACCOUNT`** $\leftarrow$ Bitext `ACCOUNT`
- **`REFUND`** $\leftarrow$ Bitext `REFUND`
*(Unrelated categories `FEEDBACK` and `NEWSLETTER` are excluded).*

### V1 Ships (MVP — 7 Days)

| Feature | Priority |
|:--------|:---------|
| RAG retrieval over filtered Bitext knowledge base with citation extraction | P0 |
| Citation-grounded reply drafting with inline source tags `[N]` | P0 |
| 3 tool schemas: `escalate_ticket`, `initiate_refund`, `send_followup_email` | P0 |
| HITL approval gates for all tool calls via LangGraph interrupt/resume | P0 |
| Streamlit chat interface with citation cards and approval buttons | P0 |
| Basic metrics logging (latency, token usage, citations per response) | P1 |
| Evaluation pipeline with 50+ test cases | P1 |
| Docker Compose one-command setup | P1 |
| GitHub Actions CI (lint + test) | P2 |

### V1 Does NOT Ship

- Multi-tenant support (single-user only)
- Real email/payment integrations (tools are simulated/mocked)
- User authentication or RBAC
- Production vector store (Qdrant/pgvector) — Chroma is sufficient for demo
- Next.js/React frontend — Streamlit is the V1 UI
- Real-time collaboration between agents
- Webhook integrations (Slack, Jira)
- Fine-tuned embedding models

---

## Data Model

```
KnowledgeArticle
├── id: str (UUID)
├── title: str (canonical policy or representative query)
├── category: str (ORDER | ACCOUNT | BILLING | SHIPPING | CANCELLATION | REFUND)
├── content: str (canonical resolution / policy guidance)
├── source_id: str (deterministic identifier or hash)
├── created_at: datetime
└── metadata: dict
    ├── intent: str (e.g., track_refund, check_invoice)
    ├── sample_queries: List[str] (aggregated customer query variants for dense retrieval)
    ├── sample_count: int (number of raw interactions consolidated)
    └── raw_categories: List[str] (original Bitext category names)

ConversationSession
├── session_id: str (UUID)
├── messages: List[Message]
├── created_at: datetime
└── metrics: SessionMetrics

Message
├── role: str (user | assistant | tool)
├── content: str
├── citations: List[Citation]
├── tool_calls: List[ToolCall] (optional)
└── timestamp: datetime

Citation
├── source_id: str → KnowledgeArticle.id
├── title: str
├── excerpt: str (relevant chunk text)
└── relevance_score: float

ToolCall
├── tool_name: str (escalate_ticket | initiate_refund | send_followup_email)
├── parameters: dict
├── status: str (proposed | approved | rejected | executed)
├── approval_timestamp: datetime (optional)
└── rejection_reason: str (optional)

SessionMetrics
├── total_tokens: int
├── response_latencies: List[float]
├── citations_per_response: List[int]
└── tools_proposed: int
```

---

## Edge Cases & Failure States

| Scenario | Expected Behavior |
|:---------|:------------------|
| Query has zero relevant KB articles | Copilot responds "I don't have enough information to answer this. Consider escalating." — never fabricates |
| LLM hallucinates a citation index that doesn't exist | Post-processing validates citation indices against retrieved chunk IDs; strips invalid citations |
| User approves a tool action but backend execution fails | Return error to UI, log failure, allow retry. Never silently fail |
| Embedding API is down | Return cached results if available; otherwise surface error: "Knowledge base temporarily unavailable" |
| Query contains PII (credit card, SSN) | Input sanitization layer strips PII before sending to LLM; log warning |
| User sends adversarial/injection prompt | System prompt hardening + input validation; log attempt; respond with safe default |
| Session state exceeds context window | Sliding window summarization of older messages; preserve most recent 10 turns verbatim |
| Concurrent tool approvals on same session | Queue-based processing; only one tool action executes at a time per session |

---

## Open Questions

1. **LLM Provider:** GPT-4o is the default. Should we support Anthropic Claude as a fallback? Cost implications?
2. **Knowledge Base Refresh:** How often should the KB be re-ingested? On-demand only, or scheduled?
3. **Simulated vs. Real Tools:** V1 mocks all tools. Should any tool have real side effects for demo purposes (e.g., actually sending an email via SendGrid)?
4. **Evaluation Frequency:** Run eval suite on every PR, or only on release branches?
5. **Observability:** LangSmith (free tier) or self-hosted Phoenix/OpenTelemetry?
