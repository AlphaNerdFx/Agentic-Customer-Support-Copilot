"""Integration and unit tests for ADR-006 Empirical HITL Approval TTL and SQLite WAL mode.

Validates:
1. Two-Tier Empirical TTL (600s soft staleness, 1200s/3600s hard expiry).
2. SQLite WAL mode (PRAGMA journal_mode=WAL, PRAGMA busy_timeout=5000) and thread concurrency.
3. Telemetry isolation: tool_expired counts TTL expirations without polluting tool_rejections.
4. FastAPI lazy-evaluation gates:
   - Bypassing 409 Conflict on POST /chat when a stale gate has expired.
   - Guarding POST /approve with 400 Bad Request on expired proposals.
   - Guarding POST /reject with 400 Bad Request on expired proposals.
"""

from __future__ import annotations

import concurrent.futures
from datetime import UTC, datetime, timedelta

import chromadb
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_metrics_service,
    get_session_service,
    reset_api_state,
)
from src.api.main import app
from src.rag.chunking import chunk_article, deduplicate_and_enrich
from src.rag.ingest import DEFAULT_BITEXT_SEED_ROWS
from src.rag.store import ChromaVectorStore, DeterministicDomainEmbeddingFunction
from src.services.metrics import MetricsService
from src.services.session import (
    DEFAULT_HARD_TTL_SECONDS,
    HARD_TTL_BY_TOOL,
    SOFT_TTL_SECONDS,
    SessionService,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database file for testing WAL and session isolation."""
    db_file = tmp_path / "test_ttl.db"
    return str(db_file)


@pytest.fixture(scope="module")
def api_ttl_client(tmp_path_factory):
    """FastAPI TestClient with initialized Chroma store and isolated SQLite database."""
    temp_dir = tmp_path_factory.mktemp("api_ttl_test_env")
    test_db_path = str(temp_dir / "test_api_ttl.db")

    client = chromadb.EphemeralClient()
    embed_fn = DeterministicDomainEmbeddingFunction(dim=384)
    store = ChromaVectorStore(
        collection_name="test_api_ttl_kb",
        client=client,
        embedding_function=embed_fn,
    )
    articles = deduplicate_and_enrich(DEFAULT_BITEXT_SEED_ROWS)
    chunks = []
    for art in articles:
        chunks.extend(chunk_article(art))
    store.add_documents(chunks)

    reset_api_state(db_path=test_db_path, custom_vector_store=store)

    with TestClient(app) as test_client:
        yield test_client

    reset_api_state()


# -----------------------------------------------------------------------------
# 1. Empirical TTL Logic & SessionService Tests
# -----------------------------------------------------------------------------


def test_ttl_constants_adhere_to_spec():
    """Verify empirical constants match ADR-006 specifications."""
    assert SOFT_TTL_SECONDS == 600
    assert HARD_TTL_BY_TOOL["initiate_refund"] == 1200
    assert HARD_TTL_BY_TOOL["escalate_ticket"] == 3600
    assert HARD_TTL_BY_TOOL["send_followup_email"] == 3600
    assert DEFAULT_HARD_TTL_SECONDS == 1200


def test_sqlite_wal_pragmas(temp_db):
    """Verify SessionService and MetricsService enforce WAL mode and busy_timeout."""
    sess_svc = SessionService(db_path=temp_db)
    with sess_svc._get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert journal_mode.lower() == "wal"
        assert busy_timeout == 5000

    metrics_svc = MetricsService(db_path=temp_db)
    with metrics_svc._get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert journal_mode.lower() == "wal"
        assert busy_timeout == 5000


def test_get_pending_tool_fresh_stale_and_expired(temp_db):
    """Test get_pending_tool correctly evaluates staleness and expiry at various simulated timestamps."""
    sess_svc = SessionService(db_path=temp_db)
    sid = sess_svc.ensure_session()

    sess_svc.log_tool_proposal(
        session_id=sid,
        tool_call_id="call-ref-1",
        tool_name="initiate_refund",
        parameters={"amount": 49.99},
        rationale="Customer requested refund",
    )

    # 1. Fresh state (e.g. 100 seconds later)
    now_fresh = datetime.now(UTC) + timedelta(seconds=100)
    pending = sess_svc.get_pending_tool(sid, now=now_fresh)
    assert pending is not None
    assert pending["tool_name"] == "initiate_refund"
    assert pending["hard_ttl_seconds"] == 1200
    assert not pending["is_stale"]
    assert not pending["is_expired"]
    assert 99.0 <= pending["elapsed_seconds"] <= 102.0

    # 2. Stale state (700 seconds later: > 600s soft TTL, < 1200s hard TTL)
    now_stale = datetime.now(UTC) + timedelta(seconds=700)
    pending_stale = sess_svc.get_pending_tool(sid, now=now_stale)
    assert pending_stale is not None
    assert pending_stale["is_stale"] is True
    assert pending_stale["is_expired"] is False

    # 3. Expired state (1300 seconds later: > 1200s hard TTL)
    now_expired = datetime.now(UTC) + timedelta(seconds=1300)
    pending_expired = sess_svc.get_pending_tool(sid, now=now_expired)
    assert pending_expired is not None
    assert pending_expired["is_stale"] is True
    assert pending_expired["is_expired"] is True


def test_expire_tool_proposal_updates_audit_status(temp_db):
    """expire_tool_proposal marks proposal status to 'expired' and hides from pending tools."""
    sess_svc = SessionService(db_path=temp_db)
    sid = sess_svc.ensure_session()

    sess_svc.log_tool_proposal(
        session_id=sid,
        tool_call_id="call-esc-1",
        tool_name="escalate_ticket",
        parameters={"priority": "high"},
        rationale="Urgent issue",
    )

    assert sess_svc.get_pending_tool(sid) is not None

    sess_svc.expire_tool_proposal(sid, tool_call_id="call-esc-1")
    assert sess_svc.get_pending_tool(sid) is None

    # Check database row has status 'expired'
    with sess_svc._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, reason FROM tool_audit_log WHERE tool_call_id = 'call-esc-1'"
        )
        row = cursor.fetchone()
        assert row["status"] == "expired"
        assert "Expired" in row["reason"]


# -----------------------------------------------------------------------------
# 2. Telemetry Isolation Tests
# -----------------------------------------------------------------------------


def test_telemetry_tool_expired_isolation(temp_db):
    """Verify tool_expired counts TTL expirations without incrementing tool_rejections."""
    metrics_svc = MetricsService(db_path=temp_db)

    initial_metrics = metrics_svc.get_system_metrics()
    assert initial_metrics["tool_expired"] == 0
    assert initial_metrics["tool_rejections"] == 0
    assert initial_metrics["tool_approvals"] == 0

    # Record TTL expiry
    metrics_svc.record_tool_expiry("sess-metrics-1")
    after_expiry = metrics_svc.get_system_metrics()
    assert after_expiry["tool_expired"] == 1
    assert after_expiry["tool_rejections"] == 0  # Rejections must NOT increase
    assert after_expiry["tool_approvals"] == 0

    # Record explicit human rejection
    metrics_svc.record_tool_decision("sess-metrics-1", approved=False)
    after_reject = metrics_svc.get_system_metrics()
    assert after_reject["tool_expired"] == 1
    assert after_reject["tool_rejections"] == 1
    assert after_reject["tool_approvals"] == 0

    # Record human approval
    metrics_svc.record_tool_decision("sess-metrics-1", approved=True)
    after_approve = metrics_svc.get_system_metrics()
    assert after_approve["tool_expired"] == 1
    assert after_approve["tool_rejections"] == 1
    assert after_approve["tool_approvals"] == 1


def test_sqlite_concurrent_access_wal(temp_db):
    """Verify concurrent reads and writes execute without lock contention under WAL mode."""
    sess_svc = SessionService(db_path=temp_db)
    sid = sess_svc.ensure_session()

    def write_op(idx: int) -> None:
        sess_svc.add_message(session_id=sid, role="user", content=f"Message {idx}")

    def read_op() -> int:
        return len(sess_svc.get_session_history(sid))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(20):
            futures.append(executor.submit(write_op, i))
            futures.append(executor.submit(read_op))
        for f in concurrent.futures.as_completed(futures):
            f.result()

    history = sess_svc.get_session_history(sid)
    assert len(history) == 20


# -----------------------------------------------------------------------------
# 3. FastAPI Lazy-Evaluation Gate Integration Tests
# -----------------------------------------------------------------------------


def test_api_active_proposal_blocks_chat_turn(api_ttl_client):
    """POST /chat returns 409 Conflict when an unexpired tool proposal is awaiting approval."""
    # 1. Propose tool via refund trigger
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $40 for order ORD-99881 right now."},
    )
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    assert data["response"]["tool_proposal"] is not None
    assert data["response"]["tool_proposal"]["status"] == "awaiting_approval"

    # 2. Immediately send another message -> must return 409 Conflict
    conflict_res = api_ttl_client.post(
        "/chat",
        json={"session_id": sid, "message": "Are you still there?"},
    )
    assert conflict_res.status_code == 409
    assert "pending approval" in conflict_res.json()["detail"]


def test_api_approve_blocked_on_expired_proposal(api_ttl_client):
    """POST /approve returns 400 Bad Request if the proposal exceeded Hard TTL."""
    sess_svc = get_session_service()
    metrics_svc = get_metrics_service()

    # 1. Create a proposal
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $25 for order ORD-11223."},
    )
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    tool_proposal = data["response"]["tool_proposal"]
    tool_call_id = tool_proposal["tool_call_id"]

    # 2. Artificially age the proposal in SQLite audit log past 1200s (e.g. 1500s ago)
    old_time = (datetime.now(UTC) - timedelta(seconds=1500)).isoformat()
    with sess_svc._get_connection() as conn:
        conn.execute(
            "UPDATE tool_audit_log SET created_at = ? WHERE tool_call_id = ?",
            (old_time, tool_call_id),
        )
        conn.commit()

    initial_expired = metrics_svc.get_system_metrics()["tool_expired"]

    # 3. Attempt approval
    approve_res = api_ttl_client.post(
        "/approve",
        json={
            "session_id": sid,
            "tool_call_id": tool_call_id,
            "approved": True,
        },
    )
    assert approve_res.status_code == 400
    assert "Hard TTL exceeded" in approve_res.json()["detail"]

    # 4. Verify telemetry recorded tool_expired and no approvals
    updated_metrics = metrics_svc.get_system_metrics()
    assert updated_metrics["tool_expired"] == initial_expired + 1


def test_api_reject_blocked_on_expired_proposal(api_ttl_client):
    """POST /reject returns 400 Bad Request if the proposal exceeded Hard TTL."""
    sess_svc = get_session_service()

    # 1. Create a proposal
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $15 for order ORD-33445."},
    )
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    tool_proposal = data["response"]["tool_proposal"]
    tool_call_id = tool_proposal["tool_call_id"]

    # 2. Artificially age proposal past 1200s
    old_time = (datetime.now(UTC) - timedelta(seconds=1500)).isoformat()
    with sess_svc._get_connection() as conn:
        conn.execute(
            "UPDATE tool_audit_log SET created_at = ? WHERE tool_call_id = ?",
            (old_time, tool_call_id),
        )
        conn.commit()

    # 3. Attempt rejection
    reject_res = api_ttl_client.post(
        "/reject",
        json={
            "session_id": sid,
            "tool_call_id": tool_call_id,
            "reason": "Customer changed mind",
        },
    )
    assert reject_res.status_code == 400
    assert "Hard TTL exceeded" in reject_res.json()["detail"]


def test_api_chat_lazy_eval_unblocks_expired_gate(api_ttl_client):
    """POST /chat lazily expires a stale gate past Hard TTL and allows the new turn to execute."""
    sess_svc = get_session_service()
    metrics_svc = get_metrics_service()

    # 1. Create a proposal
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $80 for order ORD-77889."},
    )
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    tool_call_id = data["response"]["tool_proposal"]["tool_call_id"]

    # 2. Artificially age proposal past Hard TTL
    old_time = (datetime.now(UTC) - timedelta(seconds=1600)).isoformat()
    with sess_svc._get_connection() as conn:
        conn.execute(
            "UPDATE tool_audit_log SET created_at = ? WHERE tool_call_id = ?",
            (old_time, tool_call_id),
        )
        conn.commit()

    initial_expired = metrics_svc.get_system_metrics()["tool_expired"]

    # 3. Send new message to POST /chat -> Should NOT return 409 Conflict
    new_res = api_ttl_client.post(
        "/chat",
        json={
            "session_id": sid,
            "message": "What is the return policy for international orders?",
        },
    )
    assert new_res.status_code == 200
    new_data = new_res.json()
    assert new_data["response"]["content"] != ""
    assert len(new_data["response"]["citations"]) > 0

    # 4. Check telemetry was updated with tool_expired
    metrics = metrics_svc.get_system_metrics()
    assert metrics["tool_expired"] == initial_expired + 1


def test_ttl_exact_boundary_timestamps(temp_db):
    """Verify exact boundary timestamps at 599s/601s, 1199s/1201s, and 3599s/3601s."""
    sess_svc = SessionService(db_path=temp_db)
    base_time = datetime.now(UTC)

    # 1. Soft TTL boundary (600s) on any tool
    sid1 = sess_svc.ensure_session()
    sess_svc.log_tool_proposal(sid1, "call-b1", "initiate_refund", {"amount": 10})

    # At 599s: fresh
    p_599 = sess_svc.get_pending_tool(sid1, now=base_time + timedelta(seconds=599))
    assert not p_599["is_stale"]
    assert not p_599["is_expired"]

    # At 600s: not stale yet (elapsed > 600)
    p_600 = sess_svc.get_pending_tool(sid1, now=base_time + timedelta(seconds=600))
    assert not p_600["is_stale"]
    assert not p_600["is_expired"]

    # At 601s: stale, but not expired (1200s hard TTL)
    p_601 = sess_svc.get_pending_tool(sid1, now=base_time + timedelta(seconds=601))
    assert p_601["is_stale"] is True
    assert p_601["is_expired"] is False

    # 2. Hard TTL boundary for initiate_refund (1200s)
    # At 1199s: stale, not expired
    p_1199 = sess_svc.get_pending_tool(sid1, now=base_time + timedelta(seconds=1199))
    assert p_1199["is_stale"] is True
    assert p_1199["is_expired"] is False

    # At 1200s: not expired yet
    p_1200 = sess_svc.get_pending_tool(sid1, now=base_time + timedelta(seconds=1200))
    assert p_1200["is_expired"] is False

    # At 1201s: expired!
    p_1201 = sess_svc.get_pending_tool(sid1, now=base_time + timedelta(seconds=1201))
    assert p_1201["is_stale"] is True
    assert p_1201["is_expired"] is True

    # 3. Hard TTL boundary for escalate_ticket (3600s)
    sid2 = sess_svc.ensure_session()
    sess_svc.log_tool_proposal(sid2, "call-b2", "escalate_ticket", {"priority": "high"})

    p_esc_3599 = sess_svc.get_pending_tool(sid2, now=base_time + timedelta(seconds=3599))
    assert p_esc_3599["is_stale"] is True
    assert p_esc_3599["is_expired"] is False

    p_esc_3600 = sess_svc.get_pending_tool(sid2, now=base_time + timedelta(seconds=3600))
    assert p_esc_3600["is_expired"] is False

    p_esc_3601 = sess_svc.get_pending_tool(sid2, now=base_time + timedelta(seconds=3601))
    assert p_esc_3601["is_stale"] is True
    assert p_esc_3601["is_expired"] is True

    # 4. Hard TTL boundary for send_followup_email (3600s)
    sid3 = sess_svc.ensure_session()
    sess_svc.log_tool_proposal(sid3, "call-b3", "send_followup_email", {"email": "user@example.com"})

    p_eml_3599 = sess_svc.get_pending_tool(sid3, now=base_time + timedelta(seconds=3599))
    assert p_eml_3599["is_stale"] is True
    assert p_eml_3599["is_expired"] is False

    p_eml_3601 = sess_svc.get_pending_tool(sid3, now=base_time + timedelta(seconds=3601))
    assert p_eml_3601["is_stale"] is True
    assert p_eml_3601["is_expired"] is True


def test_api_approve_double_call_idempotence(api_ttl_client):
    """Calling POST /approve twice on an expired gate does NOT increment tool_expired twice."""
    sess_svc = get_session_service()
    metrics_svc = get_metrics_service()

    # 1. Create a proposal
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $33 for order ORD-44556."},
    )
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    tool_call_id = data["response"]["tool_proposal"]["tool_call_id"]

    # 2. Artificially age the proposal past 1200s
    old_time = (datetime.now(UTC) - timedelta(seconds=1400)).isoformat()
    with sess_svc._get_connection() as conn:
        conn.execute(
            "UPDATE tool_audit_log SET created_at = ? WHERE tool_call_id = ?",
            (old_time, tool_call_id),
        )
        conn.commit()

    initial_expired = metrics_svc.get_system_metrics()["tool_expired"]
    initial_rejections = metrics_svc.get_system_metrics()["tool_rejections"]

    # 3. First call to POST /approve: returns 400 Bad Request, records expiry
    res1 = api_ttl_client.post(
        "/approve",
        json={"session_id": sid, "tool_call_id": tool_call_id, "approved": True},
    )
    assert res1.status_code == 400
    assert "Hard TTL exceeded" in res1.json()["detail"]

    m1 = metrics_svc.get_system_metrics()
    assert m1["tool_expired"] == initial_expired + 1
    assert m1["tool_rejections"] == initial_rejections  # Rejections must NOT increase

    # 4. Second call to POST /approve: returns 400 Bad Request, must NOT re-increment tool_expired
    res2 = api_ttl_client.post(
        "/approve",
        json={"session_id": sid, "tool_call_id": tool_call_id, "approved": True},
    )
    assert res2.status_code == 400
    assert "No pending tool approval gate found" in res2.json()["detail"]

    m2 = metrics_svc.get_system_metrics()
    assert m2["tool_expired"] == initial_expired + 1
    assert m2["tool_rejections"] == initial_rejections


def test_api_reject_double_call_idempotence(api_ttl_client):
    """Calling POST /reject twice on an expired gate does NOT increment tool_expired twice."""
    sess_svc = get_session_service()
    metrics_svc = get_metrics_service()

    # 1. Create a proposal
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $55 for order ORD-66778."},
    )
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    tool_call_id = data["response"]["tool_proposal"]["tool_call_id"]

    # 2. Artificially age proposal past 1200s
    old_time = (datetime.now(UTC) - timedelta(seconds=1400)).isoformat()
    with sess_svc._get_connection() as conn:
        conn.execute(
            "UPDATE tool_audit_log SET created_at = ? WHERE tool_call_id = ?",
            (old_time, tool_call_id),
        )
        conn.commit()

    initial_expired = metrics_svc.get_system_metrics()["tool_expired"]
    initial_rejections = metrics_svc.get_system_metrics()["tool_rejections"]

    # 3. First call to POST /reject: returns 400, records expiry
    res1 = api_ttl_client.post(
        "/reject",
        json={"session_id": sid, "tool_call_id": tool_call_id, "reason": "Customer cancellation"},
    )
    assert res1.status_code == 400
    assert "Hard TTL exceeded" in res1.json()["detail"]

    m1 = metrics_svc.get_system_metrics()
    assert m1["tool_expired"] == initial_expired + 1
    assert m1["tool_rejections"] == initial_rejections

    # 4. Second call to POST /reject: returns 400, must NOT re-increment tool_expired
    res2 = api_ttl_client.post(
        "/reject",
        json={"session_id": sid, "tool_call_id": tool_call_id, "reason": "Customer cancellation"},
    )
    assert res2.status_code == 400
    assert "No pending tool approval gate found" in res2.json()["detail"]

    m2 = metrics_svc.get_system_metrics()
    assert m2["tool_expired"] == initial_expired + 1
    assert m2["tool_rejections"] == initial_rejections


def test_api_chat_multiple_subsequent_turns_after_expiry(api_ttl_client):
    """Subsequent chat turns after an expired gate execute cleanly without interference."""
    sess_svc = get_session_service()
    metrics_svc = get_metrics_service()

    # 1. Propose tool
    res = api_ttl_client.post(
        "/chat",
        json={"message": "Please refund $70 for order ORD-88990."},
    )
    sid = res.json()["session_id"]
    tool_call_id = res.json()["response"]["tool_proposal"]["tool_call_id"]

    # 2. Age past Hard TTL
    old_time = (datetime.now(UTC) - timedelta(seconds=1500)).isoformat()
    with sess_svc._get_connection() as conn:
        conn.execute(
            "UPDATE tool_audit_log SET created_at = ? WHERE tool_call_id = ?",
            (old_time, tool_call_id),
        )
        conn.commit()

    initial_expired = metrics_svc.get_system_metrics()["tool_expired"]

    # 3. Turn 2: triggers lazy evaluation unblock
    turn2 = api_ttl_client.post(
        "/chat",
        json={"session_id": sid, "message": "How do I track my order delivery?"},
    )
    assert turn2.status_code == 200
    assert turn2.json()["response"]["content"] != ""
    assert metrics_svc.get_system_metrics()["tool_expired"] == initial_expired + 1

    # 4. Turn 3: normal turn with no gate in place
    turn3 = api_ttl_client.post(
        "/chat",
        json={"session_id": sid, "message": "Can I change my shipping address?"},
    )
    assert turn3.status_code == 200
    assert turn3.json()["response"]["content"] != ""
    assert metrics_svc.get_system_metrics()["tool_expired"] == initial_expired + 1


def test_concurrent_lazy_eval_expiration_atomic(temp_db):
    """Concurrent expire_tool_proposal calls atomically transition exactly once."""
    sess_svc = SessionService(db_path=temp_db)
    sid = sess_svc.ensure_session()
    sess_svc.log_tool_proposal(sid, "call-concurrent-1", "initiate_refund", {"amount": 100})

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                sess_svc.expire_tool_proposal,
                session_id=sid,
                tool_call_id="call-concurrent-1",
            )
            for _ in range(4)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Exactly one thread successfully transitions the row from awaiting_approval to expired
    assert results.count(True) == 1
    assert results.count(False) == 3

