"""SQLite-backed session persistence service for conversation history and tool audit trail."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

from src.core.config import settings

logger = logging.getLogger(__name__)

# Empirical HITL Approval TTL Constants (ADR-006)
SOFT_TTL_SECONDS: int = 600  # 10 minutes — Zendesk Messaging active-to-inactive standard
HARD_TTL_BY_TOOL: dict[str, int] = {
    "initiate_refund": 1200,      # 20 minutes for live financial refund
    "escalate_ticket": 3600,      # 60 minutes for tier-2 escalation
    "send_followup_email": 3600,  # 60 minutes for outbound email communication
}
DEFAULT_HARD_TTL_SECONDS: int = 1200


class SessionService:
    """Service managing session lifecycles, chat message history, and tool audit logs."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.sqlite_db_path
        self._init_db()

    @contextlib.contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create and yield a new SQLite connection with WAL mode enabled, closing it upon exit."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize required SQLite tables for sessions, messages, and tool audits."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    tool_proposal_json TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_audit_log (
                    tool_call_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_audit_session ON tool_audit_log (session_id)"
            )
            conn.commit()

    def ensure_session(self, session_id: str | None = None) -> str:
        """Ensure session exists in database; create and return new session_id if None."""
        now = datetime.now(UTC).isoformat()
        sid = (session_id or f"sess-{uuid.uuid4().hex[:12]}").strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (sid, now, now),
            )
            conn.commit()

        return sid

    def session_exists(self, session_id: str) -> bool:
        """Check whether a session exists in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            return cursor.fetchone() is not None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        tool_proposal: dict[str, Any] | None = None,
    ) -> int:
        """Store a chat message with citations and tool proposals in the database."""
        now = datetime.now(UTC).isoformat()
        citations_json = json.dumps(citations) if citations else None
        tool_proposal_json = json.dumps(tool_proposal) if tool_proposal else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Update session timestamp
            cursor.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            cursor.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, citations_json, tool_proposal_json, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, citations_json, tool_proposal_json, now),
            )
            message_id = cursor.lastrowid
            conn.commit()

        return int(message_id)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve full message history for a session ordered chronologically."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, citations_json, tool_proposal_json, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        history: list[dict[str, Any]] = []
        for r in rows:
            citations = json.loads(r["citations_json"]) if r["citations_json"] else []
            tool_proposal = (
                json.loads(r["tool_proposal_json"])
                if r["tool_proposal_json"]
                else None
            )
            history.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "citations": citations,
                    "tool_proposal": tool_proposal,
                    "timestamp": r["timestamp"],
                }
            )
        return history

    def log_tool_proposal(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        rationale: str = "",
    ) -> None:
        """Record an initial tool proposal awaiting human decision."""
        now = datetime.now(UTC).isoformat()
        params_json = json.dumps(parameters)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_audit_log (
                    tool_call_id, session_id, tool_name, parameters_json,
                    status, result, reason, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_call_id) DO UPDATE SET
                    parameters_json = excluded.parameters_json,
                    updated_at = excluded.updated_at
                """,
                (
                    tool_call_id,
                    session_id,
                    tool_name,
                    params_json,
                    "awaiting_approval",
                    None,
                    rationale,
                    now,
                    now,
                ),
            )
            conn.commit()

    def update_tool_decision(
        self,
        tool_call_id: str,
        status: str,
        result: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Update a tool audit record following an approval or rejection decision."""
        now = datetime.now(UTC).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE tool_audit_log
                SET status = ?, result = COALESCE(?, result), reason = COALESCE(?, reason), updated_at = ?
                WHERE tool_call_id = ?
                """,
                (status, result, reason, now, tool_call_id),
            )
            conn.commit()

    def get_pending_tool(
        self, session_id: str, now: datetime | None = None
    ) -> dict[str, Any] | None:
        """Return the most recent tool proposal awaiting approval with staleness/expiry metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tool_call_id, session_id, tool_name, parameters_json, status, reason, created_at
                FROM tool_audit_log
                WHERE session_id = ? AND status = 'awaiting_approval'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        # Parse creation timestamp
        created_raw = row["created_at"]
        try:
            created_dt = datetime.fromisoformat(created_raw)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=UTC)
        except Exception:  # noqa: BLE001
            created_dt = datetime.now(UTC)

        current_time = now if now is not None else datetime.now(UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)

        elapsed_seconds = max(0.0, (current_time - created_dt).total_seconds())
        tool_name = row["tool_name"]
        hard_ttl = HARD_TTL_BY_TOOL.get(tool_name, DEFAULT_HARD_TTL_SECONDS)

        is_stale = elapsed_seconds > SOFT_TTL_SECONDS
        is_expired = elapsed_seconds > hard_ttl

        return {
            "tool_call_id": row["tool_call_id"],
            "session_id": row["session_id"],
            "tool_name": tool_name,
            "parameters": json.loads(row["parameters_json"]),
            "status": row["status"],
            "rationale": row["reason"],
            "created_at": row["created_at"],
            "elapsed_seconds": round(elapsed_seconds, 2),
            "hard_ttl_seconds": hard_ttl,
            "is_stale": is_stale,
            "is_expired": is_expired,
        }

    def expire_tool_proposal(
        self,
        session_id: str,
        tool_call_id: str | None = None,
        reason: str = "Expired (TTL exceeded)",
    ) -> bool:
        """Mark an awaiting_approval tool proposal row as expired in the audit trail.

        Returns:
            bool: True if an awaiting_approval proposal was transitioned to expired, False otherwise.
        """
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tool_call_id:
                cursor.execute(
                    """
                    UPDATE tool_audit_log
                    SET status = 'expired', reason = ?, updated_at = ?
                    WHERE tool_call_id = ? AND status = 'awaiting_approval'
                    """,
                    (reason, now, tool_call_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE tool_audit_log
                    SET status = 'expired', reason = ?, updated_at = ?
                    WHERE session_id = ? AND status = 'awaiting_approval'
                    """,
                    (reason, now, session_id),
                )
            conn.commit()
            return cursor.rowcount > 0


__all__ = [
    "DEFAULT_HARD_TTL_SECONDS",
    "HARD_TTL_BY_TOOL",
    "SOFT_TTL_SECONDS",
    "SessionService",
]
