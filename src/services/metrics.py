"""Metrics tracking and aggregation service for observability and KPI reporting."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

from src.core.config import settings

logger = logging.getLogger(__name__)


class MetricsService:
    """Service tracking end-to-end latency, tokens, citation fidelity, and HITL decisions."""

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
        """Initialize required SQLite tables for metrics tracking."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    chunks_retrieved INTEGER NOT NULL,
                    chunks_cited INTEGER NOT NULL,
                    tool_proposed INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_expiries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_session ON query_metrics (session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_expiries_session ON tool_expiries (session_id)"
            )
            conn.commit()

    def record_query_turn(
        self,
        session_id: str,
        latency_ms: float,
        tokens_used: int,
        chunks_retrieved: int,
        chunks_cited: int,
        tool_proposed: bool,
    ) -> None:
        """Record telemetry data for a single conversational turn."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_metrics (
                    session_id, latency_ms, tokens_used, chunks_retrieved,
                    chunks_cited, tool_proposed, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    float(latency_ms),
                    int(tokens_used),
                    int(chunks_retrieved),
                    int(chunks_cited),
                    1 if tool_proposed else 0,
                    now,
                ),
            )
            conn.commit()

    def record_tool_decision(self, session_id: str, approved: bool) -> None:
        """Record a human-in-the-loop tool approval or rejection decision."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_decisions (session_id, approved, timestamp)
                VALUES (?, ?, ?)
                """,
                (session_id, 1 if approved else 0, now),
            )
            conn.commit()

    def record_tool_expiry(self, session_id: str) -> None:
        """Record an empirical HITL tool expiration event."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_expiries (session_id, timestamp)
                VALUES (?, ?)
                """,
                (session_id, now),
            )
            conn.commit()

    def get_system_metrics(self) -> dict[str, Any]:
        """Aggregate and compute system KPI metrics matching SystemMetricsResponse."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Query turn metrics
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(*) as total_queries,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN chunks_cited > 0 THEN 1 ELSE 0 END) as cited_queries,
                    SUM(tool_proposed) as total_tool_proposals,
                    AVG(chunks_cited) as avg_citations
                FROM query_metrics
                """
            )
            qm_row = cursor.fetchone()

            # Decision metrics
            cursor.execute(
                """
                SELECT
                    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) as approvals,
                    SUM(CASE WHEN approved = 0 THEN 1 ELSE 0 END) as rejections
                FROM tool_decisions
                """
            )
            td_row = cursor.fetchone()

            # Expiration metrics
            cursor.execute(
                """
                SELECT COUNT(*) as expiries
                FROM tool_expiries
                """
            )
            te_row = cursor.fetchone()

        total_sessions = int(qm_row["total_sessions"] or 0)
        total_queries = int(qm_row["total_queries"] or 0)
        avg_latency = float(qm_row["avg_latency"] or 0.0)
        cited_queries = int(qm_row["cited_queries"] or 0)
        tool_proposals = int(qm_row["total_tool_proposals"] or 0)
        avg_citations = float(qm_row["avg_citations"] or 0.0)

        tool_approvals = int(td_row["approvals"] or 0)
        tool_rejections = int(td_row["rejections"] or 0)
        tool_expired = int(te_row["expiries"] or 0)

        citation_rate = round(cited_queries / total_queries, 3) if total_queries > 0 else 0.0

        return {
            "total_sessions": total_sessions,
            "total_queries": total_queries,
            "avg_response_time_ms": round(avg_latency, 1),
            "citation_rate": citation_rate,
            "tool_proposals": tool_proposals,
            "tool_approvals": tool_approvals,
            "tool_rejections": tool_rejections,
            "tool_expired": tool_expired,
            "avg_citations_per_response": round(avg_citations, 2),
        }


__all__ = ["MetricsService"]
