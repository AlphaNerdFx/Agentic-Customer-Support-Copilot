"""Schemas for metrics aggregation and reporting."""
from pydantic import BaseModel


class SystemMetricsResponse(BaseModel):
    total_sessions: int
    total_queries: int
    avg_response_time_ms: float
    citation_rate: float
    tool_proposals: int
    tool_approvals: int
    tool_rejections: int
    avg_citations_per_response: float
