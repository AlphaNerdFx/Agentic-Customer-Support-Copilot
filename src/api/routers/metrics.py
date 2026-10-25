"""Metrics reporting router for observability and KPI dashboard."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_metrics_service
from src.api.schemas.metrics import SystemMetricsResponse
from src.services.metrics import MetricsService

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_metrics(
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> SystemMetricsResponse:
    """Retrieve aggregated system observability and adoption metrics."""
    data = metrics_service.get_system_metrics()
    return SystemMetricsResponse(**data)


__all__ = ["router"]
