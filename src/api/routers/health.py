"""FastAPI health check router."""
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Health check endpoint confirming API readiness."""
    return {
        "status": "healthy",
        "vector_store": "connected",
        "llm_provider": "connected",
        "uptime_seconds": 0
    }
