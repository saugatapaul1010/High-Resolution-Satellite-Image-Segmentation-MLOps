# api/routers/health.py
from fastapi import APIRouter, Depends
from api.services.health import HealthService

router = APIRouter()

@router.get("/health")
async def health_check(health_service: HealthService = Depends()):
    """Health check endpoint."""
    return health_service.check_health()