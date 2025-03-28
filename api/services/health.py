# api/services/health.py
import os
import shutil
from fastapi import Depends
import mlflow
import redis

from api.config import settings
from api.dependencies import get_redis

class HealthService:
    """Service for checking system health."""
    
    def __init__(self, redis_client: redis.Redis = Depends(get_redis)):
        self.redis_client = redis_client
    
    def check_health(self):
        """Check health of all services."""
        status = {
            "status": "ok",
            "redis": self._check_redis(),
            "mlflow": self._check_mlflow(),
            "disk": self._check_disk_space(),
        }
        
        # Overall status is only ok if all services are ok
        if not all(v == "ok" for k, v in status.items() if k != "status"):
            status["status"] = "error"
            
        return status
    
    def _check_redis(self):
        """Check Redis connection."""
        try:
            self.redis_client.ping()
            return "ok"
        except Exception:
            return "error"
    
    def _check_mlflow(self):
        """Check MLflow connection."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.list_experiments()
            return "ok"
        except Exception:
            return "error"
    
    def _check_disk_space(self):
        """Check available disk space."""
        try:
            total, used, free = shutil.disk_usage("/")
            # If less than 10% free space, report warning
            if (free / total) < 0.1:
                return "warning"
            return "ok"
        except Exception:
            return "error"