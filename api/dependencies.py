# api/dependencies.py
import redis
from fastapi import Depends

from api.config import settings

# Redis client
def get_redis():
    """Get Redis client."""
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True
    )
    try:
        yield redis_client
    finally:
        redis_client.close()