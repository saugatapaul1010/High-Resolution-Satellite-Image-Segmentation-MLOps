# core/config/settings.py
import os
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """Core settings loaded from environment variables."""
    
    # Data and model paths
    DATA_DIR: str = Field("/data", env="DATA_DIR")
    MODELS_DIR: str = Field("/models", env="MODELS_DIR")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field("http://mlflow:5000", env="MLFLOW_TRACKING_URI")
    
    # Redis & Celery
    REDIS_HOST: str = Field("redis", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    
    # Model configuration
    DEFAULT_BATCH_SIZE: int = Field(4, env="DEFAULT_BATCH_SIZE")
    DEFAULT_LEARNING_RATE: float = Field(0.001, env="DEFAULT_LEARNING_RATE")
    DEFAULT_EPOCHS: int = Field(5, env="DEFAULT_EPOCHS")
    
    # Training
    PATCH_SIZE: int = Field(512, env="PATCH_SIZE")
    
    # DVC
    DVC_REMOTE: str = Field("local", env="DVC_REMOTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    """Get cached settings singleton."""
    return Settings()