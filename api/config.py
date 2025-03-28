# api/config.py
import os
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """API settings loaded from environment variables."""
    
    # API settings
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    DEBUG: bool = Field(False, env="DEBUG")
    
    # CORS
    CORS_ORIGINS: list = Field(["*"], env="CORS_ORIGINS")
    
    # Data and model paths
    DATA_DIR: str = Field("/data", env="DATA_DIR")
    MODELS_DIR: str = Field("/models", env="MODELS_DIR")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field("http://mlflow:5000", env="MLFLOW_TRACKING_URI")
    
    # Redis & Celery
    REDIS_HOST: str = Field("redis", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    
    # DVC
    DVC_REMOTE: str = Field("local", env="DVC_REMOTE")
    
    class Config:
        env_file = ".env"

settings = Settings()