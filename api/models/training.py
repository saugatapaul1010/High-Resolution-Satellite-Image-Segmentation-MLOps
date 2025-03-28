# api/models/training.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any, Optional

class ModelType(str, Enum):
    LINKNET = "linknet"
    UNET = "unet"
    MANET = "manet"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingRequest(BaseModel):
    """Request model for starting a training job."""
    model_type: ModelType = Field(ModelType.LINKNET, description="Model architecture")
    dataset_name: str = Field(..., description="Name of the dataset to use")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training parameters"
    )
    experiment_name: Optional[str] = Field(
        None,
        description="MLflow experiment name"
    )
    
class TrainingResponse(BaseModel):
    """Response model for training job status."""
    task_id: str
    status: TrainingStatus
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    
class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    metrics: Dict[str, float]
    created_at: str
    experiment_id: str
    run_id: str