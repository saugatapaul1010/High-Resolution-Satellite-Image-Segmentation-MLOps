# api/routers/training.py
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from typing import List

from api.models.training import TrainingRequest, TrainingResponse, ModelInfo
from api.services.training import TrainingService

router = APIRouter()

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    training_service: TrainingService = Depends()
):
    """Start a training job."""
    return training_service.start_training(background_tasks, request)

@router.get("/status/{task_id}", response_model=TrainingResponse)
async def get_training_status(
    task_id: str,
    training_service: TrainingService = Depends()
):
    """Get status of a training job."""
    return training_service.get_training_status(task_id)

@router.get("/models", response_model=List[ModelInfo])
async def list_models(training_service: TrainingService = Depends()):
    """List all available models."""
    return training_service.list_models()

@router.post("/register/{run_id}", response_model=ModelInfo)
async def register_model(
    run_id: str,
    model_name: str,
    training_service: TrainingService = Depends()
):
    """Register a model from an MLflow run."""
    return training_service.register_model(run_id, model_name)