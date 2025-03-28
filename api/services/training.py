# api/services/training.py
import os
import uuid
from datetime import datetime
from fastapi import Depends, BackgroundTasks, HTTPException
import mlflow
from typing import List, Dict, Any

from api.config import settings
from api.models.training import TrainingRequest, TrainingResponse, ModelInfo, TrainingStatus
from tasks.training_tasks import train_model

class TrainingService:
    """Service for model training and management."""
    
    def __init__(self):
        self.mlflow_uri = settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.mlflow_uri)
    
    # api/services/training.py
    def start_training(self, background_tasks: BackgroundTasks, request: TrainingRequest) -> TrainingResponse:
        """Start a training job."""
        # Check if dataset exists
        dataset_path = os.path.join(settings.DATA_DIR, "processed", request.dataset_name)
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Processed dataset {request.dataset_name} not found"
            )
        
        # Set up experiment
        experiment_name = request.experiment_name or "Default"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")
        
        # Create default parameters if not provided
        parameters = request.parameters or {}
        
        # Add default values for missing parameters
        default_params = {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 5,
            "backbone": "efficientnetb0"
        }
        
        for key, value in default_params.items():
            if key not in parameters:
                parameters[key] = value
        
        # Start async training task and get its ID
        train_task = train_model.delay(
            str(request.model_type),
            request.dataset_name,
            parameters,
            experiment_name
        )
        
        # Use the actual Celery task ID instead of generating a random UUID
        task_id = train_task.id
        
        return TrainingResponse(
            task_id=task_id,
            status=TrainingStatus.PENDING,
            experiment_id=experiment_id,
            run_id=None
        )

    def get_training_status(self, task_id: str) -> TrainingResponse:
        """Get status of a training job."""
        try:
            # Try to get task result
            result = train_model.AsyncResult(task_id)
            
            if result.state == 'PENDING':
                return TrainingResponse(
                    task_id=task_id,
                    status=TrainingStatus.PENDING
                )
            elif result.state == 'PROGRESS':
                # Get info from task meta
                meta = result.info or {}
                return TrainingResponse(
                    task_id=task_id,
                    status=TrainingStatus.RUNNING,
                    experiment_id=meta.get('experiment_id'),
                    run_id=meta.get('run_id')
                )
            elif result.state == 'SUCCESS':
                result_data = result.get()
                return TrainingResponse(
                    task_id=task_id,
                    status=TrainingStatus.COMPLETED,
                    experiment_id=result_data.get('experiment_id'),
                    run_id=result_data.get('run_id')
                )
            else:
                return TrainingResponse(
                    task_id=task_id,
                    status=TrainingStatus.FAILED
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        try:
            # Get registered models
            models = []
            
            for rm in mlflow.search_registered_models():
                latest_version = rm.latest_versions[0]
                
                # Get run info for metrics
                run = mlflow.get_run(latest_version.run_id)
                metrics = run.data.metrics
                
                models.append(ModelInfo(
                    name=rm.name,
                    version=latest_version.version,
                    metrics=metrics,
                    created_at=latest_version.creation_timestamp.isoformat(),
                    experiment_id=run.info.experiment_id,
                    run_id=latest_version.run_id
                ))
            
            return models
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")
    
    def register_model(self, run_id: str, model_name: str) -> ModelInfo:
        """Register a model from an MLflow run."""
        try:
            # Get run info
            run = mlflow.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
            
            # Register model
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            
            # Return model info
            return ModelInfo(
                name=model_name,
                version=result.version,
                metrics=run.data.metrics,
                created_at=datetime.fromtimestamp(result.creation_timestamp / 1000).isoformat(),
                experiment_id=run.info.experiment_id,
                run_id=run_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")