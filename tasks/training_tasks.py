# tasks/training_tasks.py
import os
import mlflow
import json
from celery import shared_task

from core.config.settings import get_settings
from core.models.trainer import ModelTrainer
from core.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

@shared_task(bind=True)
def train_model(self, model_type, dataset_name, parameters, experiment_name=None):
    """Train a model as a background task.
    
    Args:
        model_type: Model architecture type
        dataset_name: Name of the dataset to use
        parameters: Training parameters
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with task result information
    """
    try:
        # Set up MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        else:
            experiment = mlflow.get_experiment_by_name("Default")
            if experiment is None:
                experiment_id = mlflow.create_experiment("Default")
            else:
                experiment_id = experiment.experiment_id
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_params(parameters)
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'experiment_id': experiment_id, 'run_id': run.info.run_id}
            )
            
            # Configure and run training
            config = {
                "model_type": model_type,
                "dataset_name": dataset_name,
                "parameters": parameters,
                "paths": {
                    "data_dir": os.path.join(settings.DATA_DIR, "processed", dataset_name),
                    "models_dir": os.path.join(settings.MODELS_DIR, run.info.run_id),
                    "output_dir": os.path.join(settings.MODELS_DIR, "outputs", run.info.run_id)
                }
            }
            
            # Initialize trainer and train model
            trainer = ModelTrainer(config)
            model, history = trainer.train()
            
            # Log metrics from history
            for metric_name, values in history.history.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=step)
            
            # Log final metrics
            final_metrics = {
                metric: values[-1] for metric, values in history.history.items()
            }
            
            # Log model to MLflow
            mlflow.tensorflow.log_model(model, "model")
            
            return {
                "status": "completed",
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
                "metrics": final_metrics
            }
    
    except Exception as e:
        # Log the error
        logger.error(f"Training error: {str(e)}")
        error_message = str(e)
        try:
            mlflow.log_param("error", error_message)
        except:
            pass
        
        return {
            "status": "failed",
            "error": error_message
        }