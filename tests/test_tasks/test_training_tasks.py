# tests/test_tasks/test_training_tasks.py
import pytest
from unittest.mock import patch, MagicMock
import os

from tasks.training_tasks import train_model
from core.models.trainer import ModelTrainer

@pytest.fixture
def mock_mlflow():
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.create_experiment') as mock_create_exp, \
         patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.tensorflow.log_model') as mock_log_model:
        
        # Configure mock experiment
        mock_exp = MagicMock()
        mock_exp.experiment_id = "test-experiment-id"
        mock_get_exp.return_value = mock_exp
        
        # Configure mock run
        mock_run = MagicMock()
        mock_run_info = MagicMock()
        mock_run_info.run_id = "test-run-id"
        mock_run.info = mock_run_info
        mock_run.__enter__.return_value = mock_run
        mock_start_run.return_value = mock_run
        
        yield {
            "get_experiment": mock_get_exp,
            "create_experiment": mock_create_exp,
            "start_run": mock_start_run,
            "log_params": mock_log_params,
            "log_metric": mock_log_metric,
            "log_model": mock_log_model,
            "run": mock_run
        }

@pytest.fixture
def mock_trainer():
    with patch.object(ModelTrainer, '__init__', return_value=None), \
         patch.object(ModelTrainer, 'train') as mock_train:
        
        # Configure mock training
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {
            "loss": [0.5, 0.3],
            "val_loss": [0.6, 0.4],
            "iou_score": [0.7, 0.8],
            "val_iou_score": [0.65, 0.75]
        }
        mock_train.return_value = (mock_model, mock_history)
        
        yield mock_train

@pytest.mark.skip(reason="Requires Celery task")
def test_train_model(mock_mlflow, mock_trainer):
    """Test model training task."""
    # Call the task
    result = train_model(
        "linknet",
        "test_dataset",
        {"batch_size": 4, "learning_rate": 0.001},
        "Test Experiment"
    )
    
    # Check MLflow interactions
    mock_mlflow["get_experiment"].assert_called_with("Test Experiment")
    mock_mlflow["log_params"].assert_called_with({"batch_size": 4, "learning_rate": 0.001})
    
    # Check that trainer was called
    mock_trainer.assert_called_once()
    
    # Check result
    assert result["status"] == "completed"
    assert result["experiment_id"] == "test-experiment-id"
    assert result["run_id"] == "test-run-id"
    assert "metrics" in result