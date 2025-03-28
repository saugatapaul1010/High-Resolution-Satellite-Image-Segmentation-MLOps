# tests/test_api/test_training_routes.py
import pytest
from unittest.mock import patch, MagicMock
import json

from api.models.training import TrainingStatus, ModelInfo

@pytest.fixture
def mock_training_service():
    with patch('api.services.training.TrainingService') as mock:
        instance = mock.return_value
        yield instance

def test_start_training(api_client, mock_training_service):
    """Test starting a training job."""
    # Set up mock return value
    mock_training_service.start_training.return_value = {
        "task_id": "test-task-id",
        "status": TrainingStatus.PENDING,
        "experiment_id": "test-experiment-id",
        "run_id": None
    }
    
    # Request data
    request_data = {
        "model_type": "linknet",
        "dataset_name": "test_dataset",
        "parameters": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 5
        }
    }
    
    # Call endpoint
    response = api_client.post(
        "/training/start",
        json=request_data
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id"
    assert data["status"] == "pending"
    assert data["experiment_id"] == "test-experiment-id"
    assert data["run_id"] is None

def test_get_training_status(api_client, mock_training_service):
    """Test getting training status."""
    # Set up mock return value
    mock_training_service.get_training_status.return_value = {
        "task_id": "test-task-id",
        "status": TrainingStatus.RUNNING,
        "experiment_id": "test-experiment-id",
        "run_id": "test-run-id"
    }
    
    # Call endpoint
    response = api_client.get("/training/status/test-task-id")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id"
    assert data["status"] == "running"
    assert data["experiment_id"] == "test-experiment-id"
    assert data["run_id"] == "test-run-id"

def test_list_models(api_client, mock_training_service):
    """Test listing models."""
    # Set up mock return value
    mock_training_service.list_models.return_value = [
        ModelInfo(
            name="test-model",
            version="1",
            metrics={"iou_score": 0.85, "dice_coef": 0.9},
            created_at="2023-01-01T00:00:00",
            experiment_id="test-experiment-id",
            run_id="test-run-id"
        )
    ]
    
    # Call endpoint
    response = api_client.get("/training/models")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "test-model"
    assert data[0]["version"] == "1"
    assert data[0]["metrics"]["iou_score"] == 0.85
    assert data[0]["metrics"]["dice_coef"] == 0.9
    assert data[0]["created_at"] == "2023-01-01T00:00:00"
    assert data[0]["experiment_id"] == "test-experiment-id"
    assert data[0]["run_id"] == "test-run-id"