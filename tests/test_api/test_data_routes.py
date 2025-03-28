# tests/test_api/test_data_routes.py
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import json

from api.models.data import DatasetInfo, DatasetType

@pytest.fixture
def mock_data_service():
    with patch('api.services.data.DataService') as mock:
        instance = mock.return_value
        yield instance

def test_list_datasets(api_client, mock_data_service):
    """Test listing datasets."""
    # Set up mock return value
    mock_data_service.list_datasets.return_value = [
        DatasetInfo(
            name="test_dataset",
            type=DatasetType.RAW,
            file_count=10,
            size_bytes=1024,
            created_at="2023-01-01T00:00:00"
        )
    ]
    
    # Call endpoint
    response = api_client.get("/data/datasets")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "test_dataset"
    assert data[0]["type"] == "raw"
    assert data[0]["file_count"] == 10
    assert data[0]["size_bytes"] == 1024
    assert data[0]["created_at"] == "2023-01-01T00:00:00"

def test_upload_file(api_client, mock_data_service):
    """Test file upload endpoint."""
    # Set up mock return value
    mock_data_service.upload_file.return_value = {
        "filename": "test.json",
        "size_bytes": 100,
        "status": "success"
    }
    
    # Create test file
    test_content = json.dumps({"test": "data"}).encode("utf-8")
    
    # Call endpoint
    response = api_client.post(
        "/data/upload",
        files={"file": ("test.json", BytesIO(test_content), "application/json")}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.json"
    assert data["size_bytes"] == 100
    assert data["status"] == "success"

def test_process_data(api_client, mock_data_service):
    """Test data processing endpoint."""
    # Set up mock return value
    mock_data_service.process_data.return_value = {
        "task_id": "test-task-id",
        "status": "pending"
    }
    
    # Call endpoint
    response = api_client.post("/data/process?dataset_name=test_dataset")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id"
    assert data["status"] == "pending"