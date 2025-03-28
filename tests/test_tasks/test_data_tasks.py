# tests/test_tasks/test_data_tasks.py
import os
import pytest
from unittest.mock import patch, MagicMock
import shutil
import tempfile

from tasks.data_tasks import process_dataset
from core.data.data_preparation import DataPreparation

@pytest.fixture
def mock_prepare_patches():
    with patch.object(DataPreparation, 'prepare_patches') as mock:
        yield mock

@pytest.fixture
def temp_dataset_dirs():
    # Create temporary directories
    data_dir = tempfile.mkdtemp()
    source_dir = os.path.join(data_dir, "raw", "test_dataset")
    os.makedirs(source_dir, exist_ok=True)
    
    # Create a test JSON file
    json_path = os.path.join(source_dir, "test.json")
    with open(json_path, "w") as f:
        f.write('{"shapes": []}')
    
    yield data_dir
    
    # Clean up
    shutil.rmtree(data_dir)

@pytest.mark.skip(reason="Requires filesystem operations")
def test_process_dataset(temp_dataset_dirs, mock_prepare_patches):
    """Test dataset processing task."""
    # Mock os.system to avoid actual DVC operations
    with patch('os.system') as mock_system:
        # Call the task
        result = process_dataset("test_dataset", temp_dataset_dirs)
        
        # Check that prepare_patches was called
        mock_prepare_patches.assert_called_once()
        
        # Check that DVC commands were called
        mock_system.assert_any_call("dvc add processed/test_dataset")
        mock_system.assert_any_call("dvc push")
        
        # Check result
        assert result["status"] == "success"
        assert result["dataset"] == "test_dataset"