# tests/test_core/test_model_trainer.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf

from core.models.trainer import ModelTrainer

@pytest.fixture
def sample_trainer_config():
    """Sample trainer configuration."""
    return {
        "model_type": "linknet",
        "dataset_name": "test_dataset",
        "parameters": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "epochs": 1,
            "backbone": "efficientnetb0",
            "img_size": (128, 128),
            "seed": 42
        },
        "paths": {
            "data_dir": "/tmp/data",
            "models_dir": "/tmp/models",
            "output_dir": "/tmp/output"
        }
    }

@pytest.mark.skip(reason="Requires TensorFlow and segmentation_models")
def test_build_model(sample_trainer_config):
    """Test model building."""
    # Mock segmentation_models
    with patch('segmentation_models.Linknet') as mock_linknet:
        # Create trainer
        trainer = ModelTrainer(sample_trainer_config)
        
        # Build model
        model = trainer.build_model()
        
        # Verify that Linknet was called with correct arguments
        mock_linknet.assert_called_once_with(
            backbone_name='efficientnetb0',
            input_shape=(128, 128, 3),
            activation='sigmoid',
            classes=1,
            encoder_weights='imagenet'
        )

@pytest.mark.skip(reason="Requires TensorFlow and segmentation_models")
def test_compile_model(sample_trainer_config):
    """Test model compilation."""
    # Create a mock model
    mock_model = MagicMock()
    
    # Create trainer
    trainer = ModelTrainer(sample_trainer_config)
    
    # Compile model
    compiled_model = trainer.compile_model(mock_model)
    
    # Verify that model.compile was called
    mock_model.compile.assert_called_once()
    
    # Check if the compiled model is returned
    assert compiled_model == mock_model

@pytest.mark.skip(reason="Requires TensorFlow and segmentation_models")
def test_create_callbacks(sample_trainer_config):
    """Test callback creation."""
    # Create trainer
    trainer = ModelTrainer(sample_trainer_config)
    
    # Create callbacks
    callbacks = trainer.create_callbacks()
    
    # Check that we have the expected number of callbacks
    assert len(callbacks) == 4
    
    # Check that we have the expected types of callbacks
    callback_types = [type(callback).__name__ for callback in callbacks]
    assert 'EarlyStopping' in callback_types
    assert 'ReduceLROnPlateau' in callback_types
    assert 'ModelCheckpoint' in callback_types
    assert 'TensorBoard' in callback_types