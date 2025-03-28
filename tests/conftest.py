# tests/conftest.py
import os
import pytest
from fastapi.testclient import TestClient
import tempfile
import numpy as np
import cv2
import json

from api.main import app

@pytest.fixture
def api_client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data": {
            "processing": {
                "patch_size": 128,  # Smaller for tests
                "binary_mask_dir": tempfile.mkdtemp(),
                "binary_masks_patches_dir": tempfile.mkdtemp(),
                "raw_images_patches_dir": tempfile.mkdtemp()
            }
        },
        "paths": {
            "data_dir": tempfile.mkdtemp(),
            "models_dir": tempfile.mkdtemp(),
            "output_dir": tempfile.mkdtemp()
        }
    }

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create a simple pattern
    img[64:192, 64:192] = [255, 0, 0]
    return img

@pytest.fixture
def sample_json(tmp_path, sample_image):
    """Create a sample annotation JSON for testing."""
    # Save the sample image
    img_path = os.path.join(tmp_path, "sample.png")
    cv2.imwrite(img_path, sample_image)
    
    # Create annotation
    annotation = {
        "shapes": [
            {
                "label": "feature",
                "points": [[64, 64], [192, 64], [192, 192], [64, 192]]
            }
        ],
        "imagePath": img_path,
        "imageHeight": 256,
        "imageWidth": 256
    }
    
    # Save annotation
    json_path = os.path.join(tmp_path, "sample.json")
    with open(json_path, "w") as f:
        json.dump(annotation, f)
    
    return json_path, img_path