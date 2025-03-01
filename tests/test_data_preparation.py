# tests/test_data_preparation.py

import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, mock_open
import cv2


# Add project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
#import src.initialize

from src.data_preparation import DataPreparation
from src.constants import Constants

# Mock Constants class with necessary attributes
@pytest.fixture
def mock_constants():
    """Fixture to mock the Constants class with required attributes."""
    mock = Mock(spec=Constants)
    mock.RAW = "data/raw"
    mock.PROCESSED = "data/processed"
    mock.BINARY_MASKS_PATCHES = "data/processed/binary_masks_patches"
    mock.RAW_IMAGES_PATCHES = "data/processed/raw_images_patches"
    mock.BINARY_MASKS = "data/processed/binary_masks"
    mock.PATCH_SIZE = 512
    return mock

@pytest.fixture
def data_preparation(mock_constants):
    """Fixture to instantiate DataPreparation with mocked Constants."""
    with patch('src.data_preparation.Constants', mock_constants):
        dp = DataPreparation(None)  # Config is unused
        return dp

def test_get_poly():
    """Test the get_poly method with a mocked JSON file."""
    sample_json = '{"shapes": [{"points": [[100, 100], [200, 100], [200, 200], [100, 200]]}]}'
    with patch("builtins.open", mock_open(read_data=sample_json)):
        dp = DataPreparation(None)
        shapes = dp.get_poly("dummy/path.json")
        assert len(shapes) == 1
        assert shapes[0]["points"] == [[100, 100], [200, 100], [200, 200], [100, 200]]

def test_create_binary_masks():
    """Test create_binary_masks with a sample image and shapes."""
    dp = DataPreparation(None)
    # Mock a 512x512 image (RGB)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    shape_dicts = [{"points": [[100, 100], [200, 100], [200, 200], [100, 200]]}]
    
    binary_mask = dp.create_binary_masks(image, shape_dicts)
    assert binary_mask.shape == (512, 512)
    assert binary_mask.dtype == np.float32
    assert np.max(binary_mask) == 255  # Filled polygon
    assert np.sum(binary_mask) > 0  # Non-zero area filled

def test_pad_input_img_single_channel():
    """Test padding a single-channel image."""
    dp = DataPreparation(None)
    image = np.ones((500, 500), dtype=np.float32)  # Smaller than patch_size
    padded = dp.pad_input_img_single_channel(image, 512)
    
    assert padded.shape == (512, 512)  # Padded to nearest multiple + 512
    assert np.all(padded[:500, :500] == 1)  # Original content preserved
    assert np.all(padded[500:, :] == 0)  # Padding is zeros
    assert np.all(padded[:, 500:] == 0)

def test_pad_input_img():
    """Test padding a 3-channel image."""
    dp = DataPreparation(None)
    image = np.ones((500, 500, 3), dtype=np.uint8)
    padded = dp.pad_input_img(image, 512)
    
    assert padded.shape == (512, 512, 3)
    assert np.all(padded[:500, :500, :] == 1)
    assert np.all(padded[500:, :, :] == 0)
    assert np.all(padded[:, 500:, :] == 0)

def test_prepare_patches(data_preparation, mocker):
    """Test prepare_patches with mocked file system and dependencies."""
    dp = data_preparation
    
    # Mock os.listdir to return a sample JSON file
    mocker.patch('os.listdir', return_value=['test.json'])
    mocker.patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    
    # Mock cv2.imread to return a sample image
    sample_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mocker.patch('cv2.imread', return_value=sample_image)
    
    # Mock get_poly
    sample_shapes = [{"points": [[100, 100], [200, 100], [200, 200], [100, 200]]}]
    mocker.patch.object(dp, 'get_poly', return_value=sample_shapes)
    
    # Mock cv2.imwrite to avoid actual file writing
    mocker.patch('cv2.imwrite', return_value=None)
    
    # Run the method
    dp.prepare_patches()
    
    # Verify directories were created
    os.makedirs.assert_any_call("data/processed/binary_masks_patches")
    os.makedirs.assert_any_call("data/processed/raw_images_patches")
    os.makedirs.assert_any_call("data/processed/binary_masks")
    
    # Verify patchify was called (indirectly via shape checks)
    assert cv2.imwrite.call_count > 0  # At least some patches were "written"

if __name__ == "__main__":
    pytest.main(["-v"])