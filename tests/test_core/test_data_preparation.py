# tests/test_core/test_data_preparation.py
import os
import numpy as np
import cv2
import pytest
import json

from core.data.data_preparation import DataPreparation

def test_create_binary_masks(sample_image):
    """Test binary mask creation."""
    # Create shape dictionaries
    shape_dicts = [
        {
            "points": [[64, 64], [192, 64], [192, 192], [64, 192]]
        }
    ]
    
    # Create binary mask
    mask = DataPreparation.create_binary_masks(sample_image, shape_dicts)
    
    # Check mask shape
    assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
    
    # Check that the mask has the correct values
    assert np.sum(mask[64:192, 64:192]) > 0
    assert np.sum(mask[0:64, 0:64]) == 0

def test_pad_input_img(sample_image, sample_config):
    """Test image padding."""
    # Get patch size
    patch_size = sample_config["data"]["processing"]["patch_size"]
    
    # Create data preparation instance
    data_prep = DataPreparation(sample_config)
    
    # Pad image
    padded_img = data_prep.pad_input_img(sample_image, patch_size)
    
    # Check that padded image dimensions are divisible by patch size
    assert padded_img.shape[0] % patch_size == 0
    assert padded_img.shape[1] % patch_size == 0
    assert padded_img.shape[2] == sample_image.shape[2]

def test_closest_number():
    """Test closest number calculation."""
    # Test cases
    assert DataPreparation.closest_number(10, 5) == 10
    assert DataPreparation.closest_number(12, 5) == 10
    assert DataPreparation.closest_number(13, 5) == 15
    assert DataPreparation.closest_number(0, 5) == 0
    assert DataPreparation.closest_number(-7, 5) == -5
    assert DataPreparation.closest_number(-12, 5) == -10

def test_pix_add(sample_image):
    """Test pixel padding calculation."""
    # Test with different patch sizes
    h_pad, w_pad = DataPreparation.pix_add(sample_image, 128)
    
    # Check that the resulting dimensions are divisible by the patch size
    assert (sample_image.shape[0] + h_pad) % 128 == 0
    assert (sample_image.shape[1] + w_pad) % 128 == 0