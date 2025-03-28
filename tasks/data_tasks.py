# tasks/data_tasks.py
import os
import shutil
import numpy as np
import cv2
import json
import random
from celery import shared_task
from tqdm import tqdm
from patchify import patchify

from core.data.data_preparation import DataPreparation
from core.utils.logging import get_logger

logger = get_logger(__name__)

@shared_task
def process_dataset(dataset_name, data_dir):
    """Process a dataset into training patches.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Data directory root
        
    Returns:
        Dict with processing results
    """
    try:
        # Set up paths
        source_dir = os.path.join(data_dir, "raw", dataset_name)
        processed_dir = os.path.join(data_dir, "processed", dataset_name)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Set up subdirectories
        binary_mask_dir = os.path.join(processed_dir, "binary_mask")
        binary_masks_patches_dir = os.path.join(processed_dir, "binary_masks_patches")
        raw_images_patches_dir = os.path.join(processed_dir, "raw_images_patches")
        
        os.makedirs(binary_mask_dir, exist_ok=True)
        os.makedirs(binary_masks_patches_dir, exist_ok=True)
        os.makedirs(raw_images_patches_dir, exist_ok=True)
        
        # Get all JSON annotation files
        json_files = [
            os.path.join(source_dir, f) for f in os.listdir(source_dir) 
            if f.lower().endswith(".json")
        ]
        
        # Create configuration
        config = {
            "data": {
                "processing": {
                    "patch_size": 512,
                    "binary_mask_dir": binary_mask_dir,
                    "binary_masks_patches_dir": binary_masks_patches_dir,
                    "raw_images_patches_dir": raw_images_patches_dir
                }
            }
        }
        
        # Process data
        data_prep = DataPreparation(config)
        data_prep.prepare_patches(json_files)
        
        # Generate train/val split
        train_dir = os.path.join(processed_dir, "train_data")
        val_dir = os.path.join(processed_dir, "val_data")
        
        os.makedirs(os.path.join(train_dir, "train_images", "train"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "train_masks", "train"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "val_images", "val"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "val_masks", "val"), exist_ok=True)
        
        # Get all image files in patches dir
        image_files = [
            f for f in os.listdir(raw_images_patches_dir) 
            if f.lower().endswith(".png")
        ]
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(image_files)
        
        split_ratio = 0.9
        split_index = int(len(image_files) * split_ratio)
        train_list = image_files[:split_index]
        val_list = image_files[split_index:]
        
        # Copy files to respective directories
        for image_file in train_list:
            shutil.copy(
                os.path.join(raw_images_patches_dir, image_file),
                os.path.join(train_dir, "train_images", "train", image_file)
            )
            shutil.copy(
                os.path.join(binary_masks_patches_dir, image_file),
                os.path.join(train_dir, "train_masks", "train", image_file)
            )
        
        for image_file in val_list:
            shutil.copy(
                os.path.join(raw_images_patches_dir, image_file),
                os.path.join(val_dir, "val_images", "val", image_file)
            )
            shutil.copy(
                os.path.join(binary_masks_patches_dir, image_file),
                os.path.join(val_dir, "val_masks", "val", image_file)
            )
        
        # Record the dataset in DVC
        os.chdir(data_dir)
        os.system(f"dvc add processed/{dataset_name}")
        os.system(f"dvc push")
        
        return {
            "status": "success",
            "dataset": dataset_name,
            "training_samples": len(train_list),
            "validation_samples": len(val_list)
        }
    
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }