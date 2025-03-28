# tasks/monitoring_tasks.py
import os
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime
from celery import shared_task

from core.config.settings import get_settings
from core.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

@shared_task
def check_data_drift(reference_dataset, current_dataset, threshold=0.05):
    """Check for data drift between reference and current datasets.
    
    Args:
        reference_dataset: Name of reference dataset
        current_dataset: Name of current dataset
        threshold: P-value threshold for drift detection
        
    Returns:
        Dict with drift detection results
    """
    try:
        from scipy import stats
        
        logger.info(f"Checking data drift: {reference_dataset} vs {current_dataset}")
        
        # Load datasets
        reference_dir = os.path.join(settings.DATA_DIR, "processed", reference_dataset, "raw_images_patches")
        current_dir = os.path.join(settings.DATA_DIR, "processed", current_dataset, "raw_images_patches")
        
        if not os.path.exists(reference_dir) or not os.path.exists(current_dir):
            return {
                "status": "error", 
                "message": "Dataset directories not found"
            }
        
        # Sample images from each dataset
        reference_samples = []
        current_samples = []
        
        # Get list of files
        reference_files = [f for f in os.listdir(reference_dir) if f.endswith('.png')]
        current_files = [f for f in os.listdir(current_dir) if f.endswith('.png')]
        
        # Limit to 100 samples per dataset for performance
        max_samples = 100
        reference_files = reference_files[:max_samples]
        current_files = current_files[:max_samples]
        
        # Extract image statistics
        import cv2
        
        for f in reference_files:
            img = cv2.imread(os.path.join(reference_dir, f))
            if img is not None:
                # Calculate basic statistics
                means = np.mean(img, axis=(0, 1))
                stds = np.std(img, axis=(0, 1))
                reference_samples.append(np.concatenate([means, stds]))
        
        for f in current_files:
            img = cv2.imread(os.path.join(current_dir, f))
            if img is not None:
                # Calculate basic statistics
                means = np.mean(img, axis=(0, 1))
                stds = np.std(img, axis=(0, 1))
                current_samples.append(np.concatenate([means, stds]))
        
        # Convert to numpy arrays
        reference_array = np.array(reference_samples)
        current_array = np.array(current_samples)
        
        # Calculate drift for each feature
        drift_metrics = {}
        drift_detected = False
        
        for i in range(reference_array.shape[1]):
            feature_name = f"feature_{i}"
            ks_statistic, p_value = stats.ks_2samp(
                reference_array[:, i],
                current_array[:, i]
            )
            
            drift_metrics[feature_name] = {
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < threshold
            }
            
            if p_value < threshold:
                drift_detected = True
        
        # Log to MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name="drift_detection"):
            mlflow.log_param("reference_dataset", reference_dataset)
            mlflow.log_param("current_dataset", current_dataset)
            mlflow.log_param("threshold", threshold)
            
            for feature, metrics in drift_metrics.items():
                mlflow.log_metric(f"{feature}_ks", metrics["ks_statistic"])
                mlflow.log_metric(f"{feature}_pvalue", metrics["p_value"])
            
            mlflow.log_param("drift_detected", drift_detected)
        
        return {
            "status": "success",
            "drift_detected": drift_detected,
            "metrics": drift_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@shared_task
def check_model_performance(model_name, dataset_name):
    """Check model performance on a dataset.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Dict with model performance metrics
    """
    try:
        import tensorflow as tf
        from core.models.metrics import Metrics
        
        logger.info(f"Checking model performance: {model_name} on {dataset_name}")
        
        # Set up MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Get latest model version
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name)[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        
        # Load model
        model = mlflow.tensorflow.load_model(model_uri)
        
        # Create data generator
        val_img_dir = os.path.join(settings.DATA_DIR, "processed", dataset_name, "val_data", "val_images")
        val_mask_dir = os.path.join(settings.DATA_DIR, "processed", dataset_name, "val_data", "val_masks")
        
        if not os.path.exists(val_img_dir) or not os.path.exists(val_mask_dir):
            return {
                "status": "error", 
                "message": "Dataset directories not found"
            }
        
        # Configure data generator
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        val_datagen = ImageDataGenerator(rescale=1.0/255)
        
        val_img_generator = val_datagen.flow_from_directory(
            val_img_dir,
            target_size=(512, 512),
            color_mode="rgb",
            class_mode=None,
            batch_size=8,
            seed=42
        )
        
        val_mask_generator = val_datagen.flow_from_directory(
            val_mask_dir,
            target_size=(512, 512),
            color_mode="grayscale",
            class_mode=None,
            batch_size=8,
            seed=42
        )
        
        val_generator = zip(val_img_generator, val_mask_generator)
        
        # Evaluate model
        steps = len(os.listdir(os.path.join(val_img_dir, "val"))) // 8
        results = model.evaluate(val_generator, steps=steps)
        
        # Get metric names
        metric_names = model.metrics_names
        
        # Create metrics dict
        metrics = {name: float(value) for name, value in zip(metric_names, results)}
        
        # Log to MLflow
        with mlflow.start_run(run_name="model_performance"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_version", latest_version.version)
            mlflow.log_param("dataset_name", dataset_name)
            
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_version": latest_version.version,
            "dataset_name": dataset_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in model performance check: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }