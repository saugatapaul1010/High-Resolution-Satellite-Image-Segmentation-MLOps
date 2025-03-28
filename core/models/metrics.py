# core/models/metrics.py
"""Metrics for model evaluation."""
import tensorflow as tf
from tensorflow.keras import backend as K

class Metrics:
    """Metrics for segmentation models."""
    
    @staticmethod
    def iou_score(y_true, y_pred, smooth=1):
        """Calculate IoU score.
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            smooth: Smoothing factor
            
        Returns:
            IoU score
        """
        # Flatten the tensors
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # Calculate intersection and union
        intersection = K.sum(K.abs(y_true * y_pred))
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        
        # Calculate IoU
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        """Calculate Dice coefficient.
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        """Calculate Dice coefficient loss.
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            
        Returns:
            Dice coefficient loss
        """
        return 1 - Metrics.dice_coef(y_true, y_pred)