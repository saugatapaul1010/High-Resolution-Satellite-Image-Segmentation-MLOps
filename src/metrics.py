# src/metrics.py

from tensorflow.keras import backend as K

class Metrics:
    @staticmethod
    def iou_score(y_pred, y_true, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return 1 - Metrics.dice_coef(y_true, y_pred)