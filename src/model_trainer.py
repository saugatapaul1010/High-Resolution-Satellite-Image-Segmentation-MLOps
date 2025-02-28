# src/model_trainer.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import segmentation_models as sm
import plotly.express as px
import src.initialize  # Ensure Constants is set
from .constants import Constants
from .metrics import Metrics

class ModelTrainer:
    def __init__(self, config):  # Kept config param for compatibility, though unused
        pass  # No need for self.config since we use Constants

    def linknet_keras(self):
        model = sm.Linknet(
            backbone_name=Constants.BACKBONE,
            input_shape=(Constants.IMG_HEIGHT, Constants.IMG_WIDTH, Constants.INPUT_CHANNELS),
            activation=Constants.ACTIVATION,
            classes=Constants.NUM_CLASSES,
            encoder_weights='imagenet'
        )
        return model

    def compile_model(self, model):
        model.compile(
            optimizer=Adam(learning_rate=Constants.LEARNING_RATE),
            loss=sm.losses.DiceLoss(),
            metrics=[sm.metrics.IOUScore()]
        )

    def add_callbacks_keras(self, run_id):
        early_stop = EarlyStopping(
            monitor='val_iou_score',
            min_delta=Constants.EARLYSTOP_MIN_DELTA,
            patience=Constants.EARLYSTOP_PATIENCE,
            verbose=Constants.VERBOSE,
            mode=Constants.CALLBACK_MODE
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_iou_score',
            factor=Constants.REDUCELR_FACTOR,
            patience=Constants.REDUCELR_PATIENCE,
            verbose=Constants.VERBOSE,
            mode=Constants.CALLBACK_MODE
        )

        tensorboard_log_dir = os.path.join(Constants.TENSORBOARD_LOGS, run_id)
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir)

        checkpoint_path = os.path.join(Constants.MODELS, 'model_checkpoint_{epoch}.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_iou_score',
            verbose=Constants.VERBOSE,
            save_best_only=False,
            save_weights_only=False,
            mode=Constants.CALLBACK_MODE,
            period=10
        )

        return [early_stop, reduce_lr, tensorboard, checkpoint], tensorboard_log_dir

    def train_generator(self, img_path, mask_path):
        img_data_gen_args = dict(
            rescale=1./Constants.RESCALE,
            horizontal_flip=True,
            vertical_flip=True
        )
        image_datagen = ImageDataGenerator(**img_data_gen_args)
        mask_datagen = ImageDataGenerator(**img_data_gen_args)

        image_generator = image_datagen.flow_from_directory(
            img_path,
            class_mode=None,
            color_mode=Constants.COLOR_MODE_RGB,
            target_size=(Constants.IMG_HEIGHT, Constants.IMG_WIDTH),
            batch_size=Constants.BATCH_SIZE,
            seed=Constants.SEED
        )

        mask_generator = mask_datagen.flow_from_directory(
            mask_path,
            class_mode=None,
            color_mode=Constants.COLOR_MODE_GRAYSCALE,
            target_size=(Constants.IMG_HEIGHT, Constants.IMG_WIDTH),
            batch_size=Constants.BATCH_SIZE,
            seed=Constants.SEED
        )

        return zip(image_generator, mask_generator)

    def train_model_keras(self, model, train_gen, val_gen, train_steps_epoch, val_steps_epoch, callbacks):
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps_epoch,
            epochs=Constants.EPOCHS,
            verbose=Constants.VERBOSE,
            validation_data=val_gen,
            validation_steps=val_steps_epoch,
            callbacks=callbacks
        )
        return history

    def save_plots_keras(self, history, run_id):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        iou = history.history['iou_score']
        val_iou = history.history['val_iou_score']

        loss_hist = {'loss': loss, 'val_loss': val_loss}
        iou_hist = {'iou_score': iou, 'val_iou_score': val_iou}

        fig1 = px.line(loss_hist, title='Dice Loss vs Epochs').update_layout(xaxis_title='Epochs', yaxis_title='Dice Loss')
        fig2 = px.line(iou_hist, title='IOU Score vs Epochs').update_layout(xaxis_title='Epochs', yaxis_title='IOU Score')

        plots_dir = os.path.join(Constants.MODELS, run_id)
        os.makedirs(plots_dir, exist_ok=True)
        fig1.write_image(os.path.join(plots_dir, 'loss.png'))
        fig2.write_image(os.path.join(plots_dir, 'iou_score.png'))

# Critical Comment: The rescale value is used as 1./Constants.RESCALE to normalize images correctly,
# assuming the YAML value (255.0) is the divisor. Verify this matches your data preprocessing intent.