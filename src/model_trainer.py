import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import segmentation_models as sm
import plotly.express as px
from src.config import Config
from src.metrics import Metrics

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def linknet_keras(self):
        model = sm.Linknet(
            backbone_name=self.config.model.backbone,
            input_shape=(self.config.model.img_height, self.config.model.img_width, self.config.model.input_channels),
            activation=self.config.model.activation,
            classes=self.config.model.num_classes,
            encoder_weights='imagenet'
        )
        return model

    def compile_model(self, model):
        model.compile(
            optimizer=Adam(learning_rate=self.config.model.learning_rate),
            loss=sm.losses.DiceLoss(),
            metrics=[sm.metrics.IOUScore()]
        )

    def add_callbacks_keras(self, run_id):
        early_stop = EarlyStopping(
            monitor='val_iou_score',
            min_delta=self.config.model.earlystop_min_delta,
            patience=self.config.model.earlystop_patience,
            verbose=self.config.model.verbose,
            mode=self.config.general.callback_mode
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_iou_score',
            factor=self.config.model.reducelr_factor,
            patience=self.config.model.reducelr_patience,
            verbose=self.config.model.verbose,
            mode=self.config.general.callback_mode
        )

        tensorboard_log_dir = os.path.join(self.config.tensorboard_logs, run_id)
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir)

        checkpoint_path = os.path.join(self.config.models, 'model_checkpoint_{epoch}.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_iou_score',
            verbose=self.config.model.verbose,
            save_best_only=False,
            save_weights_only=False,
            mode=self.config.general.callback_mode,
            period=10
        )

        return [early_stop, reduce_lr, tensorboard, checkpoint], tensorboard_log_dir

    def train_generator(self, img_path, mask_path):
        img_data_gen_args = dict(
            rescale=1./self.config.model.rescale,
            horizontal_flip=True,
            vertical_flip=True
        )
        image_datagen = ImageDataGenerator(**img_data_gen_args)
        mask_datagen = ImageDataGenerator(**img_data_gen_args)

        image_generator = image_datagen.flow_from_directory(
            img_path,
            class_mode=None,
            color_mode=self.config.general.color_mode_rgb,
            target_size=(self.config.model.img_height, self.config.model.img_width),
            batch_size=self.config.model.batch_size,
            seed=self.config.model.seed
        )

        mask_generator = mask_datagen.flow_from_directory(
            mask_path,
            class_mode=None,
            color_mode=self.config.general.color_mode_grayscale,
            target_size=(self.config.model.img_height, self.config.model.img_width),
            batch_size=self.config.model.batch_size,
            seed=self.config.model.seed
        )

        return zip(image_generator, mask_generator)

    def train_model_keras(self, model, train_gen, val_gen, train_steps_epoch, val_steps_epoch, callbacks):
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps_epoch,
            epochs=self.config.model.epochs,
            verbose=self.config.model.verbose,
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

        plots_dir = os.path.join(self.config.models, run_id)
        os.makedirs(plots_dir, exist_ok=True)
        fig1.write_image(os.path.join(plots_dir, 'loss.png'))
        fig2.write_image(os.path.join(plots_dir, 'iou_score.png'))