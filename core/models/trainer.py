# core/models/trainer.py
import os
import mlflow
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from core.models.metrics import Metrics
from core.utils.logging import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Model trainer for satellite image segmentation."""
    
    def __init__(self, config):
        """Initialize model trainer.
        
        Args:
            config: Configuration dictionary with model_type, dataset_name, parameters, paths
        """
        self.config = config
        self.model_type = config["model_type"]
        self.dataset_name = config["dataset_name"]
        self.parameters = config["parameters"]
        self.paths = config["paths"]
        
        # Setup paths
        os.makedirs(self.paths["models_dir"], exist_ok=True)
        os.makedirs(self.paths["output_dir"], exist_ok=True)
        
        # Set training parameters with defaults
        self.batch_size = self.parameters.get("batch_size", 4)
        self.learning_rate = self.parameters.get("learning_rate", 0.001)
        self.epochs = self.parameters.get("epochs", 5)
        self.backbone = self.parameters.get("backbone", "efficientnetb0")
        self.img_size = self.parameters.get("img_size", (512, 512))
        self.seed = self.parameters.get("seed", 42)
        
        logger.info(f"Initialized trainer for {self.model_type} model with {self.backbone} backbone")
    
    def build_model(self):
        """Build segmentation model.
        
        Returns:
            TensorFlow Keras model
        """
        logger.info(f"Building {self.model_type} model with {self.backbone} backbone")
        
        img_shape = (*self.img_size, 3)  # (width, height, channels)
        
        if self.model_type.lower() == "linknet":
            model = sm.Linknet(
                backbone_name=self.backbone,
                input_shape=img_shape,
                activation="sigmoid",
                classes=1,
                encoder_weights="imagenet"
            )
        elif self.model_type.lower() == "unet":
            model = sm.Unet(
                backbone_name=self.backbone,
                input_shape=img_shape,
                activation="sigmoid",
                classes=1,
                encoder_weights="imagenet"
            )
        elif self.model_type.lower() == "manet":
            model = sm.MAnet(
                backbone_name=self.backbone,
                input_shape=img_shape,
                activation="sigmoid",
                classes=1,
                encoder_weights="imagenet"
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def compile_model(self, model):
        """Compile the model with optimizer and loss function.
        
        Args:
            model: TensorFlow Keras model
            
        Returns:
            Compiled model
        """
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=Metrics.dice_coef_loss,
            metrics=[Metrics.dice_coef, Metrics.iou_score, 'accuracy']
        )
        
        logger.info(f"Compiled model with learning rate: {self.learning_rate}")
        return model
    
    def create_callbacks(self):
        """Create callbacks for model training.
        
        Returns:
            List of callbacks
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.paths["models_dir"], "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create tensorboard directory
        tensorboard_dir = os.path.join(self.paths["output_dir"], "tensorboard_logs")
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Create callbacks
        early_stop = EarlyStopping(
            monitor="val_iou_score",
            min_delta=0,
            patience=10,
            verbose=1,
            mode="max"
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_iou_score",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode="max"
        )
        
        checkpoint_path = os.path.join(checkpoint_dir, "model_{epoch:02d}.h5")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_iou_score",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max"
        )
        
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch"
        )
        
        return [early_stop, reduce_lr, checkpoint, tensorboard]
    
    def create_data_generators(self):
        """Create data generators for training and validation.
        
        Returns:
            Tuple of (train_generator, val_generator, train_steps, val_steps)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            zoom_range=0.2
        )
        
        # No augmentation for validation, just rescaling
        val_datagen = ImageDataGenerator(rescale=1.0/255)
        
        # Training data generator
        train_img_dir = os.path.join(self.paths["data_dir"], "train_data", "train_images")
        train_mask_dir = os.path.join(self.paths["data_dir"], "train_data", "train_masks")
        
        train_img_generator = train_datagen.flow_from_directory(
            train_img_dir,
            target_size=self.img_size,
            color_mode="rgb",
            class_mode=None,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        train_mask_generator = train_datagen.flow_from_directory(
            train_mask_dir,
            target_size=self.img_size,
            color_mode="grayscale",
            class_mode=None,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        # Validation data generator
        val_img_dir = os.path.join(self.paths["data_dir"], "val_data", "val_images")
        val_mask_dir = os.path.join(self.paths["data_dir"], "val_data", "val_masks")
        
        val_img_generator = val_datagen.flow_from_directory(
            val_img_dir,
            target_size=self.img_size,
            color_mode="rgb",
            class_mode=None,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        val_mask_generator = val_datagen.flow_from_directory(
            val_mask_dir,
            target_size=self.img_size,
            color_mode="grayscale",
            class_mode=None,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        # Combine generators
        train_generator = zip(train_img_generator, train_mask_generator)
        val_generator = zip(val_img_generator, val_mask_generator)
        
        # Calculate steps per epoch
        train_samples = len(os.listdir(os.path.join(train_img_dir, "train")))
        val_samples = len(os.listdir(os.path.join(val_img_dir, "val")))
        
        train_steps = train_samples // self.batch_size
        val_steps = val_samples // self.batch_size
        
        logger.info(f"Training samples: {train_samples}, steps: {train_steps}")
        logger.info(f"Validation samples: {val_samples}, steps: {val_steps}")
        
        return train_generator, val_generator, train_steps, val_steps
    
    def train(self):
        """Train the model.
        
        Returns:
            Tuple of (trained model, training history)
        """
        logger.info("Starting model training")
        
        # Build and compile model
        model = self.build_model()
        model = self.compile_model(model)
        
        # Create data generators
        train_gen, val_gen, train_steps, val_steps = self.create_data_generators()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.paths["models_dir"], "final_model.h5")
        model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        return model, history