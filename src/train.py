import os
import mlflow
import mlflow.tensorflow
import src.initialize  # Ensure Constants is set
from src.constants import Constants
from src.model_trainer import ModelTrainer

def run_training(config):  # Kept config param for compatibility, though unused
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id

        trainer = ModelTrainer(None)  # Pass None since config is not used
        model = trainer.linknet_keras()
        trainer.compile_model(model)

        train_gen = trainer.train_generator(Constants.TRAIN_IMAGES, Constants.TRAIN_MASKS)
        val_gen = trainer.train_generator(Constants.VAL_IMAGES, Constants.VAL_MASKS)

        train_steps_epoch = len(os.listdir(Constants.TRAIN_IMAGES)) // Constants.BATCH_SIZE
        val_steps_epoch = len(os.listdir(Constants.VAL_IMAGES)) // Constants.BATCH_SIZE

        callbacks, tensorboard_log_dir = trainer.add_callbacks_keras(run_id)
        history = trainer.train_model_keras(model, train_gen, val_gen, train_steps_epoch, val_steps_epoch, callbacks)

        model.save(os.path.join(Constants.MODELS, f'model_{run_id}.h5'))
        trainer.save_plots_keras(history, run_id)

        # Log parameters (manually since config.model.dict() is no longer available)
        mlflow.log_param("backbone", Constants.BACKBONE)
        mlflow.log_param("activation", Constants.ACTIVATION)
        mlflow.log_param("num_classes", Constants.NUM_CLASSES)
        mlflow.log_param("img_width", Constants.IMG_WIDTH)
        mlflow.log_param("img_height", Constants.IMG_HEIGHT)
        mlflow.log_param("input_channels", Constants.INPUT_CHANNELS)
        mlflow.log_param("batch_size", Constants.BATCH_SIZE)
        mlflow.log_param("learning_rate", Constants.LEARNING_RATE)
        mlflow.log_param("epochs", Constants.EPOCHS)
        mlflow.log_param("earlystop_patience", Constants.EARLYSTOP_PATIENCE)
        mlflow.log_param("earlystop_min_delta", Constants.EARLYSTOP_MIN_DELTA)
        mlflow.log_param("reducelr_factor", Constants.REDUCELR_FACTOR)
        mlflow.log_param("reducelr_patience", Constants.REDUCELR_PATIENCE)
        mlflow.log_param("verbose", Constants.VERBOSE)
        mlflow.log_param("seed", Constants.SEED)
        mlflow.log_param("rescale", Constants.RESCALE)
        mlflow.log_param("patch_size", Constants.PATCH_SIZE)

        for epoch in range(Constants.EPOCHS):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("train_iou_score", history.history['iou_score'][epoch], step=epoch)
            mlflow.log_metric("val_iou_score", history.history['val_iou_score'][epoch], step=epoch)

        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(os.path.join(Constants.MODELS, run_id, 'loss.png'))
        mlflow.log_artifact(os.path.join(Constants.MODELS, run_id, 'iou_score.png'))
        mlflow.log_artifact(tensorboard_log_dir)

if __name__ == "__main__":
    run_training(None)  # No config needed since Constants is set via src/initialize.py

# Critical Comment: The config parameter is retained but unused. MLflow logging now manually lists parameters
# from Constants since we no longer pass a Config object.