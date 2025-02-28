import os
import mlflow
import mlflow.tensorflow
from src.config import load_config
from src.model_trainer import ModelTrainer

def run_training(config):
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id

        trainer = ModelTrainer(config)
        model = trainer.linknet_keras()
        trainer.compile_model(model)

        train_gen = trainer.train_generator(config.data.train_images, config.data.train_masks)
        val_gen = trainer.train_generator(config.data.val_images, config.data.val_masks)

        train_steps_epoch = len(os.listdir(config.data.train_images)) // config.model.batch_size
        val_steps_epoch = len(os.listdir(config.data.val_images)) // config.model.batch_size

        callbacks, tensorboard_log_dir = trainer.add_callbacks_keras(run_id)
        history = trainer.train_model_keras(model, train_gen, val_gen, train_steps_epoch, val_steps_epoch, callbacks)

        model.save(os.path.join(config.models, f'model_{run_id}.h5'))
        trainer.save_plots_keras(history, run_id)

        mlflow.log_params(config.model.dict())
        for epoch in range(config.model.epochs):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("train_iou_score", history.history['iou_score'][epoch], step=epoch)
            mlflow.log_metric("val_iou_score", history.history['val_iou_score'][epoch], step=epoch)

        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(os.path.join(config.models, run_id, 'loss.png'))
        mlflow.log_artifact(os.path.join(config.models, run_id, 'iou_score.png'))
        mlflow.log_artifact(tensorboard_log_dir)

if __name__ == "__main__":
    config = load_config('config/config.yaml', 'config/hyperparameters.yaml')
    run_training(config)