from pydantic import BaseModel
from typing import Dict, List

class DataConfig(BaseModel):
    raw: str
    processed: str
    binary_masks: str
    binary_masks_patches: str
    raw_images_patches: str
    train_images: str
    train_masks: str
    val_images: str
    val_masks: str

class ModelConfig(BaseModel):
    backbone: str
    activation: str
    num_classes: int
    img_width: int
    img_height: int
    input_channels: int
    batch_size: int
    learning_rate: float
    epochs: int
    earlystop_patience: int
    earlystop_min_delta: float
    reducelr_factor: float
    reducelr_patience: int
    verbose: int
    seed: int
    rescale: float
    patch_size: int

class GeneralConfig(BaseModel):
    callback_mode: str
    color_mode_rgb: str
    color_mode_grayscale: str

class Config(BaseModel):
    data: DataConfig
    models: str
    tensorboard_logs: str
    model: ModelConfig
    general: GeneralConfig

def load_config(config_path: str, hyperparams_path: str) -> Config:
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    with open(hyperparams_path, 'r') as f:
        hyperparams_data = yaml.safe_load(f)
    config_data['model'] = hyperparams_data['model']
    config_data['general'] = hyperparams_data['general']
    return Config(**config_data)