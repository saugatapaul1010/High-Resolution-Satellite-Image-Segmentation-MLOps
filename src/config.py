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
    split_ratio: float  # Added for dataset split ratio

class Config(BaseModel):
    data: DataConfig
    models: str
    tensorboard_logs: str
    config_path: str       # Added for config file path
    hyperparams_path: str  # Added for hyperparams file path
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
    config_data['config_path'] = config_path  # Set dynamically
    config_data['hyperparams_path'] = hyperparams_path  # Set dynamically
    return Config(**config_data)