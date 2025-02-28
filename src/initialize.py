# src/initialize.py

from .config import load_config
from .constants import Constants

# Load configuration and set Constants
config = load_config('config/config.yaml', 'config/hyperparameters.yaml')
Constants.set_config(config)

