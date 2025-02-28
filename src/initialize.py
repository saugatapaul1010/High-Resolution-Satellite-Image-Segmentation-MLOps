from config import load_config
from constants import Constants

# Load configuration and set Constants
config = load_config('config/config.yaml', 'config/hyperparameters.yaml')
Constants.set_config(config)

# Critical Comment: This script assumes the YAML files are at fixed paths. In a production environment,
# these paths should be configurable via environment variables or command-line arguments to avoid hardcoding.