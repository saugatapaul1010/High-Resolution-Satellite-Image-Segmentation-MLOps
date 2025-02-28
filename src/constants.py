# src/constants.py

class Constants:
    # Data paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    BINARY_MASKS_DIR = "data/binary_masks"
    BINARY_MASKS_PATCHES_DIR = "data/binary_masks_patches"
    RAW_IMAGES_PATCHES_DIR = "data/raw_images_patches"
    TRAIN_IMAGES_DIR = "data/train/images"
    TRAIN_MASKS_DIR = "data/train/masks"
    VAL_IMAGES_DIR = "data/val/images"
    VAL_MASKS_DIR = "data/val/masks"
    MODELS_DIR = "models"
    TENSORBOARD_LOGS_DIR = "logs/tensorboard"

    # Model hyperparameters
    BACKBONE = "resnet34"
    ACTIVATION = "sigmoid"
    NUM_CLASSES = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    INPUT_CHANNELS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    EPOCHS = 50
    EARLYSTOP_PATIENCE = 10
    EARLYSTOP_MIN_DELTA = 0.001
    REDUCELR_FACTOR = 0.2
    REDUCELR_PATIENCE = 5
    VERBOSE = 1
    SEED = 42
    RESCALE = 1./255
    PATCH_SIZE = 256

    # General settings
    CALLBACK_MODE = "max"
    COLOR_MODE_RGB = "rgb"
    COLOR_MODE_GRAYSCALE = "grayscale"

# Note: These values are placeholders. In a real application, they could be set by a separate script
# that reads YAML files once and updates this class, but that step is outside the main logic.