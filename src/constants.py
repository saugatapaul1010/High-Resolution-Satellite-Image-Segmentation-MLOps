# src/constants.py

class Constants:
    @classmethod
    def set_config(cls, config):
        # Data paths
        cls.RAW = config.data.raw
        cls.PROCESSED = config.data.processed
        cls.BINARY_MASKS = config.data.binary_masks
        cls.BINARY_MASKS_PATCHES = config.data.binary_masks_patches
        cls.RAW_IMAGES_PATCHES = config.data.raw_images_patches
        cls.TRAIN_IMAGES = config.data.train_images
        cls.TRAIN_MASKS = config.data.train_masks
        cls.VAL_IMAGES = config.data.val_images
        cls.VAL_MASKS = config.data.val_masks
        cls.MODELS = config.models
        cls.TENSORBOARD_LOGS = config.tensorboard_logs
        cls.CONFIG_PATH = config.config_path
        cls.HYPERPARAMS_PATH = config.hyperparams_path

        # Model hyperparameters
        cls.BACKBONE = config.model.backbone
        cls.ACTIVATION = config.model.activation
        cls.NUM_CLASSES = config.model.num_classes
        cls.IMG_WIDTH = config.model.img_width
        cls.IMG_HEIGHT = config.model.img_height
        cls.INPUT_CHANNELS = config.model.input_channels
        cls.BATCH_SIZE = config.model.batch_size
        cls.LEARNING_RATE = config.model.learning_rate
        cls.EPOCHS = config.model.epochs
        cls.EARLYSTOP_PATIENCE = config.model.earlystop_patience
        cls.EARLYSTOP_MIN_DELTA = config.model.earlystop_min_delta
        cls.REDUCELR_FACTOR = config.model.reducelr_factor
        cls.REDUCELR_PATIENCE = config.model.reducelr_patience
        cls.VERBOSE = config.model.verbose
        cls.SEED = config.model.seed
        cls.RESCALE = config.model.rescale
        cls.PATCH_SIZE = config.model.patch_size

        # General settings
        cls.CALLBACK_MODE = config.general.callback_mode
        cls.COLOR_MODE_RGB = config.general.color_mode_rgb
        cls.COLOR_MODE_GRAYSCALE = config.general.color_mode_grayscale
        cls.SPLIT_RATIO = config.general.split_ratio