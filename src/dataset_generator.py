import os
import random
import shutil
from src.config import Config
import src.initialize  # Ensure Constants is set
from src.constants import Constants

class DatasetGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.images_patch_folder = self.config.data.raw_images_patches
        self.bin_mask_patch_folder = self.config.data.binary_masks_patches

        self.train_images_folder = self.config.data.train_images
        self.val_images_folder = self.config.data.val_images
        self.train_masks_folder = self.config.data.train_masks
        self.val_masks_folder = self.config.data.val_masks

        os.makedirs(self.train_images_folder, exist_ok=True)
        os.makedirs(self.val_images_folder, exist_ok=True)
        os.makedirs(self.train_masks_folder, exist_ok=True)
        os.makedirs(self.val_masks_folder, exist_ok=True)

        self.image_files = [f for f in os.listdir(self.images_patch_folder) if f.endswith('.png')]
        random.shuffle(self.image_files)
        split_ratio = Constants.SPLIT_RATIO
        split_index = int(len(self.image_files) * split_ratio)
        self.train_list = self.image_files[:split_index]
        self.val_list = self.image_files[split_index:]

    def generate_datasets(self):
        for image_file in self.train_list:
            shutil.copy(os.path.join(self.images_patch_folder, image_file), os.path.join(self.train_images_folder, image_file))
            shutil.copy(os.path.join(self.bin_mask_patch_folder, image_file), os.path.join(self.train_masks_folder, image_file))

        for image_file in self.val_list:
            shutil.copy(os.path.join(self.images_patch_folder, image_file), os.path.join(self.val_images_folder, image_file))
            shutil.copy(os.path.join(self.bin_mask_patch_folder, image_file), os.path.join(self.val_masks_folder, image_file))

        print("Files copied successfully!")