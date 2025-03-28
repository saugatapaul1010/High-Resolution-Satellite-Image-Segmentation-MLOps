# core/data/data_preparation.py
"""Data preparation module for satellite image segmentation."""
import json
import os
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from patchify import patchify
from tqdm import tqdm

from core.utils.logging import get_logger

logger = get_logger(__name__)


class DataPreparation:
    """Data preparation class for satellite image segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data preparation.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.patch_size = config["data"]["processing"]["patch_size"]
        self.binary_mask_dir = config["data"]["processing"]["binary_mask_dir"]
        self.binary_masks_patches_dir = config["data"]["processing"]["binary_masks_patches_dir"]
        self.raw_images_patches_dir = config["data"]["processing"]["raw_images_patches_dir"]
        
        # Create directories if they don't exist
        os.makedirs(self.binary_mask_dir, exist_ok=True)
        os.makedirs(self.binary_masks_patches_dir, exist_ok=True)
        os.makedirs(self.raw_images_patches_dir, exist_ok=True)
    
    def prepare_patches(self, json_files: List[str]) -> None:
        """Prepare patches from satellite images.
        
        Args:
            json_files: List of JSON annotation files
        """
        logger.info("Creating Binary Masks...")
        for filename in tqdm(json_files):
            png_filename = filename.split(".")[0] + ".png"
            annotation_path = filename
            
            # Skip if PNG file doesn't exist
            if not os.path.exists(png_filename):
                logger.warning(f"PNG file not found: {png_filename}")
                continue
            
            # Get base filename without directory and extension
            base_filename = os.path.basename(filename).split(".")[0]
            
            png_image = cv2.imread(png_filename)
            shape_dicts = self.get_poly(annotation_path)
            im_binary = self.create_binary_masks(png_image, shape_dicts)
            
            # Save binary mask
            mask_output_path = os.path.join(self.binary_mask_dir, f"{base_filename}.png")
            cv2.imwrite(mask_output_path, im_binary)
            
            # Create patches
            pad_mask = self.pad_input_img_single_channel(im_binary, self.patch_size)
            patches_mask = patchify(pad_mask, (self.patch_size, self.patch_size), self.patch_size)
            
            pad_img = self.pad_input_img(png_image, self.patch_size)
            patches_img = patchify(pad_img, (self.patch_size, self.patch_size, 3), self.patch_size)
            
            logger.info(f"Creating Patches for {base_filename}...")
            ctr = 0
            for i in tqdm(range(patches_mask.shape[0])):
                for j in range(patches_mask.shape[1]):
                    patch_mask = patches_mask[i, j]
                    patch_img = patches_img[i, j, 0]
                    
                    if np.sum(patch_mask) == 0:
                        # Save patches with no masks (background)
                        cv2.imwrite(
                            os.path.join(self.binary_masks_patches_dir, f"{base_filename}_{i}_{j}.png"), 
                            patch_mask
                        )
                        cv2.imwrite(
                            os.path.join(self.raw_images_patches_dir, f"{base_filename}_{i}_{j}.png"), 
                            patch_img
                        )
                    else:
                        # Create multiple copies of patches with masks (foreground)
                        ctr += 1
                        for xx in range(6):
                            cv2.imwrite(
                                os.path.join(self.binary_masks_patches_dir, f"{base_filename}_{i}_{j}_{ctr}_{xx}.png"),
                                patch_mask
                            )
                            cv2.imwrite(
                                os.path.join(self.raw_images_patches_dir, f"{base_filename}_{i}_{j}_{ctr}_{xx}.png"),
                                patch_img
                            )

    @staticmethod
    def closest_number(n: float, m: int) -> int:
        """Find the closest number to n that is divisible by m.
        
        Args:
            n: Number
            m: Divisor
            
        Returns:
            Closest number divisible by m
        """
        q = int(n / m)
        n1 = m * q
        if (n * m) > 0:
            n2 = (m * (q + 1))
        else:
            n2 = (m * (q - 1))
        if abs(n - n1) < abs(n - n2):
            return n1
        return n2

    @staticmethod
    def pix_add(image: np.ndarray, pixels: int) -> Tuple[int, int]:
        """Calculate padding size to make the image divisible by pixels.
        
        Args:
            image: Input image
            pixels: Desired pixel size
            
        Returns:
            Tuple of (height_padding, width_padding)
        """
        if DataPreparation.closest_number(image.shape[0], pixels) < image.shape[0]:
            h1 = DataPreparation.closest_number(image.shape[0], pixels) + pixels
        else:
            h1 = DataPreparation.closest_number(image.shape[0], pixels)
        if DataPreparation.closest_number(image.shape[1], pixels) < image.shape[1]:
            w1 = DataPreparation.closest_number(image.shape[1], pixels) + pixels
        else:
            w1 = DataPreparation.closest_number(image.shape[1], pixels)

        pixels_to_add_h = h1 - image.shape[0]
        pixels_to_add_w = w1 - image.shape[1]

        return pixels_to_add_h, pixels_to_add_w

    @staticmethod
    def get_poly(ann_path: str) -> List[Dict[str, Any]]:
        """Get polygon shapes from annotation file.
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of shape dictionaries
        """
        with open(ann_path) as handle:
            data = json.load(handle)
        shape_dicts = data['shapes']
        return shape_dicts

    @staticmethod
    def pad_input_img_single_channel(image: np.ndarray, pixels: int) -> np.ndarray:
        """Pad single-channel input image to be divisible by pixels.
        
        Args:
            image: Input image
            pixels: Desired pixel size
            
        Returns:
            Padded image
        """
        pixels_to_add_h, pixels_to_add_w = DataPreparation.pix_add(image, pixels)
        pad_h = np.zeros((pixels_to_add_h, image.shape[1]))
        result_h = np.vstack((image, pad_h))
        pad_w = np.zeros((result_h.shape[0], pixels_to_add_w))
        result = np.hstack((result_h, pad_w))
        return result

    @staticmethod
    def create_binary_masks(im: np.ndarray, shape_dicts: List[Dict[str, Any]]) -> np.ndarray:
        """Create binary masks from shape dictionaries.
        
        Args:
            im: Input image
            shape_dicts: List of shape dictionaries
            
        Returns:
            Binary mask image
        """
        blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
        for shape in shape_dicts:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(blank, [points], 255)
        return blank

    @staticmethod
    def pad_input_img(image: np.ndarray, pixels: int) -> np.ndarray:
        """Pad multi-channel input image to be divisible by pixels.
        
        Args:
            image: Input image
            pixels: Desired pixel size
            
        Returns:
            Padded image
        """
        pixels_to_add_h, pixels_to_add_w = DataPreparation.pix_add(image, pixels)
        pad_h = np.zeros((pixels_to_add_h, image.shape[1], 3))
        result_h = np.vstack((image, pad_h))
        pad_w = np.zeros((result_h.shape[0], pixels_to_add_w, 3))
        result = np.hstack((result_h, pad_w))
        return result