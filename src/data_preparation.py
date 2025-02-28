import os
import cv2
import json
from patchify import patchify
from tqdm import tqdm
import numpy as np
import initialize  # Ensure Constants is set
from constants import Constants

class DataPreparation:
    def __init__(self, config):  # Kept config param for compatibility, though unused
        pass  # No need for self.config since we use Constants

    def prepare_patches(self):
        raw_dir = Constants.RAW
        processed_dir = Constants.PROCESSED
        bin_mask_patches_path = Constants.BINARY_MASKS_PATCHES
        raw_image_patches_path = Constants.RAW_IMAGES_PATCHES
        bin_mask_path = Constants.BINARY_MASKS

        os.makedirs(bin_mask_patches_path, exist_ok=True)
        os.makedirs(raw_image_patches_path, exist_ok=True)
        os.makedirs(bin_mask_path, exist_ok=True)

        all_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.json')]

        print("Creating Binary Masks...")
        for filename in tqdm(all_files):
            png_filename = filename.replace('.json', '.png')
            bin_mask_file = os.path.join(bin_mask_path, os.path.basename(filename).replace('.json', '.png'))

            png_image = cv2.imread(png_filename)
            shape_dicts = self.get_poly(filename)
            im_binary = self.create_binary_masks(png_image, shape_dicts)
            cv2.imwrite(bin_mask_file, im_binary)

            patch_size = Constants.PATCH_SIZE
            pad_mask = self.pad_input_img_single_channel(im_binary, patch_size)
            patches_mask = patchify(pad_mask, (patch_size, patch_size), patch_size)

            pad_img = self.pad_input_img(png_image, patch_size)
            patches_img = patchify(pad_img, (patch_size, patch_size, 3), patch_size)

            patch_file_name = os.path.basename(png_filename).replace('.png', '')
            print(f"Creating Patches for {patch_file_name}...")
            ctr = 0
            for i in tqdm(range(patches_mask.shape[0])):
                for j in range(patches_mask.shape[1]):
                    patch_mask = patches_mask[i, j]
                    patch_img = patches_img[i, j, 0]

                    if np.sum(patch_mask) == 0:
                        cv2.imwrite(os.path.join(bin_mask_patches_path, f"{patch_file_name}_{i}_{j}.png"), patch_mask)
                        cv2.imwrite(os.path.join(raw_image_patches_path, f"{patch_file_name}_{i}_{j}.png"), patch_img)
                    else:
                        ctr += 1
                        for xx in range(6):
                            cv2.imwrite(os.path.join(bin_mask_patches_path, f"{patch_file_name}_{i}_{j}_{ctr}_{xx}.png"), patch_mask)
                            cv2.imwrite(os.path.join(raw_image_patches_path, f"{patch_file_name}_{i}_{j}_{ctr}_{xx}.png"), patch_img)

    @staticmethod
    def closest_number(n, m):
        q = int(n / m)
        n1 = m * q
        n2 = m * (q + 1 if (n * m) > 0 else q - 1)
        return n1 if abs(n - n1) < abs(n - n2) else n2

    @staticmethod
    def pix_add(image, pixels):
        h1 = DataPreparation.closest_number(image.shape[0], pixels)
        w1 = DataPreparation.closest_number(image.shape[1], pixels)
        h1 += pixels if h1 < image.shape[0] else 0
        w1 += pixels if w1 < image.shape[1] else 0
        return h1 - image.shape[0], w1 - image.shape[1]

    @staticmethod
    def get_poly(ann_path):
        with open(ann_path) as handle:
            data = json.load(handle)
        return data['shapes']

    @staticmethod
    def pad_input_img_single_channel(image, pixels):
        pixels_to_add_h, pixels_to_add_w = DataPreparation.pix_add(image, pixels)
        pad_h = np.zeros((pixels_to_add_h, image.shape[1]))
        result_h = np.vstack((image, pad_h))
        pad_w = np.zeros((result_h.shape[0], pixels_to_add_w))
        return np.hstack((result_h, pad_w))

    @staticmethod
    def create_binary_masks(im, shape_dicts):
        blank = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
        for shape in shape_dicts:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(blank, [points], 255)
        return blank

    @staticmethod
    def pad_input_img(image, pixels):
        pixels_to_add_h, pixels_to_add_w = DataPreparation.pix_add(image, pixels)
        pad_h = np.zeros((pixels_to_add_h, image.shape[1], 3))
        result_h = np.vstack((image, pad_h))
        pad_w = np.zeros((result_h.shape[0], pixels_to_add_w, 3))
        return np.hstack((result_h, pad_w))

# Critical Comment: The config parameter is retained for compatibility but is unused since Constants is now global.
# Ensure src/initialize.py is imported before any Constants attribute is accessed to avoid AttributeError.



from initialize import initialize
from data_preparation import DataPreparation

initialize()  # Set up Constants
dp = DataPreparation(None)
dp.prepare_patches()  # Run data preparation