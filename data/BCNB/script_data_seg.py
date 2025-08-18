import glob
import os
import random
from shutil import move, copy2

import numpy as np
import pandas as pd
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

YOLO_IMG_DIR =       './images/'          # Output directory for copied/organized images (optional)
YOLO_LABEL_DIR =     './labels/'        # Output directory for YOLO-format .txt files
YOLO_MASKS_DIR =     './masks/'        # Output directory for YOLO-format .txt files


def create_segmentation_masks():
    from PIL import Image, ImageDraw

    os.makedirs(YOLO_MASKS_DIR, exist_ok=True)

    for split in ['train', 'val', 'test']:
        img_split_dir = os.path.join(YOLO_IMG_DIR, split)
        label_split_dir = os.path.join(YOLO_LABEL_DIR, split)
        mask_split_dir = os.path.join(YOLO_MASKS_DIR, split)
        os.makedirs(mask_split_dir, exist_ok=True)

        for img_name in os.listdir(img_split_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(img_split_dir, img_name)
            label_path = os.path.join(label_split_dir, os.path.splitext(img_name)[0] + '.txt')

            if not os.path.exists(label_path):
                print(f"No label found for {img_name}, skipping.")
                continue

            # Load image size
            with Image.open(img_path) as img:
                w, h = img.size

            # Create blank mask (black)
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)

            # Parse YOLO label and draw rectangles
            # Multi-class mask (start all pixels as background class: 0)
            mask = np.zeros((h, w), dtype=np.uint8)

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_idx, x_c, y_c, box_w, box_h = map(float, parts)
                    cls_idx = int(cls_idx)

                    x_center = x_c * w
                    y_center = y_c * h
                    box_width = box_w * w
                    box_height = box_h * h

                    xmin = int(x_center - box_width / 2)
                    ymin = int(y_center - box_height / 2)
                    xmax = int(x_center + box_width / 2)
                    ymax = int(y_center + box_height / 2)

                    mask[ymin:ymax, xmin:xmax] = cls_idx + 1  # Add 1 to separate from background (0)

            mask_path = os.path.join(mask_split_dir, f"{img_name}")
            Image.fromarray(mask).save(mask_path)

    print("Segmentation masks created.")

if __name__ == "__main__":
    create_segmentation_masks()