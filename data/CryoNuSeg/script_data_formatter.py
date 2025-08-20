import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import List, Tuple


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['./images/', './masks/', './labels/',
            './images/train/', './images/val/', './images/test/',
            './masks/train/', './masks/val/', './masks/test/',
            './labels/train/', './labels/val/', './labels/test/']

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Created necessary directories")


def get_organ_groups(image_folder: str) -> dict:
    """Extract organ groups from image filenames"""
    groups = defaultdict(list)

    for filename in os.listdir(image_folder):
        if filename.startswith('Human_') and ('_01' in filename or '_02' in filename or '_03' in filename):
            # Extract organ name (X in Human_X_01)
            parts = filename.split('_')
            if len(parts) >= 3:
                organ = parts[1]  # This is the X part
                groups[organ].append(filename)

    print(f"Found {len(groups)} organ groups:")
    for organ, files in groups.items():
        print(f"  {organ}: {len(files)} files")

    return groups


def mask_to_yolo_bbox(mask: np.ndarray, img_width: int, img_height: int) -> List[str]:
    """Convert binary mask to YOLO format bounding boxes"""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_boxes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Convert to YOLO format (normalized coordinates)
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        # YOLO format: class_id center_x center_y width height
        # Using class_id = 0 for all objects
        yolo_boxes.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return yolo_boxes


def process_images_and_masks():
    """Process all images and masks, convert to YOLO format"""
    image_folder = "./original_images/tissue images/"
    mask_folder = "./original_images/Annotator 1 (biologist second round of manual marks up)/mask binary/"

    if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
        print("Error: Source folders not found!")
        return

    # Get organ groups for splitting
    organ_groups = get_organ_groups(image_folder)

    # Randomly assign groups to train/val/test (8/1/1)
    group_names = list(organ_groups.keys())
    random.shuffle(group_names)

    train_groups = group_names[:8]
    val_groups = group_names[8:9]
    test_groups = group_names[9:10]

    print(f"Train groups: {train_groups}")
    print(f"Val groups: {val_groups}")
    print(f"Test groups: {test_groups}")

    def get_split(filename):
        """Determine which split a file belongs to based on organ group"""
        for organ in train_groups:
            if f"Human_{organ}_" in filename:
                return "train"
        for organ in val_groups:
            if f"Human_{organ}_" in filename:
                return "val"
        for organ in test_groups:
            if f"Human_{organ}_" in filename:
                return "test"
        return "train"  # default

    processed_count = 0

    # Process each image
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            image_path = os.path.join(image_folder, filename)

            # Find corresponding mask
            base_name = os.path.splitext(filename)[0]
            mask_filename = None

            # Look for mask with same base name
            for mask_file in os.listdir(mask_folder):
                if os.path.splitext(mask_file)[0] == base_name:
                    mask_filename = mask_file
                    break

            if mask_filename is None:
                print(f"Warning: No mask found for {filename}")
                continue

            mask_path = os.path.join(mask_folder, mask_filename)

            # Determine split
            split = get_split(filename)

            # Copy image to appropriate split folder
            dst_image_path = f"./images/{split}/{filename}"
            shutil.copy2(image_path, dst_image_path)

            # Copy mask to appropriate split folder
            dst_mask_path = f"./masks/{split}/{mask_filename}"
            shutil.copy2(mask_path, dst_mask_path)

            # Process mask and create YOLO labels
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_filename}")
                continue

            # Get image dimensions for normalization
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {filename}")
                continue

            img_height, img_width = img.shape[:2]

            # Convert mask to YOLO bounding boxes
            yolo_boxes = mask_to_yolo_bbox(mask, img_width, img_height)

            # Save YOLO labels
            label_filename = f"{base_name}.txt"
            label_path = f"./labels/{split}/{label_filename}"

            with open(label_path, 'w') as f:
                for box in yolo_boxes:
                    f.write(box + '\n')

            processed_count += 1
            print(f"Processed {filename} -> {split} split")

    print(f"\nProcessed {processed_count} image-mask pairs")


def visualize_results():
    """Visualize one sample: grayscale image with red mask overlay vs image with bounding boxes"""
    # Find any available sample from train folder
    image_dir = "./images/train/"
    mask_dir = "./masks/train/"
    label_dir = "./labels/train/"

    if not os.path.exists(image_dir):
        print("No train images found for visualization")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    if not image_files:
        print("No image files found for visualization")
        return

    # Pick the first available sample
    filename = image_files[0]
    base_name = os.path.splitext(filename)[0]

    # Load image
    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load mask
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith(base_name)]
    if not mask_files:
        print(f"No mask found for {filename}")
        return

    mask_path = os.path.join(mask_dir, mask_files[0])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create grayscale image with transparent red mask overlay
    img_with_mask = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    red_overlay = np.zeros_like(img_with_mask)
    red_overlay[mask > 0] = [255, 0, 0]  # Red overlay on mask regions

    # Blend with alpha transparency (0.3 = 30% red overlay, 70% original image)
    img_with_mask = cv2.addWeighted(img_with_mask.astype(np.uint8), 0.7, red_overlay.astype(np.uint8), 0.3, 0)

    # Load YOLO labels and draw bounding boxes on color image
    img_with_bbox = img_rgb.copy()
    label_path = os.path.join(label_dir, f"{base_name}.txt")

    if os.path.exists(label_path):
        h, w = img_rgb.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, center_x, center_y, width, height = map(float, parts)

                    # Convert back to pixel coordinates
                    x1 = int((center_x - width / 2) * w)
                    y1 = int((center_y - height / 2) * h)
                    x2 = int((center_x + width / 2) * w)
                    y2 = int((center_y + height / 2) * h)

                    # Draw bounding box in green
                    cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(img_with_mask)
    ax1.set_title('Grayscale Image + Red Mask Overlay')
    ax1.axis('off')

    ax2.imshow(img_with_bbox)
    ax2.set_title('Image + YOLO Bounding Boxes')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('visualization_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Visualized sample: {filename}")


def main():
    """Main function to execute all processing steps"""
    print("Starting image processing pipeline...")

    # Set random seed for reproducible splits
    random.seed(42)

    # Step 1: Create directories
    create_directories()

    # Step 2: Process images and masks
    process_images_and_masks()

    # Step 3: Visualize results
    print("\nGenerating visualization...")
    visualize_results()

    print("\nProcessing complete!")
    print("Files have been organized into:")
    print("  ./images/{train,val,test}/")
    print("  ./masks/{train,val,test}/")
    print("  ./labels/{train,val,test}/")
    print("\nVisualization saved as 'visualization_results.png'")


if __name__ == "__main__":
    main()