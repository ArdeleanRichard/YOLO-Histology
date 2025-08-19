import cv2
import numpy as np
import os
from pathlib import Path


def mask_to_yolo_bboxes(mask_path, output_path, class_id=0, min_area=10):
    """
    Convert a binary segmentation mask to YOLO bounding box format.

    Args:
        mask_path (str): Path to the binary mask image
        output_path (str): Path to save the YOLO label file
        class_id (int): Class ID for all objects (default: 0)
        min_area (int): Minimum area threshold to filter small blobs
    """
    # Read the binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask {mask_path}")
        return

    # Get image dimensions
    height, width = mask.shape

    # Find connected components (individual blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Prepare output lines for YOLO format
    yolo_lines = []

    # Process each connected component (skip background label 0)
    for label_id in range(1, num_labels):
        # Filter by minimum area
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # Get bounding box coordinates from connected component stats
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Convert to YOLO format: [x_center, y_center, width, height] (normalized 0-1)
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        bbox_width = w / width
        bbox_height = h / height

        # Format: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_lines.append(yolo_line)

    # Save YOLO labels to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    print(f"Processed {mask_path}: Found {len(yolo_lines)} objects")
    return len(yolo_lines)


def convert_masks_to_yolo(masks_folder, labels_folder, class_id=0, min_area=10):
    """
    Convert all masks in a folder to YOLO format labels.

    Args:
        masks_folder (str): Path to folder containing mask images
        labels_folder (str): Path to folder where YOLO labels will be saved
        class_id (int): Class ID for all objects
        min_area (int): Minimum area threshold to filter small blobs
    """
    masks_path = Path(masks_folder)
    labels_path = Path(labels_folder)

    # Create output directory
    labels_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    # Process all mask files
    total_objects = 0
    processed_files = 0

    for mask_file in masks_path.iterdir():
        if mask_file.suffix.lower() in supported_extensions:
            # Create corresponding label filename
            label_filename = mask_file.stem
            label_file = labels_path / f"{label_filename}.txt"

            # Convert mask to YOLO format
            num_objects = mask_to_yolo_bboxes(
                str(mask_file),
                str(label_file),
                class_id=class_id,
                min_area=min_area
            )

            if num_objects is not None:
                total_objects += num_objects
                processed_files += 1

    print(f"\nConversion completed!")
    print(f"Processed {processed_files} mask files")
    print(f"Total objects detected: {total_objects}")


def visualize_yolo_labels(image_path, mask_path, label_path, output_path=None):
    """
    Create a 3-subplot visualization showing:
    1. Grayscale image with red mask overlay
    2. Extracted bounding boxes from connected components
    3. YOLO bounding box labels

    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the binary mask
        label_path (str): Path to the YOLO label file
        output_path (str): Path to save visualization (required)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Read image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    if mask is None:
        print(f"Error: Could not read mask {mask_path}")
        return

    # Convert image to RGB and grayscale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('YOLO Bounding Box Conversion', fontsize=16, fontweight='bold')

    # Subplot 1: Grayscale image with red mask overlay
    axes[0].imshow(image_gray, cmap='gray')
    # Create red overlay where mask is white
    red_overlay = np.zeros((*mask.shape, 4))
    red_overlay[mask > 0] = [1, 0, 0, 0.5]  # Red with 50% transparency
    axes[0].imshow(red_overlay)
    axes[0].set_title('Original Image + Mask Overlay')
    axes[0].axis('off')


    # Subplot 3: YOLO bounding box labels
    axes[1].imshow(image_gray, cmap='gray')

    # Read and visualize YOLO labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        yolo_bbox_count = 0

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:  # class_id x_center y_center width height
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            # Convert normalized YOLO coordinates back to pixel coordinates
            x_center_px = x_center * width
            y_center_px = y_center * height
            bbox_width_px = bbox_width * width
            bbox_height_px = bbox_height * height

            # Calculate top-left corner
            x_topleft = x_center_px - bbox_width_px / 2
            y_topleft = y_center_px - bbox_height_px / 2

            # Draw bounding box
            rect = patches.Rectangle((x_topleft, y_topleft), bbox_width_px, bbox_height_px,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            axes[1].add_patch(rect)

            # Add class ID text at center
            # axes[1].text(x_center_px, y_center_px, str(class_id),
            #              color='white', fontsize=10, fontweight='bold',
            #              ha='center', va='center',
            #              bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

            yolo_bbox_count += 1

        axes[1].set_title(f'YOLO Bounding Boxes ({yolo_bbox_count} objects)')
    else:
        axes[1].set_title('YOLO Bounding Boxes (No labels found)')

    axes[1].axis('off')

    # Adjust layout and save
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


# Example usage
if __name__ == "__main__":
    # Configure paths

    for x in ["train", "val", "test"]:
        masks_folder =  f"./masks/{x}"
        labels_folder = f"./labels/{x}"
        images_folder = f"./images/{x}"

        # Convert all masks to YOLO format
        print("Converting masks to YOLO bounding box format...")
        convert_masks_to_yolo(
            masks_folder=masks_folder,
            labels_folder=labels_folder,
            class_id=0,  # Change this if you have multiple classes
            min_area=10  # Minimum blob area (pixels) to include
        )

    # Optional: Visualize results for verification
    import glob
    mask_files = glob.glob(os.path.join(masks_folder, "*.png"))
    mask_file = mask_files[0]
    mask_name = os.path.splitext(os.path.basename(mask_file))[0]
    image_file = os.path.join(images_folder, f"{mask_name}.png")
    label_file = os.path.join(labels_folder, f"{mask_name}.txt")

    visualize_yolo_labels(image_file, mask_file, label_file,f"./visualization_{mask_name}.png")