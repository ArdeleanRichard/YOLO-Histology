import json
import os
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def polygon_to_bbox(vertices):
    """Convert polygon vertices to bounding box (x_center, y_center, width, height) normalized"""
    if not vertices:
        return None

    vertices = np.array(vertices)
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    return x_min, y_min, x_max, y_max


def normalize_bbox(bbox, img_width, img_height):
    """Convert bbox to YOLO format (normalized center coordinates and dimensions)"""
    x_min, y_min, x_max, y_max = bbox

    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def yolo_to_bbox(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    """Convert YOLO format back to pixel coordinates for visualization"""
    # Denormalize
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height

    # Calculate corners
    x_min = x_center - width / 2
    y_min = y_center - height / 2

    return x_min, y_min, width, height


def visualize_annotations(processed_img_path, original_json_path, output_path=None, scale_factor=0.05):
    """
    Visualize original polygon annotations vs converted YOLO bounding boxes on processed images

    Args:
        processed_img_path (str): Path to the processed/downsampled image file
        original_json_path (str): Path to the original JSON annotation file
        output_path (str): Path to save the visualization (if None, will show instead)
        scale_factor (float): Scale factor used for downsampling (to scale the polygons)
    """

    # Load processed image
    img_small = Image.open(processed_img_path)
    new_width, new_height = img_small.size

    # Load annotations
    with open(original_json_path, 'r') as f:
        annotations = json.load(f)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Left subplot: Original image with polygon annotations
    ax1.imshow(img_small)
    ax1.set_title('Original Polygon Annotations', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw polygons on original image
    polygon_count = 0

    # Process positive regions (red polygons)
    positive_regions = annotations.get("positive", [])
    for region in positive_regions:
        vertices = region.get("vertices", [])
        if vertices:
            # Scale vertices to match downsampled image
            scaled_vertices = [(x * scale_factor, y * scale_factor) for x, y in vertices]

            if len(scaled_vertices) >= 3:  # Need at least 3 points for a polygon
                polygon = patches.Polygon(scaled_vertices, linewidth=2, edgecolor='red',
                                          facecolor='red', alpha=0.3, label='Positive' if polygon_count == 0 else "")
                ax1.add_patch(polygon)
                polygon_count += 1

    # Process negative regions (blue polygons)
    negative_regions = annotations.get("negative", [])
    negative_count = 0
    for region in negative_regions:
        vertices = region.get("vertices", [])
        if vertices:
            # Scale vertices to match downsampled image
            scaled_vertices = [(x * scale_factor, y * scale_factor) for x, y in vertices]

            if len(scaled_vertices) >= 3:  # Need at least 3 points for a polygon
                polygon = patches.Polygon(scaled_vertices, linewidth=2, edgecolor='blue',
                                          facecolor='blue', alpha=0.3, label='Negative' if negative_count == 0 else "")
                ax1.add_patch(polygon)
                negative_count += 1

    # Right subplot: Downsampled image with YOLO bounding boxes
    ax2.imshow(img_small)
    ax2.set_title('Converted YOLO Bounding Boxes', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Convert annotations to YOLO format and draw bounding boxes
    bbox_count = 0

    # Process positive regions (class 0 - green boxes)
    for region in positive_regions:
        vertices = region.get("vertices", [])
        if vertices:
            # Scale vertices to match downsampled image
            scaled_vertices = [(x * scale_factor, y * scale_factor) for x, y in vertices]

            # Convert polygon to bounding box
            bbox = polygon_to_bbox(scaled_vertices)
            if bbox:
                x_min, y_min, x_max, y_max = bbox

                # Draw bounding box
                width = x_max - x_min
                height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
                                         edgecolor='lime', facecolor='none',
                                         label='Positive' if bbox_count == 0 else "")
                ax2.add_patch(rect)
                bbox_count += 1

    # Process negative regions (class 1 - orange boxes)
    bbox_neg_count = 0
    for region in negative_regions:
        vertices = region.get("vertices", [])
        if vertices:
            # Scale vertices to match downsampled image
            scaled_vertices = [(x * scale_factor, y * scale_factor) for x, y in vertices]

            # Convert polygon to bounding box
            bbox = polygon_to_bbox(scaled_vertices)
            if bbox:
                x_min, y_min, x_max, y_max = bbox

                # Draw bounding box
                width = x_max - x_min
                height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
                                         edgecolor='orange', facecolor='none',
                                         label='Negative BBox' if bbox_neg_count == 0 else "")
                ax2.add_patch(rect)
                bbox_neg_count += 1

    # Add legends
    if polygon_count > 0 or negative_count > 0:
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
    if bbox_count > 0 or bbox_neg_count > 0:
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Add image information
    fig.suptitle(f'Annotation Conversion Visualization',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    # Clean up
    img_small.close()
    plt.close(fig)

    return {
        'positive_polygons': len(positive_regions),
        'negative_polygons': len(negative_regions),
        'positive_bboxes': bbox_count,
        'negative_bboxes': bbox_neg_count
    }


def process_bcnb_dataset(original_images_dir, output_dir, scale_factor=0.05, resampling_method="box"):
    """
    Process BCNB dataset: downsample images and convert annotations to YOLO format

    Args:
        original_images_dir (str): Path to folder containing original images and JSON files
        output_dir (str): Path to output directory
        scale_factor (float): Downsampling factor (default: 0.05 for 5% of original size)
        resampling_method (str): Resampling method - "box" (area averaging), "bilinear", or "lanczos"
    """

    # Map resampling methods
    resampling_map = {
        "box": Image.Resampling.BOX,  # Area averaging - best for histology
        "bilinear": Image.Resampling.BILINEAR,  # Linear interpolation
        "lanczos": Image.Resampling.LANCZOS,  # High-quality but may over-smooth
        "nearest": Image.Resampling.NEAREST  # Preserves exact pixel values
    }

    if resampling_method not in resampling_map:
        print(f"Warning: Unknown resampling method '{resampling_method}', using 'box'")
        resampling_method = "box"

    resampling_filter = resampling_map[resampling_method]

    # Create output directories
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Set PIL limit for large images
    Image.MAX_IMAGE_PIXELS = None

    # Get all image files
    original_path = Path(original_images_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    image_files = [f for f in original_path.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process")

    processed_count = 0

    for img_file in image_files:
        try:
            # Corresponding JSON file
            json_file = original_path / f"{img_file.stem}.json"

            if not json_file.exists():
                print(f"Warning: No JSON file found for {img_file.name}")
                continue

            print(f"Processing {img_file.name}...")

            # Load and downsample image
            img = Image.open(img_file)
            original_width, original_height = img.size

            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            new_size = (new_width, new_height)

            print(f"  Resizing from {original_width}x{original_height} to {new_width}x{new_height} using {resampling_method}")

            # Use area averaging (BOX) for better preservation of histological features
            # This method averages pixel intensities within regions, preserving diagnostic information
            img_small = img.resize(new_size, resampling_filter)

            # Save downsampled image
            output_img_path = images_dir / f"{img_file.stem}.jpg"
            img_small.save(output_img_path, "JPEG", quality=95)

            # Load annotations
            with open(json_file, 'r') as f:
                annotations = json.load(f)

            # Convert annotations to YOLO format
            yolo_annotations = []

            # Process positive regions (class 0 for positive/cancer)
            positive_regions = annotations.get("positive", [])
            for region in positive_regions:
                vertices = region.get("vertices", [])
                if vertices:
                    # Scale vertices to match downsampled image
                    scaled_vertices = [(x * scale_factor, y * scale_factor) for x, y in vertices]

                    # Convert polygon to bounding box
                    bbox = polygon_to_bbox(scaled_vertices)
                    if bbox:
                        # Normalize bbox for YOLO format
                        x_center, y_center, width, height = normalize_bbox(bbox, new_width, new_height)

                        # Ensure coordinates are within valid range [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # Class 0 for positive (cancer)
                        yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Process negative regions if they exist (class 1 for negative/benign)
            negative_regions = annotations.get("negative", [])
            for region in negative_regions:
                vertices = region.get("vertices", [])
                if vertices:
                    # Scale vertices to match downsampled image
                    scaled_vertices = [(x * scale_factor, y * scale_factor) for x, y in vertices]

                    # Convert polygon to bounding box
                    bbox = polygon_to_bbox(scaled_vertices)
                    if bbox:
                        # Normalize bbox for YOLO format
                        x_center, y_center, width, height = normalize_bbox(bbox, new_width, new_height)

                        # Ensure coordinates are within valid range [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # Class 1 for negative (benign)
                        yolo_annotations.append(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Save YOLO format annotations
            output_label_path = labels_dir / f"{img_file.stem}.txt"
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            processed_count += 1
            print(f"  Saved: {output_img_path.name} with {len(yolo_annotations)} annotations")

            # Clean up memory
            img.close()

        except Exception as e:
            print(f"Error processing {img_file.name}: {str(e)}")
            continue

    print(f"\nProcessing complete! Successfully processed {processed_count} images.")
    print(f"Downsampled images saved to: {images_dir}")
    print(f"YOLO labels saved to: {labels_dir}")

    # Create classes.txt file for YOLO
    classes_file = Path(output_dir) / "classes.txt"
    with open(classes_file, 'w') as f:
        f.write("positive\nnegative\n")  # or "cancer\nbenign"

    print(f"Classes file saved to: {classes_file}")


def verify_dataset(output_dir):
    """Verify the processed dataset"""
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"

    image_files = list(images_dir.glob("*.jpg"))
    label_files = list(labels_dir.glob("*.txt"))

    print(f"\nDataset verification:")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")

    # Check for missing pairs
    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        print(f"Images without labels: {missing_labels}")
    if missing_images:
        print(f"Labels without images: {missing_images}")

    if not missing_labels and not missing_images:
        print("✓ All images have corresponding labels!")


# Example usage
if __name__ == "__main__":
    # # Configuration
    ORIGINAL_IMAGES_DIR = "./original_images"  # Folder with original images and JSON files
    OUTPUT_DIR = "./"  # Output directory for processed dataset
    SCALE_FACTOR = 0.05  # 5% of original size - adjust as needed
    RESAMPLING_METHOD = "box"  # "box" (area averaging) recommended for histology

    # # Process the dataset
    # process_bcnb_dataset(ORIGINAL_IMAGES_DIR, OUTPUT_DIR, SCALE_FACTOR, RESAMPLING_METHOD)
    #
    # # Verify the results
    # verify_dataset(OUTPUT_DIR)
    #
    #
    # print(f"\nYour dataset is ready for YOLO training!")
    # print(f"Dataset structure:")
    # print(f"  {OUTPUT_DIR}/")
    # print(f"  ├── images/        # Downsampled images")
    # print(f"  ├── labels/        # YOLO format annotations")
    # print(f"  └── classes.txt    # Class names")


    # Example: Visualize a single processed image's annotations
    processed_img_path = "./images/train/1.jpg"  # Path to processed image
    original_json_path = "./original_images/1.json"  # Path to original JSON
    viz_output_path = "./visualization_sample.png"  # Where to save the visualization

    print("\nCreating visualization...")
    stats = visualize_annotations(
        processed_img_path=processed_img_path,
        original_json_path=original_json_path,
        output_path=viz_output_path,
        scale_factor=SCALE_FACTOR
    )

    print(f"Visualization stats: {stats}")

    print(f"\nTo visualize annotations for a specific image, use:")
    print(f"visualize_annotations('path/to/processed_image.jpg', 'path/to/original_annotations.json', 'output_viz.png')")