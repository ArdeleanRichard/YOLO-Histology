import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# Increase PIL's image size limit for large WSI images
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely


def polygon_to_bbox(vertices):
    """
    Convert polygon vertices to bounding box coordinates.

    Args:
        vertices: List of [x, y] coordinate pairs

    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    vertices = np.array(vertices)
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    x_min = np.min(x_coords)
    y_min = np.min(y_coords)
    x_max = np.max(x_coords)
    y_max = np.max(y_coords)

    return x_min, y_min, x_max, y_max


def bbox_to_yolo_format(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO format (normalized center coordinates and dimensions).

    Args:
        bbox: tuple (x_min, y_min, x_max, y_max)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        tuple: (x_center, y_center, width, height) in normalized coordinates
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return x_center, y_center, width, height


def get_image_dimensions(image_path):
    """
    Get image dimensions without fully loading the image.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def resize_image_for_display(image_path, max_size=2048):
    """
    Resize large image for display while maintaining aspect ratio.
    Uses a more memory-efficient approach for extremely large images.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height)

    Returns:
        tuple: (resized_image_array, scale_factor)
    """
    try:
        # First, get image info without loading the full image
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            print(f"Original image size: {original_width} x {original_height}")

            # Calculate scale factor to fit within max_size
            scale_factor = min(max_size / original_width, max_size / original_height)

            # Calculate new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            print(f"Resizing to: {new_width} x {new_height} (scale factor: {scale_factor:.4f})")

            # For extremely large images, use thumbnail method which is more memory efficient
            if original_width > 10000 or original_height > 10000:
                print("Using memory-efficient thumbnail method for large image...")
                # Create a copy to avoid modifying the original
                img_copy = img.copy()
                img_copy.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                resized_img = img_copy
            else:
                # Use standard resize for smaller images
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            return np.array(resized_img), scale_factor

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        print("This might be due to the extremely large size of WSI images.")
        print("Try reducing max_size parameter or using a smaller test image.")
        raise


def scale_coordinates(coordinates, scale_factor):
    """
    Scale coordinates by a given factor.

    Args:
        coordinates: List of [x, y] pairs or bounding box
        scale_factor: Scale factor to apply

    Returns:
        Scaled coordinates
    """
    if isinstance(coordinates[0], list):  # List of coordinate pairs
        return [[x * scale_factor, y * scale_factor] for x, y in coordinates]
    else:  # Single coordinate pair or bounding box
        return [coord * scale_factor for coord in coordinates]


def plot_annotations_comparison(image_path, json_data, output_path=None, max_size=1024):
    """
    Plot original image with JSON polygons and YOLO bounding boxes side by side.

    Args:
        image_path: Path to the image file
        json_data: JSON annotation data
        output_path: Path to save the plot (optional)
        max_size: Maximum dimension for display (reduced default for memory efficiency)
    """
    print(f"Loading image: {image_path}")

    # Load and resize image with error handling
    try:
        img_array, scale_factor = resize_image_for_display(image_path, max_size)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # Get original image dimensions for YOLO conversion
    try:
        with Image.open(image_path) as orig_img:
            orig_width, orig_height = orig_img.size
    except Exception as e:
        print(f"Failed to get original image dimensions: {e}")
        return

    print(
        f"Creating visualization with {len(json_data.get('positive', []))} positive and {len(json_data.get('negative', []))} negative annotations...")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: Original polygons
    ax1.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    ax1.set_title('Original JSON Polygons', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw polygons on first subplot
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan']

    # Process positive annotations
    for i, annotation in enumerate(json_data.get('positive', [])):
        vertices = annotation['vertices']
        scaled_vertices = scale_coordinates(vertices, scale_factor)

        # Create polygon patch
        polygon = Polygon(scaled_vertices, fill=False, edgecolor=colors[i % len(colors)],
                          linewidth=2, alpha=0.8)
        ax1.add_patch(polygon)

        # Add annotation label
        centroid_x = np.mean([v[0] for v in scaled_vertices])
        centroid_y = np.mean([v[1] for v in scaled_vertices])
        ax1.text(centroid_x, centroid_y, f'P{i}', fontsize=10, color=colors[i % len(colors)],
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Process negative annotations
    for i, annotation in enumerate(json_data.get('negative', [])):
        vertices = annotation['vertices']
        scaled_vertices = scale_coordinates(vertices, scale_factor)

        # Create polygon patch
        polygon = Polygon(scaled_vertices, fill=False, edgecolor='darkred',
                          linewidth=2, alpha=0.8, linestyle='--')
        ax1.add_patch(polygon)

        # Add annotation label
        centroid_x = np.mean([v[0] for v in scaled_vertices])
        centroid_y = np.mean([v[1] for v in scaled_vertices])
        ax1.text(centroid_x, centroid_y, f'N{i}', fontsize=10, color='darkred',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Plot 2: YOLO bounding boxes
    ax2.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    ax2.set_title('YOLO Bounding Boxes', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Convert polygons to bounding boxes and draw them
    # Process positive annotations
    for i, annotation in enumerate(json_data.get('positive', [])):
        vertices = annotation['vertices']

        # Convert to bounding box
        bbox = polygon_to_bbox(vertices)
        x_min, y_min, x_max, y_max = bbox

        # Scale bounding box coordinates
        scaled_bbox = scale_coordinates([x_min, y_min, x_max, y_max], scale_factor)
        x_min_s, y_min_s, x_max_s, y_max_s = scaled_bbox

        # Create rectangle patch
        width = x_max_s - x_min_s
        height = y_max_s - y_min_s
        rect = patches.Rectangle((x_min_s, y_min_s), width, height,
                                 fill=False, edgecolor=colors[i % len(colors)],
                                 linewidth=2, alpha=0.8)
        ax2.add_patch(rect)

        # Add YOLO format coordinates as text
        x_center, y_center, norm_width, norm_height = bbox_to_yolo_format(bbox, orig_width, orig_height)
        ax2.text(x_min_s, y_min_s - 10,
                 f'P{i}: ({x_center:.3f}, {y_center:.3f}, {norm_width:.3f}, {norm_height:.3f})',
                 fontsize=8, color=colors[i % len(colors)],
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Process negative annotations
    for i, annotation in enumerate(json_data.get('negative', [])):
        vertices = annotation['vertices']

        # Convert to bounding box
        bbox = polygon_to_bbox(vertices)
        x_min, y_min, x_max, y_max = bbox

        # Scale bounding box coordinates
        scaled_bbox = scale_coordinates([x_min, y_min, x_max, y_max], scale_factor)
        x_min_s, y_min_s, x_max_s, y_max_s = scaled_bbox

        # Create rectangle patch
        width = x_max_s - x_min_s
        height = y_max_s - y_min_s
        rect = patches.Rectangle((x_min_s, y_min_s), width, height,
                                 fill=False, edgecolor='darkred',
                                 linewidth=2, alpha=0.8, linestyle='--')
        ax2.add_patch(rect)

        # Add YOLO format coordinates as text
        x_center, y_center, norm_width, norm_height = bbox_to_yolo_format(bbox, orig_width, orig_height)
        ax2.text(x_min_s, y_min_s - 10,
                 f'N{i}: ({x_center:.3f}, {y_center:.3f}, {norm_width:.3f}, {norm_height:.3f})',
                 fontsize=8, color='darkred',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=2, label='Positive'),
        plt.Line2D([0], [0], color='darkred', linewidth=2, linestyle='--', label='Negative')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")

    plt.show()
    plt.close()  # Close the figure to free memory


def visualize_conversion(json_path, images_folder, output_folder=None):
    """
    Visualize the conversion from JSON polygons to YOLO bounding boxes.

    Args:
        json_path: Path to the JSON annotation file
        images_folder: Path to the folder containing images
        output_folder: Path to save visualization plots (optional)
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get the base name of the JSON file to match with image
    json_name = Path(json_path).stem

    # Find corresponding image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    image_path = None

    for ext in image_extensions:
        potential_path = os.path.join(images_folder, json_name + ext)
        if os.path.exists(potential_path):
            image_path = potential_path
            break

    if image_path is None:
        print(f"Warning: No image found for {json_name}")
        return

    # Create output path for visualization
    output_path = None
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{json_name}_visualization.png")

    # Plot comparison
    plot_annotations_comparison(image_path, data, output_path)

    # Print statistics
    num_positive = len(data.get('positive', []))
    num_negative = len(data.get('negative', []))
    print(f"\nAnnotation Statistics for {json_name}:")
    print(f"  Positive annotations: {num_positive}")
    print(f"  Negative annotations: {num_negative}")
    print(f"  Total annotations: {num_positive + num_negative}")
    """
    Get image dimensions without fully loading the image.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def convert_json_to_yolo(json_path, images_folder, labels_folder):
    """
    Convert BCNB JSON annotations to YOLO format.

    Args:
        json_path: Path to the JSON annotation file
        images_folder: Path to the folder containing images
        labels_folder: Path to the output labels folder
    """
    # Create labels folder if it doesn't exist
    os.makedirs(labels_folder, exist_ok=True)

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get the base name of the JSON file to match with image
    json_name = Path(json_path).stem

    # Find corresponding image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    image_path = None

    for ext in image_extensions:
        potential_path = os.path.join(images_folder, json_name + ext)
        if os.path.exists(potential_path):
            image_path = potential_path
            break

    if image_path is None:
        print(f"Warning: No image found for {json_name}")
        return

    # Get image dimensions
    try:
        img_width, img_height = get_image_dimensions(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return

    # Prepare YOLO format annotations
    yolo_annotations = []

    # Process positive annotations (class 0)
    for annotation in data.get('positive', []):
        vertices = annotation['vertices']

        # Convert polygon to bounding box
        bbox = polygon_to_bbox(vertices)

        # Convert to YOLO format
        x_center, y_center, width, height = bbox_to_yolo_format(bbox, img_width, img_height)

        # Class 0 for positive annotations
        class_id = 0
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Process negative annotations (class 1) if they exist
    for annotation in data.get('negative', []):
        vertices = annotation['vertices']

        # Convert polygon to bounding box
        bbox = polygon_to_bbox(vertices)

        # Convert to YOLO format
        x_center, y_center, width, height = bbox_to_yolo_format(bbox, img_width, img_height)

        # Class 1 for negative annotations
        class_id = 1
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save YOLO format annotations
    output_file = os.path.join(labels_folder, json_name + '.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    print(f"Converted {json_name}: {len(yolo_annotations)} annotations -> {output_file}")


def batch_convert_and_visualize(json_folder, images_folder, labels_folder):
    """
    Convert all JSON files to YOLO format and create visualizations.

    Args:
        json_folder: Path to folder containing JSON files
        images_folder: Path to folder containing images
        labels_folder: Path to output labels folder
        visualizations_folder: Path to save visualization plots (optional)
    """
    json_files = list(Path(json_folder).glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {json_folder}")
        return

    print(f"Found {len(json_files)} JSON files to convert...")

    for json_file in json_files:
        # Convert to YOLO format
        convert_json_to_yolo(str(json_file), images_folder, labels_folder)


    visualize_conversion(str(json_files[0]), images_folder, ".")

    print(f"\nConversion complete! Labels saved to {labels_folder}")




def create_classes_file(labels_folder):
    """
    Create a classes.txt file for YOLO training.

    Args:
        labels_folder: Path to labels folder
    """
    classes = ["positive", "negative"]  # Adjust class names as needed

    classes_file = os.path.join(labels_folder, 'classes.txt')
    with open(classes_file, 'w') as f:
        f.write('\n'.join(classes))

    print(f"Created classes file: {classes_file}")


# Example usage
if __name__ == "__main__":
    # Configuration
    json_folder = "./images"  # Folder containing JSON annotation files
    images_folder = "./images"  # Folder containing images
    labels_folder = "./labels"  # Output folder for YOLO labels

    # Convert and visualize single file
    # convert_json_to_yolo("1.json", images_folder, labels_folder)
    # visualize_conversion("1.json", images_folder, visualizations_folder)

    # Convert entire directory with visualizations
    batch_convert_and_visualize(json_folder, images_folder, labels_folder)

    # Create classes file
    create_classes_file(labels_folder)

    # Create classes file
    create_classes_file(labels_folder)

    print("\nYOLO Format Structure:")
    print("- Each line in .txt file: class_id x_center y_center width height")
    print("- All coordinates are normalized (0.0 to 1.0)")
    print("- Class 0: positive annotations")
    print("- Class 1: negative annotations")
    print("\nVisualization Features:")
    print("- Left plot: Original JSON polygon annotations")
    print("- Right plot: Converted YOLO bounding boxes")
    print("- Images automatically resized for display (max 2048px)")
    print("- YOLO coordinates shown as normalized values")
