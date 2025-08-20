import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from PIL import Image
import numpy as np
import glob

def create_directories(base_path):
    """Create the required directory structure"""
    directories = [
        'images/train',
        'images/val',
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]

    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")


def get_class_mapping():
    """Define the class mapping for YOLO format"""
    return {
        'Epithelial': 0,
        'Lymphocyte': 1,
        'Neutrophil': 2,
        'Macrophage': 3
    }


def polygon_to_bbox(vertices):
    """Convert polygon vertices to bounding box (x_min, y_min, x_max, y_max)"""
    x_coords = [float(vertex.get('X')) for vertex in vertices]
    y_coords = [float(vertex.get('Y')) for vertex in vertices]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return x_min, y_min, x_max, y_max


def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert bounding box to YOLO format (normalized center_x, center_y, width, height)"""
    x_min, y_min, x_max, y_max = bbox

    # Calculate center point and dimensions
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize by image dimensions
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height

    return center_x, center_y, width, height


def parse_xml_annotation(xml_path, img_width, img_height):
    """Parse XML annotation file and extract YOLO format annotations"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_mapping = get_class_mapping()
    annotations = []

    # Iterate through all annotations
    for annotation in root.findall('Annotation'):
        # Get cell type from attributes
        cell_type = None
        for attr in annotation.find('Attributes').findall('Attribute'):
            attr_name = attr.get('Name')
            if attr_name in class_mapping:
                cell_type = attr_name
                break

        if cell_type is None:
            continue

        class_id = class_mapping[cell_type]

        # Process all regions for this annotation
        regions = annotation.find('Regions')
        if regions is not None:
            for region in regions.findall('Region'):
                vertices = region.find('Vertices')
                if vertices is not None:
                    vertex_list = vertices.findall('Vertex')
                    if len(vertex_list) >= 3:  # Need at least 3 points for a polygon
                        # Convert polygon to bounding box
                        bbox = polygon_to_bbox(vertex_list)

                        # Convert to YOLO format
                        yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)

                        # Add to annotations list
                        annotations.append((class_id, *yolo_bbox))

    return annotations


def get_image_dimensions(img_path):
    """Get image width and height"""
    try:
        with Image.open(img_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None, None


def process_dataset(original_path, output_path, val_split=0.15, test_split=0.15):
    """Process the entire dataset"""

    # Create output directories
    create_directories(output_path)

    # Define paths
    train_source = os.path.join(original_path, 'MoNuSAC_images_and_annotations')
    test_source = os.path.join(original_path, 'MoNuSAC Testing Data and Annotations')

    # Process training data
    print("\nProcessing training data...")
    if os.path.exists(train_source):
        process_folder(train_source, output_path, 'train')
    else:
        print(f"Training folder not found: {train_source}")

    # Process test data and split into val/test
    print("\nProcessing test data...")
    if os.path.exists(test_source):
        # Get all subfolders in test source
        test_folders = [f for f in os.listdir(test_source)
                        if os.path.isdir(os.path.join(test_source, f))]

        if test_folders:
            # Shuffle and split
            random.shuffle(test_folders)
            val_count = int(len(test_folders) * val_split / (val_split + test_split))

            val_folders = test_folders[:val_count]
            test_folders = test_folders[val_count:]

            # Process validation folders
            print(f"Processing {len(val_folders)} folders for validation...")
            for folder in val_folders:
                folder_path = os.path.join(test_source, folder)
                process_folder(folder_path, output_path, 'val', single_folder=True)

            # Process test folders
            print(f"Processing {len(test_folders)} folders for testing...")
            for folder in test_folders:
                folder_path = os.path.join(test_source, folder)
                process_folder(folder_path, output_path, 'test', single_folder=True)
        else:
            print("No subfolders found in test source")
    else:
        print(f"Test folder not found: {test_source}")


def process_folder(source_folder, output_path, split, single_folder=False):
    """Process a folder containing images and annotations"""

    if single_folder:
        # Process single folder directly
        subfolders = [source_folder]
    else:
        # Get all subfolders
        subfolders = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
                      if os.path.isdir(os.path.join(source_folder, f))]

    processed_count = 0

    for subfolder in subfolders:
        if not os.path.isdir(subfolder):
            continue

        # Get all .tif files in the subfolder
        tif_files = [f for f in os.listdir(subfolder) if f.lower().endswith('.tif')]

        for tif_file in tif_files:
            # Process each image and its annotation
            img_path = os.path.join(subfolder, tif_file)
            xml_file = tif_file.replace('.tif', '.xml')
            xml_path = os.path.join(subfolder, xml_file)

            if os.path.exists(xml_path):
                # Copy image
                dest_img_path = os.path.join(output_path, 'images', split, tif_file)

                # shutil.copy2(img_path, dest_img_path)

                img = Image.open(img_path).convert("RGB")  # force 3 channels
                img.save(dest_img_path, format="TIFF", compression="tiff_deflate")

                # Get image dimensions
                img_width, img_height = get_image_dimensions(img_path)
                if img_width is None or img_height is None:
                    print(f"Skipping {tif_file} - couldn't read image dimensions")
                    continue

                # Parse XML and create YOLO annotation
                annotations = parse_xml_annotation(xml_path, img_width, img_height)

                # Save YOLO format annotation
                txt_file = tif_file.replace('.tif', '.txt')
                dest_txt_path = os.path.join(output_path, 'labels', split, txt_file)

                with open(dest_txt_path, 'w') as f:
                    for annotation in annotations:
                        class_id, center_x, center_y, width, height = annotation
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} images for {split}")
            else:
                print(f"Warning: No XML annotation found for {tif_file}")

    print(f"Completed {split}: processed {processed_count} images")


def create_dataset_yaml(output_path, class_names):
    """Create a YOLO dataset configuration file"""
    yaml_content = f"""# MoNuSAC Dataset Configuration
path: {os.path.abspath(output_path)}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(class_names)}  # number of classes
names: {list(class_names.keys())}  # class names
"""

    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created dataset configuration: {yaml_path}")


def main():
    # Set paths
    original_path = './original_images'  # Adjust this path as needed
    output_path = '.'  # Current directory, adjust as needed

    # Set random seed for reproducible splits
    random.seed(42)

    print("Starting MoNuSAC dataset processing...")
    print(f"Source: {original_path}")
    print(f"Output: {output_path}")

    # Process the dataset
    process_dataset(original_path, output_path, val_split=0.15, test_split=0.15)

    # Create YOLO dataset configuration
    class_mapping = get_class_mapping()
    create_dataset_yaml(output_path, class_mapping)

    print("\nDataset processing completed!")
    print("\nClass mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")


if __name__ == "__main__":
    main()