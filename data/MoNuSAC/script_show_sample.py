import os
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random


def get_class_mapping():
    """Define the class mapping for YOLO format"""
    return {
        'Epithelial': 0,
        'Lymphocyte': 1,
        'Neutrophil': 2,
        'Macrophage': 3
    }


def get_class_colors():
    """Define colors for each cell type"""
    return {
        'Epithelial': '#FF0000',  # Red
        'Lymphocyte': '#00FF00',  # Green
        'Neutrophil': '#0000FF',  # Blue
        'Macrophage': '#FFFF00'  # Yellow
    }


def parse_xml_polygons(xml_path):
    """Parse XML annotation file and extract polygon data"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_mapping = get_class_mapping()
    polygons_data = []

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

        # Process all regions for this annotation
        regions = annotation.find('Regions')
        if regions is not None:
            for region in regions.findall('Region'):
                vertices = region.find('Vertices')
                if vertices is not None:
                    vertex_list = vertices.findall('Vertex')
                    if len(vertex_list) >= 3:  # Need at least 3 points for a polygon
                        # Extract polygon coordinates
                        polygon_coords = [(float(v.get('X')), float(v.get('Y'))) for v in vertex_list]
                        polygons_data.append((cell_type, polygon_coords))

    return polygons_data


def read_yolo_annotations(yolo_path):
    """Read YOLO annotations from txt file"""
    yolo_annotations = []
    if os.path.exists(yolo_path):
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:5])
                    yolo_annotations.append((class_id, center_x, center_y, width, height))
    return yolo_annotations


def yolo_to_bbox_coords(yolo_bbox, img_width, img_height):
    """Convert YOLO format back to pixel coordinates (x_min, y_min, x_max, y_max)"""
    class_id, center_x, center_y, width, height = yolo_bbox

    # Convert from normalized to pixel coordinates
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height

    # Calculate corners
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2

    return x_min, y_min, x_max, y_max


def compare_annotations(image_path, xml_path, yolo_path, figsize=(15, 7)):
    """
    Plot original XML polygon annotations and YOLO bounding boxes side by side

    Args:
        image_path: Path to the image file
        xml_path: Path to the XML annotation file
        yolo_path: Path to the YOLO txt file
        figsize: Figure size tuple
    """
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Parse XML polygons
    polygons_data = parse_xml_polygons(xml_path)

    # Read YOLO annotations
    yolo_annotations = read_yolo_annotations(yolo_path)

    # Get class mapping and colors
    class_mapping = get_class_mapping()
    class_colors = get_class_colors()
    id_to_class = {v: k for k, v in class_mapping.items()}

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot original XML polygons
    ax1.imshow(img)
    ax1.set_title('Original XML Polygon Annotations', fontsize=14)
    ax1.axis('off')

    for cell_type, polygon_coords in polygons_data:
        color = class_colors[cell_type]
        polygon = Polygon(polygon_coords, linewidth=2, edgecolor=color,
                          facecolor='none', alpha=0.8)
        ax1.add_patch(polygon)

    # Plot YOLO bounding boxes
    ax2.imshow(img)
    ax2.set_title('YOLO Bounding Box Annotations', fontsize=14)
    ax2.axis('off')

    for yolo_bbox in yolo_annotations:
        class_id = int(yolo_bbox[0])
        cell_type = id_to_class.get(class_id, f'Class_{class_id}')
        color = class_colors.get(cell_type, '#000000')

        # Convert YOLO format to pixel coordinates
        x_min, y_min, x_max, y_max = yolo_to_bbox_coords(yolo_bbox, img_width, img_height)

        # Create rectangle
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
        ax2.add_patch(rect)

    # Add legend
    legend_elements = []
    used_classes = set()

    # Add classes from polygons
    for cell_type, _ in polygons_data:
        if cell_type not in used_classes:
            legend_elements.append(plt.Line2D([0], [0], color=class_colors[cell_type],
                                              lw=2, label=cell_type))
            used_classes.add(cell_type)

    # Add classes from YOLO (in case some are only in YOLO)
    for yolo_bbox in yolo_annotations:
        class_id = int(yolo_bbox[0])
        cell_type = id_to_class.get(class_id, f'Class_{class_id}')
        if cell_type not in used_classes and cell_type in class_colors:
            legend_elements.append(plt.Line2D([0], [0], color=class_colors[cell_type],
                                              lw=2, label=cell_type))
            used_classes.add(cell_type)

    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                   ncol=len(legend_elements), fontsize=12)
        plt.subplots_adjust(bottom=0.1)

    plt.tight_layout()
    plt.savefig("./visualization_")

    # Print statistics
    print(f"\nAnnotation Statistics for {os.path.basename(image_path)}:")
    print(f"Image size: {img_width} x {img_height}")
    print(f"XML polygons: {len(polygons_data)}")
    print(f"YOLO bounding boxes: {len(yolo_annotations)}")

    # Count by class
    xml_class_counts = {}
    for cell_type, _ in polygons_data:
        xml_class_counts[cell_type] = xml_class_counts.get(cell_type, 0) + 1

    yolo_class_counts = {}
    for yolo_bbox in yolo_annotations:
        class_id = int(yolo_bbox[0])
        cell_type = id_to_class.get(class_id, f'Class_{class_id}')
        yolo_class_counts[cell_type] = yolo_class_counts.get(cell_type, 0) + 1

    print("\nClass distribution:")
    all_classes = set(xml_class_counts.keys()) | set(yolo_class_counts.keys())
    for cell_type in sorted(all_classes):
        xml_count = xml_class_counts.get(cell_type, 0)
        yolo_count = yolo_class_counts.get(cell_type, 0)
        print(f"  {cell_type}: XML={xml_count}, YOLO={yolo_count}")


# Example usage:
if __name__ == "__main__":
    file_name = "TCGA-5P-A9K0-01Z-00-DX1_1"
    folder_name=file_name.split("_")[0]
    image_path = f"./images/train/{file_name}.tif"
    xml_path = f"./original_images/MoNuSAC_images_and_annotations/{folder_name}/{file_name}.xml"
    yolo_path = f"./labels/train/{file_name}.txt"

    # file_name = "TCGA-DW-7838-01Z-00-DX1_1"
    # folder_name=file_name.split("_")[0]
    # image_path = f"./images/val/{file_name}.tif"
    # xml_path = f"./original_images/MoNuSAC Testing Data and Annotations/{folder_name}/{file_name}.xml"
    # yolo_path = f"./labels/val/{file_name}.txt"

    compare_annotations(image_path, xml_path, yolo_path)