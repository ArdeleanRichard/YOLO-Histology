import os
import shutil
from pathlib import Path

def convert_yolo_to_seg(input_folder, output_folder):
    """
    Convert YOLO detection dataset to segmentation format

    Args:
        input_folder: Path to existing YOLO dataset
        output_folder: Path where segmentation dataset will be created
    """
    import shutil
    import yaml
    from pathlib import Path

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output structure
    output_path.mkdir(exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        input_images = input_path / 'images' / split
        input_labels = input_path / 'labels' / split

        if not input_images.exists():
            print(f"Warning: {input_images} does not exist, skipping {split}")
            continue

        # Copy images
        for img_file in input_images.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                shutil.copy2(img_file, output_path / 'images' / split / img_file.name)

        # Convert labels
        for label_file in input_labels.glob('*.txt'):
            new_labels = []

            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Skip empty files
            if not lines:
                continue

            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    try:
                        class_id = parts[0]
                        x_center, y_center, width, height = map(float, parts[1:5])

                        # Validate bbox coordinates
                        if width <= 0 or height <= 0:
                            print(f"Warning: Invalid bbox dimensions in {label_file.name}: w={width}, h={height}")
                            continue

                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            print(f"Warning: Invalid bbox center in {label_file.name}: x={x_center}, y={y_center}")
                            continue

                        # Calculate bbox corners with proper bounds checking
                        x1 = max(0.0, x_center - width / 2)
                        y1 = max(0.0, y_center - height / 2)
                        x2 = min(1.0, x_center + width / 2)
                        y2 = min(1.0, y_center + height / 2)

                        # Ensure we have a valid rectangle
                        if x2 <= x1 or y2 <= y1:
                            print(f"Warning: Degenerate bbox in {label_file.name}: ({x1},{y1}) to ({x2},{y2})")
                            continue

                        # Create rectangle polygon in correct order (counterclockwise)
                        # Format: class_id x1 y1 x2 y1 x2 y2 x1 y2
                        polygon_coords = [x1, y1, x2, y1, x2, y2, x1, y2]

                        # Format as string with proper precision
                        coord_str = ' '.join([f"{coord:.6f}" for coord in polygon_coords])
                        polygon = f"{class_id} {coord_str}"
                        new_labels.append(polygon)

                    except ValueError as e:
                        print(f"Warning: Could not parse line in {label_file.name}: {line} - {e}")
                        continue

            # Only write file if we have valid labels
            if new_labels:
                output_label_path = output_path / 'labels' / split / label_file.name
                with open(output_label_path, 'w') as f:
                    for label in new_labels:
                        f.write(label + '\n')
            else:
                print(f"Warning: No valid labels found in {label_file.name}")


    print(f"Conversion complete: {output_folder}")
    print("Dataset structure:")
    print(f"  Images: {len(list((output_path / 'images' / 'train').glob('*')))} train, {len(list((output_path / 'images' / 'val').glob('*')))} val, {len(list((output_path / 'images' / 'test').glob('*')))} test")
    print(f"  Labels: {len(list((output_path / 'labels' / 'train').glob('*')))} train, {len(list((output_path / 'labels' / 'val').glob('*')))} val, {len(list((output_path / 'labels' / 'test').glob('*')))} test")

if __name__ == "__main__":
    convert_yolo_to_seg("../MoNuSAC/", "../MoNuSAC/MoNuSAC_seg/")