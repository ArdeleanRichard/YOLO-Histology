import os
import shutil
from pathlib import Path


def read_ids_from_file(file_path):
    """Read image IDs from a text file"""
    ids = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    ids.append(line)
        print(f"Read {len(ids)} IDs from {file_path}")
        return ids
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []


def create_split_folders(dataset_dir, splits=['train', 'val', 'test']):
    """Create train/val/test subfolders in images and labels directories"""
    dataset_path = Path(dataset_dir)

    for folder_type in ['images', 'labels']:
        folder_path = dataset_path / folder_type
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist")
            continue

        for split in splits:
            split_folder = folder_path / split
            split_folder.mkdir(exist_ok=True)
            print(f"Created/verified: {split_folder}")


def move_files_by_ids(dataset_dir, split_name, ids):
    """Move image and label files to the appropriate split folder"""
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    # Target directories
    target_images_dir = images_dir / split_name
    target_labels_dir = labels_dir / split_name

    moved_images = 0
    moved_labels = 0
    missing_files = []

    for file_id in ids:
        # Look for image file (try different extensions)
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        image_file = None

        for ext in image_extensions:
            potential_image = images_dir / f"{file_id}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break

        # Look for label file
        label_file = labels_dir / f"{file_id}.txt"

        # Move image file if found
        if image_file and image_file.exists():
            target_image_path = target_images_dir / image_file.name
            try:
                shutil.move(str(image_file), str(target_image_path))
                moved_images += 1
            except Exception as e:
                print(f"Error moving image {image_file}: {str(e)}")
        else:
            missing_files.append(f"Image: {file_id}")

        # Move label file if found
        if label_file.exists():
            target_label_path = target_labels_dir / label_file.name
            try:
                shutil.move(str(label_file), str(target_label_path))
                moved_labels += 1
            except Exception as e:
                print(f"Error moving label {label_file}: {str(e)}")
        else:
            missing_files.append(f"Label: {file_id}.txt")

    print(f"{split_name.upper()}: Moved {moved_images} images and {moved_labels} labels")

    if missing_files:
        print(f"Warning: {len(missing_files)} missing files for {split_name}:")
        for missing in missing_files[:10]:  # Show first 10 missing files
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    return moved_images, moved_labels, len(missing_files)


def split_yolo_dataset(dataset_dir, train_ids_file, val_ids_file, test_ids_file):
    """
    Split YOLO dataset into train/val/test folders based on ID files

    Args:
        dataset_dir (str): Path to dataset directory containing images/ and labels/ folders
        train_ids_file (str): Path to file containing training IDs
        val_ids_file (str): Path to file containing validation IDs
        test_ids_file (str): Path to file containing test IDs
    """

    dataset_path = Path(dataset_dir)

    # Verify dataset structure
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return

    if not labels_dir.exists():
        print(f"Error: Labels directory {labels_dir} does not exist")
        return

    print(f"Processing dataset in: {dataset_path}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")

    # Read IDs from files
    id_files = {
        'train': train_ids_file,
        'val': val_ids_file,
        'test': test_ids_file
    }

    all_ids = {}
    for split_name, id_file in id_files.items():
        if id_file and os.path.exists(id_file):
            all_ids[split_name] = read_ids_from_file(id_file)
        else:
            print(f"Skipping {split_name}: file {id_file} not found")
            all_ids[split_name] = []

    # Check for overlapping IDs
    train_set = set(all_ids.get('train', []))
    val_set = set(all_ids.get('val', []))
    test_set = set(all_ids.get('test', []))

    overlaps = []
    if train_set & val_set:
        overlaps.append(f"Train-Val: {len(train_set & val_set)} overlapping IDs")
    if train_set & test_set:
        overlaps.append(f"Train-Test: {len(train_set & test_set)} overlapping IDs")
    if val_set & test_set:
        overlaps.append(f"Val-Test: {len(val_set & test_set)} overlapping IDs")

    if overlaps:
        print("Warning: Found overlapping IDs between splits:")
        for overlap in overlaps:
            print(f"  - {overlap}")
    else:
        print("✓ No overlapping IDs found between splits")

    # Create split folders
    splits_to_create = [split for split in ['train', 'val', 'test'] if all_ids.get(split)]
    create_split_folders(dataset_dir, splits_to_create)

    # Move files for each split
    total_images = 0
    total_labels = 0
    total_missing = 0

    for split_name in ['train', 'val', 'test']:
        if all_ids.get(split_name):
            print(f"\nProcessing {split_name} split...")
            moved_imgs, moved_lbls, missing = move_files_by_ids(
                dataset_dir, split_name, all_ids[split_name]
            )
            total_images += moved_imgs
            total_labels += moved_lbls
            total_missing += missing

    # Summary
    print(f"\n{'=' * 50}")
    print("DATASET SPLIT SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total files moved:")
    print(f"  - Images: {total_images}")
    print(f"  - Labels: {total_labels}")
    print(f"  - Missing files: {total_missing}")

    # Show final structure
    print(f"\nFinal dataset structure:")
    print(f"{dataset_path}/")
    for split in splits_to_create:
        img_count = len(list((images_dir / split).glob('*'))) if (images_dir / split).exists() else 0
        lbl_count = len(list((labels_dir / split).glob('*.txt'))) if (labels_dir / split).exists() else 0
        print(f"├── images/{split}/     ({img_count} images)")
        print(f"├── labels/{split}/     ({lbl_count} labels)")

    if (dataset_path / 'classes.txt').exists():
        print(f"└── classes.txt")


def verify_split_integrity(dataset_dir):
    """Verify that each split has matching images and labels"""
    dataset_path = Path(dataset_dir)

    print(f"\n{'=' * 50}")
    print("SPLIT INTEGRITY VERIFICATION")
    print(f"{'=' * 50}")

    for split in ['train', 'val', 'test']:
        images_split = dataset_path / 'images' / split
        labels_split = dataset_path / 'labels' / split

        if not images_split.exists() or not labels_split.exists():
            continue

        # Get file stems (without extensions)
        image_stems = {f.stem for f in images_split.glob('*') if f.is_file()}
        label_stems = {f.stem for f in labels_split.glob('*.txt') if f.is_file()}

        # Find mismatches
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        print(f"\n{split.upper()} split:")
        print(f"  Images: {len(image_stems)}")
        print(f"  Labels: {len(label_stems)}")

        if missing_labels:
            print(f"  ⚠️  Images without labels: {len(missing_labels)}")
            if len(missing_labels) <= 5:
                for missing in missing_labels:
                    print(f"    - {missing}")

        if missing_images:
            print(f"  ⚠️  Labels without images: {len(missing_images)}")
            if len(missing_images) <= 5:
                for missing in missing_images:
                    print(f"    - {missing}")

        if not missing_labels and not missing_images:
            print(f"  ✓ All files have matching pairs")

# Example usage
if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "./"  # Your dataset directory
    TRAIN_IDS_FILE = "./dataset-splitting/train_id.txt"
    VAL_IDS_FILE = "./dataset-splitting/val_id.txt"
    TEST_IDS_FILE = "./dataset-splitting/test_id.txt"

    # Split the dataset
    split_yolo_dataset(
        dataset_dir=DATASET_DIR,
        train_ids_file=TRAIN_IDS_FILE,
        val_ids_file=VAL_IDS_FILE,
        test_ids_file=TEST_IDS_FILE
    )

    # Verify the split
    verify_split_integrity(DATASET_DIR)