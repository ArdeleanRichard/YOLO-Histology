import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


def split_dataset(base_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test folders maintaining correspondence between images, masks, and labels.

    Args:
        base_path (str): Path to the directory containing images/, masks/, labels/ folders
        train_ratio (float): Ratio for training set (default: 0.7)
        val_ratio (float): Ratio for validation set (default: 0.2)
        test_ratio (float): Ratio for test set (default: 0.1)
        seed (int): Random seed for reproducibility
    """

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    base_path = Path(base_path)

    # Define folder paths
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    labels_dir = base_path / "labels"

    # Check if folders exist
    for folder in [images_dir, masks_dir, labels_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")

    # Get all unique IDs by examining the images folder
    image_files = list(images_dir.glob("*"))
    ids = set()

    for img_file in image_files:
        if img_file.is_file():
            # Extract ID from filename (remove _original suffix and extension)
            filename = img_file.stem  # filename without extension
            if filename.endswith("_original"):
                file_id = filename[:-9]  # remove "_original"
                ids.add(file_id)

    if not ids:
        raise ValueError("No files with '_original' suffix found in images folder")

    ids = list(ids)
    print(f"Found {len(ids)} unique IDs")

    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(ids)

    # Calculate split indices
    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Ensure all samples are assigned

    # Split IDs
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Create split directories
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    for split_name in splits.keys():
        for folder_type in ['images', 'masks', 'labels']:
            split_dir = base_path / folder_type / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

    # Move files to appropriate splits
    moved_files = defaultdict(int)
    missing_files = defaultdict(list)

    for split_name, split_ids in splits.items():
        for file_id in split_ids:
            # Define expected filenames
            image_filename = f"{file_id}_original"
            mask_filename = f"{file_id}_mask"
            label_filename = f"{file_id}_original"

            # Move image file
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
                image_src = images_dir / f"{image_filename}{ext}"
                if image_src.exists():
                    image_dst = images_dir / split_name / f"{image_filename}{ext}"
                    shutil.move(str(image_src), str(image_dst))
                    moved_files['images'] += 1
                    break
            else:
                missing_files['images'].append(file_id)

            # Move mask file
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
                mask_src = masks_dir / f"{mask_filename}{ext}"
                if mask_src.exists():
                    mask_dst = masks_dir / split_name / f"{mask_filename}{ext}"
                    shutil.move(str(mask_src), str(mask_dst))
                    moved_files['masks'] += 1
                    break
            else:
                missing_files['masks'].append(file_id)

            # Move label file
            for ext in ['.txt', '.json', '.xml', '.csv']:
                label_src = labels_dir / f"{label_filename}{ext}"
                if label_src.exists():
                    label_dst = labels_dir / split_name / f"{label_filename}{ext}"
                    shutil.move(str(label_src), str(label_dst))
                    moved_files['labels'] += 1
                    break
            else:
                missing_files['labels'].append(file_id)

    # Print summary
    print("\n=== SPLIT COMPLETE ===")
    print("Files moved:")
    for folder_type, count in moved_files.items():
        print(f"  {folder_type}: {count} files")

    if any(missing_files.values()):
        print("\nMissing files:")
        for folder_type, missing_ids in missing_files.items():
            if missing_ids:
                print(f"  {folder_type}: {len(missing_ids)} missing files")
                print(f"    IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")


if __name__ == "__main__":
    # Configuration
    BASE_PATH = "."  # Current directory - change this to your dataset path
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    RANDOM_SEED = 42

    print(f"Starting dataset split with ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Dataset path: {BASE_PATH}")
    print(f"Random seed: {RANDOM_SEED}")

    try:
        split_dataset(
            base_path=BASE_PATH,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=RANDOM_SEED
        )
    except Exception as e:
        print(f"Error: {e}")