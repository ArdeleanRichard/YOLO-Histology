import os
import shutil
from pathlib import Path

# Define base paths
base_dir = Path("./original_images")
output_images = Path("./images")
output_masks = Path("./masks")

# Train/val/test split mapping
def get_split(index: int) -> str:
    if 1 <= index <= 9:
        return "train"
    elif index == 10:
        return "val"
    elif index == 11:
        return "test"
    else:
        raise ValueError(f"Unexpected index {index}")

# Create destination subfolders if not exist
for split in ["train", "val", "test"]:
    (output_images / split).mkdir(parents=True, exist_ok=True)
    (output_masks / split).mkdir(parents=True, exist_ok=True)

# Iterate over original_images subfolders
for folder in base_dir.iterdir():
    if folder.is_dir():
        name = folder.name  # e.g., GT_01 or Slide_03
        if "_" not in name:
            continue

        prefix, idx_str = name.split("_")
        idx = int(idx_str)

        split = get_split(idx)

        if prefix == "GT":
            dest = output_masks / split
        elif prefix == "Slide":
            dest = output_images / split
        else:
            continue

        # Copy all files inside the folder
        for file in folder.iterdir():
            if file.is_file():
                shutil.copy(file, dest / file.name)

print("Images copied and organized successfully!")
