import os
import shutil

# Define folder paths
original_folder = 'original_images'
images_folder = 'images'
masks_folder = 'masks'

# Create destination folders if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)

# Iterate over files in the original_images folder
for filename in os.listdir(original_folder):
    file_path = os.path.join(original_folder, filename)

    if os.path.isfile(file_path):
        name_parts = filename.rsplit('.', 1)  # Split into name and extension
        if len(name_parts) != 2:
            continue  # Skip files without an extension

        name, ext = name_parts
        if name.endswith('_original'):
            shutil.move(file_path, os.path.join(images_folder, filename))
        elif name.endswith('_mask'):
            shutil.move(file_path, os.path.join(masks_folder, filename))

print("Files have been moved successfully.")
