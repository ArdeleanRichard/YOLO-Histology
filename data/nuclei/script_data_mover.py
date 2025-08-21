import os
import shutil
from PIL import Image

# Define folder paths
original_folder = 'original_images'
images_folder = 'images'
masks_folder = 'masks'

# Create destination folders if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)

scale_factor=0.5
resampling_method="box"

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

# Iterate over files in the original_images folder
for filename in os.listdir(original_folder):
    file_path = os.path.join(original_folder, filename)

    if os.path.isfile(file_path):
        name_parts = filename.rsplit('.', 1)  # Split into name and extension
        if len(name_parts) != 2:
            continue  # Skip files without an extension

        name, ext = name_parts
        if name.endswith('_original'):
            # Load and downsample image
            img = Image.open(file_path)
            original_width, original_height = img.size

            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            new_size = (new_width, new_height)

            print(f"  Resizing from {original_width}x{original_height} to {new_width}x{new_height} using {resampling_method}")

            # Use area averaging (BOX) for better preservation of histological features
            # This method averages pixel intensities within regions, preserving diagnostic information
            img_small = img.resize(new_size, resampling_filter)

            # Save downsampled image
            output_img_path = images_folder+f"/{name}.jpg"
            img_small.save(output_img_path, "JPEG", quality=95)

            # shutil.copy(file_path, os.path.join(images_folder, filename))
        elif name.endswith('_mask'):
            shutil.copy(file_path, os.path.join(masks_folder, filename))

print("Files have been moved successfully.")
