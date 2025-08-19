import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Update these paths
wsi_path = './original_images/1.jpg'
json_path = './original_images/1.json'

# Reduce size when opening
Image.MAX_IMAGE_PIXELS = None
img = Image.open(wsi_path)
scale_factor = 0.05  # 5% of original size
new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
img_small = img.resize(new_size, Image.Resampling.LANCZOS)

# Load annotations
with open(json_path, 'r') as f:
    ann = json.load(f)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_small)
ax.axis('off')

# Draw polygons (scaled)
for region in ann.get("positive", []):
    verts = [(x * scale_factor, y * scale_factor) for x, y in region["vertices"]]
    poly = patches.Polygon(verts, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(poly)

plt.show()

