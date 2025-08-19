import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# same color palette as before
DEFAULT_COLORS = [
    (0.121, 0.466, 0.705), (1.000, 0.498, 0.054), (0.172, 0.627, 0.172),
    (0.839, 0.153, 0.157), (0.580, 0.404, 0.741), (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811)
]


def _load_image(path):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.array(im).astype(np.float32) / 255.0


def _load_mask(path):
    return np.array(Image.open(path).convert("L")).astype(np.int32)


def plot_from_folders(image_folder, mask_folder, n=3, alpha=0.5, class_colors=None):
    """
    Load matching images + masks from folders and plot n samples.
    Filenames must match (e.g., img1.png <-> img1.png).
    """
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # match by common filenames
    common = sorted(set(image_files) & set(mask_files))
    if not common:
        raise ValueError("No matching filenames between image and mask folders!")

    for i, fname in enumerate(common[:n]):
        img = _load_image(os.path.join(image_folder, fname))
        mask = _load_mask(os.path.join(mask_folder, fname))
        H, W = mask.shape

        # resize image if not same size
        if img.shape[0] != H or img.shape[1] != W:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))).astype(np.float32) / 255.0

        labels = np.unique(mask)
        colors = {}
        next_color = 0
        for lab in labels:
            if class_colors and lab in class_colors:
                colors[lab] = class_colors[lab]
            elif lab == 0:
                colors[lab] = (0, 0, 0)
            else:
                colors[lab] = DEFAULT_COLORS[next_color % len(DEFAULT_COLORS)]
                next_color += 1

        # build mask rgb
        mask_rgb = np.zeros((H, W, 3), dtype=np.float32)
        for lab, col in colors.items():
            mask_rgb[mask == lab] = col

        overlay = (1 - alpha) * img + alpha * mask_rgb
        pad = np.ones((H, 8, 3), dtype=np.float32)
        concat = np.concatenate([img, pad, mask_rgb, pad, overlay], axis=1)

        plt.figure(figsize=(6, 3))
        plt.imshow(concat)
        plt.axis("off")
        plt.title(f"{fname}: [image] [mask] [overlay]")

        patches = [mpatches.Patch(color=colors[lab], label=str(lab)) for lab in labels if lab != 0]
        if patches:
            plt.legend(handles=patches, title="mask labels", bbox_to_anchor=(1.01, 0.5),
                       loc="center left", borderaxespad=0)
        plt.show()


plot_from_folders(
    image_folder="./original_images/",
    mask_folder="./original_masks/",
    n=3,     # how many to show
    alpha=0.5
)
