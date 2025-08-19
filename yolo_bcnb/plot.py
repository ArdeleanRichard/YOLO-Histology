import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from yolo_bcnb.functions import load_model_test


def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append((int(cls), x, y, w, h))
    return boxes

def draw_boxes(img, labels, color_map, class_names):
    h, w = img.shape[:2]
    for cls, x, y, bw, bh in labels:
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        color = color_map[(cls+1) % len(color_map)]
        # print(cls, cls+1, len(color_map), color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
        # cv2.putText(img, class_names[cls], (x1, max(20, y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)
    return img




def prepare_images():
    combined_images = []

    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(gt_label_folder, os.path.splitext(img_file)[0] + ".txt")

        # Load image and prepare copies
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        img_gt = img.copy()
        img_pred = img.copy()

        # Ground truth
        gt_boxes = load_yolo_labels(label_path)
        img_gt = draw_boxes(img_gt, gt_boxes, colors, class_names)
        cv2.putText(img_gt, "Ground Truth", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Prediction
        results = model(img_path)[0]
        pred_boxes = []
        for cls, xywh in zip(results.boxes.cls, results.boxes.xywh):
            cls = int(cls.item())
            x, y, w, h = xywh.tolist()
            pred_boxes.append((cls, x / results.orig_shape[1], y / results.orig_shape[0],
                               w / results.orig_shape[1], h / results.orig_shape[0]))
        img_pred = draw_boxes(img_pred, pred_boxes, colors, class_names)
        cv2.putText(img_pred, "Prediction", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Combine horizontally: [GT | Prediction]
        combined = np.hstack((img_gt, img_pred))
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        combined_images.append(combined_rgb)

    return combined_images

def plot(combined_images):
    rows = math.ceil(len(combined_images) / cols)

    # Creează grid-ul rând cu rând
    grid_rows = []
    for i in range(rows):
        row_imgs = combined_images[i * cols:(i + 1) * cols]

        # Dacă rândul nu e complet, adaugă imagini goale
        while len(row_imgs) < cols:
            h, w, _ = row_imgs[0].shape
            empty_img = np.zeros((h, w, 3), dtype=np.uint8)
            row_imgs.append(empty_img)

        row = np.hstack(row_imgs)
        grid_rows.append(row)

    # Stack pe verticală toate rândurile
    final_image = np.vstack(grid_rows)

    plt.figure(figsize=(16, num_images * 5))
    plt.imshow(final_image)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"plot_{MODEL}.png", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    # === CONFIGURATION ===
    MODEL = "yolo12"
    image_folder = "../data/BCNB/images/test/"  # Folder with images
    gt_label_folder = "../data/BCNB/labels/test/"  # Folder with ground truth YOLO labels (txt)
    num_images = 4  # Total images to show
    img_ext = ('.jpg', '.png', '.jpeg')  # Image file extensions
    cols = 2  # Number of columns in final plot

    # === IMAGES ===
    sel_imgs = [
        "6.jpg",
        "9.jpg",
        "11.jpg",
        "21.jpg",
    ]
    images = sel_imgs[-4:]

    # === LOAD MODEL ===
    model = load_model_test(MODEL)
    class_names = model.names
    num_classes = len(class_names)

    # === COLOR MAP (tab10 with proper BGR tuples) ===
    colors = [tuple(map(int, np.array(c[:3])[::-1] * 255)) for c in plt.get_cmap("tab10").colors]
    # print(colors)
    # print(len(colors))

    combined_images = prepare_images()
    plot(combined_images)