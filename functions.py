from ultralytics import YOLO, RTDETR, YOLOE, YOLOWorld
import numpy as np
import pandas as pd
import os
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from constants import (yolo12_model_config, yolo12_model_path, yolo12_model_name,
                       detr_model_name, detr_model_path, detr_model_config,
                       yolo8_model_config, yolo8_model_name, yolo8_model_path,
                       yolo9_model_config, yolo9_model_name, yolo9_model_path,
                       yolo10_model_path, yolo10_model_name, yolo10_model_config,
                       yolo11_model_config, yolo11_model_name, yolo11_model_path,
                       yoloe_model_config, yoloe_model_name, yoloe_model_path,
                       yoloworld_model_config, yoloworld_model_name, yoloworld_model_path,
                       result_root, fig_root, data_root)



def load_model_train(model_name):
    if model_name == "rtdetr":
        return RTDETR(detr_model_config), detr_model_name, detr_model_path
    if model_name == "yolo8":
        return YOLO(yolo8_model_config), yolo8_model_name, yolo8_model_path
    if model_name == "yolo9":
        return YOLO(yolo9_model_config), yolo9_model_name, yolo9_model_path
    if model_name == "yolo10":
        return YOLO(yolo10_model_config), yolo10_model_name, yolo10_model_path
    if model_name == "yolo11":
        return YOLO(yolo11_model_config), yolo11_model_name, yolo11_model_path
    if model_name == "yolo12":
        return YOLO(yolo12_model_config), yolo12_model_name, yolo12_model_path
    if model_name == "yoloe":
        return YOLOE(yoloe_model_config), yoloe_model_name, yoloe_model_path
    if model_name == "yolow":
        return YOLOWorld(yoloworld_model_config), yoloworld_model_name, yoloworld_model_path

def load_model_test(model_name):
    if model_name == "rtdetr":
        return RTDETR(detr_model_path)
    if model_name == "yolo8":
        return YOLO(yolo8_model_path)
    if model_name == "yolo9":
        return YOLO(yolo9_model_path)
    if model_name == "yolo10":
        return YOLO(yolo10_model_path)
    if model_name == "yolo11":
        return YOLO(yolo11_model_path)
    if model_name == "yolo12":
        return YOLO(yolo12_model_path)
    if model_name == "yoloe":
        return YOLOE(yoloe_model_path)
    if model_name == "yolow":
        return YOLOWorld(yoloworld_model_path)

def evaluate_model_box_mask(model, dataset_yaml_path, device):
    """
    Evaluate YOLO model using metrics similar to the UNet evaluation.

    Args:
        model: Trained YOLO model
        device: Device to run the model on

    Returns:
        Evaluation metrics dictionary
    """
    model.to(device)

    # Run YOLO validation on the dataset
    conf=0.25
    nms=0.3
    val_results = model.val(data=dataset_yaml_path,
                            project=f"{result_root}/runs_test/",
                            conf=conf, iou=nms, split='test')

    return {
        "Box mAP@50":           val_results.box.map50,
        "Box mAP@50-95":        val_results.box.map,
        "Box Precision":        val_results.box.mp,
        "Box Recall":           val_results.box.mr,
        "Box F1":               np.mean(np.array(val_results.box.f1)),

        "Mask mAP@50":          val_results.seg.map50,
        "Mask mAP@50-95":       val_results.seg.map,
        "Mask Precision":       val_results.seg.mp,
        "Mask Recall":          val_results.seg.mr,
        "Mask F1":              np.mean(np.array(val_results.seg.f1)),
    }

def evaluate_model_box(model, dataset_yaml_path, device):
    """
    Evaluate YOLO model using metrics similar to the UNet evaluation.

    Args:
        model: Trained YOLO model
        test_loader: DataLoader for test dataset
        device: Device to run the model on

    Returns:
        Evaluation metrics dictionary
    """
    model.to(device)

    # Run YOLO validation on the dataset
    conf=0.25
    nms=0.5
    val_results = model.val(data=dataset_yaml_path,
                            project=f"{result_root}/runs_test/",
                            conf=conf, iou=nms, split='test')

    return {
        "Box mAP@50":           val_results.box.map50,
        "Box mAP@50-95":        val_results.box.map,
        "Box Precision":        val_results.box.mp,
        "Box Recall":           val_results.box.mr,
        "Box F1":               np.mean(np.array(val_results.box.f1)),
    }

def save_results_to_csv(metrics, model_name, dataset_name, csv_path):
    """
    Save evaluation results to CSV file using pandas

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        model_name (str): Name of the model being evaluated
        dataset_name (str): Name of the dataset used
        csv_path (str): Path to the CSV file
    """
    # Create a results dictionary with common fields
    results = {
        'dataset': dataset_name,
        'model': model_name,
    }

    # Add metrics based on model type
    if model_name != "yoloe":
        # Standard YOLO model metrics
        results.update({
            'box_mean_precision': metrics["Box Precision"],
            'box_mean_recall': metrics["Box Recall"],
            'box_mean_f1': metrics["Box F1"],
            'box_mAP@50': metrics["Box mAP@50"],
            'box_mAP@50-95': metrics["Box mAP@50-95"],

            # Set segmentation metrics to None for non-segmentation models
            'seg_mean_precision': None,
            'seg_mean_recall': None,
            'seg_mean_f1': None,
            'seg_mAP@50': None,
            'seg_mAP@50-95': None,
        })
    else:
        # YOLO-E model metrics (includes segmentation)
        results.update({
            'box_mean_precision': metrics["Box Precision"],
            'box_mean_recall': metrics["Box Recall"],
            'box_mean_f1': metrics["Box F1"],
            'box_mAP@50': metrics["Box mAP@50"],
            'box_mAP@50-95': metrics["Box mAP@50-95"],

            'seg_mean_precision': metrics["Mask Precision"],
            'seg_mean_recall': metrics["Mask Recall"],
            'seg_mean_f1': np.mean(np.array(metrics["Mask F1"])),
            'seg_mAP@50': metrics["Mask mAP@50"],
            'seg_mAP@50-95': metrics["Mask mAP@50-95"],
        })

    # Convert to DataFrame
    df_new = pd.DataFrame([results])

    # Check if CSV already exists
    if os.path.exists(csv_path):
        # Load existing data and append new results
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Save to CSV
    df_combined.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def print_metrics(metrics, model_name):
    """
    Print metrics in a formatted way

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        model_name (str): Name of the model being evaluated
    """
    print("Evaluation Metrics:")

    if model_name != "yoloe":
        print(f"mAP@50:         {metrics['Box mAP@50']:.3f}")
        print(f"mAP@50-95:      {metrics['Box mAP@50-95']:.3f}")
        print(f"Mean F1:        {metrics['Box F1']:.3f}")
        print(f"Mean Precision: {metrics['Box Precision']:.3f}")
        print(f"Mean Recall:    {metrics['Box Recall']:.3f}")
    else:
        print(f"Box mAP@50:         {metrics['Box mAP@50']:.3f}")
        print(f"Box mAP@50-95:      {metrics['Box mAP@50-95']:.3f}")
        print(f"Box Mean F1:        {np.mean(np.array(metrics['Box F1'])):.3f}")
        print(f"Box Mean Precision: {metrics['Box Precision']:.3f}")
        print(f"Box Mean Recall:    {metrics['Box Recall']:.3f}")
        print(f"Seg mAP@50:         {metrics['Mask mAP@50']:.3f}")
        print(f"Seg mAP@50-95:      {metrics['Mask mAP@50-95']:.3f}")
        print(f"Seg Mean F1:        {np.mean(np.array(metrics['Mask F1'])):.3f}")
        print(f"Seg Mean Precision: {metrics['Mask Precision']:.3f}")
        print(f"Seg Mean Recall:    {metrics['Mask Recall']:.3f}")


def get_file_names(folder_path, n):
    """
    Returns up to n file names from the given folder.
    """
    # List all files in the folder (ignores subdirectories)
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Return only the first n files
    return all_files[:n]


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
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        color = color_map[(cls + 1) % len(color_map)]
        # print(cls, cls+1, len(color_map), color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
        # cv2.putText(img, class_names[cls], (x1, max(20, y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)
    return img


def prepare_images(image_folder, label_folder, class_names):
    # === COLOR MAP (tab10 with proper BGR tuples) ===
    colors = [tuple(map(int, np.array(c[:3])[::-1] * 255)) for c in plt.get_cmap("tab10").colors]
    # print(colors)
    # print(len(colors))

    combined_images = []

    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")

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
    num_images = 4  # Total images to show
    cols = 2  # Number of columns in final plot

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
    plt.savefig(f"{fig_root}/plot_{MODEL}.png", bbox_inches="tight", pad_inches=0)
    plt.show()