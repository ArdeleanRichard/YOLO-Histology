from ultralytics import YOLO, RTDETR, YOLOE, YOLOWorld
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class YOLOtoSegmentationDataset(Dataset):
    def __init__(self, base_dir, image_size=512, split="train"):
        self.base_dir = base_dir
        self.image_size = image_size
        self.split = split

        self.img_dir = os.path.join(base_dir, "images", split)
        self.label_dir = os.path.join(base_dir, "labels", split)

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def yolo_to_mask(self, labels, img_width, img_height):
        # Create an empty mask
        mask = np.zeros((img_height, img_width), dtype=np.float32)

        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:  # class_id x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                # Calculate box coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)

                # Fill the box in the mask
                mask[y1:y2, x1:x2] = 1.0  # Binary mask for all objects

        return mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Read YOLO labels
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            mask = self.yolo_to_mask(labels, orig_width, orig_height)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            # Create empty mask if no label file exists
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Convert to tensors
        image = image.transpose(2, 0, 1) / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension

        return torch.FloatTensor(image), torch.FloatTensor(mask)


def evaluate_model(model, dataset_yaml_path, test_loader, device):
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
    nms=0.3
    val_results = model.val(data=dataset_yaml_path, conf=conf, iou=nms, split='test')

    # Extract mAP metrics directly from YOLO validation
    map50 = val_results.box.map50
    map50_95 = val_results.box.map

    # Compute pixel-wise metrics
    iou_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for batch in tqdm(test_loader, desc="Computing pixel metrics"):
        images, labels = batch
        images = images.to(device)

        results = model.predict(images, conf=conf, iou=nms)

        for i, result in enumerate(results):
            gt_mask = labels[i]
            gt_mask = gt_mask.to(device)

            img_h, img_w = images.shape[2], images.shape[3]

            # Create prediction mask
            pred_mask = torch.zeros((img_h, img_w), device=device)
            if len(result.boxes) > 0:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.int().cpu().numpy()
                    pred_mask[y1:y2, x1:x2] = 1

            pred_mask = pred_mask.to(device)

            # Calculate IoU
            intersection = (gt_mask * pred_mask).sum().float()
            union = ((gt_mask + pred_mask) > 0).sum().float()
            iou = (intersection / (union + 1e-6)).cpu().numpy()
            iou_scores.append(iou)

            # Calculate precision, recall, F1
            true_positive = intersection
            false_positive = pred_mask.sum() - intersection
            false_negative = gt_mask.sum() - intersection

            precision = (true_positive / (true_positive + false_positive + 1e-6)).cpu().numpy()
            recall = (true_positive / (true_positive + false_negative + 1e-6)).cpu().numpy()
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

    return {
        "mAP@50": map50,
        "mAP@50-95": map50_95,
        "Mean IoU": np.mean(iou_scores),
        "Mean F1": np.mean(f1_scores),
        "Mean Precision": np.mean(precision_scores),
        "Mean Recall": np.mean(recall_scores)
    }



def evaluate_model_box_mask(model, dataset_yaml_path, test_loader, device):
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
    nms=0.3
    val_results = model.val(data=dataset_yaml_path, conf=conf, iou=nms, split='test')

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
    val_results = model.val(data=dataset_yaml_path, conf=conf, iou=nms, split='test')

    return {
        "Box mAP@50":           val_results.box.map50,
        "Box mAP@50-95":        val_results.box.map,
        "Box Precision":        val_results.box.mp,
        "Box Recall":           val_results.box.mr,
        "Box F1":               np.mean(np.array(val_results.box.f1)),
    }


def visualize_predictions(model, test_loader, device, num_samples=5, output_dir="yolo_predictions"):
    """Visualize model predictions with bounding boxes"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.to(device)

    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        images = images.to(device)
        results = model.predict(images)

        for j, result in enumerate(results):
            plt.figure(figsize=(12, 10))

            # Plot original image with predictions
            plt.imshow(result.plot())
            plt.title(f"YOLO Predictions - Sample {i + 1}")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pred_sample_{i}_{j}.png"))
            plt.close()

    print(f"Predictions saved to {output_dir}")


def visualize_predictions_with_ground_truth(model, test_loader, device, num_samples=5, output_dir="yolo_gt_vs_pred"):
    """
    Visualize ground truth masks/bounding boxes alongside model predictions.

    Args:
        model: Trained YOLO model
        test_loader: DataLoader for test dataset
        device: Device to run the model on
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.to(device)

    for i, (images, masks) in enumerate(test_loader):
        if i >= num_samples:
            break

        images = images.to(device)
        results = model.predict(images)

        for j, (image, mask, result) in enumerate(zip(images, masks, results)):
            # Convert tensors to numpy arrays for visualization
            image_np = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWH
            mask_np = mask.cpu().numpy()[0]  # Remove channel dimension

            # Create the figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 9))

            # Plot original image
            axes[0].imshow(image_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            # Plot ground truth mask overlay
            axes[1].imshow(image_np)
            axes[1].imshow(mask_np, alpha=0.5, cmap='jet')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')

            # Also draw ground truth bounding boxes
            # Extract bounding boxes from the mask
            gt_contours, _ = cv2.findContours(
                (mask_np * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in gt_contours:
                x, y, w, h = cv2.boundingRect(contour)
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
                axes[1].add_patch(rect)

            # Plot YOLO predictions
            axes[2].imshow(result.plot())
            axes[2].set_title("YOLO Predictions")
            axes[2].axis('off')

            # Create prediction mask for visualization
            pred_mask = np.zeros_like(mask_np)
            if len(result.boxes) > 0:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.int().cpu().numpy()
                    # Draw filled box on the prediction mask
                    pred_mask[y1:y2, x1:x2] = 1

            # Overlay prediction mask
            pred_overlay = axes[2].imshow(pred_mask, alpha=0.3, cmap='plasma')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{i}_{j}_comparison.png"))
            plt.close()

    print(f"Visualizations saved to {output_dir}")



from constants import (yolo12_model_config, yolo12_model_path, yolo12_model_name,
                       detr_model_name, detr_model_path, detr_model_config, yolo8_model_config, yolo8_model_name, yolo9_model_config, yolo10_model_config, yolo11_model_config, yolo11_model_name, yolo10_model_name, yolo9_model_name, yolo9_model_path,
                       yolo8_model_path, yolo10_model_path, yolo11_model_path, yoloe_model_config, yoloe_model_name, yoloe_model_path, yoloworld_model_config, yoloworld_model_name, yoloworld_model_path)


def load_model_train(model_name):
    if model_name == "rtdetr":
        return RTDETR(detr_model_config), detr_model_name, detr_model_path
    if model_name == "yolo8":
        return YOLO(yolo8_model_config), yolo8_model_name, yolo8_model_path
    if model_name == "yolo9":
        return YOLO(yolo9_model_config), yolo9_model_name, yolo9_model_path
    if model_name == "yolo10":
        return YOLO(yolo10_model_config), yolo10_model_name, yolo10_model_path
    if model_name == "yolo12":
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
    if model_name == "yolo12":
        return YOLO(yolo11_model_path)
    if model_name == "yolo12":
        return YOLO(yolo12_model_path)
    if model_name == "yoloe":
        return YOLOE(yoloe_model_path)
    if model_name == "yolow":
        return YOLOWorld(yoloworld_model_path)