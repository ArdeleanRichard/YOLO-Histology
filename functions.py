import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import math

from ultralytics import YOLO, RTDETR, YOLOE, YOLOWorld
from constants import (yolo12_model_config, yolo12_model_path, yolo12_model_name,
                       detr_model_name, detr_model_path, detr_model_config,
                       yolo8_model_config, yolo8_model_name, yolo8_model_path,
                       yolo9_model_config, yolo9_model_name, yolo9_model_path,
                       yolo10_model_path, yolo10_model_name, yolo10_model_config,
                       yolo11_model_config, yolo11_model_name, yolo11_model_path,
                       yoloe_model_config, yoloe_model_name, yoloe_model_path,
                       yoloworld_model_config, yoloworld_model_name, yoloworld_model_path,
                       result_root, fig_root, dataset_yaml_path, dataset_SEG_yaml_path, image_folder, label_folder)



def load_model_train(model_name):
    if model_name == "rtdetr":
        return RTDETR(detr_model_config), detr_model_name, detr_model_path
    elif model_name == "yolo8":
        return YOLO(yolo8_model_config), yolo8_model_name, yolo8_model_path
    elif model_name == "yolo9":
        return YOLO(yolo9_model_config), yolo9_model_name, yolo9_model_path
    elif model_name == "yolo10":
        return YOLO(yolo10_model_config), yolo10_model_name, yolo10_model_path
    elif model_name == "yolo11":
        return YOLO(yolo11_model_config), yolo11_model_name, yolo11_model_path
    elif model_name == "yolo12":
        return YOLO(yolo12_model_config), yolo12_model_name, yolo12_model_path
    elif model_name == "yoloe":
        return YOLOE(yoloe_model_config), yoloe_model_name, yoloe_model_path
    elif model_name == "yolow":
        return YOLOWorld(yoloworld_model_config), yoloworld_model_name, yoloworld_model_path
    else:
        raise Exception("Model missing")


def load_model_test(model_name):
    if model_name == "rtdetr":
        return RTDETR(detr_model_path)
    elif model_name == "yolo8":
        return YOLO(yolo8_model_path)
    elif model_name == "yolo9":
        return YOLO(yolo9_model_path)
    elif model_name == "yolo10":
        return YOLO(yolo10_model_path)
    elif model_name == "yolo11":
        return YOLO(yolo11_model_path)
    elif model_name == "yolo12":
        return YOLO(yolo12_model_path)
    elif model_name == "yoloe":
        return YOLOE(yoloe_model_path)
    elif model_name == "yolow":
        return YOLOWorld(yoloworld_model_path)
    else:
        raise Exception("Model missing")


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        if self.model_name == "rtdetr":
            self.model = RTDETR(detr_model_path)
        elif self.model_name == "yolo8":
            self.model = YOLO(yolo8_model_path)
        elif self.model_name == "yolo9":
            self.model = YOLO(yolo9_model_path)
        elif self.model_name == "yolo10":
            self.model = YOLO(yolo10_model_path)
        elif self.model_name == "yolo11":
            self.model = YOLO(yolo11_model_path)
        elif self.model_name == "yolo12":
            self.model = YOLO(yolo12_model_path)
        elif self.model_name == "yoloe":
            self.model = YOLOE(yoloe_model_path)
        elif self.model_name == "yolow":
            self.model = YOLOWorld(yoloworld_model_path)
        else:
            raise Exception("Model missing")

        print(f"Model: {self.model_name}")
        print(f"Model classes: {self.model.names}")


class ModelEvaluator(ModelLoader):
    def __init__(self, MODEL_NAME, DATA, device, conf=0.5, iou=0.5):
        super().__init__(MODEL_NAME)
        self.dataset_name = DATA
        print(f"Dataset: {self.dataset_name}")

        self.model_name = MODEL_NAME
        self.model = None
        self.device = device

        self.conf = conf
        self.iou = iou
        self.metrics = None


    def evaluate(self):
        """
        Evaluate YOLO model using metrics similar to the UNet evaluation.

        Returns:
            Evaluation metrics dictionary
        """
        self.model.to(self.device)

        os.makedirs(f"{result_root}/runs_test/{self.model_name}/", exist_ok=True)

        # Run YOLO validation on the dataset
        val_results = self.model.val(data=dataset_yaml_path if self.model_name != "yoloe" else dataset_SEG_yaml_path,
                                     project=f"{result_root}/runs_test/{self.model_name}/",
                                     conf=self.conf, iou=self.iou, split='test')

        self.metrics = {
            "Box mAP@50":           val_results.box.map50,
            "Box mAP@50-95":        val_results.box.map,
            "Box Precision":        val_results.box.mp,
            "Box Recall":           val_results.box.mr,
            "Box F1":               np.mean(np.array(val_results.box.f1)),

            "Mask mAP@50":          None if self.model_name != "yoloe" else val_results.seg.map50,
            "Mask mAP@50-95":       None if self.model_name != "yoloe" else val_results.seg.map,
            "Mask Precision":       None if self.model_name != "yoloe" else val_results.seg.mp,
            "Mask Recall":          None if self.model_name != "yoloe" else val_results.seg.mr,
            "Mask F1":              None if self.model_name != "yoloe" else np.mean(np.array(val_results.seg.f1)),
        }

    def save_results_to_csv(self, csv_path):
        """
        Save evaluation results to CSV file using pandas

        Args:
            csv_path (str): Path to the CSV file
        """
        # Create a results dictionary with common fields
        results = {
            'dataset': self.dataset_name,
            'model': self.model_name,
        }

        # YOLO-E model metrics (includes segmentation)
        results.update({
            'box_mAP@50':               self.metrics["Box mAP@50"],
            'box_mAP@50-95':            self.metrics["Box mAP@50-95"],
            'box_mean_f1':              self.metrics["Box F1"],
            'box_mean_precision':       self.metrics["Box Precision"],
            'box_mean_recall':          self.metrics["Box Recall"],

            # Set segmentation metrics to None for non-segmentation models
            'seg_mAP@50':               None if self.model_name != "yoloe" else self.metrics["Mask mAP@50"],
            'seg_mAP@50-95':            None if self.model_name != "yoloe" else self.metrics["Mask mAP@50-95"],
            'seg_mean_f1':              None if self.model_name != "yoloe" else self.metrics["Mask F1"],
            'seg_mean_precision':       None if self.model_name != "yoloe" else self.metrics["Mask Precision"],
            'seg_mean_recall':          None if self.model_name != "yoloe" else self.metrics["Mask Recall"],
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

    def print_metrics(self):
        """
        Print metrics in a formatted way

        """
        print("Evaluation Metrics:")

        if self.model_name != "yoloe":
            print(f"mAP@50:         {self.metrics['Box mAP@50']:.3f}")
            print(f"mAP@50-95:      {self.metrics['Box mAP@50-95']:.3f}")
            print(f"Mean F1:        {self.metrics['Box F1']:.3f}")
            print(f"Mean Precision: {self.metrics['Box Precision']:.3f}")
            print(f"Mean Recall:    {self.metrics['Box Recall']:.3f}")
        else:
            print(f"Box mAP@50:         {self.metrics['Box mAP@50']:.3f}")
            print(f"Box mAP@50-95:      {self.metrics['Box mAP@50-95']:.3f}")
            print(f"Box Mean F1:        {self.metrics['Box F1']:.3f}")
            print(f"Box Mean Precision: {self.metrics['Box Precision']:.3f}")
            print(f"Box Mean Recall:    {self.metrics['Box Recall']:.3f}")
            print(f"Seg mAP@50:         {self.metrics['Mask mAP@50']:.3f}")
            print(f"Seg mAP@50-95:      {self.metrics['Mask mAP@50-95']:.3f}")
            print(f"Seg Mean F1:        {self.metrics['Mask F1']:.3f}")
            print(f"Seg Mean Precision: {self.metrics['Mask Precision']:.3f}")
            print(f"Seg Mean Recall:    {self.metrics['Mask Recall']:.3f}")


class PlotHelper:
    def __init__(self):
        # === COLOR MAP (tab10 with proper BGR tuples) ===
        self.colors = [tuple(map(int, np.array(c[:3])[::-1] * 255)) for c in plt.get_cmap("tab10").colors]
        # print(colors)
        # print(len(colors))

    def get_file_names(self, folder_path, n):
        """
        Returns up to n file names from the given folder.
        """
        # List all files in the folder (ignores subdirectories)
        all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Return only the first n files
        return all_files[:n]

    def load_yolo_labels(self, label_path):
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                boxes.append((int(cls), x, y, w, h))
        return boxes

    def draw_boxes(self, img, labels, color_map, class_names):
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




class ModelPlotter(PlotHelper, ModelLoader):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.model_name = MODEL_NAME

        self.combined_images = None


    def prepare_images(self, nr_images=4):
        images = self.get_file_names(image_folder, nr_images)  # Get 4 images

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
            gt_boxes = self.load_yolo_labels(label_path)
            img_gt = self.draw_boxes(img_gt, gt_boxes, self.colors, class_names=self.model.names)
            cv2.putText(img_gt, "Ground Truth", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Prediction
            results = self.model(img_path)[0]
            pred_boxes = []
            for cls, xywh in zip(results.boxes.cls, results.boxes.xywh):
                cls = int(cls.item())
                x, y, w, h = xywh.tolist()
                pred_boxes.append((cls, x / results.orig_shape[1], y / results.orig_shape[0],
                                   w / results.orig_shape[1], h / results.orig_shape[0]))
            img_pred = self.draw_boxes(img_pred, pred_boxes, self.colors, class_names=self.model.names)
            cv2.putText(img_pred, "Prediction", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Combine horizontally: [GT | Prediction]
            combined = np.hstack((img_gt, img_pred))
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            combined_images.append(combined_rgb)

        self.combined_images = combined_images


    def plot(self):
        num_images = 4  # Total images to show
        cols = 2  # Number of columns in final plot

        rows = math.ceil(len(self.combined_images) / cols)

        # Create grid by rows
        grid_rows = []
        for i in range(rows):
            row_imgs = self.combined_images[i * cols:(i + 1) * cols]

            # If row is not complete, add empty
            while len(row_imgs) < cols:
                h, w, _ = row_imgs[0].shape
                empty_img = np.zeros((h, w, 3), dtype=np.uint8)
                row_imgs.append(empty_img)

            row = np.hstack(row_imgs)
            grid_rows.append(row)

        # Vertical stack all rows
        final_image = np.vstack(grid_rows)

        plt.figure(figsize=(16, num_images * 5))
        plt.imshow(final_image)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"{fig_root}/plot_{self.model_name}.png", bbox_inches="tight", pad_inches=0)
        # plt.show()
        plt.close()


class ResultPlotter(PlotHelper, ModelLoader):
    def __init__(self, model_names):
        super().__init__()
        self.model_names = model_names
        self.model = None

        self.combined_images = None
        self.image_filename = None

    def prepare_single_image_all_models(self, image_index=0):
        """
        Prepare a single image with ground truth and predictions from all models

        Args:
            image_index: Index of image to use (default: 0 for first image)

        Returns:
            List of combined images (GT + Prediction) for each model
        """
        # Get image files and select one
        images = self.get_file_names(image_folder, 10)  # Get more files to have options
        if image_index >= len(images):
            image_index = 0  # Fallback to first image

        img_file = images[image_index]
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")


        # Load and prepare base image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))

        # Load ground truth once
        gt_boxes = self.load_yolo_labels(label_path)

        combined_images = []
        for model_name in self.model_names:
            # Load model
            self.model_name = model_name
            self.load_model()

            class_names = self.model.names

            # Prepare copies for this model
            img_gt = img.copy()
            img_pred = img.copy()

            # Ground truth (same for all models)
            img_gt = self.draw_boxes(img_gt, gt_boxes, self.colors, class_names)
            cv2.putText(img_gt, "Ground Truth", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Prediction for this model
            results = self.model(img_path)[0]
            pred_boxes = []
            for cls, xywh in zip(results.boxes.cls, results.boxes.xywh):
                cls = int(cls.item())
                x, y, w, h = xywh.tolist()
                pred_boxes.append((cls, x / results.orig_shape[1], y / results.orig_shape[0],
                                   w / results.orig_shape[1], h / results.orig_shape[0]))

            img_pred = self.draw_boxes(img_pred, pred_boxes, self.colors, class_names)
            cv2.putText(img_pred, f"{model_name.upper()}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Combine horizontally: [GT | Prediction]
            combined = np.hstack((img_gt, img_pred))
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            combined_images.append(combined_rgb)

            # print(f"Processed {model_name}")

        self.combined_images = combined_images
        self.image_filename = img_file


    def plot_all_models_single_image(self):
        """
        Plot all 8 models' inferences on a single image in one figure with separating lines

        """
        num_models = len(self.combined_images)  # Should be 8
        cols = 2  # Number of columns in final plot
        rows = math.ceil(num_models / cols)  # Should be 4 rows

        # Get dimensions
        h, w, _ = self.combined_images[0].shape
        line_thickness = 4  # Thickness of separator lines

        # Create grid rows with horizontal separators
        grid_rows = []
        for i in range(rows):
            row_imgs = self.combined_images[i * cols:(i + 1) * cols]

            # If row is not complete, add empty images
            while len(row_imgs) < cols:
                empty_img = np.zeros((h, w, 3), dtype=np.uint8)
                row_imgs.append(empty_img)

            # Add vertical separator between images in the row
            row_with_separators = []
            for j, img in enumerate(row_imgs):
                row_with_separators.append(img)
                # Add vertical separator (except for last image in row)
                if j < len(row_imgs) - 1:
                    vertical_separator = np.ones((h, line_thickness, 3), dtype=np.uint8) * 255  # White line
                    row_with_separators.append(vertical_separator)

            row = np.hstack(row_with_separators)
            grid_rows.append(row)

            # Add horizontal separator (except for last row)
            if i < rows - 1:
                horizontal_separator = np.ones((line_thickness, row.shape[1], 3), dtype=np.uint8) * 255  # White line
                grid_rows.append(horizontal_separator)

        # Stack all rows vertically
        final_image = np.vstack(grid_rows)

        plt.figure(figsize=(16, num_models * 2.5))  # Adjust height based on number of models
        plt.imshow(final_image)
        plt.axis("off")
        plt.title(f"All Models Inference Comparison - {self.image_filename}", fontsize=16, pad=20)
        plt.tight_layout(pad=0)

        # Save with descriptive filename
        safe_filename = os.path.splitext(self.image_filename)[0]
        plt.savefig(f"{fig_root}/visualization_{safe_filename}.png",
                    bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()
        print(f"Saved comparison plot to {fig_root}/visualization_{safe_filename}.png")


    def create_all_models_comparison(self, image_index=0):
        """
        Create a comparison plot showing all 8 models' inferences on a single image

        Args:
            image_index: Which image to use (default: 0 for first image)
        """
        self.prepare_single_image_all_models(image_index)
        self.plot_all_models_single_image()


