from ultralytics import YOLO, RTDETR, YOLOE, YOLOWorld
import numpy as np

from constants import (yolo12_model_config, yolo12_model_path, yolo12_model_name,
                       detr_model_name, detr_model_path, detr_model_config, yolo8_model_config, yolo8_model_name, yolo9_model_config, yolo10_model_config, yolo11_model_config, yolo11_model_name, yolo10_model_name, yolo9_model_name, yolo9_model_path,
                       yolo8_model_path, yolo10_model_path, yolo11_model_path, yoloe_model_config, yoloe_model_name, yoloe_model_path, yoloworld_model_config, yoloworld_model_name, yoloworld_model_path)



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