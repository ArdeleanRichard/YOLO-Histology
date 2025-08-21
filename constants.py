import os

# === CONFIGURATION ===

# MODEL OPTIONS: ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yoloe", "yolow"]
MODEL = "yolow"

# DATA OPTIONS: ["BCNB", "nuclei", "TNBC", "MoNuSAC", "CryoNuSeg"]
DATA = "nuclei"

data_root = f"./data/{DATA}/"
result_root = f"./results_data_{DATA}/"
saved_model_root = f"{result_root}/saved_models/"
download_model_root = f"./models/"
fig_root = f"{result_root}/figs/"
os.makedirs(result_root, exist_ok=True)
os.makedirs(saved_model_root, exist_ok=True)
os.makedirs(fig_root, exist_ok=True)

image_folder = f"{data_root}/images/test/"  # Folder with images
label_folder = f"{data_root}/labels/test/"  # Folder with ground truth YOLO labels (txt)

# Define your dataset YAML config.
dataset_yaml_path = data_root + 'data.yaml'
dataset_SEG_yaml_path = data_root + 'data_seg.yaml'


# === MODELS ===
yolo8_model_name = "yolov8s"
yolo8_model_config = f"{download_model_root}/{yolo8_model_name}.pt"
yolo8_model_path = f"{saved_model_root}/{yolo8_model_name}.pt"

yolo9_model_name = "yolov9s"
yolo9_model_config = f"{download_model_root}/{yolo9_model_name}.pt"
yolo9_model_path = f"{saved_model_root}/{yolo9_model_name}.pt"

yolo10_model_name = "yolov10s"
yolo10_model_config = f"{download_model_root}/{yolo10_model_name}.pt"
yolo10_model_path = f"{saved_model_root}/{yolo10_model_name}.pt"

yolo11_model_name = "yolo11s"
yolo11_model_config = f"{download_model_root}/{yolo11_model_name}.pt"
yolo11_model_path = f"{saved_model_root}/{yolo11_model_name}.pt"

yolo12_model_name = "yolo12s"
yolo12_model_config = f"{download_model_root}/{yolo12_model_name}.pt"
yolo12_model_path = f"{saved_model_root}/{yolo12_model_name}.pt"

detr_model_name = "rtdetr-l"
detr_model_config = f"{download_model_root}/{detr_model_name}.pt"
detr_model_path = f"{saved_model_root}/{detr_model_name}.pt"

yoloe_model_name = "yoloe-11s-seg"
yoloe_model_config = f"{download_model_root}/{yoloe_model_name}.pt"
yoloe_model_path = f"{saved_model_root}/{yoloe_model_name}.pt"

yoloworld_model_name = "yolov8s-worldv2"
yoloworld_model_config = f"{download_model_root}/{yoloworld_model_name}.pt"
yoloworld_model_path = f"{saved_model_root}/{yoloworld_model_name}.pt"
