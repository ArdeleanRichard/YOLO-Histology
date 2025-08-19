import os

MODEL = "yolo12"
DATA = "BCNB"
# DATA = "nuclei"

data_root = f"./data/{DATA}/"
result_root = f"./results_data_{DATA}/"
model_root = f"{result_root}/saved_models/"
fig_root = f"{result_root}/figs/"
os.makedirs(result_root, exist_ok=True)
os.makedirs(model_root, exist_ok=True)
os.makedirs(fig_root, exist_ok=True)

# Step 1. Define your dataset YAML config.
dataset_yaml_path = data_root + 'data.yaml'
dataset_SEG_yaml_path = data_root + 'data_seg.yaml'

# Instantiate the model.
yolo8_model_name = "yolo12s"
yolo8_model_config = f"{yolo8_model_name}.pt"
yolo8_model_path = model_root + yolo8_model_name + ".pt"

yolo9_model_name = "yolo12s"
yolo9_model_config = f"{yolo9_model_name}.pt"
yolo9_model_path = model_root + yolo9_model_name + ".pt"

yolo10_model_name = "yolo12s"
yolo10_model_config = f"{yolo10_model_name}.pt"
yolo10_model_path = model_root + yolo10_model_name + ".pt"

yolo11_model_name = "yolo12s"
yolo11_model_config = f"{yolo11_model_name}.pt"
yolo11_model_path = model_root + yolo11_model_name + ".pt"

yolo12_model_name = "yolo12s"
yolo12_model_config = f"{yolo12_model_name}.pt"
yolo12_model_path = model_root + yolo12_model_name + ".pt"


detr_model_name = "rtdetr-l"
detr_model_config = f"{detr_model_name}.pt"
detr_model_path = model_root + detr_model_name + ".pt"



yoloe_model_name = "yoloe-11s-seg"
yoloe_model_config = f"{yoloe_model_name}.pt"
yoloe_model_path = model_root + yoloe_model_name + ".pt"



yoloworld_model_name = "yolov8s-worldv2"
yoloworld_model_config = f"{yoloworld_model_name}.pt"
yoloworld_model_path = model_root + yoloworld_model_name + ".pt"