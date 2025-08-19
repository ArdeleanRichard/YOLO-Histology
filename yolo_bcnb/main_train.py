from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

from constants import dataset_yaml_path, dataset_SEG_yaml_path
from yolo_bcnb.functions import load_model_train

if __name__ == '__main__':
    # If needed for frozen executables, uncomment the next line:
    # from multiprocessing import freeze_support; freeze_support()

    # Step 2. Initialize the YOLOv12 model.
    # Option 1: Start from scratch with config
    # model = YOLO(model_config)

    # Option 2: Start from pretrained weights (recommended)
    # Use a pre-trained YOLOv8 model (or YOLOv12 if available)

    MODEL = "yolo12"
    model, model_name, model_path = load_model_train(MODEL)
    print(model.names)

    # Step 3. Train the model with enhanced parameters and augmentation
    results = model.train(
        data=dataset_yaml_path if MODEL != "yoloe" else dataset_SEG_yaml_path,  # dataset YAML config
        epochs=1,               # number of epochs
        imgsz=640,              # training image size
        batch=16,               # adjust according to your GPU memory
        optimizer="AdamW",      # try different optimizers
        lr0=0.001,              # initial learning rate
        lrf=0.01,               # final learning rate as a fraction of initial lr
        weight_decay=0.0005 if MODEL != "yoloe" else 0.025,  # weight decay
        momentum=0.937,         # SGD momentum
        warmup_epochs=0,        # warmup epochs
        plots=True,
        cos_lr=False,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        trainer=None if MODEL != "yoloe" else YOLOEPESegTrainer,


        # Augmentation settings
        hsv_h=0.015,            # hue augmentation
        hsv_s=0.7,              # saturation augmentation
        hsv_v=0.4,              # value augmentation (brightness)
        degrees=10.0,           # rotation (+/- deg)
        translate=0.1,          # translation (+/- fraction)
        scale=0.2,              # scale (+/- gain)
        fliplr=0.5,             # flip left-right probability
        mosaic=1.0,             # mosaic probability
        mixup=0.05,             # mixup probability
        close_mosaic=10,

        # Early stopping patience (if needed)
        patience=15,            # early stopping patience (epochs)

        # Save best model during training
        save_period=10,         # save checkpoint every x epochs
        project="runs",         # project name
        name=model_name,        # experiment name
    )
    print(model.names)

    print("Training completed.")
    # Save the current state of the model to a file.
    model.save(model_path)

    # evaluation_results = model.val(
    #     data=dataset_yaml_path if MODEL != "yoloe" else dataset_SEG_yaml_path,
    #     imgsz=640,
    #     project="runs",
    #     name=model_name,
    # )
    #
    # print(evaluation_results)
