from constants import ALL_MODELS
from functions import BatchInferenceSaver


if __name__ == "__main__":
    # Configuration
    conf = 0.0  # Confidence threshold
    iou = 0.5  # IoU threshold for NMS

    # Create batch inference saver
    batch_saver = BatchInferenceSaver(
        model_names=ALL_MODELS,
        conf=conf,
        iou=iou
    )

    # Save inferences for all models
    batch_saver.save_all_models_inferences()
