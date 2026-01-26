from constants import ALL_MODELS
from functions import InferenceSaver


if __name__ == "__main__":
    # Configuration
    conf = 0.0  # Confidence threshold
    iou = 0.5  # IoU threshold for NMS

    # Create batch inference saver
    for model_name in ALL_MODELS:
        print(f"{'=' * 60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'=' * 60}")
        saver = InferenceSaver(model_name, conf=conf, iou=iou)
        saver.load_model()
        saver.save_all_inferences()
