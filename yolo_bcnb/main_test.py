import numpy as np
import torch

from constants import dataset_yaml_path, dataset_SEG_yaml_path
from functions import evaluate_model_box, load_model_test, evaluate_model_box_mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL = "yolo12"
    model = load_model_test(MODEL)
    print(model.names)

    ### Evaluate the model
    if MODEL != "yoloe":
        metrics = evaluate_model_box(model, dataset_yaml_path, device)

        # Print metrics in the same format as the UNet evaluation
        print("Evaluation Metrics:")
        print(f"mAP@50:         {metrics["Box mAP@50"   ]:.3f}")
        print(f"mAP@50-95:      {metrics["Box mAP@50-95"]:.3f}")
        print(f"Mean F1:        {metrics["Box F1"]:.3f}")
        print(f"Mean Precision: {metrics["Box Precision"  ]:.3f}")
        print(f"Mean Recall:    {metrics["Box Recall"     ]:.3f}")

    else:
        metrics = evaluate_model_box_mask(model, dataset_SEG_yaml_path, device)

        # Print metrics in the same format as the UNet evaluation
        print("Evaluation Metrics:")
        print(f"Box mAP@50:         {metrics["Box mAP@50"]:.3f}")
        print(f"Box mAP@50-95:      {metrics["Box mAP@50-95"]:.3f}")
        print(f"Box Mean F1:        {np.mean(np.array(metrics["Box F1"])):.3f}")
        print(f"Box Mean Precision:       {metrics["Box Precision"]:.3f}")
        print(f"Box Mean Recall:        {metrics["Box Recall"]:.3f}")
        print(f"Seg mAP@50:      {metrics["Mask mAP@50"]:.3f}")
        print(f"Seg mAP@50-95:   {metrics["Mask mAP@50-95"]:.3f}")
        print(f"Seg Mean F1:        {np.mean(np.array(metrics["Mask F1"])):.3f}")
        print(f"Seg Mean Precision:  {metrics["Mask Precision"]:.3f}")
        print(f"Seg Mean Recall: {metrics["Mask Recall"]:.3f}")