from constants import dataset_yaml_path, data_root
import torch
from torch.utils.data import DataLoader

from functions import YOLOtoSegmentationDataset, evaluate_model, evaluate_model_box, load_model_test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL = "yolo12"
    model = load_model_test(MODEL)
    print(model.names)

    ### Evaluate the model
    metrics = evaluate_model_box(model, dataset_yaml_path, device)

    # Print metrics in the same format as the UNet evaluation
    print("Evaluation Metrics:")
    print(f"mAP@50:         {metrics["Box mAP@50"   ]:.3f}")
    print(f"mAP@50-95:      {metrics["Box mAP@50-95"]:.3f}")
    print(f"Mean F1:        {metrics["Box F1"]:.3f}")
    print(f"Mean Precision: {metrics["Box Precision"  ]:.3f}")
    print(f"Mean Recall:    {metrics["Box Recall"     ]:.3f}")

    # Visualize some predictions
    # test_dataset = YOLOtoSegmentationDataset(data_root, image_size=640, split="test")
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    # visualize_predictions(model, test_loader, device)
    # visualize_predictions_with_ground_truth(model, test_loader, device)