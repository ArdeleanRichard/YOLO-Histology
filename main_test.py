import numpy as np
import torch
import pandas as pd
import os

from constants import dataset_yaml_path, dataset_SEG_yaml_path, MODEL, DATA, result_root
from functions import evaluate_model_box, load_model_test, evaluate_model_box_mask




def save_results_to_csv(metrics, model_name, dataset_name, csv_path):
    """
    Save evaluation results to CSV file using pandas

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        model_name (str): Name of the model being evaluated
        dataset_name (str): Name of the dataset used
        csv_path (str): Path to the CSV file
    """
    # Create a results dictionary with common fields
    results = {
        'dataset': dataset_name,
        'model': model_name,
    }

    # Add metrics based on model type
    if model_name != "yoloe":
        # Standard YOLO model metrics
        results.update({
            'box_mean_precision': metrics["Box Precision"],
            'box_mean_recall': metrics["Box Recall"],
            'box_mean_f1': metrics["Box F1"],
            'box_mAP@50': metrics["Box mAP@50"],
            'box_mAP@50-95': metrics["Box mAP@50-95"],

            # Set segmentation metrics to None for non-segmentation models
            'seg_mean_precision': None,
            'seg_mean_recall': None,
            'seg_mean_f1': None,
            'seg_mAP@50': None,
            'seg_mAP@50-95': None,
        })
    else:
        # YOLO-E model metrics (includes segmentation)
        results.update({
            'box_mean_precision': metrics["Box Precision"],
            'box_mean_recall': metrics["Box Recall"],
            'box_mean_f1': metrics["Box F1"],
            'box_mAP@50': metrics["Box mAP@50"],
            'box_mAP@50-95': metrics["Box mAP@50-95"],

            'seg_mean_precision': metrics["Mask Precision"],
            'seg_mean_recall': metrics["Mask Recall"],
            'seg_mean_f1': np.mean(np.array(metrics["Mask F1"])),
            'seg_mAP@50': metrics["Mask mAP@50"],
            'seg_mAP@50-95': metrics["Mask mAP@50-95"],
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


def print_metrics(metrics, model_name):
    """
    Print metrics in a formatted way

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        model_name (str): Name of the model being evaluated
    """
    print("Evaluation Metrics:")

    if model_name != "yoloe":
        print(f"mAP@50:         {metrics['Box mAP@50']:.3f}")
        print(f"mAP@50-95:      {metrics['Box mAP@50-95']:.3f}")
        print(f"Mean F1:        {metrics['Box F1']:.3f}")
        print(f"Mean Precision: {metrics['Box Precision']:.3f}")
        print(f"Mean Recall:    {metrics['Box Recall']:.3f}")
    else:
        print(f"Box mAP@50:         {metrics['Box mAP@50']:.3f}")
        print(f"Box mAP@50-95:      {metrics['Box mAP@50-95']:.3f}")
        print(f"Box Mean F1:        {np.mean(np.array(metrics['Box F1'])):.3f}")
        print(f"Box Mean Precision: {metrics['Box Precision']:.3f}")
        print(f"Box Mean Recall:    {metrics['Box Recall']:.3f}")
        print(f"Seg mAP@50:         {metrics['Mask mAP@50']:.3f}")
        print(f"Seg mAP@50-95:      {metrics['Mask mAP@50-95']:.3f}")
        print(f"Seg Mean F1:        {np.mean(np.array(metrics['Mask F1'])):.3f}")
        print(f"Seg Mean Precision: {metrics['Mask Precision']:.3f}")
        print(f"Seg Mean Recall:    {metrics['Mask Recall']:.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_test(MODEL)
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATA}")
    print(f"Model classes: {model.names}")

    # Evaluate the model
    if MODEL != "yoloe":
        metrics = evaluate_model_box(model, dataset_yaml_path, device)
    else:
        metrics = evaluate_model_box_mask(model, dataset_SEG_yaml_path, device)

    # Print metrics
    print_metrics(metrics, MODEL)

    # Save results to CSV
    RESULTS_CSV_PATH = f"{result_root}/results.csv"
    save_results_to_csv(metrics, MODEL, DATA, RESULTS_CSV_PATH)

    print(f"\nResults for model '{MODEL}' on dataset '{DATA}' have been saved.")