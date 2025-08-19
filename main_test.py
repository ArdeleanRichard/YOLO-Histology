import torch

from constants import dataset_yaml_path, dataset_SEG_yaml_path, MODEL, DATA, result_root
from functions import evaluate_model_box, load_model_test, evaluate_model_box_mask, print_metrics, save_results_to_csv

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_CSV_PATH = f"{result_root}/results.csv"

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
    save_results_to_csv(metrics, MODEL, DATA, RESULTS_CSV_PATH)

    print(f"\nResults for model '{MODEL}' on dataset '{DATA}' have been saved.")