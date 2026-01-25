import torch

from constants import MODEL, DATA, results_root
from functions import ModelEvaluator

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_CSV_PATH = f"{results_root}/results.csv"

    evaluator = ModelEvaluator(MODEL, DATA, device, conf=0.25, iou=0.5)
    evaluator.load_model()

    # Evaluate the model
    evaluator.evaluate()

    # Print metrics
    evaluator.print_metrics()

    # Save results to CSV
    evaluator.save_results_to_csv(RESULTS_CSV_PATH)