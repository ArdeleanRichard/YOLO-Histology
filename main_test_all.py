import torch

from constants import DATA, results_root, ALL_MODELS
from functions import ModelEvaluator

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_CSV_PATH = f"{results_root}/results.csv"

    for MODEL_NAME in ALL_MODELS:
        evaluator = ModelEvaluator(MODEL_NAME, DATA, device, conf=0.25, iou=0.5)
        evaluator.load_model()

        # Evaluate the model
        evaluator.evaluate()

        # Print metrics
        evaluator.print_metrics()

        # Save results to CSV
        evaluator.save_results_to_csv(RESULTS_CSV_PATH)