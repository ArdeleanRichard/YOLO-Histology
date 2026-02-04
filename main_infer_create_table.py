import os
import pandas as pd

from constants import ALL_MODELS


def combine_comparison_results(datasets, models, results_root_template="./results_data_{}/inferences_all/{}/comparison/comparison_results.csv"):
    """
    Extract F1 scores from multiple comparison results and combine into a single CSV.

    Args:
        datasets: List of dataset names (e.g., ["BCNB", "nuclei", "TNBC", "MoNuSAC", "CryoNuSeg"])
        models: List of model names (e.g., ["rtdetr", "yolo8", "yolo9", ...])
        results_root_template: Template string for the path to comparison_results.csv
                              Should contain two {} placeholders for dataset and model

    Returns:
        DataFrame with models as rows and dataset-method combinations as columns
    """

    # Dictionary to store all data
    # Structure: {model: {(dataset, method): f1_score}}
    all_data = {model: {} for model in models}

    # Collect all unique methods across all datasets
    all_methods = set()

    # Read all CSV files
    for dataset in datasets:
        for model in models:
            csv_path = results_root_template.format(dataset, model)

            if not os.path.exists(csv_path):
                print(f"Warning: File not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)

                # Extract method and f1 columns
                for _, row in df.iterrows():
                    method = row['method']
                    f1_score = row['f1']

                    # Store in dictionary
                    column_name = f"{dataset}_{method}"
                    all_data[model][column_name] = f1_score
                    all_methods.add(method)

            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

    # Create DataFrame
    # Get all unique column names (dataset-method combinations) in sorted order
    all_columns = sorted(set(col for model_data in all_data.values() for col in model_data.keys()))

    # Build the final dataframe
    rows = []
    for model in models:
        row = {'model': model}
        for col in all_columns:
            row[col] = all_data[model].get(col, None)  # None if data doesn't exist
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Set model as index
    result_df.set_index('model', inplace=True)

    return result_df


# Example usage
if __name__ == "__main__":
    # Define your datasets and models
    DATASETS = ["nuclei", "TNBC", "CryoNuSeg"]

    # Combine all results
    combined_df = combine_comparison_results(
        datasets=DATASETS,
        models=ALL_MODELS,
        results_root_template="./results_data_{}/inferences_all/{}/comparison/comparison_results.csv"
    )

    # Save to CSV
    output_path = "./results/inference_f1_scores.csv"
    combined_df.to_csv(output_path)

    print(f"\nCombined results saved to: {output_path}")
    print(f"\nShape: {combined_df.shape}")
    print(f"\nPreview:")
    print(combined_df)