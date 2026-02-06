import pandas as pd
import numpy as np
from constants import ALL_MODELS, results_root, DATA


def process_results_csv(csv_path, output_stats_path, output_mean_path):
    """
    Process a results CSV file and generate statistics and mean CSV files.

    Parameters:
    - csv_path: Path to the input results.csv file
    - output_stats_path: Path to save the results_stats.csv (mean ± std)
    - output_mean_path: Path to save the results_mean.csv (mean only)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Group by model name
    grouped = df.groupby('model')

    # Get numeric columns (exclude 'dataset' and 'model')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate mean and std for each model
    stats_data = []
    mean_data = []

    for model in ALL_MODELS:
        if model not in grouped.groups:
            print(f"Warning: Model '{model}' not found in {csv_path}")
            continue

        model_data = grouped.get_group(model)

        # Get the dataset name (should be the same for all rows)
        dataset_name = model_data['dataset'].iloc[0]

        # Calculate statistics for numeric columns
        means = model_data[numeric_cols].mean()
        stds = model_data[numeric_cols].std()

        # Create stats row (mean ± std)
        stats_row = {'dataset': dataset_name, 'model': model}
        for col in numeric_cols:
            mean_val = means[col]
            std_val = stds[col]

            # Handle NaN values
            if pd.isna(mean_val):
                stats_row[col] = ''
            elif pd.isna(std_val) or std_val == 0:
                # If std is NaN or 0 (single sample), just show mean
                stats_row[col] = f"{mean_val:.3f}"
            else:
                stats_row[col] = f"{mean_val:.3f}±{std_val:.3f}"

        stats_data.append(stats_row)

        # Create mean row
        mean_row = {'dataset': dataset_name, 'model': model}
        for col in numeric_cols:
            mean_val = means[col]
            if pd.isna(mean_val):
                mean_row[col] = ''
            else:
                mean_row[col] = f"{mean_val:.3f}"

        mean_data.append(mean_row)

    # Create DataFrames
    stats_df = pd.DataFrame(stats_data)
    mean_df = pd.DataFrame(mean_data)

    # Ensure columns are in the same order as original
    original_cols = df.columns.tolist()
    stats_df = stats_df[original_cols]
    mean_df = mean_df[original_cols]

    # Save to CSV
    stats_df.to_csv(output_stats_path, index=False)
    mean_df.to_csv(output_mean_path, index=False)

    print(f"Generated: {output_stats_path}")
    print(f"Generated: {output_mean_path}")

    return stats_df, mean_df


def process_all_datasets(datasets):
    """
    Process results CSV files for all specified datasets.

    Parameters:
    - datasets: List of dataset names
    """
    for dataset in datasets:
        csv_path = f"./results_data_{dataset}/results.csv"
        stats_path = f"./results_data_{dataset}/results_stats.csv"
        mean_path = f"./results_data_{dataset}/results_mean.csv"

        try:
            print(f"\nProcessing dataset: {dataset}")
            process_results_csv(csv_path, stats_path, mean_path)
        except FileNotFoundError:
            print(f"Error: File not found - {csv_path}")
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")


if __name__ == "__main__":
    datasets = ["BCNB", "CryoNuSeg", "MoNuSAC", "nuclei", "TNBC"]

    process_all_datasets(datasets)
