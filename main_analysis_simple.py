import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_stats import (ObjectSizeAnalyzer, StatisticalAnalyzer,
                            FailureModeAnalyzer, generate_analysis_report)
from analysis_adv import (EnsembleAnalyzer, HyperparameterAnalyzer,
                               create_comprehensive_tables_for_paper)
from constants import ALL_MODELS, data_root, results_root
import pandas as pd


def main():
    gt_folder = f"{data_root}/labels/test/"
    image_folder = f"{data_root}/images/test/"
    inference_root = f"{results_root}/inferences/"
    results_csv = f"{results_root}/results.csv"

    # Create analysis output directory
    analysis_output = f"{results_root}/analysis/"
    os.makedirs(analysis_output, exist_ok=True)

    print("=" * 80)
    print(f"COMPREHENSIVE ANALYSIS FOR {data_root}")
    print("=" * 80)

    # Check if results CSV exists
    if not os.path.exists(results_csv):
        print(f"ERROR: Results CSV not found at {results_csv}")
        print("Please run main_test_all.py first to generate results.")
        return

    # Check if inferences exist
    if not os.path.exists(inference_root):
        print(f"ERROR: Inferences not found at {inference_root}")
        print("Please run main_save_inferences.py first to generate predictions.")
        return

    # Storage for results
    all_dataframes = {}

    # ========================================================================
    # CONTRIBUTION #1: Object Size Category Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #1: Object Size Category Analysis")
    print("=" * 80)

    size_output = os.path.join(analysis_output, '01_size_category')
    analyzer = ObjectSizeAnalyzer(
        gt_folder=gt_folder,
        inference_root=inference_root,
        image_folder=image_folder,
        models=ALL_MODELS,
        output_dir=size_output,
        iou_threshold=0.5
    )

    size_df = analyzer.analyze_all()
    analyzer.plot_results(size_df)
    all_dataframes['size'] = size_df

    print("\nâœ“ Size category analysis complete!")
    print(f"  Results saved to: {size_output}")

    # ========================================================================
    # CONTRIBUTION #2: Statistical Significance Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #2: Statistical Significance Testing")
    print("=" * 80)

    stats_output = os.path.join(analysis_output, '02_statistical_tests')
    analyzer = StatisticalAnalyzer(
        results_csv_path=results_csv,
        output_dir=stats_output
    )

    # Compute confidence intervals
    print("\nComputing confidence intervals...")
    ci_df = analyzer.compute_confidence_intervals(metric='box_mAP@50')
    analyzer.plot_confidence_intervals(ci_df, metric='box_mAP@50')

    # Pairwise tests
    print("\nPerforming pairwise statistical tests...")
    tests_df = analyzer.pairwise_statistical_tests(metric='box_mAP@50')
    analyzer.create_significance_heatmap(tests_df, metric='box_mAP@50')

    all_dataframes['stats_ci'] = ci_df
    all_dataframes['stats_tests'] = tests_df

    print("\nâœ“ Statistical analysis complete!")
    print(f"  Results saved to: {stats_output}")

    # ========================================================================
    # CONTRIBUTION #3: Failure Mode Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #3: Failure Mode Analysis")
    print("=" * 80)

    failure_output = os.path.join(analysis_output, '03_failure_modes')
    analyzer = FailureModeAnalyzer(
        gt_folder=gt_folder,
        inference_root=inference_root,
        image_folder=image_folder,
        models=ALL_MODELS,
        output_dir=failure_output,
        iou_threshold=0.5
    )

    failure_df = analyzer.analyze_all()
    analyzer.plot_failure_modes(failure_df)
    all_dataframes['failure'] = failure_df

    print("\nâœ“ Failure mode analysis complete!")
    print(f"  Results saved to: {failure_output}")

    # ========================================================================
    # CONTRIBUTION #5: Ensemble Methods
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #5: Ensemble Methods")
    print("=" * 80)

    ensemble_output = os.path.join(analysis_output, '05_ensemble_methods')
    analyzer = EnsembleAnalyzer(
        gt_folder=gt_folder,
        inference_root=inference_root,
        image_folder=image_folder,
        models=ALL_MODELS,
        output_dir=ensemble_output,
        iou_threshold=0.5
    )

    # You can optionally provide model weights based on their performance
    # For example, weight models by their F1 score
    results_df = pd.read_csv(results_csv)
    model_weights = {}
    for model in ALL_MODELS:
        model_data = results_df[results_df['model'] == model]
        if len(model_data) > 0:
            # Weight by F1 score (normalized)
            f1 = model_data['box_mean_f1'].mean()
            model_weights[model] = f1

    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v / total_weight for k, v in model_weights.items()}

    ensemble_df = analyzer.compare_all_ensembles(model_weights=model_weights)
    analyzer.plot_ensemble_comparison(ensemble_df, results_df)
    all_dataframes['ensemble'] = ensemble_df

    print("\nâœ“ Ensemble analysis complete!")
    print(f"  Results saved to: {ensemble_output}")
    print(f"\nBest ensemble method:")
    best_ensemble = ensemble_df.loc[ensemble_df['f1'].idxmax()]
    print(f"  Method: {best_ensemble['ensemble_method']}")
    print(f"  F1: {best_ensemble['f1']:.4f}")
    print(f"  Precision: {best_ensemble['precision']:.4f}")
    print(f"  Recall: {best_ensemble['recall']:.4f}")

    # ========================================================================
    # CONTRIBUTION #8: Hyperparameter Sensitivity Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #8: Hyperparameter Sensitivity Analysis")
    print("=" * 80)

    hyperparam_output = os.path.join(analysis_output, '08_hyperparameter_sensitivity')
    analyzer = HyperparameterAnalyzer(
        results_csv_path=results_csv,
        output_dir=hyperparam_output
    )

    # Confidence threshold sensitivity
    print("\nAnalyzing confidence threshold sensitivity...")
    conf_df = analyzer.analyze_conf_threshold_sensitivity(
        gt_folder=gt_folder,
        inference_root=inference_root,
        image_folder=image_folder,
        models=ALL_MODELS,
        conf_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    )

    # IoU threshold sensitivity
    print("\nAnalyzing IoU threshold sensitivity...")
    iou_df = analyzer.analyze_iou_threshold_sensitivity(
        gt_folder=gt_folder,
        inference_root=inference_root,
        image_folder=image_folder,
        models=ALL_MODELS,
        iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
    )

    analyzer.plot_threshold_sensitivity(conf_df, iou_df)
    all_dataframes['hyperparam_conf'] = conf_df
    all_dataframes['hyperparam_iou'] = iou_df

    print("\nâœ“ Hyperparameter sensitivity analysis complete!")
    print(f"  Results saved to: {hyperparam_output}")



if __name__ == "__main__":
    main()
