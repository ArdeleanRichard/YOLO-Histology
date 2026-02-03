import os
import sys

from constants import results_all_root

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_stats import (ObjectSizeAnalyzer, StatisticalAnalyzer, FailureModeAnalyzer)
from analysis_stats_aggregated import (AggregatedObjectSizeAnalyzer, AggregatedStatisticalAnalyzer, AggregatedFailureModeAnalyzer)



def main_one_dataset():
    from constants import ALL_MODELS, data_root, results_root
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

    print("\n✓ Size category analysis complete!")
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

    print("\n✓ Statistical analysis complete!")
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

    print("\n✓ Failure mode analysis complete!")
    print(f"  Results saved to: {failure_output}")


def get_dataset_paths(dataset: str):
    """Get paths for a specific dataset"""
    data_root = f"./data/{dataset}/"
    results_root = f"./results_data_{dataset}/"

    return {
        'data_root': data_root,
        'results_root': results_root,
        'gt_folder': f"{data_root}/labels/test/",
        'image_folder': f"{data_root}/images/test/",
        'inference_root': f"{results_root}/inferences/",
        'results_csv': f"{results_root}/results.csv"
    }


def verify_dataset_exists(paths: dict, dataset: str) -> bool:
    """Check if dataset results exist"""
    if not os.path.exists(paths['results_csv']):
        print(f"  ⚠️  WARNING: Results CSV not found for {dataset} at {paths['results_csv']}")
        return False

    if not os.path.exists(paths['inference_root']):
        print(f"  ⚠️  WARNING: Inferences not found for {dataset} at {paths['inference_root']}")
        return False

    return True


def main_all_datasets():
    """
    IMPROVED VERSION: Run comprehensive cross-dataset analysis with proper statistical tests

    This analysis uses:
    - Friedman test (non-parametric repeated measures ANOVA)
    - Nemenyi post-hoc test for pairwise comparisons
    - Wilcoxon signed-rank test with Bonferroni correction
    - Effect size calculations (Cohen's d, Hedges' g)
    - Critical difference diagrams
    """

    # All datasets to analyze
    ALL_DATASETS = ["BCNB", "nuclei", "TNBC", "MoNuSAC", "CryoNuSeg"]
    ALL_MODELS = ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yoloe", "yolow"]

    print("=" * 80)
    print("AGGREGATED CROSS-DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 80)


    # Verify all datasets
    print("\nVerifying dataset availability...")
    available_datasets = []
    for dataset in ALL_DATASETS:
        paths = get_dataset_paths(dataset)
        print(f"\nChecking {dataset}:")
        if verify_dataset_exists(paths, dataset):
            available_datasets.append(dataset)
            print(f"  ✓ {dataset} is available")
        else:
            print(f"  ✗ {dataset} is NOT available (skipping)")

    if len(available_datasets) == 0:
        print("\nERROR: No datasets available for analysis!")
        print("Please run main_test_all.py and main_save_inferences.py for each dataset first.")
        return

    print(f"\n{'=' * 80}")
    print(f"Proceeding with {len(available_datasets)} datasets: {', '.join(available_datasets)}")
    print(f"{'=' * 80}")

    # Prepare dataset information
    datasets_info = []
    for dataset in available_datasets:
        paths = get_dataset_paths(dataset)
        datasets_info.append({
            'name': dataset,
            **paths
        })

    # Storage for aggregated results
    all_dataframes = {}

    # ========================================================================
    # CONTRIBUTION #1: Aggregated Object Size Category Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #1: Aggregated Object Size Category Analysis")
    print("=" * 80)

    size_output = os.path.join(results_all_root, '01_size_category')
    analyzer = AggregatedObjectSizeAnalyzer(
        datasets_info=datasets_info,
        models=ALL_MODELS,
        output_dir=size_output,
        iou_threshold=0.5
    )

    size_df = analyzer.analyze_all_datasets()
    analyzer.plot_aggregated_results(size_df)
    all_dataframes['size'] = size_df

    print("\n✓ Aggregated size category analysis complete!")
    print(f"  Results saved to: {size_output}")

    # ========================================================================
    # CONTRIBUTION #2: IMPROVED Statistical Significance Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #2: IMPROVED Statistical Significance Testing")
    print("=" * 80)
    print("\nThis analysis includes:")
    print("  • Friedman test (non-parametric repeated measures)")
    print("  • Nemenyi post-hoc test")
    print("  • Wilcoxon signed-rank test with Bonferroni correction")
    print("  • Effect size calculations (Cohen's d, Hedges' g)")
    print("  • Critical difference diagrams")
    print("=" * 80)

    stats_output = os.path.join(results_all_root, '02_statistical_tests')
    analyzer = AggregatedStatisticalAnalyzer(
        datasets_info=datasets_info,
        output_dir=stats_output
    )

    # Define which metrics to analyze
    metrics_to_analyze = ['box_mAP@50', 'box_mAP@50-95', 'box_mean_f1']

    # Run the complete statistical analysis pipeline
    analyzer.run_complete_analysis(metrics=metrics_to_analyze)

    # The run_complete_analysis method performs:
    # 1. Confidence intervals computation and visualization
    # 2. Friedman test for overall significance
    # 3. Nemenyi post-hoc test (if Friedman is significant)
    # 4. Wilcoxon post-hoc test with Bonferroni correction (if Friedman is significant)
    # 5. Critical difference diagrams
    # 6. Comprehensive significance heatmaps
    # 7. Comprehensive text report with all results

    print("\n✓ IMPROVED statistical analysis complete!")
    print(f"  Results saved to: {stats_output}")
    print("\n  Key files generated:")
    print("    • comprehensive_statistical_report.txt - Full analysis report")
    print("    • *_friedman_test.csv - Overall significance test")
    print("    • *_nemenyi_post_hoc.csv - Pairwise comparisons (Nemenyi)")
    print("    • *_wilcoxon_bonferroni.csv - Pairwise comparisons (Wilcoxon)")
    print("    • *_effect_sizes.csv - Effect size calculations")
    print("    • *_critical_difference_diagram.png - Visual comparison")
    print("    • *_significance_heatmap_comprehensive.png - Detailed heatmaps")

    # ========================================================================
    # CONTRIBUTION #3: Aggregated Failure Mode Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #3: Aggregated Failure Mode Analysis")
    print("=" * 80)

    failure_output = os.path.join(results_all_root, '03_failure_modes')
    analyzer = AggregatedFailureModeAnalyzer(
        datasets_info=datasets_info,
        models=ALL_MODELS,
        output_dir=failure_output,
        iou_threshold=0.5
    )

    failure_df = analyzer.analyze_all_datasets()
    analyzer.plot_aggregated_failure_modes(failure_df)
    all_dataframes['failure'] = failure_df

    print("\n✓ Aggregated failure mode analysis complete!")
    print(f"  Results saved to: {failure_output}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nTotal datasets analyzed: {len(available_datasets)}")
    print(f"Total models compared: {len(ALL_MODELS)}")
    print(f"\nAll results saved to: {results_all_root}")
    print("\nKey Outputs:")
    print(f"  1. Size Category Analysis: {size_output}")
    print(f"  2. Statistical Tests: {stats_output}")
    print(f"     → READ: comprehensive_statistical_report.txt for detailed findings")
    print(f"  3. Failure Mode Analysis: {failure_output}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # main_one_dataset()
    main_all_datasets()