import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_stats_aggregated import (
    AggregatedObjectSizeAnalyzer,
    AggregatedStatisticalAnalyzer,
    AggregatedFailureModeAnalyzer,
    generate_aggregated_report
)
from analysis_adv_aggregated import (
    AggregatedEnsembleAnalyzer,
    AggregatedHyperparameterAnalyzer,
    create_aggregated_comprehensive_tables
)
import pandas as pd


# All datasets to analyze
ALL_DATASETS = ["BCNB", "nuclei", "TNBC", "MoNuSAC", "CryoNuSeg"]
ALL_MODELS = ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yoloe", "yolow"]


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


def main():
    print("=" * 80)
    print("AGGREGATED CROSS-DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Create aggregated output directory
    aggregated_output = "./results_aggregated_all_datasets/"
    os.makedirs(aggregated_output, exist_ok=True)
    
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
    
    print(f"\n{'='*80}")
    print(f"Proceeding with {len(available_datasets)} datasets: {', '.join(available_datasets)}")
    print(f"{'='*80}")
    
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
    
    size_output = os.path.join(aggregated_output, '01_size_category_aggregated')
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
    # CONTRIBUTION #2: Aggregated Statistical Significance Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #2: Aggregated Statistical Significance Testing")
    print("=" * 80)
    
    stats_output = os.path.join(aggregated_output, '02_statistical_tests_aggregated')
    analyzer = AggregatedStatisticalAnalyzer(
        datasets_info=datasets_info,
        output_dir=stats_output
    )
    
    # Compute aggregated confidence intervals
    print("\nComputing aggregated confidence intervals...")
    ci_df = analyzer.compute_aggregated_confidence_intervals(metric='box_mAP@50')
    analyzer.plot_aggregated_confidence_intervals(ci_df, metric='box_mAP@50')
    
    # Perform cross-dataset statistical tests
    print("\nPerforming cross-dataset statistical tests...")
    tests_df = analyzer.cross_dataset_statistical_tests(metric='box_mAP@50')
    analyzer.create_aggregated_significance_heatmap(tests_df, metric='box_mAP@50')
    
    all_dataframes['stats_ci'] = ci_df
    all_dataframes['stats_tests'] = tests_df
    
    print("\n✓ Aggregated statistical analysis complete!")
    print(f"  Results saved to: {stats_output}")
    
    # ========================================================================
    # CONTRIBUTION #3: Aggregated Failure Mode Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #3: Aggregated Failure Mode Analysis")
    print("=" * 80)
    
    failure_output = os.path.join(aggregated_output, '03_failure_modes_aggregated')
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
    # CONTRIBUTION #5: Aggregated Ensemble Methods
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #5: Aggregated Ensemble Methods")
    print("=" * 80)
    
    ensemble_output = os.path.join(aggregated_output, '05_ensemble_methods_aggregated')
    analyzer = AggregatedEnsembleAnalyzer(
        datasets_info=datasets_info,
        models=ALL_MODELS,
        output_dir=ensemble_output,
        iou_threshold=0.5
    )
    
    # Compute model weights across all datasets
    all_results = []
    for dataset_info in datasets_info:
        if os.path.exists(dataset_info['results_csv']):
            df = pd.read_csv(dataset_info['results_csv'])
            df['dataset'] = dataset_info['name']
            all_results.append(df)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Weight models by average F1 across datasets
    model_weights = {}
    for model in ALL_MODELS:
        model_data = combined_results[combined_results['model'] == model]
        if len(model_data) > 0:
            avg_f1 = model_data['box_mean_f1'].mean()
            model_weights[model] = avg_f1
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v / total_weight for k, v in model_weights.items()}
    
    ensemble_df = analyzer.compare_all_ensembles_aggregated(model_weights=model_weights)
    analyzer.plot_aggregated_ensemble_comparison(ensemble_df, combined_results)
    all_dataframes['ensemble'] = ensemble_df
    
    print("\n✓ Aggregated ensemble analysis complete!")
    print(f"  Results saved to: {ensemble_output}")
    
    if len(ensemble_df) > 0:
        print(f"\nBest ensemble method across all datasets:")
        best_ensemble = ensemble_df.loc[ensemble_df['f1'].idxmax()]
        print(f"  Method: {best_ensemble['ensemble_method']}")
        print(f"  F1: {best_ensemble['f1']:.4f}")
        print(f"  Precision: {best_ensemble['precision']:.4f}")
        print(f"  Recall: {best_ensemble['recall']:.4f}")
    
    # ========================================================================
    # CONTRIBUTION #8: Aggregated Hyperparameter Sensitivity Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRIBUTION #8: Aggregated Hyperparameter Sensitivity Analysis")
    print("=" * 80)
    
    hyperparam_output = os.path.join(aggregated_output, '08_hyperparameter_sensitivity_aggregated')
    analyzer = AggregatedHyperparameterAnalyzer(
        datasets_info=datasets_info,
        output_dir=hyperparam_output
    )
    
    # Confidence threshold sensitivity across datasets
    print("\nAnalyzing confidence threshold sensitivity across datasets...")
    conf_df = analyzer.analyze_conf_threshold_sensitivity_aggregated(
        models=ALL_MODELS,
        conf_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    )
    
    # IoU threshold sensitivity across datasets
    print("\nAnalyzing IoU threshold sensitivity across datasets...")
    iou_df = analyzer.analyze_iou_threshold_sensitivity_aggregated(
        models=ALL_MODELS,
        iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
    )
    
    analyzer.plot_aggregated_threshold_sensitivity(conf_df, iou_df)
    all_dataframes['hyperparam_conf'] = conf_df
    all_dataframes['hyperparam_iou'] = iou_df
    
    print("\n✓ Aggregated hyperparameter sensitivity analysis complete!")
    print(f"  Results saved to: {hyperparam_output}")
    
    # ========================================================================
    # Generate Comprehensive Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Comprehensive Aggregated Report")
    print("=" * 80)
    
    report_path = os.path.join(aggregated_output, 'AGGREGATED_ANALYSIS_REPORT.md')
    generate_aggregated_report(
        size_df=all_dataframes.get('size'),
        stat_df_ci=all_dataframes.get('stats_ci'),
        stat_df_tests=all_dataframes.get('stats_tests'),
        failure_df=all_dataframes.get('failure'),
        ensemble_df=all_dataframes.get('ensemble'),
        datasets=available_datasets,
        output_path=report_path
    )
    
    # Generate LaTeX tables
    print("\nGenerating publication-ready tables...")
    tables_output = os.path.join(aggregated_output, 'latex_tables')
    create_aggregated_comprehensive_tables(
        size_df=all_dataframes.get('size'),
        ensemble_df=all_dataframes.get('ensemble'),
        failure_df=all_dataframes.get('failure'),
        stat_df=combined_results,
        output_dir=tables_output
    )
    
    print("\n" + "=" * 80)
    print("✓ AGGREGATED ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {aggregated_output}")
    print(f"Report available at: {report_path}")
    print(f"\nDatasets analyzed: {', '.join(available_datasets)}")
    print(f"Total datasets: {len(available_datasets)}")
    print(f"Models analyzed: {len(ALL_MODELS)}")


if __name__ == "__main__":
    main()
