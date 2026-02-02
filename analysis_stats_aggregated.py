import numpy as np
import pandas as pd
import os
import cv2
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

# Import from original analysis_stats
from analysis_stats import (
    BoundingBox, load_yolo_boxes, load_inference_boxes,
    calculate_iou
)


class AggregatedObjectSizeAnalyzer:
    """
    Aggregated Object Size Analysis across multiple datasets
    """
    
    def __init__(self, datasets_info: List[Dict], models: List[str],
                 output_dir: str, iou_threshold: float = 0.5):
        self.datasets_info = datasets_info
        self.models = models
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define size categories based on normalized area
        self.size_categories = {
            'tiny': (0, 0.0001),      # < 0.01% of image
            'small': (0.0001, 0.001),  # 0.01% - 0.1%
            'medium': (0.001, 0.01),   # 0.1% - 1%
            'large': (0.01, 1.0)       # > 1%
        }
    
    def categorize_by_size(self, box: BoundingBox) -> str:
        """Categorize a box by its size"""
        area = box.area()
        for category, (min_area, max_area) in self.size_categories.items():
            if min_area <= area < max_area:
                return category
        return 'large'
    
    def analyze_single_dataset(self, dataset_info: Dict) -> pd.DataFrame:
        """Analyze object size performance for a single dataset"""
        dataset_name = dataset_info['name']
        gt_folder = dataset_info['gt_folder']
        inference_root = dataset_info['inference_root']
        image_folder = dataset_info['image_folder']
        
        print(f"  Analyzing dataset: {dataset_name}")
        
        # Get all image files
        image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                      if f.endswith('.txt')]
        
        all_results = []
        
        for model in self.models:
            # Initialize counters for each size category
            category_stats = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0}
                            for cat in self.size_categories.keys()}
            
            for img_name in image_files:
                gt_path = os.path.join(gt_folder, f"{img_name}.txt")
                inf_path = os.path.join(inference_root, model, f"{img_name}.txt")

                img_height, img_width = None, None
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    image_path = os.path.join(image_folder, f"{img_name}{ext}")
                    if os.path.exists(image_path):
                        import cv2
                        img = cv2.imread(image_path)
                        if img is not None:
                            img_height, img_width = img.shape[:2]
                            break

                if img_width is None and img_height is None:
                    img_width, img_height = 640, 640
                
                gt_boxes = load_yolo_boxes(gt_path)
                pred_boxes = load_inference_boxes(inf_path, img_width, img_height)
                
                # Categorize ground truth boxes
                gt_by_category = {cat: [] for cat in self.size_categories.keys()}
                for gt_box in gt_boxes:
                    cat = self.categorize_by_size(gt_box)
                    gt_by_category[cat].append(gt_box)
                    category_stats[cat]['total_gt'] += 1
                
                # Match predictions to ground truth
                matched_gt = set()
                matched_pred = set()
                
                for i, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_gt_idx = -1
                    best_category = None
                    
                    for cat, gt_list in gt_by_category.items():
                        for j, gt_box in enumerate(gt_list):
                            if gt_box.cls != pred_box.cls:
                                continue
                            
                            # Create unique identifier for gt_box
                            gt_global_idx = (cat, j)
                            if gt_global_idx in matched_gt:
                                continue
                            
                            iou = calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                                best_category = cat
                    
                    if best_iou >= self.iou_threshold and best_category is not None:
                        # True positive
                        category_stats[best_category]['tp'] += 1
                        matched_gt.add((best_category, best_gt_idx))
                        matched_pred.add(i)
                    else:
                        # False positive - categorize by prediction size
                        pred_cat = self.categorize_by_size(pred_box)
                        category_stats[pred_cat]['fp'] += 1
                
                # Count false negatives (unmatched GT)
                for cat, gt_list in gt_by_category.items():
                    for j, gt_box in enumerate(gt_list):
                        if (cat, j) not in matched_gt:
                            category_stats[cat]['fn'] += 1
            
            # Calculate metrics for each category
            for cat, stats_dict in category_stats.items():
                tp = stats_dict['tp']
                fp = stats_dict['fp']
                fn = stats_dict['fn']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                all_results.append({
                    'dataset': dataset_name,
                    'model': model,
                    'size_category': cat,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'total_gt': stats_dict['total_gt'],
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        return pd.DataFrame(all_results)
    
    def analyze_all_datasets(self) -> pd.DataFrame:
        """Analyze all datasets and aggregate results"""
        all_dfs = []
        
        for dataset_info in self.datasets_info:
            df = self.analyze_single_dataset(dataset_info)
            all_dfs.append(df)
        
        # Combine all results
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined results
        output_path = os.path.join(self.output_dir, 'size_category_all_datasets.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\nAggregated size category results saved to: {output_path}")
        
        # Also save aggregated summary (summed across datasets)
        aggregated_df = self._aggregate_across_datasets(combined_df)
        agg_output_path = os.path.join(self.output_dir, 'size_category_aggregated_summary.csv')
        aggregated_df.to_csv(agg_output_path, index=False)
        print(f"Aggregated summary saved to: {agg_output_path}")
        
        return combined_df
    
    def _aggregate_across_datasets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics across all datasets"""
        # Group by model and size_category, sum TP/FP/FN
        grouped = df.groupby(['model', 'size_category']).agg({
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'total_gt': 'sum'
        }).reset_index()
        
        # Recalculate metrics
        grouped['precision'] = grouped.apply(
            lambda row: row['tp'] / (row['tp'] + row['fp']) if (row['tp'] + row['fp']) > 0 else 0,
            axis=1
        )
        grouped['recall'] = grouped.apply(
            lambda row: row['tp'] / (row['tp'] + row['fn']) if (row['tp'] + row['fn']) > 0 else 0,
            axis=1
        )
        grouped['f1'] = grouped.apply(
            lambda row: 2 * row['precision'] * row['recall'] / (row['precision'] + row['recall'])
            if (row['precision'] + row['recall']) > 0 else 0,
            axis=1
        )
        
        return grouped
    
    def plot_aggregated_results(self, df: pd.DataFrame):
        """Create visualizations for aggregated results"""
        # Aggregate across datasets for plotting
        agg_df = self._aggregate_across_datasets(df)
        
        # Plot 1: F1 score by size category (aggregated)
        plt.figure(figsize=(12, 6))
        
        categories = ['tiny', 'small', 'medium', 'large']
        x = np.arange(len(categories))
        width = 0.1
        
        for i, model in enumerate(self.models):
            model_data = agg_df[agg_df['model'] == model]
            f1_scores = [model_data[model_data['size_category'] == cat]['f1'].values[0]
                        if len(model_data[model_data['size_category'] == cat]) > 0 else 0
                        for cat in categories]
            
            plt.bar(x + i * width, f1_scores, width, label=model.upper())
        
        plt.xlabel('Object Size Category', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score by Object Size (All Datasets)', fontsize=14, fontweight='bold')
        categories_plot = ['tiny\n<0.01%', 'small\n0.01% - 0.1%', 'medium\n0.1% - 1%', 'large\n>1%']
        plt.xticks(x + width * (len(self.models) - 1) / 2, categories_plot)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_size_aggregated.png'), dpi=300)
        plt.close()
        
        # Plot 2: Heatmap of F1 scores (aggregated)
        pivot_df = agg_df.pivot(index='model', columns='size_category', values='f1')
        pivot_df = pivot_df[categories]  # Ensure correct order
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'F1 Score'})
        plt.title('F1 Score by Size Category', fontsize=14, fontweight='bold')
        plt.xlabel('Object Size Category', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_heatmap_aggregated.png'), dpi=300)
        plt.close()
        
        # Plot 3: Per-dataset comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        datasets = df['dataset'].unique()
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]
            
            for model in self.models:
                model_data = dataset_df[dataset_df['model'] == model]
                f1_scores = [model_data[model_data['size_category'] == cat]['f1'].values[0]
                            if len(model_data[model_data['size_category'] == cat]) > 0 else 0
                            for cat in categories]

                categories_plot = ['tiny\n<0.01%', 'small\n0.01% - 0.1%', 'medium\n0.1% - 1%', 'large\n>1%']
                ax.plot(categories_plot, f1_scores, marker='o', label=model.upper(), linewidth=2)
            
            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Size Category', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('F1 Score by Size Category - Per Dataset Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_size_per_dataset.png'), dpi=300)
        plt.close()
        
        print(f"Aggregated size analysis plots saved to: {self.output_dir}")


class AggregatedStatisticalAnalyzer:
    """
    Aggregated Statistical Analysis across multiple datasets
    """
    
    def __init__(self, datasets_info: List[Dict], output_dir: str):
        self.datasets_info = datasets_info
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_all_results(self) -> pd.DataFrame:
        """Load and combine results from all datasets"""
        all_dfs = []
        
        for dataset_info in self.datasets_info:
            csv_path = dataset_info['results_csv']
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['dataset'] = dataset_info['name']
                all_dfs.append(df)
        
        if len(all_dfs) == 0:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    
    def compute_aggregated_confidence_intervals(self, metric: str = 'box_mAP@50', confidence: float = 0.95) -> pd.DataFrame:
        """Compute confidence intervals aggregated across datasets"""
        combined_df = self.load_all_results()
        
        if len(combined_df) == 0:
            return pd.DataFrame()
        
        results = []
        
        for model in combined_df['model'].unique():
            model_data = combined_df[combined_df['model'] == model][metric].values
            
            if len(model_data) == 0:
                continue
            
            mean_val = np.mean(model_data)
            std_val = np.std(model_data, ddof=1)
            n = len(model_data)
            
            # Calculate confidence interval
            from scipy.stats import t
            confidence_level = confidence
            df = n - 1
            t_crit = t.ppf((1 + confidence_level) / 2, df)
            margin_error = t_crit * (std_val / np.sqrt(n))
            
            results.append({
                'model': model,
                'mean': mean_val,
                'std': std_val,
                'n_datasets': n,
                'ci_lower': mean_val - margin_error,
                'ci_upper': mean_val + margin_error,
                'confidence_level': confidence_level
            })
        
        df = pd.DataFrame(results)
        
        output_path = os.path.join(self.output_dir, f'{metric}_confidence_intervals_aggregated.csv')
        df.to_csv(output_path, index=False)
        print(f"Confidence intervals saved to: {output_path}")
        
        return df
    
    def plot_aggregated_confidence_intervals(self, df: pd.DataFrame, metric: str):
        """Plot confidence intervals across models"""
        if len(df) == 0:
            return
        
        plt.figure(figsize=(12, 6))
        
        df_sorted = df.sort_values('mean', ascending=False)
        models = df_sorted['model'].values
        means = df_sorted['mean'].values
        ci_lower = df_sorted['ci_lower'].values
        ci_upper = df_sorted['ci_upper'].values
        
        x = np.arange(len(models))
        
        plt.errorbar(x, means, yerr=[means - ci_lower, ci_upper - means],
                    fmt='o', capsize=5, capthick=2, markersize=8,
                    linewidth=2, color='steelblue')
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(f'{metric} (Mean ± CI)', fontsize=12)
        plt.title(f'Model Performance with Confidence Intervals\n(Aggregated across {df["n_datasets"].iloc[0]:.0f} datasets)',
                 fontsize=14, fontweight='bold')
        plt.xticks(x, [m.upper() for m in models], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric}_ci_plot_aggregated.png'), dpi=300)
        plt.close()
    
    def cross_dataset_statistical_tests(self, metric: str = 'box_mAP@50') -> pd.DataFrame:
        """Perform statistical tests comparing models across datasets"""
        combined_df = self.load_all_results()
        
        if len(combined_df) == 0:
            return pd.DataFrame()
        
        models = combined_df['model'].unique()
        results = []
        
        # Pairwise comparisons using Wilcoxon signed-rank test
        # (paired test since same datasets are used for all models)
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:
                    continue
                
                # Get paired data (same datasets)
                data1 = []
                data2 = []
                
                for dataset in combined_df['dataset'].unique():
                    m1_data = combined_df[(combined_df['model'] == model1) & (combined_df['dataset'] == dataset)]
                    m2_data = combined_df[(combined_df['model'] == model2) & (combined_df['dataset'] == dataset)]
                    
                    if len(m1_data) > 0 and len(m2_data) > 0:
                        data1.append(m1_data[metric].values[0])
                        data2.append(m2_data[metric].values[0])
                
                if len(data1) < 2:  # Need at least 2 datasets for test
                    continue
                
                # Perform Wilcoxon signed-rank test
                try:
                    stat, p_value = wilcoxon(data1, data2)
                except:
                    p_value = 1.0
                
                mean1 = np.mean(data1)
                mean2 = np.mean(data2)
                
                # Determine significance
                if p_value < 0.001:
                    sig = '***'
                elif p_value < 0.01:
                    sig = '**'
                elif p_value < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                better_model = model1 if mean1 > mean2 else model2
                
                results.append({
                    'model_a': model1,
                    'model_b': model2,
                    'mean_a': mean1,
                    'mean_b': mean2,
                    'p_value': p_value,
                    'significance': sig,
                    'better_model': better_model,
                    'n_datasets': len(data1)
                })
        
        df = pd.DataFrame(results)
        
        output_path = os.path.join(self.output_dir, f'{metric}_statistical_tests_aggregated.csv')
        df.to_csv(output_path, index=False)
        print(f"Statistical tests saved to: {output_path}")
        
        return df
    
    def create_aggregated_significance_heatmap(self, df: pd.DataFrame, metric: str):
        """Create heatmap of pairwise significance"""
        if len(df) == 0:
            return
        
        models = list(set(df['model_a'].unique()) | set(df['model_b'].unique()))
        models.sort()
        
        # Create matrix for p-values
        p_matrix = np.ones((len(models), len(models)))
        
        for _, row in df.iterrows():
            i = models.index(row['model_a'])
            j = models.index(row['model_b'])
            p_matrix[i, j] = row['p_value']
            p_matrix[j, i] = row['p_value']
        
        # Apply significance thresholds for visualization
        sig_matrix = np.zeros_like(p_matrix)
        sig_matrix[p_matrix < 0.05] = 1
        sig_matrix[p_matrix < 0.01] = 2
        sig_matrix[p_matrix < 0.001] = 3
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sig_matrix, annot=p_matrix, fmt='.3f',
                   cmap='RdYlGn_r', cbar_kws={'label': 'Significance Level'},
                   xticklabels=[m.upper() for m in models],
                   yticklabels=[m.upper() for m in models])
        
        plt.title(f'Statistical Significance of Pairwise Comparisons\n({metric}, aggregated across datasets)',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric}_significance_heatmap_aggregated.png'), dpi=300)
        plt.close()


class AggregatedFailureModeAnalyzer:
    """
    Aggregated Failure Mode Analysis across multiple datasets
    """
    
    def __init__(self, datasets_info: List[Dict], models: List[str],
                 output_dir: str, iou_threshold: float = 0.5):
        self.datasets_info = datasets_info
        self.models = models
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.failure_modes = {
            'missed_detection': 'Ground truth object not detected',
            'background_fp': 'False positive on background',
            'boundary_error': 'Incorrect bounding box size/location',
            'class_confusion': 'Wrong class predicted',
            'duplicate_detection': 'Multiple predictions for same object',
            'split_detection': 'Object split into multiple detections'
        }
    
    def analyze_image(self, gt_path: str, inf_path: str, img_width: int, img_height: int) -> Dict:
        """Analyze failure modes for a single image"""
        gt_boxes = load_yolo_boxes(gt_path)
        pred_boxes = load_inference_boxes(inf_path, img_width, img_height)
        
        failures = defaultdict(int)
        
        # Match predictions to ground truth
        matched_pred = set()
        
        for gt_box in gt_boxes:
            # Find all predictions that match this GT
            matches = []
            for i, pred_box in enumerate(pred_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > 0:
                    matches.append((i, iou, pred_box.cls == gt_box.cls))
            
            if len(matches) == 0:
                # Missed detection
                failures['missed_detection'] += 1
            else:
                # Sort by IoU
                matches.sort(key=lambda x: x[1], reverse=True)
                best_match = matches[0]
                
                if best_match[1] < self.iou_threshold:
                    # Poor localization
                    failures['boundary_error'] += 1
                
                if not best_match[2]:
                    # Wrong class
                    failures['class_confusion'] += 1
                
                # Check for duplicates
                high_iou_matches = [m for m in matches if m[1] >= self.iou_threshold]
                if len(high_iou_matches) > 1:
                    failures['duplicate_detection'] += len(high_iou_matches) - 1
                
                # Mark predictions as matched
                for pred_idx, _, _ in high_iou_matches:
                    matched_pred.add(pred_idx)
        
        # Analyze unmatched predictions (false positives)
        for i, pred_box in enumerate(pred_boxes):
            if i not in matched_pred:
                # Check if it overlaps with any ground truth
                has_overlap = False
                for gt_box in gt_boxes:
                    if calculate_iou(pred_box, gt_box) > 0.1:
                        has_overlap = True
                        break
                
                if has_overlap:
                    failures['split_detection'] += 1
                else:
                    failures['background_fp'] += 1
        
        return dict(failures)
    
    def analyze_single_dataset(self, dataset_info: Dict) -> pd.DataFrame:
        """Analyze failure modes for a single dataset"""
        dataset_name = dataset_info['name']
        gt_folder = dataset_info['gt_folder']
        inference_root = dataset_info['inference_root']
        image_folder = dataset_info['image_folder']
        
        print(f"  Analyzing dataset: {dataset_name}")
        
        image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                      if f.endswith('.txt')]
        
        all_results = []
        
        for model in self.models:
            model_failures = defaultdict(int)
            
            for img_name in image_files:
                # Get image dimensions
                image_path = os.path.join(image_folder, f"{img_name}.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{img_name}.png")
                
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        img_width, img_height = 640, 640
                else:
                    img_width, img_height = 640, 640
                
                gt_path = os.path.join(gt_folder, f"{img_name}.txt")
                inf_path = os.path.join(inference_root, model, f"{img_name}.txt")
                
                img_failures = self.analyze_image(gt_path, inf_path, img_width, img_height)
                
                for mode, count in img_failures.items():
                    model_failures[mode] += count
            
            # Add results
            result = {
                'dataset': dataset_name,
                'model': model
            }
            result.update(model_failures)
            
            # Calculate percentages
            total_failures = sum(model_failures.values())
            for mode in self.failure_modes.keys():
                count = model_failures.get(mode, 0)
                result[f'{mode}_pct'] = (count / total_failures * 100) if total_failures > 0 else 0
            
            all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def analyze_all_datasets(self) -> pd.DataFrame:
        """Analyze all datasets and aggregate results"""
        all_dfs = []
        
        for dataset_info in self.datasets_info:
            df = self.analyze_single_dataset(dataset_info)
            all_dfs.append(df)
        
        # Combine all results
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined results
        output_path = os.path.join(self.output_dir, 'failure_modes_all_datasets.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\nAggregated failure mode results saved to: {output_path}")
        
        # Also save aggregated summary (summed across datasets)
        aggregated_df = self._aggregate_across_datasets(combined_df)
        agg_output_path = os.path.join(self.output_dir, 'failure_modes_aggregated_summary.csv')
        aggregated_df.to_csv(agg_output_path, index=False)
        print(f"Aggregated summary saved to: {agg_output_path}")
        
        return combined_df
    
    def _aggregate_across_datasets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate failure counts across all datasets"""
        failure_cols = [col for col in df.columns
                       if col not in ['dataset', 'model'] and not col.endswith('_pct')]
        
        # Group by model and sum failure counts
        grouped = df.groupby('model')[failure_cols].sum().reset_index()
        
        # Recalculate percentages
        for mode in self.failure_modes.keys():
            if mode in grouped.columns:
                grouped['total_failures'] = grouped[list(self.failure_modes.keys())].sum(axis=1)
                grouped[f'{mode}_pct'] = (grouped[mode] / grouped['total_failures'] * 100).fillna(0)
        
        return grouped
    
    def plot_aggregated_failure_modes(self, df: pd.DataFrame):
        """Visualize aggregated failure mode distributions"""
        # Aggregate across datasets
        agg_df = self._aggregate_across_datasets(df)
        
        failure_cols = [col for col in agg_df.columns if col in self.failure_modes.keys()]
        
        if len(failure_cols) == 0:
            return
        
        # Plot 1: Stacked bar chart (aggregated)
        plt.figure(figsize=(12, 6))
        df_plot = agg_df.set_index('model')[failure_cols]
        df_plot.plot(kind='bar', stacked=True, figsize=(12, 6),
                    colormap='Set3', edgecolor='black', linewidth=0.5)
        
        plt.title('Failure Mode Distribution by Model (All Datasets)', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Number of Failures', fontsize=12)
        plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'failure_modes_stacked_aggregated.png'), dpi=300)
        plt.close()
        
        # Plot 2: Per-dataset comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        datasets = df['dataset'].unique()
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset].set_index('model')[failure_cols]
            
            dataset_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Failures', fontsize=10)
            ax.legend().set_visible(False)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        # Add legend to the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Failure Mode', loc='center left', bbox_to_anchor=(0.95, 0.5))
        
        plt.suptitle('Failure Modes - Per Dataset Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(self.output_dir, 'failure_modes_per_dataset.png'), dpi=300)
        plt.close()
        
        print(f"Aggregated failure mode plots saved to: {self.output_dir}")


def generate_aggregated_report(size_df: pd.DataFrame,
                              stat_df_ci: pd.DataFrame,
                              stat_df_tests: pd.DataFrame,
                              failure_df: pd.DataFrame,
                              ensemble_df: Optional[pd.DataFrame],
                              datasets: List[str],
                              output_path: str):
    """
    Generate a comprehensive markdown report for aggregated analysis
    """
    with open(output_path, 'w') as f:
        f.write("# Comprehensive Aggregated Analysis Report\n\n")
        f.write(f"**Datasets Analyzed:** {', '.join(datasets)}\n")
        f.write(f"**Total Datasets:** {len(datasets)}\n\n")
        
        f.write("## 1. Aggregated Object Size Category Analysis\n\n")
        
        if size_df is not None and len(size_df) > 0:
            # Aggregate across datasets
            agg_size = size_df.groupby(['model', 'size_category']).agg({
                'tp': 'sum', 'fp': 'sum', 'fn': 'sum'
            }).reset_index()
            
            agg_size['precision'] = agg_size.apply(
                lambda row: row['tp'] / (row['tp'] + row['fp']) if (row['tp'] + row['fp']) > 0 else 0,
                axis=1
            )
            agg_size['recall'] = agg_size.apply(
                lambda row: row['tp'] / (row['tp'] + row['fn']) if (row['tp'] + row['fn']) > 0 else 0,
                axis=1
            )
            agg_size['f1'] = agg_size.apply(
                lambda row: 2 * row['precision'] * row['recall'] / (row['precision'] + row['recall'])
                if (row['precision'] + row['recall']) > 0 else 0,
                axis=1
            )
            
            f.write("### Best Models by Size Category (Aggregated)\n\n")
            for cat in ['tiny', 'small', 'medium', 'large']:
                cat_data = agg_size[agg_size['size_category'] == cat].nlargest(3, 'f1')
                f.write(f"**{cat.capitalize()} objects:**\n")
                for _, row in cat_data.iterrows():
                    f.write(f"- {row['model'].upper()}: F1={row['f1']:.3f}, ")
                    f.write(f"Precision={row['precision']:.3f}, Recall={row['recall']:.3f}\n")
                f.write("\n")
        
        f.write("## 2. Aggregated Statistical Significance Analysis\n\n")
        
        if stat_df_tests is not None and len(stat_df_tests) > 0:
            sig_tests = stat_df_tests[stat_df_tests['significance'] != 'ns']
            f.write(f"Found {len(sig_tests)} statistically significant differences (p < 0.05)\n\n")
            
            if len(sig_tests) > 0:
                f.write("### Significant Pairwise Comparisons\n\n")
                for _, row in sig_tests.iterrows():
                    f.write(f"- **{row['model_a'].upper()} vs {row['model_b'].upper()}**: ")
                    f.write(f"p={row['p_value']:.4f} {row['significance']}, ")
                    f.write(f"{row['better_model'].upper()} performs better ")
                    f.write(f"(across {row['n_datasets']:.0f} datasets)\n")
        
        f.write("\n## 3. Aggregated Failure Mode Analysis\n\n")
        
        if failure_df is not None and len(failure_df) > 0:
            # Aggregate across datasets
            failure_cols = [col for col in failure_df.columns
                           if col not in ['dataset', 'model'] and not col.endswith('_pct')]
            
            agg_failure = failure_df.groupby('model')[failure_cols].sum().reset_index()
            agg_failure['total_failures'] = agg_failure[failure_cols].sum(axis=1)
            
            f.write("### Total Failures by Model (Aggregated)\n\n")
            f.write(agg_failure[['model', 'total_failures']].sort_values('total_failures').to_markdown(index=False))
            f.write("\n\n")
            
            f.write("### Failure Mode Distribution\n\n")
            for col in failure_cols:
                if col in agg_failure.columns:
                    f.write(f"\n**{col.replace('_', ' ').title()}:**\n")
                    top_models = agg_failure.nlargest(3, col)[['model', col]]
                    for _, row in top_models.iterrows():
                        f.write(f"- {row['model'].upper()}: {row[col]:.0f}\n")
        
        if ensemble_df is not None and len(ensemble_df) > 0:
            f.write("\n## 4. Ensemble Methods (Aggregated)\n\n")
            f.write("### Best Ensemble Methods\n\n")
            best_ensembles = ensemble_df.nlargest(3, 'f1')
            for _, row in best_ensembles.iterrows():
                f.write(f"- **{row['ensemble_method']}**: ")
                f.write(f"F1={row['f1']:.3f}, ")
                f.write(f"Precision={row['precision']:.3f}, ")
                f.write(f"Recall={row['recall']:.3f}\n")
    
    print(f"\nAggregated report saved to: {output_path}")
