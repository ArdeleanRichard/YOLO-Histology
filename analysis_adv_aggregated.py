import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from analysis_stats import (load_yolo_boxes, load_inference_boxes,
                            calculate_iou, BoundingBox)
from analysis_adv import EnsembleAnalyzer


class AggregatedEnsembleAnalyzer:
    """
    Aggregated Ensemble Analysis across multiple datasets
    """
    
    def __init__(self, datasets_info: List[Dict], models: List[str],
                 output_dir: str, iou_threshold: float = 0.5):
        self.datasets_info = datasets_info
        self.models = models
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create ensemble analyzers for each dataset
        self.dataset_analyzers = {}
        for dataset_info in datasets_info:
            analyzer = EnsembleAnalyzer(
                gt_folder=dataset_info['gt_folder'],
                inference_root=dataset_info['inference_root'],
                image_folder=dataset_info['image_folder'],
                models=models,
                output_dir=os.path.join(output_dir, f"dataset_{dataset_info['name']}"),
                iou_threshold=iou_threshold
            )
            self.dataset_analyzers[dataset_info['name']] = analyzer
    
    def compare_all_ensembles_aggregated(self, model_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Compare ensemble methods across all datasets"""
        all_results = []
        
        ensemble_methods = ['nms', 'weighted_box_fusion', 'voting_2', 'voting_3', 'average_confidence']
        
        print("\nEvaluating ensemble methods across all datasets:")
        
        for method in ensemble_methods:
            print(f"  Evaluating {method}...")
            
            # Aggregate metrics across datasets
            total_tp, total_fp, total_fn = 0, 0, 0
            
            for dataset_name, analyzer in self.dataset_analyzers.items():
                # Get image files for this dataset
                dataset_info = next(d for d in self.datasets_info if d['name'] == dataset_name)
                gt_folder = dataset_info['gt_folder']
                image_folder = dataset_info['image_folder']
                
                image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                              if f.endswith('.txt')]
                
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
                    
                    # Load all predictions
                    all_preds = analyzer.load_all_predictions(img_name, img_width, img_height)
                    
                    # Flatten to list of (model, box) tuples
                    all_boxes = []
                    for model, boxes in all_preds.items():
                        for box in boxes:
                            all_boxes.append((model, box))
                    
                    # Apply ensemble method
                    if method == 'nms':
                        ensemble_boxes = analyzer.nms_ensemble(all_boxes, self.iou_threshold)
                    elif method == 'weighted_box_fusion':
                        ensemble_boxes = analyzer.weighted_box_fusion(all_boxes, model_weights, self.iou_threshold)
                    elif method == 'voting_2':
                        ensemble_boxes = analyzer.voting_ensemble(all_boxes, min_votes=2, iou_threshold=self.iou_threshold)
                    elif method == 'voting_3':
                        ensemble_boxes = analyzer.voting_ensemble(all_boxes, min_votes=3, iou_threshold=self.iou_threshold)
                    elif method == 'average_confidence':
                        ensemble_boxes = analyzer.average_confidence_ensemble(all_boxes, self.iou_threshold)
                    else:
                        continue
                    
                    # Load ground truth
                    gt_path = os.path.join(gt_folder, f"{img_name}.txt")
                    gt_boxes = load_yolo_boxes(gt_path)
                    
                    # Calculate metrics
                    matched_gt = set()
                    
                    for pred_box in ensemble_boxes:
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for j, gt_box in enumerate(gt_boxes):
                            if j in matched_gt or gt_box.cls != pred_box.cls:
                                continue
                            
                            iou = calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                        
                        if best_iou >= self.iou_threshold and best_gt_idx != -1:
                            total_tp += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            total_fp += 1
                    
                    total_fn += len(gt_boxes) - len(matched_gt)
            
            # Calculate aggregated metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            all_results.append({
                'ensemble_method': method,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            })
        
        df = pd.DataFrame(all_results)
        
        output_path = os.path.join(self.output_dir, 'ensemble_comparison_aggregated.csv')
        df.to_csv(output_path, index=False)
        print(f"\nAggregated ensemble comparison saved to: {output_path}")
        
        return df
    
    def plot_aggregated_ensemble_comparison(self, ensemble_df: pd.DataFrame,
                                           individual_results_df: pd.DataFrame):
        """Plot ensemble vs individual model performance (aggregated)"""
        if len(ensemble_df) == 0:
            return
        
        # Calculate average individual model performance across datasets
        individual_avg = individual_results_df.groupby('model').agg({
            'box_mean_precision': 'mean',
            'box_mean_recall': 'mean',
            'box_mean_f1': 'mean'
        }).reset_index()
        
        individual_avg = individual_avg.rename(columns={
            'box_mean_precision': 'precision',
            'box_mean_recall': 'recall',
            'box_mean_f1': 'f1'
        })
        individual_avg['type'] = 'Individual'
        individual_avg['name'] = individual_avg['model']
        
        ensemble_df_plot = ensemble_df.copy()
        ensemble_df_plot['type'] = 'Ensemble'
        ensemble_df_plot['name'] = ensemble_df_plot['ensemble_method']
        
        # Combine for plotting
        combined = pd.concat([
            individual_avg[['name', 'type', 'precision', 'recall', 'f1']],
            ensemble_df_plot[['name', 'type', 'precision', 'recall', 'f1']]
        ], ignore_index=True)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(['precision', 'recall', 'f1']):
            ax = axes[idx]
            
            individual_data = combined[combined['type'] == 'Individual']
            ensemble_data = combined[combined['type'] == 'Ensemble']
            
            x_individual = np.arange(len(individual_data))
            x_ensemble = np.arange(len(ensemble_data)) + len(individual_data) + 1
            
            ax.bar(x_individual, individual_data[metric].values,
                  color='steelblue', alpha=0.7, label='Individual Models')
            ax.bar(x_ensemble, ensemble_data[metric].values,
                  color='coral', alpha=0.7, label='Ensemble Methods')
            
            # Add labels
            all_x = np.concatenate([x_individual, x_ensemble])
            all_labels = list(individual_data['name'].values) + list(ensemble_data['name'].values)
            
            ax.set_xticks(all_x)
            ax.set_xticklabels([l.upper() if len(l) < 10 else l for l in all_labels],
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.suptitle('Aggregated Ensemble vs Individual Model Performance\n(Across All Datasets)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ensemble_vs_individual_aggregated.png'), dpi=300)
        plt.close()
        
        print(f"Aggregated ensemble comparison plots saved to: {self.output_dir}")


class AggregatedHyperparameterAnalyzer:
    """
    Aggregated Hyperparameter Sensitivity Analysis across multiple datasets
    """
    
    def __init__(self, datasets_info: List[Dict], output_dir: str):
        self.datasets_info = datasets_info
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_conf_threshold_sensitivity_aggregated(self, models: List[str],
                                                      conf_thresholds: List[float]) -> pd.DataFrame:
        """Analyze confidence threshold sensitivity across all datasets"""
        all_results = []
        
        print(f"Analyzing confidence threshold sensitivity across datasets...")
        
        for conf_thresh in conf_thresholds:
            print(f"  Testing confidence threshold: {conf_thresh}")
            
            for model in models:
                # Aggregate across datasets
                total_tp, total_fp, total_fn = 0, 0, 0
                
                for dataset_info in self.datasets_info:
                    gt_folder = dataset_info['gt_folder']
                    inference_root = dataset_info['inference_root']
                    image_folder = dataset_info['image_folder']
                    
                    image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                                  if f.endswith('.txt')]
                    
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
                        
                        gt_boxes = load_yolo_boxes(gt_path)
                        all_pred_boxes = load_inference_boxes(inf_path, img_width, img_height)
                        
                        # Filter by confidence threshold
                        pred_boxes = [box for box in all_pred_boxes if box.conf >= conf_thresh]
                        
                        # Match predictions to ground truth
                        matched_gt = set()
                        
                        for pred_box in pred_boxes:
                            best_iou = 0
                            best_gt_idx = -1
                            
                            for j, gt_box in enumerate(gt_boxes):
                                if j in matched_gt or gt_box.cls != pred_box.cls:
                                    continue
                                
                                iou = calculate_iou(pred_box, gt_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = j
                            
                            if best_iou >= 0.5 and best_gt_idx != -1:
                                total_tp += 1
                                matched_gt.add(best_gt_idx)
                            else:
                                total_fp += 1
                        
                        total_fn += len(gt_boxes) - len(matched_gt)
                
                # Calculate aggregated metrics
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                all_results.append({
                    'model': model,
                    'conf_threshold': conf_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        df = pd.DataFrame(all_results)
        
        output_path = os.path.join(self.output_dir, 'conf_threshold_sensitivity_aggregated.csv')
        df.to_csv(output_path, index=False)
        print(f"Aggregated confidence threshold sensitivity saved to: {output_path}")
        
        return df
    
    def analyze_iou_threshold_sensitivity_aggregated(self, models: List[str],
                                                     iou_thresholds: List[float]) -> pd.DataFrame:
        """Analyze IoU threshold sensitivity across all datasets"""
        all_results = []
        
        print(f"Analyzing IoU threshold sensitivity across datasets...")
        
        for iou_thresh in iou_thresholds:
            print(f"  Testing IoU threshold: {iou_thresh}")
            
            for model in models:
                # Aggregate across datasets
                total_tp, total_fp, total_fn = 0, 0, 0
                
                for dataset_info in self.datasets_info:
                    gt_folder = dataset_info['gt_folder']
                    inference_root = dataset_info['inference_root']
                    image_folder = dataset_info['image_folder']
                    
                    image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                                  if f.endswith('.txt')]
                    
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
                        
                        gt_boxes = load_yolo_boxes(gt_path)
                        pred_boxes = load_inference_boxes(inf_path, img_width, img_height)
                        
                        # Match predictions to ground truth
                        matched_gt = set()
                        
                        for pred_box in pred_boxes:
                            best_iou = 0
                            best_gt_idx = -1
                            
                            for j, gt_box in enumerate(gt_boxes):
                                if j in matched_gt or gt_box.cls != pred_box.cls:
                                    continue
                                
                                iou = calculate_iou(pred_box, gt_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = j
                            
                            if best_iou >= iou_thresh and best_gt_idx != -1:
                                total_tp += 1
                                matched_gt.add(best_gt_idx)
                            else:
                                total_fp += 1
                        
                        total_fn += len(gt_boxes) - len(matched_gt)
                
                # Calculate aggregated metrics
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                all_results.append({
                    'model': model,
                    'iou_threshold': iou_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        df = pd.DataFrame(all_results)
        
        output_path = os.path.join(self.output_dir, 'iou_threshold_sensitivity_aggregated.csv')
        df.to_csv(output_path, index=False)
        print(f"Aggregated IoU threshold sensitivity saved to: {output_path}")
        
        return df
    
    def plot_aggregated_threshold_sensitivity(self, df_conf: pd.DataFrame, df_iou: pd.DataFrame):
        """Plot aggregated threshold sensitivity"""
        # Plot 1: Confidence threshold sensitivity
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for metric, ax in zip(['precision', 'recall', 'f1'], axes):
            for model in df_conf['model'].unique():
                model_data = df_conf[df_conf['model'] == model]
                ax.plot(model_data['conf_threshold'], model_data[metric],
                       marker='o', label=model.upper(), linewidth=2)
            
            ax.set_xlabel('Confidence Threshold', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} vs Confidence Threshold\n(Aggregated)',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'conf_threshold_sensitivity_aggregated.png'), dpi=300)
        plt.close()
        
        # Plot 2: IoU threshold sensitivity
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for metric, ax in zip(['precision', 'recall', 'f1'], axes):
            for model in df_iou['model'].unique():
                model_data = df_iou[df_iou['model'] == model]
                ax.plot(model_data['iou_threshold'], model_data[metric],
                       marker='s', label=model.upper(), linewidth=2)
            
            ax.set_xlabel('IoU Threshold', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} vs IoU Threshold\n(Aggregated)',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'iou_threshold_sensitivity_aggregated.png'), dpi=300)
        plt.close()
        
        print(f"Aggregated threshold sensitivity plots saved to: {self.output_dir}")


def create_aggregated_comprehensive_tables(size_df: pd.DataFrame,
                                           ensemble_df: pd.DataFrame,
                                           failure_df: pd.DataFrame,
                                           stat_df: pd.DataFrame,
                                           output_dir: str):
    """
    Create publication-ready LaTeX tables for aggregated analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: Aggregated Performance by Size Category
    if size_df is not None and len(size_df) > 0:
        # Aggregate across datasets
        agg_size = size_df.groupby(['model', 'size_category']).agg({
            'tp': 'sum', 'fp': 'sum', 'fn': 'sum'
        }).reset_index()
        
        agg_size['f1'] = agg_size.apply(
            lambda row: 2 * (row['tp'] / (row['tp'] + row['fp'])) * (row['tp'] / (row['tp'] + row['fn'])) /
                       ((row['tp'] / (row['tp'] + row['fp'])) + (row['tp'] / (row['tp'] + row['fn'])))
            if (row['tp'] + row['fp']) > 0 and (row['tp'] + row['fn']) > 0 else 0,
            axis=1
        )
        
        table1 = agg_size.pivot_table(
            index='model',
            columns='size_category',
            values='f1',
            aggfunc='mean'
        )
        
        # Ensure correct column order
        column_order = ['tiny', 'small', 'medium', 'large']
        table1 = table1[[col for col in column_order if col in table1.columns]]
        
        with open(os.path.join(output_dir, 'table_size_category_aggregated.tex'), 'w') as f:
            f.write(table1.to_latex(float_format='%.3f',
                                   caption='Aggregated F1 Score by Object Size Category (All Datasets)',
                                   label='tab:size_category_agg'))
        
        print(f"Table 1 (Size Category) saved")
    
    # Table 2: Best Ensemble vs Best Individual (Aggregated)
    if stat_df is not None and ensemble_df is not None and len(stat_df) > 0 and len(ensemble_df) > 0:
        # Average across datasets for individual models
        individual_avg = stat_df.groupby('model').agg({
            'box_mean_precision': 'mean',
            'box_mean_recall': 'mean',
            'box_mean_f1': 'mean',
            'box_mAP@50': 'mean'
        }).reset_index()
        
        best_individual = individual_avg.loc[individual_avg['box_mAP@50'].idxmax()]
        best_ensemble = ensemble_df.loc[ensemble_df['f1'].idxmax()]
        
        comparison_data = {
            'Method': [best_individual['model'].upper(), best_ensemble['ensemble_method']],
            'Type': ['Individual', 'Ensemble'],
            'Precision': [best_individual['box_mean_precision'], best_ensemble['precision']],
            'Recall': [best_individual['box_mean_recall'], best_ensemble['recall']],
            'F1': [best_individual['box_mean_f1'], best_ensemble['f1']]
        }
        table2 = pd.DataFrame(comparison_data)
        
        with open(os.path.join(output_dir, 'table_best_methods_aggregated.tex'), 'w') as f:
            f.write(table2.to_latex(index=False, float_format='%.3f',
                                   caption='Best Individual Model vs Best Ensemble (Aggregated)',
                                   label='tab:best_methods_agg'))
        
        print(f"Table 2 (Best Methods) saved")
    
    # Table 3: Aggregated Failure Mode Summary
    if failure_df is not None and len(failure_df) > 0:
        failure_cols = [col for col in failure_df.columns
                       if col not in ['dataset', 'model'] and not col.endswith('_pct')]
        
        # Aggregate across datasets
        agg_failure = failure_df.groupby('model')[failure_cols].sum().reset_index()
        
        # Select key failure modes
        key_modes = ['missed_detection', 'background_fp', 'boundary_error', 'duplicate_detection']
        key_modes = [m for m in key_modes if m in agg_failure.columns]
        
        if len(key_modes) > 0:
            table3 = agg_failure[['model'] + key_modes].set_index('model')
            
            with open(os.path.join(output_dir, 'table_failure_modes_aggregated.tex'), 'w') as f:
                f.write(table3.to_latex(float_format='%.0f',
                                       caption='Aggregated Failure Mode Distribution by Model (All Datasets)',
                                       label='tab:failure_modes_agg'))
            
            print(f"Table 3 (Failure Modes) saved")
    
    print(f"\nAll LaTeX tables saved to: {output_dir}")
