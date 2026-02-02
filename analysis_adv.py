"""
Advanced analysis module for research contributions
Implements:
5. Ensemble Methods
6. Domain-Specific Augmentation Study (requires retraining)
7. Transfer Learning Analysis (requires retraining)
8. Hyperparameter Sensitivity Analysis
"""

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import itertools
from analysis_stats import (load_yolo_boxes, load_inference_boxes,
                                    calculate_iou, BoundingBox)


class EnsembleAnalyzer:
    """
    CONTRIBUTION #5: Ensemble Methods
    Combines predictions from multiple models using various strategies
    """

    def __init__(self, gt_folder: str, inference_root: str, image_folder: str,
                 models: List[str], output_dir: str, iou_threshold: float = 0.5):
        self.gt_folder = gt_folder
        self.inference_root = inference_root
        self.image_folder = image_folder
        self.models = models
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ensemble_predictions'), exist_ok=True)

    def load_all_predictions(self, image_name: str, img_width: int, img_height: int) -> Dict[str, List[BoundingBox]]:
        """Load predictions from all models for a single image"""
        all_preds = {}

        for model in self.models:
            inf_path = os.path.join(self.inference_root, model, f"{image_name}.txt")
            boxes = load_inference_boxes(inf_path, img_width, img_height)
            all_preds[model] = boxes

        return all_preds

    def nms_ensemble(self, all_boxes: List[Tuple[str, BoundingBox]],
                     iou_threshold: float = 0.5) -> List[BoundingBox]:
        """
        Non-Maximum Suppression across all model predictions
        Keeps the box with highest confidence among overlapping boxes
        """
        if len(all_boxes) == 0:
            return []

        # Sort by confidence
        all_boxes = sorted(all_boxes, key=lambda x: x[1].conf, reverse=True)

        keep = []

        while len(all_boxes) > 0:
            # Take the box with highest confidence
            current_model, current_box = all_boxes.pop(0)
            keep.append(current_box)

            # Remove all boxes that overlap significantly
            filtered = []
            for model, box in all_boxes:
                if calculate_iou(current_box, box) < iou_threshold:
                    filtered.append((model, box))

            all_boxes = filtered

        return keep

    def weighted_box_fusion(self, all_boxes: List[Tuple[str, BoundingBox]],
                            weights: Optional[Dict[str, float]] = None,
                            iou_threshold: float = 0.5) -> List[BoundingBox]:
        """
        Weighted Box Fusion - fuses overlapping boxes from different models
        """
        if weights is None:
            weights = {model: 1.0 for model in self.models}

        if len(all_boxes) == 0:
            return []

        # Group overlapping boxes
        clusters = []
        used = set()

        for i, (model_i, box_i) in enumerate(all_boxes):
            if i in used:
                continue

            cluster = [(model_i, box_i)]
            used.add(i)

            for j, (model_j, box_j) in enumerate(all_boxes):
                if j in used:
                    continue

                # Check if overlaps with any box in cluster
                overlaps = False
                for _, box_c in cluster:
                    if calculate_iou(box_j, box_c) >= iou_threshold:
                        overlaps = True
                        break

                if overlaps:
                    cluster.append((model_j, box_j))
                    used.add(j)

            clusters.append(cluster)

        # Fuse each cluster
        fused_boxes = []

        for cluster in clusters:
            if len(cluster) == 0:
                continue

            # Weighted average of coordinates
            total_weight = sum(weights.get(model, 1.0) * box.conf for model, box in cluster)

            if total_weight == 0:
                continue

            x_center = sum(weights.get(model, 1.0) * box.conf * box.x_center
                           for model, box in cluster) / total_weight
            y_center = sum(weights.get(model, 1.0) * box.conf * box.y_center
                           for model, box in cluster) / total_weight
            width = sum(weights.get(model, 1.0) * box.conf * box.width
                        for model, box in cluster) / total_weight
            height = sum(weights.get(model, 1.0) * box.conf * box.height
                         for model, box in cluster) / total_weight

            # Average confidence weighted by model weights
            conf = sum(weights.get(model, 1.0) * box.conf for model, box in cluster) / len(cluster)

            # Use most common class
            classes = [box.cls for _, box in cluster]
            cls = max(set(classes), key=classes.count)

            fused_box = BoundingBox(cls, x_center, y_center, width, height, conf)
            fused_boxes.append(fused_box)

        return fused_boxes

    def voting_ensemble(self, all_boxes: List[Tuple[str, BoundingBox]],
                        min_votes: int = 2,
                        iou_threshold: float = 0.5) -> List[BoundingBox]:
        """
        Voting ensemble - only keeps boxes that appear in at least min_votes models
        """
        if len(all_boxes) == 0:
            return []

        # Group overlapping boxes and count votes
        clusters = []
        used = set()

        for i, (model_i, box_i) in enumerate(all_boxes):
            if i in used:
                continue

            cluster = [(model_i, box_i)]
            models_in_cluster = {model_i}
            used.add(i)

            for j, (model_j, box_j) in enumerate(all_boxes):
                if j in used or model_j in models_in_cluster:
                    continue

                # Check if overlaps with any box in cluster
                overlaps = False
                for _, box_c in cluster:
                    if calculate_iou(box_j, box_c) >= iou_threshold:
                        overlaps = True
                        break

                if overlaps:
                    cluster.append((model_j, box_j))
                    models_in_cluster.add(model_j)
                    used.add(j)

            clusters.append((cluster, len(models_in_cluster)))

        # Keep clusters with enough votes
        kept_boxes = []

        for cluster, votes in clusters:
            if votes >= min_votes:
                # Average the coordinates
                x_center = np.mean([box.x_center for _, box in cluster])
                y_center = np.mean([box.y_center for _, box in cluster])
                width = np.mean([box.width for _, box in cluster])
                height = np.mean([box.height for _, box in cluster])
                conf = np.mean([box.conf for _, box in cluster])

                # Use most common class
                classes = [box.cls for _, box in cluster]
                cls = max(set(classes), key=classes.count)

                kept_boxes.append(BoundingBox(cls, x_center, y_center, width, height, conf))

        return kept_boxes

    def evaluate_ensemble(self, ensemble_method: str,
                          ensemble_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Evaluate a specific ensemble method on all images
        """
        if ensemble_params is None:
            ensemble_params = {}

        image_files = [f.replace('.txt', '') for f in os.listdir(self.gt_folder)
                       if f.endswith('.txt')]

        print(f"Evaluating {ensemble_method} ensemble on {len(image_files)} images...")

        # Metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Save ensemble predictions
        ensemble_pred_dir = os.path.join(self.output_dir, 'ensemble_predictions', ensemble_method)
        os.makedirs(ensemble_pred_dir, exist_ok=True)

        for img_name in image_files:
            # Get image dimensions
            image_path = os.path.join(self.image_folder, f"{img_name}.jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_folder, f"{img_name}.png")

            if os.path.exists(image_path):
                import cv2
                img = cv2.imread(image_path)
                if img is not None:
                    img_height, img_width = img.shape[:2]
                else:
                    img_width, img_height = 640, 640
            else:
                img_width, img_height = 640, 640

            # Load ground truth
            gt_path = os.path.join(self.gt_folder, f"{img_name}.txt")
            gt_boxes = load_yolo_boxes(gt_path)

            # Load all model predictions
            all_preds = self.load_all_predictions(img_name, img_width, img_height)

            # Flatten predictions with model names
            all_boxes = []
            for model, boxes in all_preds.items():
                for box in boxes:
                    all_boxes.append((model, box))

            # Apply ensemble method
            if ensemble_method == 'nms':
                ensemble_boxes = self.nms_ensemble(all_boxes,
                                                   ensemble_params.get('iou_threshold', 0.5))
            elif ensemble_method == 'wbf':
                ensemble_boxes = self.weighted_box_fusion(all_boxes,
                                                          ensemble_params.get('weights'),
                                                          ensemble_params.get('iou_threshold', 0.5))
            elif ensemble_method == 'voting':
                ensemble_boxes = self.voting_ensemble(all_boxes,
                                                      ensemble_params.get('min_votes', 2),
                                                      ensemble_params.get('iou_threshold', 0.5))
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")

            # Save ensemble predictions
            with open(os.path.join(ensemble_pred_dir, f"{img_name}.txt"), 'w') as f:
                for box in ensemble_boxes:
                    f.write(f"{box.cls} {box.x_center:.6f} {box.y_center:.6f} "
                            f"{box.width:.6f} {box.height:.6f} {box.conf:.6f}\n")

            # Evaluate
            matched_gt = set()
            matched_pred = set()

            # Find true positives
            for i, pred_box in enumerate(ensemble_boxes):
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    if gt_box.cls != pred_box.cls:
                        continue

                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= self.iou_threshold and best_gt_idx != -1:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
                else:
                    total_fp += 1

            # False negatives
            total_fn += len(gt_boxes) - len(matched_gt)

        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'ensemble_method': ensemble_method,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }

        # Add ensemble parameters
        results.update({f'param_{k}': v for k, v in ensemble_params.items()})

        return pd.DataFrame([results])

    def compare_all_ensembles(self, model_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Compare all ensemble methods
        """
        all_results = []

        # NMS ensemble
        print("\nEvaluating NMS ensemble...")
        results = self.evaluate_ensemble('nms', {'iou_threshold': 0.5})
        all_results.append(results)

        # Weighted Box Fusion with different configurations
        print("\nEvaluating Weighted Box Fusion...")
        for iou_thresh in [0.5, 0.6]:
            results = self.evaluate_ensemble('wbf', {
                'iou_threshold': iou_thresh,
                'weights': model_weights
            })
            all_results.append(results)

        # Voting ensemble with different vote thresholds
        print("\nEvaluating Voting ensemble...")
        for min_votes in [2, 3, 4]:
            if min_votes <= len(self.models):
                results = self.evaluate_ensemble('voting', {
                    'min_votes': min_votes,
                    'iou_threshold': 0.5
                })
                all_results.append(results)

        df = pd.concat(all_results, ignore_index=True)

        output_path = os.path.join(self.output_dir, 'ensemble_comparison.csv')
        df.to_csv(output_path, index=False)
        print(f"\nEnsemble comparison saved to: {output_path}")

        return df

    def plot_ensemble_comparison(self, df: pd.DataFrame, individual_results: pd.DataFrame):
        """
        Compare ensemble methods against individual models
        """
        # Combine individual and ensemble results
        # Map the correct column names from results.csv
        individual_summary = individual_results.groupby('model').agg({
            'box_mean_precision': 'mean',
            'box_mean_recall': 'mean',
            'box_mean_f1': 'mean'
        }).rename(columns={
            'box_mean_precision': 'precision',
            'box_mean_recall': 'recall',
            'box_mean_f1': 'f1'
        })
        individual_summary['method_type'] = 'Individual Model'
        individual_summary['method'] = individual_summary.index

        ensemble_summary = df[['ensemble_method', 'precision', 'recall', 'f1']].copy()
        ensemble_summary['method_type'] = 'Ensemble'
        ensemble_summary['method'] = ensemble_summary['ensemble_method']

        # Plot F1 scores
        plt.figure(figsize=(14, 6))

        # Individual models
        x_pos = np.arange(len(individual_summary))
        plt.bar(x_pos, individual_summary['f1'], alpha=0.7, label='Individual Models', color='skyblue')

        # Ensemble methods
        x_pos_ensemble = np.arange(len(ensemble_summary)) + len(individual_summary) + 0.5
        plt.bar(x_pos_ensemble, ensemble_summary['f1'], alpha=0.7, label='Ensemble Methods', color='lightcoral')

        # Set all x-tick positions and labels
        all_positions = list(x_pos) + list(x_pos_ensemble)
        all_labels = list(individual_summary.index.str.upper()) + list(ensemble_summary['ensemble_method'])

        plt.xticks(all_positions, all_labels, rotation=45, ha='right')
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Individual Models vs Ensemble Methods', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ensemble_vs_individual.png'), dpi=300)
        plt.close()

        # Plot precision-recall tradeoff
        plt.figure(figsize=(10, 8))

        # Individual models
        colors_individual = plt.cm.Set3(np.linspace(0, 1, len(individual_summary)))
        for idx, (model, row) in enumerate(individual_summary.iterrows()):
            plt.scatter(row['recall'], row['precision'],
                        s=150, alpha=0.6, label=model.upper(), marker='o',
                        color=colors_individual[idx])
            plt.annotate(model.upper(), (row['recall'], row['precision']),
                         fontsize=8, ha='right', va='bottom')

        # Ensemble methods
        colors_ensemble = plt.cm.Set1(np.linspace(0, 1, len(ensemble_summary)))
        for idx, (_, row) in enumerate(ensemble_summary.iterrows()):
            plt.scatter(row['recall'], row['precision'],
                        s=200, alpha=0.7, marker='^',
                        color=colors_ensemble[idx], edgecolors='black', linewidths=2)
            plt.annotate(row['ensemble_method'], (row['recall'], row['precision']),
                         fontsize=9, ha='left', va='bottom', fontweight='bold')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall: Individual vs Ensemble\n(Triangles = Ensemble, Circles = Individual)',
                  fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add legend for individual models only (ensembles are annotated)
        plt.legend(loc='lower left', fontsize=8, ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_ensemble.png'), dpi=300)
        plt.close()

        print(f"Ensemble comparison plots saved to: {self.output_dir}")


class HyperparameterAnalyzer:
    """
    CONTRIBUTION #8: Hyperparameter Sensitivity Analysis
    Analyzes how different hyperparameters affect model performance
    """

    def __init__(self, results_csv_path: str, output_dir: str):
        self.results_csv_path = results_csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.df = pd.read_csv(results_csv_path)

    def analyze_conf_threshold_sensitivity(self, gt_folder: str, inference_root: str,
                                           image_folder: str, models: List[str],
                                           conf_thresholds: List[float] = None) -> pd.DataFrame:
        """
        Analyze how confidence threshold affects performance
        """
        if conf_thresholds is None:
            conf_thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

        image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                       if f.endswith('.txt')]

        all_results = []

        print(f"Analyzing confidence threshold sensitivity...")

        for model in models:
            print(f"  Processing model: {model}")

            for conf_thresh in conf_thresholds:
                tp, fp, fn = 0, 0, 0

                for img_name in image_files:
                    # Get image dimensions
                    image_path = os.path.join(image_folder, f"{img_name}.jpg")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(image_folder, f"{img_name}.png")

                    if os.path.exists(image_path):
                        import cv2
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

                    # Filter by confidence threshold
                    pred_boxes = [box for box in pred_boxes if box.conf >= conf_thresh]

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
                            tp += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            fp += 1

                    fn += len(gt_boxes) - len(matched_gt)

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                all_results.append({
                    'model': model,
                    'conf_threshold': conf_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                })

        df = pd.DataFrame(all_results)

        output_path = os.path.join(self.output_dir, 'conf_threshold_sensitivity.csv')
        df.to_csv(output_path, index=False)
        print(f"Confidence threshold sensitivity saved to: {output_path}")

        return df

    def analyze_iou_threshold_sensitivity(self, gt_folder: str, inference_root: str,
                                          image_folder: str, models: List[str],
                                          iou_thresholds: List[float] = None) -> pd.DataFrame:
        """
        Analyze how IoU threshold for matching affects performance metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

        image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                       if f.endswith('.txt')]

        all_results = []

        print(f"Analyzing IoU threshold sensitivity...")

        for model in models:
            print(f"  Processing model: {model}")

            for iou_thresh in iou_thresholds:
                tp, fp, fn = 0, 0, 0

                for img_name in image_files:
                    # Get image dimensions
                    image_path = os.path.join(image_folder, f"{img_name}.jpg")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(image_folder, f"{img_name}.png")

                    if os.path.exists(image_path):
                        import cv2
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
                            tp += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            fp += 1

                    fn += len(gt_boxes) - len(matched_gt)

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                all_results.append({
                    'model': model,
                    'iou_threshold': iou_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        df = pd.DataFrame(all_results)

        output_path = os.path.join(self.output_dir, 'iou_threshold_sensitivity.csv')
        df.to_csv(output_path, index=False)
        print(f"IoU threshold sensitivity saved to: {output_path}")

        return df

    def plot_threshold_sensitivity(self, df_conf: pd.DataFrame, df_iou: pd.DataFrame):
        """
        Plot how thresholds affect performance
        """
        # Plot 1: Confidence threshold sensitivity
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for metric, ax in zip(['precision', 'recall', 'f1'], axes):
            for model in df_conf['model'].unique():
                model_data = df_conf[df_conf['model'] == model]
                ax.plot(model_data['conf_threshold'], model_data[metric],
                        marker='o', label=model.upper(), linewidth=2)

            ax.set_xlabel('Confidence Threshold', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} vs Confidence Threshold',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'conf_threshold_sensitivity.png'), dpi=300)
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
            ax.set_title(f'{metric.capitalize()} vs IoU Threshold',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'iou_threshold_sensitivity.png'), dpi=300)
        plt.close()

        print(f"Threshold sensitivity plots saved to: {self.output_dir}")

    def analyze_image_resolution_effect(self, results_by_resolution: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze effect of image resolution on performance
        Note: This requires results from training at different resolutions

        Args:
            results_by_resolution: Dict mapping resolution (e.g., 640) to results DataFrame
        """
        all_results = []

        for resolution, df in results_by_resolution.items():
            df_copy = df.copy()
            df_copy['resolution'] = resolution
            all_results.append(df_copy)

        combined_df = pd.concat(all_results, ignore_index=True)

        output_path = os.path.join(self.output_dir, 'resolution_sensitivity.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Resolution sensitivity saved to: {output_path}")

        return combined_df


def create_comprehensive_tables_for_paper(size_df: pd.DataFrame,
                                          ensemble_df: pd.DataFrame,
                                          failure_df: pd.DataFrame,
                                          stat_df: pd.DataFrame,
                                          output_dir: str):
    """
    Create publication-ready LaTeX tables for the paper
    """
    os.makedirs(output_dir, exist_ok=True)

    # Table 1: Performance by Size Category
    table1 = size_df.pivot_table(
        index='model',
        columns='size_category',
        values='f1',
        aggfunc='mean'
    )
    table1 = table1[['tiny', 'small', 'medium', 'large']]

    with open(os.path.join(output_dir, 'table_size_category.tex'), 'w') as f:
        f.write(table1.to_latex(float_format='%.3f', caption='F1 Score by Object Size Category',
                                label='tab:size_category'))

    # Table 2: Best Ensemble vs Best Individual
    best_individual = stat_df.loc[stat_df['box_mAP@50'].idxmax()]
    best_ensemble = ensemble_df.loc[ensemble_df['f1'].idxmax()]

    comparison_data = {
        'Method': [best_individual['model'].upper(), best_ensemble['ensemble_method']],
        'Type': ['Individual', 'Ensemble'],
        'Precision': [best_individual['box_mean_precision'], best_ensemble['precision']],
        'Recall': [best_individual['box_mean_recall'], best_ensemble['recall']],
        'F1': [best_individual['box_mean_f1'], best_ensemble['f1']]
    }
    table2 = pd.DataFrame(comparison_data)

    with open(os.path.join(output_dir, 'table_best_methods.tex'), 'w') as f:
        f.write(table2.to_latex(index=False, float_format='%.3f',
                                caption='Best Individual Model vs Best Ensemble',
                                label='tab:best_methods'))

    # Table 3: Failure Mode Summary
    failure_cols = [col for col in failure_df.columns
                    if col in ['missed_detection', 'background_fp', 'boundary_error', 'duplicate_detection']]

    if len(failure_cols) > 0:
        table3 = failure_df[['model'] + failure_cols].set_index('model')

        with open(os.path.join(output_dir, 'table_failure_modes.tex'), 'w') as f:
            f.write(table3.to_latex(float_format='%.0f',
                                    caption='Failure Mode Distribution by Model',
                                    label='tab:failure_modes'))

    print(f"LaTeX tables saved to: {output_dir}")

