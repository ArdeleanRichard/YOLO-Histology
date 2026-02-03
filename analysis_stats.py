import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BoundingBox:
    """Represents a bounding box with normalized coordinates"""
    cls: int
    x_center: float
    y_center: float
    width: float
    height: float
    conf: float = 1.0

    def area(self) -> float:
        """Calculate normalized area"""
        return self.width * self.height

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to x1, y1, x2, y2 format"""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return x1, y1, x2, y2



def load_yolo_boxes(label_path: str) -> List[BoundingBox]:
    """Load YOLO format boxes from a label file"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                boxes.append(BoundingBox(cls, x, y, w, h, conf))

    return boxes


def load_inference_boxes(inference_path: str, image_width: int = 640, image_height: int = 640) -> List[BoundingBox]:
    """
    Load boxes from inference file and normalize pixel coordinates

    Args:
        inference_path: Path to inference txt file
        image_width: Width of the image (default 640)
        image_height: Height of the image (default 640)
    """
    boxes = []
    if not os.path.exists(inference_path):
        return boxes

    with open(inference_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Handle comma-separated format (from _save_inference_ints)
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    cls = int(parts[0])
                    x_pixel = float(parts[1])
                    y_pixel = float(parts[2])
                    w_pixel = float(parts[3])
                    h_pixel = float(parts[4])
                    conf = float(parts[5])


                    # Normalize to 0-1 range
                    x_norm = x_pixel / image_width
                    y_norm = y_pixel / image_height
                    w_norm = w_pixel / image_width
                    h_norm = h_pixel / image_height


                    boxes.append(BoundingBox(cls, x_norm, y_norm, w_norm, h_norm, conf))
            else:
                # Handle space-separated normalized format (standard YOLO)
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    boxes.append(BoundingBox(cls, x, y, w, h, conf))

    return boxes


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1.to_xyxy()
    x1_2, y1_2, x2_2, y2_2 = box2.to_xyxy()

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = box1.area()
    area2 = box2.area()
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class ObjectSizeAnalyzer:
    """
    CONTRIBUTION #1: Error Analysis by Object Size Categories
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

    def analyze_single_image(self, image_name: str, model: str) -> Dict:
        """Analyze detections for a single image and model"""
        gt_path = os.path.join(self.gt_folder, f"{image_name}.txt")
        inf_path = os.path.join(self.inference_root, model, f"{image_name}.txt")


        img_height, img_width = None, None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            image_path = os.path.join(self.image_folder, f"{image_name}{ext}")
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

        # Categorize ground truth boxes by size
        gt_by_size = defaultdict(list)
        for box in gt_boxes:
            category = self.categorize_by_size(box)
            gt_by_size[category].append(box)

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        results = {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in self.size_categories.keys()}

        # Find true positives and false positives
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            best_category = None

            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if gt_box.cls != pred_box.cls:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    best_category = self.categorize_by_size(gt_box)

            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                # True positive
                results[best_category]['tp'] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                # False positive - categorize by predicted box size
                pred_category = self.categorize_by_size(pred_box)
                results[pred_category]['fp'] += 1

        # False negatives - unmatched ground truth
        for j, gt_box in enumerate(gt_boxes):
            if j not in matched_gt:
                category = self.categorize_by_size(gt_box)
                results[category]['fn'] += 1

        return results

    def analyze_all(self) -> pd.DataFrame:
        """Analyze all images for all models"""
        all_results = []

        # Get list of images from ground truth folder
        image_files = [f.replace('.txt', '') for f in os.listdir(self.gt_folder)
                      if f.endswith('.txt')]

        print(f"Analyzing {len(image_files)} images for {len(self.models)} models...")

        for model in self.models:
            print(f"  Processing model: {model}")
            model_results = {cat: {'tp': 0, 'fp': 0, 'fn': 0}
                           for cat in self.size_categories.keys()}

            for img_name in image_files:
                img_results = self.analyze_single_image(img_name, model)

                # Aggregate results
                for cat in self.size_categories.keys():
                    for metric in ['tp', 'fp', 'fn']:
                        model_results[cat][metric] += img_results[cat][metric]

            # Calculate metrics per size category
            for category in self.size_categories.keys():
                tp = model_results[category]['tp']
                fp = model_results[category]['fp']
                fn = model_results[category]['fn']

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                all_results.append({
                    'model': model,
                    'size_category': category,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        df = pd.DataFrame(all_results)

        # Save results
        output_path = os.path.join(self.output_dir, 'size_category_analysis.csv')
        df.to_csv(output_path, index=False)
        print(f"Size category analysis saved to: {output_path}")

        return df

    def plot_results(self, df: pd.DataFrame):
        """Create visualizations for size category analysis"""
        # Plot 1: F1 score by size category
        plt.figure(figsize=(12, 6))
        pivot_f1 = df.pivot(index='model', columns='size_category', values='f1')
        pivot_f1 = pivot_f1[['tiny', 'small', 'medium', 'large']]  # Order categories

        ax = pivot_f1.plot(kind='bar', figsize=(12, 6))
        plt.title('F1 Score by Object Size Category', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.legend(title='Size Category', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_size_category.png'), dpi=300)
        plt.close()

        # Plot 2: Precision vs Recall by size for each model
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, model in enumerate(sorted(df['model'].unique())):
            if idx >= len(axes):
                break

            model_data = df[df['model'] == model]

            for _, row in model_data.iterrows():
                axes[idx].scatter(row['recall'], row['precision'],
                                s=100, label=row['size_category'], alpha=0.7)

            axes[idx].set_xlabel('Recall', fontsize=10)
            axes[idx].set_ylabel('Precision', fontsize=10)
            axes[idx].set_title(model.upper(), fontsize=11, fontweight='bold')
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(alpha=0.3)
            axes[idx].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_by_size.png'), dpi=300)
        plt.close()

        # Plot 3: Heatmap of F1 scores
        plt.figure(figsize=(10, 6))
        pivot_f1 = df.pivot(index='model', columns='size_category', values='f1')
        pivot_f1 = pivot_f1[['tiny', 'small', 'medium', 'large']]

        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'F1 Score'})
        plt.title('F1 Score Heatmap by Model and Size Category',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Size Category', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_heatmap.png'), dpi=300)
        plt.close()

        print(f"Plots saved to: {self.output_dir}")


class StatisticalAnalyzer:
    """
    CONTRIBUTION #2: Statistical Significance Testing
    """

    def __init__(self, results_csv_path: str, output_dir: str):
        self.results_csv_path = results_csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.df = pd.read_csv(results_csv_path)

    def compute_confidence_intervals(self, metric: str = 'box_mAP@50',
                                   confidence: float = 0.95) -> pd.DataFrame:
        """
        Compute confidence intervals using bootstrap
        Note: This requires per-image results. If only aggregate results available,
        we'll use theoretical standard errors.
        """
        results = []

        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]

            for dataset in model_data['dataset'].unique():
                dataset_model = model_data[model_data['dataset'] == dataset]

                if len(dataset_model) > 0:
                    mean_val = dataset_model[metric].mean()

                    # Since we only have aggregate results, compute theoretical CI
                    # assuming normal distribution (conservative estimate)
                    # For proper CI, would need per-image results
                    n = len(dataset_model)
                    std_val = dataset_model[metric].std() if n > 1 else 0

                    # Standard error
                    se = std_val / np.sqrt(n) if n > 1 else 0

                    # Critical value for confidence interval
                    z = stats.norm.ppf((1 + confidence) / 2)

                    ci_lower = mean_val - z * se
                    ci_upper = mean_val + z * se

                    results.append({
                        'model': model,
                        'dataset': dataset,
                        'metric': metric,
                        'mean': mean_val,
                        'std': std_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_upper - ci_lower
                    })

        df_ci = pd.DataFrame(results)

        output_path = os.path.join(self.output_dir, f'confidence_intervals_{metric}.csv')
        df_ci.to_csv(output_path, index=False)
        print(f"Confidence intervals saved to: {output_path}")

        return df_ci

    def pairwise_statistical_tests(self, metric: str = 'box_mAP@50') -> pd.DataFrame:
        """
        Perform pairwise statistical tests between models
        Uses Wilcoxon signed-rank test (non-parametric) for paired comparisons across datasets
        For single dataset, performs simple comparison with effect size
        """
        models = sorted(self.df['model'].unique())
        datasets = sorted(self.df['dataset'].unique())

        results = []

        # Check if we have multiple datasets
        if len(datasets) < 3:
            print(f"Warning: Only {len(datasets)} dataset(s) found. Statistical tests require at least 3 datasets.")
            print("Performing simple pairwise comparisons with effect sizes instead.")

            # For single/few datasets, just compute differences and effect sizes
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:
                        continue

                    val1 = self.df[self.df['model'] == model1][metric].mean()
                    val2 = self.df[self.df['model'] == model2][metric].mean()

                    diff = val1 - val2
                    rel_diff = (diff / val2 * 100) if val2 != 0 else 0

                    results.append({
                        'model_a': model1,
                        'model_b': model2,
                        'metric': metric,
                        'model_a_mean': val1,
                        'model_b_mean': val2,
                        'absolute_diff': diff,
                        'relative_diff_pct': rel_diff,
                        'better_model': model1 if diff > 0 else model2,
                        'note': 'Insufficient datasets for statistical testing'
                    })
        else:
            # Multiple datasets - use Wilcoxon test
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:
                        continue

                    values1 = []
                    values2 = []

                    for dataset in datasets:
                        val1 = self.df[(self.df['model'] == model1) & (self.df['dataset'] == dataset)][metric].values
                        val2 = self.df[(self.df['model'] == model2) & (self.df['dataset'] == dataset)][metric].values

                        if len(val1) > 0 and len(val2) > 0:
                            values1.append(val1[0])
                            values2.append(val2[0])

                    if len(values1) >= 3:
                        try:
                            statistic, p_value = wilcoxon(values1, values2, alternative='two-sided')
                            mean_diff = np.mean(np.array(values1) - np.array(values2))

                            if p_value < 0.001:
                                significance = '***'
                            elif p_value < 0.01:
                                significance = '**'
                            elif p_value < 0.05:
                                significance = '*'
                            else:
                                significance = 'ns'

                            results.append({
                                'model_a': model1,
                                'model_b': model2,
                                'metric': metric,
                                'mean_diff': mean_diff,
                                'p_value': p_value,
                                'significance': significance,
                                'better_model': model1 if mean_diff > 0 else model2
                            })
                        except Exception as e:
                            print(f"Could not perform test for {model1} vs {model2}: {e}")

        df_tests = pd.DataFrame(results)

        if len(df_tests) > 0:
            output_path = os.path.join(self.output_dir, f'pairwise_comparison_{metric}.csv')
            df_tests.to_csv(output_path, index=False)
            print(f"Pairwise comparisons saved to: {output_path}")

        return df_tests

    def plot_confidence_intervals(self, df_ci: pd.DataFrame, metric: str = 'box_mAP@50'):
        """Plot confidence intervals for each model across datasets"""
        datasets = sorted(df_ci['dataset'].unique())

        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 6))
        if len(datasets) == 1:
            axes = [axes]

        for idx, dataset in enumerate(datasets):
            data = df_ci[df_ci['dataset'] == dataset].sort_values('mean', ascending=False)

            y_pos = np.arange(len(data))
            axes[idx].barh(y_pos, data['mean'], xerr=data['ci_width']/2,
                          alpha=0.7, capsize=5)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(data['model'].str.upper())
            axes[idx].set_xlabel(metric, fontsize=11)
            axes[idx].set_title(f'{dataset}', fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'confidence_intervals_{metric}.png'), dpi=300)
        plt.close()
        print(f"CI plot saved to: {self.output_dir}")

    def create_significance_heatmap(self, df_tests: pd.DataFrame, metric: str = 'box_mAP@50'):
        """Create heatmap showing statistical significance between models"""
        if len(df_tests) == 0:
            print("No pairwise comparisons available for heatmap")
            return

        # Determine which columns we have
        if 'model_a' in df_tests.columns:
            model_col_a = 'model_a'
            model_col_b = 'model_b'
        elif 'model1' in df_tests.columns:
            model_col_a = 'model1'
            model_col_b = 'model2'
        else:
            print("Error: Cannot find model columns in comparison results")
            return

        models = sorted(set(df_tests[model_col_a].unique()) | set(df_tests[model_col_b].unique()))

        # Create matrix for values
        n = len(models)
        value_matrix = np.zeros((n, n))
        text_matrix = np.full((n, n), '', dtype=object)

        model_to_idx = {m: i for i, m in enumerate(models)}

        # Fill matrices
        for _, row in df_tests.iterrows():
            i = model_to_idx[row[model_col_a]]
            j = model_to_idx[row[model_col_b]]

            # For statistical tests
            if 'p_value' in df_tests.columns and 'significance' in df_tests.columns:
                p_val = row['p_value']
                sig = row['significance']

                value_matrix[i, j] = -np.log10(p_val + 1e-10)
                value_matrix[j, i] = -np.log10(p_val + 1e-10)
                text_matrix[i, j] = sig
                text_matrix[j, i] = sig
            # For simple comparisons
            elif 'absolute_diff' in df_tests.columns:
                diff = abs(row['absolute_diff'])
                value_matrix[i, j] = diff
                value_matrix[j, i] = diff
                # Show which is better
                better = row['better_model']
                if better == row[model_col_a]:
                    text_matrix[i, j] = '▲'
                    text_matrix[j, i] = '▼'
                else:
                    text_matrix[i, j] = '▼'
                    text_matrix[j, i] = '▲'

        # Plot heatmap
        plt.figure(figsize=(10, 8))

        if 'p_value' in df_tests.columns:
            # Statistical test version
            sns.heatmap(value_matrix, annot=text_matrix, fmt='',
                       xticklabels=[m.upper() for m in models],
                       yticklabels=[m.upper() for m in models],
                       cmap='RdYlGn_r', cbar_kws={'label': '-log10(p-value)'})
            title = f'Statistical Significance Between Models ({metric})'
        else:
            # Simple comparison version
            sns.heatmap(value_matrix, annot=text_matrix, fmt='',
                       xticklabels=[m.upper() for m in models],
                       yticklabels=[m.upper() for m in models],
                       cmap='RdYlGn', cbar_kws={'label': 'Absolute Difference'})
            title = f'Pairwise Performance Comparison ({metric})\n▲=better, ▼=worse'

        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'comparison_heatmap_{metric}.png'), dpi=300)
        plt.close()
        print(f"Comparison heatmap saved to: {self.output_dir}")


class FailureModeAnalyzer:
    """
    CONTRIBUTION #3: Failure Mode Analysis
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

        # Define failure mode categories
        self.failure_modes = {
            'missed_detection': 'Object not detected (FN)',
            'background_fp': 'False positive in background',
            'boundary_error': 'Poor localization (low IoU)',
            'class_confusion': 'Wrong class predicted',
            'duplicate_detection': 'Multiple boxes for same object',
            'split_detection': 'Object split into multiple boxes'
        }

    def analyze_image(self, image_name: str, model: str) -> Dict:
        """Analyze failure modes for a single image"""
        gt_path = os.path.join(self.gt_folder, f"{image_name}.txt")
        inf_path = os.path.join(self.inference_root, model, f"{image_name}.txt")

        # Get image dimensions
        image_path = os.path.join(self.image_folder, f"{image_name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_folder, f"{image_name}.png")

        if os.path.exists(image_path):
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
            else:
                img_width, img_height = 640, 640
        else:
            img_width, img_height = 640, 640

        gt_boxes = load_yolo_boxes(gt_path)
        pred_boxes = load_inference_boxes(inf_path, img_width, img_height)

        failures = defaultdict(int)

        # Track matches
        matched_gt = {}  # gt_idx -> list of (pred_idx, iou)
        matched_pred = set()

        # Find all matches above a low threshold to detect duplicates
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)

                if iou > 0.1:  # Low threshold to catch all potential matches
                    if j not in matched_gt:
                        matched_gt[j] = []
                    matched_gt[j].append((i, iou, pred_box.cls == gt_box.cls))

        # Analyze each ground truth box
        for j, gt_box in enumerate(gt_boxes):
            if j not in matched_gt or len(matched_gt[j]) == 0:
                # Missed detection
                failures['missed_detection'] += 1
            else:
                matches = matched_gt[j]
                best_match = max(matches, key=lambda x: x[1])
                best_iou = best_match[1]
                correct_class = best_match[2]

                if best_iou < self.iou_threshold:
                    # Boundary error
                    failures['boundary_error'] += 1
                elif not correct_class:
                    # Class confusion
                    failures['class_confusion'] += 1

                # Check for duplicate detections
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

    def analyze_all(self) -> pd.DataFrame:
        """Analyze failure modes for all images and models"""
        all_results = []

        image_files = [f.replace('.txt', '') for f in os.listdir(self.gt_folder)
                      if f.endswith('.txt')]

        print(f"Analyzing failure modes for {len(image_files)} images and {len(self.models)} models...")

        for model in self.models:
            print(f"  Processing model: {model}")

            model_failures = defaultdict(int)

            for img_name in image_files:
                img_failures = self.analyze_image(img_name, model)

                for mode, count in img_failures.items():
                    model_failures[mode] += count

            # Add results
            result = {'model': model}
            result.update(model_failures)

            # Calculate percentages
            total_failures = sum(model_failures.values())
            for mode in self.failure_modes.keys():
                count = model_failures.get(mode, 0)
                result[f'{mode}_pct'] = (count / total_failures * 100) if total_failures > 0 else 0

            all_results.append(result)

        df = pd.DataFrame(all_results)

        output_path = os.path.join(self.output_dir, 'failure_mode_analysis.csv')
        df.to_csv(output_path, index=False)
        print(f"Failure mode analysis saved to: {output_path}")

        return df

    def plot_failure_modes(self, df: pd.DataFrame):
        """Visualize failure mode distributions"""
        # Plot 1: Stacked bar chart of failure counts
        failure_cols = [col for col in df.columns if col in self.failure_modes.keys()]

        if len(failure_cols) > 0:
            plt.figure(figsize=(12, 6))
            df_plot = df.set_index('model')[failure_cols]
            df_plot.plot(kind='bar', stacked=True, figsize=(12, 6),
                        colormap='Set3', edgecolor='black', linewidth=0.5)

            plt.title('Failure Mode Distribution by Model', fontsize=14, fontweight='bold')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Number of Failures', fontsize=12)
            plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'failure_modes_stacked.png'), dpi=300)
            plt.close()

        # Plot 2: Percentage breakdown
        pct_cols = [col for col in df.columns if col.endswith('_pct')]

        if len(pct_cols) > 0:
            plt.figure(figsize=(12, 6))
            df_pct = df.set_index('model')[pct_cols]
            df_pct.columns = [col.replace('_pct', '') for col in df_pct.columns]

            df_pct.plot(kind='bar', stacked=True, figsize=(12, 6),
                       colormap='Set3', edgecolor='black', linewidth=0.5)

            plt.title('Failure Mode Percentage by Model', fontsize=14, fontweight='bold')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Percentage of Failures (%)', fontsize=12)
            plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'failure_modes_percentage.png'), dpi=300)
            plt.close()

        # Plot 3: Heatmap
        if len(failure_cols) > 0:
            plt.figure(figsize=(10, 6))
            df_heatmap = df.set_index('model')[failure_cols]

            sns.heatmap(df_heatmap, annot=True, fmt='.0f', cmap='YlOrRd',
                       cbar_kws={'label': 'Number of Failures'})

            plt.title('Failure Mode Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Failure Mode', fontsize=12)
            plt.ylabel('Model', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'failure_modes_heatmap.png'), dpi=300)
            plt.close()

        print(f"Failure mode plots saved to: {self.output_dir}")
