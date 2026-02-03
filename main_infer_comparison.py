import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA


class DetectionEvaluator:
    """
    Evaluate object detection performance using COCO-style metrics.
    Compatible with Ultralytics format and methodology.
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def load_labels_ground_truth(self, label_dir, img_width=None, img_height=None):
        """
        Load ground truth labels.

        Format: class_id x y w h (space-separated, normalized floats)

        Args:
            label_dir: Directory with label files
            img_width: Image width for denormalization (required if labels are normalized)
            img_height: Image height for denormalization (required if labels are normalized)

        Returns dict: {img_name: [(class_id, x, y, w, h), ...]}
        """
        labels = {}
        label_files = os.listdir(label_dir)

        for label_file in label_files:
            if not label_file.endswith('.txt'):
                continue

            img_name = os.path.splitext(label_file)[0]

            label_path = os.path.join(label_dir, label_file)
            bboxes = []

            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Space-separated floats
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = [float(p) for p in parts[1:5]]

                        # Check if normalized (values between 0 and 1)
                        if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                            # Normalized coordinates - need to denormalize
                            if img_width is None or img_height is None:
                                # Try to get image dimensions
                                img_dims = self._get_image_dimensions(img_name)
                                if img_dims:
                                    img_width, img_height = img_dims
                                else:
                                    print(f"Warning: Cannot denormalize {img_name} - using normalized coords")
                                    bboxes.append((class_id, x, y, w, h))
                                    continue

                            # Convert from normalized to pixel coordinates
                            # YOLO format: center_x, center_y, width, height
                            x = x * img_width
                            y = y * img_height
                            w = w * img_width
                            h = h * img_height

                        bboxes.append((class_id, x, y, w, h))

            labels[img_name] = bboxes

        return labels

    def _get_image_dimensions(self, img_name):
        """Try to get image dimensions from corresponding image file."""
        # Go up from label dir to find images

        img_dir = os.path.join(data_root, 'images/test')
        if os.path.exists(img_dir):
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = os.path.join(img_dir, img_name + ext)
                if os.path.exists(img_path):
                    try:
                        from PIL import Image
                        img = Image.open(img_path)
                        return img.size  # (width, height)
                    except:
                        pass
        return None

    def load_labels_predictions(self, pred_dir, img_names=None):
        """
        Load predictions.

        Format: class_id, x, y, w, h, score (comma-separated ints/floats)
        Note: x, y, w, h are in pixel coordinates (integers)

        Returns dict: {img_name: [(class_id, x, y, w, h, score), ...]}
        """
        predictions = {}
        pred_files = os.listdir(pred_dir)

        for pred_file in pred_files:
            if not pred_file.endswith('.txt'):
                continue

            img_name = os.path.splitext(pred_file)[0]

            # If specific image names provided, filter
            if img_names is not None and img_name not in img_names:
                continue

            pred_path = os.path.join(pred_dir, pred_file)
            bboxes = []

            with open(pred_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Comma-separated values
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        score = float(parts[5])
                        bboxes.append((class_id, x, y, w, h, score))

            predictions[img_name] = bboxes

        return predictions

    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes.

        Box formats:
        - Ground truth: (class, x_center, y_center, w, h) - all in pixels
        - Predictions: (class, x_center, y_center, w, h, score) - all in pixels

        Both use center coordinates (YOLO format).
        """
        # Extract coordinates (skip class and score if present)
        if len(box1) == 4:
            x1, y1, w1, h1 = box1
        elif len(box1) == 5:
            _, x1, y1, w1, h1 = box1
        else:
            _, x1, y1, w1, h1 = box1[:5]

        if len(box2) == 4:
            x2, y2, w2, h2 = box2
        elif len(box2) == 5:
            _, x2, y2, w2, h2 = box2
        else:
            _, x2, y2, w2, h2 = box2[:5]

        # Convert from center coordinates to corner coordinates
        # YOLO format: (center_x, center_y, width, height)
        # Need: (x1, y1, x2, y2)
        box1_x1 = x1 - w1 / 2
        box1_y1 = y1 - h1 / 2
        box1_x2 = x1 + w1 / 2
        box1_y2 = y1 + h1 / 2

        box2_x1 = x2 - w2 / 2
        box2_y1 = y2 - h2 / 2
        box2_x2 = x2 + w2 / 2
        box2_y2 = y2 + h2 / 2

        # Intersection
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def match_predictions_to_labels(self, image_name, predictions, labels, iou_threshold):
        """
        Match predictions to ground truth labels using IoU threshold.

        Returns:
            tp: List of (score, iou) for true positives
            fp: List of (score,) for false positives
            fn_count: Number of false negatives (unmatched ground truths)
        """
        tp = []
        fp = []

        # Sort predictions by score (descending)
        predictions = np.array(predictions)
        preds_argsorted = predictions[:, 5].argsort()[::-1]

        preds_sorted = predictions[preds_argsorted]

        # Track which labels have been matched
        matched_labels = set()

        if DEBUG:
            labs = []
            alps = []
        for id, pred in enumerate(preds_sorted):
            pred_class, pred_x, pred_y, pred_w, pred_h, pred_score = pred

            best_iou = 0.0
            best_label_idx = -1

            # Find best matching label
            for idx, label in enumerate(labels):
                if idx in matched_labels:
                    continue

                label_class = label[0]

                # Only match same class
                if pred_class != label_class:
                    continue

                iou = self.compute_iou(pred, label)

                if iou > best_iou:
                    best_iou = iou
                    best_label_idx = idx

            # Check if match is good enough
            if best_iou >= iou_threshold:
                tp.append((pred_score, best_iou))
                matched_labels.add(best_label_idx)
                if DEBUG:
                    labs.append(1)
                    alps.append(pred_score)
            else:
                fp.append((pred_score,))
                if DEBUG:
                    alps.append(pred_score)
                    labs.append(0)

        if DEBUG:
            features = np.load(f"{results_inf_all_root}/{MODEL}/cluster_refined2/features/{image_name}.npy", allow_pickle=True)

            features = features[preds_argsorted]

            # pca = PCA(n_components=2)
            # X = pca.fit_transform(features)

            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42
            )

            X = reducer.fit_transform(features)

            alps = np.array(alps)
            labs = np.array(labs)
            color_dict = {0: 'red', 1: "green"}
            plt.scatter(X[:, 0], X[:, 1], c=[color_dict[x] for x in labs], alpha=((alps / max(alps)) + 3) / 4, edgecolors="k")
            plt.figure()
            plt.hist(alps[labs == 0], bins=100, alpha=0.6, label='FP')
            plt.hist(alps[labs == 1], bins=100, alpha=0.6, label='TP')
            plt.legend()
            plt.show()

        # False negatives are unmatched labels
        fn_count = len(labels) - len(matched_labels)

        return tp, fp, fn_count

    def compute_precision_recall_curve(self, tp_list, fp_list, total_gt):
        """
        Compute precision-recall curve from matched predictions.

        Args:
            tp_list: List of (score, iou) tuples for true positives
            fp_list: List of (score,) tuples for false positives
            total_gt: Total number of ground truth boxes

        Returns:
            precisions, recalls, scores
        """
        # Combine and sort by score
        all_dets = []
        for score, iou in tp_list:
            all_dets.append((score, 1, iou))  # 1 = TP
        for score, in fp_list:
            all_dets.append((score, 0, 0.0))  # 0 = FP

        all_dets.sort(key=lambda x: x[0], reverse=True)

        precisions = []
        recalls = []
        scores = []

        tp_cumsum = 0
        fp_cumsum = 0

        for score, is_tp, iou in all_dets:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0.0
            recall = tp_cumsum / total_gt if total_gt > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            scores.append(score)

        return np.array(precisions), np.array(recalls), np.array(scores)

    # def compute_ap(self, precisions, recalls):
    #     """
    #     Compute Average Precision using 101-point interpolation (COCO style).
    #     """
    #     # Add sentinel values
    #     recalls = np.concatenate(([0.0], recalls, [1.0]))
    #     precisions = np.concatenate(([0.0], precisions, [0.0]))
    #
    #     # Compute the precision envelope
    #     for i in range(len(precisions) - 1, 0, -1):
    #         precisions[i - 1] = max(precisions[i - 1], precisions[i])
    #
    #     # Integrate using 101-point interpolation
    #     recall_thresholds = np.linspace(0, 1, 101)
    #     ap = 0.0
    #
    #     for r_thresh in recall_thresholds:
    #         # Find precisions where recall >= r_thresh
    #         precs = precisions[recalls >= r_thresh]
    #         ap += precs[0] if len(precs) > 0 else 0.0
    #
    #     ap /= 101
    #
    #     return ap

    def evaluate(self, labels_gt, predictions, iou_thresholds=None):
        """
        Evaluate predictions against labels.

        Args:
            labels_gt: Dict of {img_name: [(class, x, y, w, h), ...]}
            predictions: Dict of {img_name: [(class, x, y, w, h, score), ...]}
            iou_thresholds: List of IoU thresholds (default: [0.5])

        Returns:
            metrics: Dictionary of metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]

        # Ensure all images are accounted for
        all_img_names = set(labels_gt.keys()) | set(predictions.keys())

        results = {}

        for iou_thresh in iou_thresholds:
            tp_all = []
            fp_all = []
            total_gt = 0

            for img_name in all_img_names:
                img_labels = labels_gt.get(img_name, [])
                img_preds = predictions.get(img_name, [])

                total_gt += len(img_labels)

                if len(img_preds) == 0:
                    continue

                # print(img_name)
                tp, fp, fn = self.match_predictions_to_labels(img_name, img_preds, img_labels, iou_thresh)

                tp_all.extend(tp)
                fp_all.extend(fp)

            # Compute precision-recall curve
            if total_gt == 0:
                results[iou_thresh] = {
                    # 'ap': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'total_gt': 0,
                    'total_pred': 0
                }
                continue

            precisions, recalls, scores = self.compute_precision_recall_curve(tp_all, fp_all, total_gt)

            # # Compute AP
            # ap = self.compute_ap(precisions, recalls)

            # Compute metrics directly from actual TP/FP/FN counts
            # This is correct for fixed threshold evaluation
            tp_count = len(tp_all)
            fp_count = len(fp_all)
            fn_count = total_gt - tp_count

            actual_precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            actual_recall = tp_count / total_gt if total_gt > 0 else 0.0
            actual_f1 = 2 * (actual_precision * actual_recall) / (actual_precision + actual_recall) if (actual_precision + actual_recall) > 0 else 0.0

            results[iou_thresh] = {
                # 'ap': ap,
                'precision': actual_precision,
                'recall': actual_recall,
                'f1': actual_f1,
                'tp': tp_count,
                'fp': fp_count,
                'fn': fn_count,
                'total_gt': total_gt,
                'total_pred': tp_count + fp_count,
                'precisions': precisions,
                'recalls': recalls,
                'scores': scores
            }

        return results

    def compute_map(self, labels, predictions):
        """
        Compute mAP@0.5 and mAP@0.5:0.95 (COCO-style).

        Returns:
            Dictionary with map50, map50_95, and other metrics
        """
        # mAP@0.5
        results_50 = self.evaluate(labels, predictions, iou_thresholds=[0.5])
        # map50 = results_50[0.5]['ap']
        #
        # # mAP@0.5:0.95 (10 IoU thresholds)
        # iou_thresholds = np.linspace(0.5, 0.95, 10)
        # results_range = self.evaluate(labels, predictions, iou_thresholds=iou_thresholds)
        #
        # aps = [results_range[thresh]['ap'] for thresh in iou_thresholds]
        # map50_95 = np.mean(aps)

        # Use metrics from IoU=0.5 for precision, recall, F1
        metrics = {
            # 'map50': map50,
            # 'map50_95': map50_95,
            'precision': results_50[0.5]['precision'],
            'recall': results_50[0.5]['recall'],
            'f1': results_50[0.5]['f1'],
            'tp': results_50[0.5]['tp'],
            'fp': results_50[0.5]['fp'],
            'fn': results_50[0.5]['fn'],
            'total_gt': results_50[0.5]['total_gt'],
            'total_pred': results_50[0.5]['total_pred'],
        }

        return metrics, results_50[0.5]


def apply_confidence_threshold(predictions, threshold):
    """
    Apply confidence threshold to predictions.

    Args:
        predictions: Dict of {img_name: [(class, x, y, w, h, score), ...]}
        threshold: Confidence threshold

    Returns:
        Filtered predictions dict
    """
    filtered = {}
    for img_name, preds in predictions.items():
        filtered[img_name] = [p for p in preds if p[5] >= threshold]
    return filtered


def compare_methods(label_dir, pred_dirs, method_names, output_dir=None):
    """
    Compare multiple detection methods.

    Args:
        label_dir: Directory with ground truth labels
        pred_dirs: List of prediction directories
        method_names: List of method names
        output_dir: Directory to save comparison results

    Returns:
        DataFrame with comparison results
    """
    evaluator = DetectionEvaluator()

    # Load ground truth
    print("Loading ground truth labels...")
    labels = evaluator.load_labels_ground_truth(label_dir)
    total_gt = sum(len(v) for v in labels.values())
    print(f"Loaded {len(labels)} images with {total_gt} ground truth boxes")

    results_list = []
    detailed_results = {}

    for method_name, pred_dir in zip(method_names, pred_dirs):
        print(f"\nEvaluating: {method_name}")
        # print(f"\t Prediction dir: {pred_dir}")

        # Load predictions
        predictions = evaluator.load_labels_predictions(pred_dir)
        total_pred = sum(len(v) for v in predictions.values())
        print(f"Loaded {total_pred} predictions")

        # Compute metrics
        metrics, detailed = evaluator.compute_map(labels, predictions)

        # Add method name
        metrics['method'] = method_name
        metrics['pred_dir'] = pred_dir

        results_list.append(metrics)
        detailed_results[method_name] = detailed

    # Create comparison DataFrame
    df = pd.DataFrame(results_list)

    # Reorder columns
    # cols = ['method', 'map50', 'map50_95', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'total_gt', 'total_pred']
    cols = ['method', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'total_gt', 'total_pred']
    df = df[cols]

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(output_dir, 'comparison_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Create comparison plots
        create_comparison_plots(df, detailed_results, output_dir)

    return df, detailed_results


def create_comparison_plots(df, detailed_results, output_dir):
    """Create visualization plots for method comparison."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Method Comparison', fontsize=16, fontweight='bold')

    # metrics = ['map50', 'map50_95', 'f1', 'precision', 'recall']
    metrics = ['f1', 'precision', 'recall']
    colors = plt.cm.Set3(range(len(df)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['method'], rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Detection counts
    ax = axes[1, 2]
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df['tp'], width, label='TP', color='green', alpha=0.7)
    ax.bar(x, df['fp'], width, label='FP', color='red', alpha=0.7)
    ax.bar(x + width, df['fn'], width, label='FN', color='orange', alpha=0.7)

    ax.set_ylabel('Count', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Detection Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall curves
    plt.figure(figsize=(10, 8))

    for method_name, details in detailed_results.items():
        if 'precisions' in details and 'recalls' in details:
            plt.plot(details['recalls'], details['precisions'], marker='', linewidth=2, label=f"{method_name}")

    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves (IoU=0.5)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {output_dir}")


def run_full_comparison(data_root, results_inf_all_root, model_name, thresholds=[0.3, 0.5, 0.7]):
    """
    Run full comparison: Hard thresholds vs TSBP vs SimProp.

    Args:
        data_root: Root data directory
        results_inf_all_root: Root results directory
        model_name: Model name
        thresholds: List of confidence thresholds to test

    Returns:
        DataFrame with comparison results
    """
    label_dir = f"{data_root}/labels/test/"
    base_pred_dir = f"{results_inf_all_root}/{model_name}/"
    output_dir = f"{results_inf_all_root}/{model_name}/comparison/"

    print("=" * 80)
    print("FULL METHOD COMPARISON")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Ground truth: {label_dir}")
    print(f"Predictions base: {base_pred_dir}")
    print("=" * 80)

    # Prepare prediction directories and method names
    pred_dirs = []
    method_names = []

    # 1. Hard thresholds
    evaluator = DetectionEvaluator()
    base_predictions = evaluator.load_labels_predictions(base_pred_dir)

    for thresh in thresholds:
        # Create temporary directory with thresholded predictions
        thresh_dir = f"{results_inf_all_root}/{model_name}/threshold_{thresh}/"
        os.makedirs(thresh_dir, exist_ok=True)

        # Apply threshold and save
        filtered_preds = apply_confidence_threshold(base_predictions, thresh)

        if len(filtered_preds.items()) == 0:
            continue

        for img_name, preds in filtered_preds.items():
            out_path = os.path.join(thresh_dir, f"{img_name}.txt")
            with open(out_path, 'w') as f:
                for pred in preds:
                    class_id, x, y, w, h, score = pred
                    f.write(f"{class_id}, {x}, {y}, {w}, {h}, {score}\n")

        pred_dirs.append(thresh_dir)
        method_names.append(f"Threshold {thresh}")

    # 2. TSBP (if exists)
    tsbp_dir = f"{results_inf_all_root}/{model_name}/tsbp/"
    if os.path.exists(tsbp_dir) and os.listdir(tsbp_dir):
        pred_dirs.append(tsbp_dir)
        method_names.append("TSBP")


    tsbp_dir = f"{results_inf_all_root}/{model_name}/tsbp_adaptive/"
    if os.path.exists(tsbp_dir) and os.listdir(tsbp_dir):
        pred_dirs.append(tsbp_dir)
        method_names.append("TSBP_A")

    tsbp_dir = f"{results_inf_all_root}/{model_name}/tsbp_hierarchical/"
    if os.path.exists(tsbp_dir) and os.listdir(tsbp_dir):
        pred_dirs.append(tsbp_dir)
        method_names.append("TSBP_H")

    tsbp_dir = f"{results_inf_all_root}/{model_name}/tsbp_plusplus/"
    if os.path.exists(tsbp_dir) and os.listdir(tsbp_dir):
        pred_dirs.append(tsbp_dir)
        method_names.append("TSBP_PP")


    # # 3. Similarity Propagation (if exists)
    # simprop = "simprop"
    # for simprop_type in ["_knn", "_density", "_voting"]:
    #     sim_dir = f"{results_inf_all_root}/{model_name}/{simprop}{simprop_type}/"
    #     if os.path.exists(sim_dir) and os.listdir(sim_dir):
    #         pred_dirs.append(sim_dir)
    #         method_names.append(f"{simprop}{simprop_type}")

    # 4
    clust_dir = f"{results_inf_all_root}/{model_name}/cluster_refined/"
    if os.path.exists(clust_dir) and os.listdir(clust_dir):
        pred_dirs.append(clust_dir)
        method_names.append("Clust")

    clust_dir = f"{results_inf_all_root}/{model_name}/cluster_refined2/"
    if os.path.exists(clust_dir) and os.listdir(clust_dir):
        pred_dirs.append(clust_dir)
        method_names.append("Clust2")

    # Run comparison
    df, detailed = compare_methods(label_dir, pred_dirs, method_names, output_dir)

    return df, detailed



if __name__ == "__main__":
    """
    File Structure:
        {data_root} /
            images / test /  # Test images
            labels / test /  # Ground truth (class, x, y, w, h)

        {results_inf_all_root} /
            {model_name} /                          # Base predictions (all boxes, no threshold)
            {model_name} / tsbp /                   # TSBP results
            {model_name} / similarity_prop /        # SimProp results
            {model_name} / comparison /             # Output comparison results
    """
    DEBUG = False

    # from constants import results_inf_all_root, data_root, MODEL
    #
    # df, detailed = run_full_comparison(
    #     data_root=data_root,
    #     results_inf_all_root=results_inf_all_root,
    #     model_name=MODEL,
    #     # thresholds=[0.00, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]  # Test multiple thresholds
    #     thresholds=[0.5],
    # )
    #
    # print(f"Results saved to: {results_inf_all_root}/{MODEL}/comparison/")


    from constants import results_inf_all_root, data_root, ALL_MODELS

    for MODEL in ALL_MODELS:
        df, detailed = run_full_comparison(
            data_root=data_root,
            results_inf_all_root=results_inf_all_root,
            model_name=MODEL,
            # thresholds=[0.00, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]  # Test multiple thresholds
            thresholds=[0.5],
        )

        print(f"Results saved to: {results_inf_all_root}/{MODEL}/comparison/")
