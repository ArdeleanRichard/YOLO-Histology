import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter


class ClusterRefinement:
    """
    Cluster-based detection refinement using feature clustering.

    Core idea:
    1. Select high-confidence anchors (top percentile per class)
    2. Extract features using EfficientNet
    3. Per-class clustering: Remove outliers based on intra-cluster distance
    4. Cross-class clustering: Remove minority classes from impure clusters
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize EfficientNet-B3 feature extractor (better than ResNet50)"""
        efficientnet = models.efficientnet_b3(pretrained=True)
        # Remove classification head, keep feature extractor
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_pil):
        """Extract deep features from image crop using EfficientNet-B3"""
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(image_tensor).squeeze().cpu()
        return features

    def load_all_detections(self, txt_path, img_path):
        """Load ALL detections from file (no threshold filtering)."""
        if not os.path.exists(txt_path):
            return []

        img = Image.open(img_path).convert('RGB')
        detections = []

        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) != 6:
                    continue

                class_id, x, y, w, h, score = [float(p.strip()) for p in parts]

                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                if x2 <= x1 + 2 or y2 <= y1 + 2:
                    continue

                crop = img.crop((x1, y1, x2, y2))
                features = self.extract_features(crop)

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'bbox_orig': (x, y, w, h),
                    'score': score,
                    'class': int(class_id),
                    'feature': features.numpy()
                }
                detections.append(detection)

        return detections

    def get_class_anchors(self, detections, top_percentile=90):
        """
        Extract anchors (high-confidence boxes) per class.

        Args:
            detections: List of all detections
            top_percentile: Percentile for anchor selection

        Returns:
            tuple: (class_detections, class_anchors)
                - class_detections: dict {class_id: [all_detections]}
                - class_anchors: dict {class_id: [anchor_detections]}
        """
        # Group by class
        class_detections = {}
        for det in detections:
            cls = det['class']
            if cls not in class_detections:
                class_detections[cls] = []
            class_detections[cls].append(det)

        # Select anchors per class
        class_anchors = {}
        for cls, dets in class_detections.items():
            scores = np.array([d['score'] for d in dets])
            threshold = np.percentile(scores, top_percentile)
            anchors = [d for d in dets if d['score'] >= threshold]
            class_anchors[cls] = anchors

        print(f"\n=== Anchor Selection (top {100 - top_percentile}% per class) ===")
        for cls, anchors in class_anchors.items():
            print(f"Class {cls}: {len(anchors)} anchors from {len(class_detections[cls])} detections")

        return class_detections, class_anchors

    def per_class_clustering(self, class_detections, class_anchors, k=10):
        """
        Run K-Means clustering per class on ALL detections and remove outliers.
        Remove entire clusters that contain no anchors.

        For remaining clusters, remove boxes whose minimum distance to other boxes
        in the cluster is larger than the mean distance.

        Args:
            class_detections: dict {class_id: [all_detections]}
            class_anchors: dict {class_id: [anchor_detections]}
            k: Number of clusters

        Returns:
            dict: {class_id: [refined_detections]}
        """
        refined_detections = {}

        print(f"\n=== Per-Class Clustering (k={k}) ===")

        for cls, detections in class_detections.items():
            anchors = class_anchors[cls]
            anchor_ids = set(id(a) for a in anchors)

            if len(detections) < k:
                print(f"Class {cls}: Too few detections ({len(detections)}), keeping all")
                refined_detections[cls] = detections
                continue

            # Extract features from ALL detections
            features = np.stack([d['feature'] for d in detections])

            # K-Means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)

            # Process each cluster
            kept_indices = []
            removed_no_anchor = 0
            removed_outliers = 0

            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]

                # Check if cluster contains any anchors
                cluster_has_anchor = any(id(detections[i]) in anchor_ids for i in cluster_indices)

                if not cluster_has_anchor:
                    # Remove entire cluster (no anchors)
                    removed_no_anchor += len(cluster_indices)
                    continue

                if len(cluster_indices) <= 1:
                    kept_indices.extend(cluster_indices.tolist())
                    continue

                # Get features for this cluster
                cluster_features = features[cluster_indices]

                # Compute pairwise distances within cluster
                distances = pairwise_distances(cluster_features, metric='euclidean')

                # For each box, find minimum distance to another box in cluster
                np.fill_diagonal(distances, np.inf)
                min_distances = np.min(distances, axis=1)

                # Compute mean minimum distance
                mean_min_dist = np.mean(min_distances)

                # anchors_in_cluster_mask = [id(detections[i]) in anchor_ids for i in cluster_indices]
                # anchor_positions = np.where(np.array(anchors_in_cluster_mask))[0]
                #
                # anchor_features = cluster_features[anchor_positions]
                #
                # dist_to_anchors = pairwise_distances(cluster_features, anchor_features, metric='euclidean')
                # min_dist_to_anchor = np.min(dist_to_anchors, axis=1)

                # Keep boxes with min_distance <= mean
                for i, idx in enumerate(cluster_indices):
                    if min_distances[i] <= mean_min_dist: # and min_dist_to_anchor[i] <= mean_min_dist :
                        kept_indices.append(idx)
                    else:
                        removed_outliers += 1

            refined_detections[cls] = [detections[i] for i in kept_indices]

            print(f"Class {cls}: {len(detections)} -> {len(refined_detections[cls])} "
                  f"(removed {removed_no_anchor} from clusters without anchors, "
                  f"{removed_outliers} outliers)")

        return refined_detections

    def cross_class_clustering(self, class_detections, k=10):
        """
        Run K-Means across all classes on ALL detections to find impure clusters.
        Remove minority classes from impure clusters.

        Args:
            class_detections: dict {class_id: [all_detections]}
            k: Number of clusters

        Returns:
            dict: {class_id: [refined_detections]}
        """
        # Flatten all detections
        all_detections = []
        for cls, detections in class_detections.items():
            all_detections.extend(detections)

        if len(all_detections) < k:
            print(f"\n=== Cross-Class Clustering: Skipped (too few detections) ===")
            return class_detections

        print(f"\n=== Cross-Class Clustering (k={k}) ===")

        # Extract features and class labels
        features = np.stack([d['feature'] for d in all_detections])
        classes = np.array([d['class'] for d in all_detections])

        # K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Identify impure clusters and remove minority classes
        indices_to_keep = set(range(len(all_detections)))

        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_classes = classes[cluster_mask]

            # Count classes in cluster
            class_counts = Counter(cluster_classes)

            if len(class_counts) > 1:
                # Impure cluster - keep only majority class
                majority_class = class_counts.most_common(1)[0][0]
                minority_count = 0

                for i in cluster_indices:
                    if classes[i] != majority_class:
                        indices_to_keep.discard(i)
                        minority_count += 1

                print(f"Cluster {cluster_id}: Impure - kept class {majority_class}, "
                      f"removed {minority_count} minority boxes")

        # Reconstruct class_detections with kept indices
        kept_detections = [all_detections[i] for i in sorted(indices_to_keep)]

        refined_detections = {}
        for detection in kept_detections:
            cls = detection['class']
            if cls not in refined_detections:
                refined_detections[cls] = []
            refined_detections[cls].append(detection)

        total_removed = len(all_detections) - len(kept_detections)
        print(f"Total: {len(all_detections)} -> {len(kept_detections)} (removed {total_removed})")

        return refined_detections

    def run(self, image_dir, infer_dir, out_dir, top_percentile=90, k_per_class=10, k_cross_class=10):
        """
        Run cluster-based refinement.

        Args:
            image_dir: Directory with images
            infer_dir: Directory with detection .txt files
            out_dir: Output directory
            top_percentile: Percentile for anchor selection per class
            k_per_class: Number of clusters for per-class clustering
            k_cross_class: Number of clusters for cross-class clustering
        """
        os.makedirs(out_dir, exist_ok=True)

        print("=" * 70)
        print("CLUSTER-BASED DETECTION REFINEMENT")
        print("=" * 70)
        print(f"Image dir: {image_dir}")
        print(f"Inference dir: {infer_dir}")
        print(f"Output dir: {out_dir}")

        # Get all detection files
        txt_files = [f for f in os.listdir(infer_dir) if f.endswith('.txt')]

        if not txt_files:
            print(f"No .txt files found in {infer_dir}")
            return

        print(f"\nProcessing {len(txt_files)} files...")

        # Load ALL detections from all images
        all_detections = []
        image_names = []

        for txt_file in tqdm(txt_files, desc="Loading detections"):
            img_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(infer_dir, txt_file)

            # Find image
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                test_path = os.path.join(image_dir, img_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            if img_path is None:
                continue

            detections = self.load_all_detections(txt_path, img_path)

            for det in detections:
                det['img_name'] = img_name

            all_detections.extend(detections)
            if detections:
                image_names.append(img_name)

        if len(all_detections) == 0:
            print("No detections found!")
            return

        # Count classes
        unique_classes = set(d['class'] for d in all_detections)
        print(f"\n=== Dataset Statistics ===")
        print(f"Total detections: {len(all_detections)}")
        print(f"Number of classes: {len(unique_classes)}")
        print(f"Classes: {sorted(unique_classes)}")

        # Step 1: Get both all class detections and anchors per class
        class_detections, class_anchors = self.get_class_anchors(all_detections, top_percentile)

        # Step 2: Per-class clustering on ALL detections (removes clusters without anchors + outliers)
        refined_detections = self.per_class_clustering(class_detections, class_anchors, k=k_per_class)

        # Step 3: Cross-class clustering on ALL remaining detections (only if multiple classes)
        if len(unique_classes) > 1:
            refined_detections = self.cross_class_clustering(refined_detections, k=k_cross_class)
        else:
            print("\n=== Cross-Class Clustering: Skipped (single class) ===")

        # Flatten final detections
        final_detections = []
        for cls, detections in refined_detections.items():
            final_detections.extend(detections)

        print(f"\n=== Final Results ===")
        print(f"Original detections: {len(all_detections)}")
        print(f"Final detections: {len(final_detections)}")
        print(f"Kept: {len(final_detections)} ({100 * len(final_detections) / len(all_detections):.1f}%)")
        print(f"Removed: {len(all_detections) - len(final_detections)} "
              f"({100 * (len(all_detections) - len(final_detections)) / len(all_detections):.1f}%)")

        # Save results
        self._save_results(final_detections, image_names, out_dir)

        # Save analysis
        self._save_analysis(all_detections, class_anchors, final_detections, out_dir)

        return {
            'all_detections': all_detections,
            'class_anchors': class_anchors,
            'final_detections': final_detections
        }

    def _save_results(self, detections, image_names, out_dir):
        """Save detection results"""
        # Group by image
        image_dets = {name: [] for name in image_names}

        for det in detections:
            img_name = det.get('img_name')
            if img_name in image_dets:
                image_dets[img_name].append(det)

        # Write files
        for img_name, dets in image_dets.items():
            out_path = os.path.join(out_dir, f"{img_name}.txt")
            with open(out_path, 'w') as f:
                for det in dets:
                    x, y, w, h = det['bbox_orig']
                    score = det['score']
                    class_id = det['class']
                    f.write(f"{class_id}, {x}, {y}, {w}, {h}, {score}\n")

        print(f"\nResults saved to: {out_dir}")

    def _save_analysis(self, all_dets, class_anchors, final_dets, out_dir):
        """Save analysis plots and statistics"""
        analysis_dir = os.path.join(out_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Get all anchors
        initial_anchors = []
        for cls, anchors in class_anchors.items():
            initial_anchors.extend(anchors)

        plt.figure(figsize=(12, 5))

        # Score distribution
        plt.subplot(1, 2, 1)
        scores_all = [d['score'] for d in all_dets]
        scores_anchors = [d['score'] for d in initial_anchors]
        scores_final = [d['score'] for d in final_dets]

        plt.hist(scores_all, bins=50, alpha=0.3, label='All', color='gray')
        plt.hist(scores_anchors, bins=50, alpha=0.5, label='Anchors', color='blue')
        plt.hist(scores_final, bins=50, alpha=0.5, label='Final (Refined)', color='green')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Detection Score Distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        # Count comparison
        plt.subplot(1, 2, 2)
        categories = ['All\nDetections', 'Anchors', 'Final\n(Refined)']
        counts = [len(all_dets), len(initial_anchors), len(final_dets)]
        colors = ['gray', 'blue', 'green']
        plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Detection Counts')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'distribution.png'), dpi=150)
        plt.close()

        print(f"Analysis saved to: {analysis_dir}")


def run_cluster_refinement(model_name, data_root, results_inf_root,
                           top_percentile=90, k_per_class=10, k_cross_class=10):
    """
    Run cluster-based detection refinement.

    Args:
        model_name: Model name
        data_root: Root data directory
        results_inf_root: Root results directory
        top_percentile: Percentile for anchor selection per class (default: 90 = top 10%)
        k_per_class: Number of clusters for per-class clustering
        k_cross_class: Number of clusters for cross-class clustering
    """
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_root}/{model_name}/"
    out_dir = f"{results_inf_root}/{model_name}/cluster_refined2/"

    refiner = ClusterRefinement()
    results = refiner.run(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        top_percentile=top_percentile,
        k_per_class=k_per_class,
        k_cross_class=k_cross_class
    )

    return results


if __name__ == "__main__":
    from constants import results_inf_root, data_root, MODEL

    results = run_cluster_refinement(
        model_name=MODEL,
        data_root=data_root,
        results_inf_root=results_inf_root,
        top_percentile=90,  # Top 10% as anchors per class
        k_per_class=10,  # 10 clusters per class
        k_cross_class=10  # 10 clusters across all classes
    )