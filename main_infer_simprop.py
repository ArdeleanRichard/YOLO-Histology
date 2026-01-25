import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimProp:
    """
    Threshold-free detection refinement using similarity propagation.

    Core idea:
    1. High-confidence detections are assumed correct (anchors)
    2. Low-confidence detections similar to anchors are kept
    3. Low-confidence detections dissimilar to all anchors are removed
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize ResNet50 feature extractor"""
        resnet_50 = models.resnet50(pretrained=True)
        modules = list(resnet_50.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.resnet.to(self.device)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_pil):
        """Extract deep features from image crop using ResNet50"""
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet(image_tensor).squeeze().cpu()
        return features

    def load_all_detections(self, txt_path, img_path):
        """
        Load ALL detections from file (no threshold filtering).

        Format per line: class_id, x, y, w, h, score
        """
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

                # Convert to integer bbox coordinates
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                # Skip invalid boxes
                if x2 <= x1 + 2 or y2 <= y1 + 2:
                    continue

                # Crop and extract features
                crop = img.crop((x1, y1, x2, y2))
                features = self.extract_features(crop)

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'bbox_orig': (x, y, w, h),
                    'score': score,
                    'class': int(class_id),
                    'feature': features.numpy()  # Store as numpy for sklearn
                }
                detections.append(detection)

        return detections

    def analyze_confidence_distribution(self, all_detections):
        """Analyze the distribution of confidence scores"""
        scores = [d['score'] for d in all_detections]

        print("\n=== Confidence Score Distribution ===")
        print(f"Total detections: {len(scores)}")
        print(f"Min score: {min(scores):.4f}")
        print(f"Max score: {max(scores):.4f}")
        print(f"Mean score: {np.mean(scores):.4f}")
        print(f"Median score: {np.median(scores):.4f}")

        # Percentiles
        percentiles = [50, 60, 70, 80, 90, 95, 99]
        print("\nPercentiles:")
        for p in percentiles:
            val = np.percentile(scores, p)
            count = sum(s >= val for s in scores)
            print(f"  {p}th percentile: {val:.4f} ({count} detections above)")

        return scores

    def automatic_anchor_selection(self, all_detections, top_percentile=90):
        """
        Automatically select high-confidence anchors.

        Args:
            all_detections: List of all detections
            top_percentile: Percentile for anchor selection (default: top 10%)

        Returns:
            anchors: High-confidence detections
            candidates: Low-confidence detections to evaluate
        """
        scores = np.array([d['score'] for d in all_detections])
        threshold = np.percentile(scores, top_percentile)

        anchors = [d for d in all_detections if d['score'] >= threshold]
        candidates = [d for d in all_detections if d['score'] < threshold]

        print(f"\n=== Anchor Selection (top {100 - top_percentile}%) ===")
        print(f"Anchor threshold: {threshold:.4f}")
        print(f"Anchors: {len(anchors)}")
        print(f"Candidates: {len(candidates)}")

        return anchors, candidates, threshold

    def compute_similarity_statistics(self, anchors):
        """
        Compute inter-anchor similarity statistics to understand
        what "similar" means for this dataset.
        """
        if len(anchors) < 2:
            return None

        features = np.stack([a['feature'] for a in anchors])

        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features, metric='euclidean'))

        # Get nearest neighbor distances (exclude self)
        np.fill_diagonal(distances, np.inf)
        nn_distances = np.min(distances, axis=1)

        stats = {
            'mean': np.mean(nn_distances),
            'median': np.median(nn_distances),
            'std': np.std(nn_distances),
            'q25': np.percentile(nn_distances, 25),
            'q75': np.percentile(nn_distances, 75),
            'q95': np.percentile(nn_distances, 95)
        }

        print("\n=== Anchor Similarity Statistics ===")
        print(f"Mean NN distance: {stats['mean']:.2f}")
        print(f"Median NN distance: {stats['median']:.2f}")
        print(f"Std NN distance: {stats['std']:.2f}")
        print(f"25th percentile: {stats['q25']:.2f}")
        print(f"75th percentile: {stats['q75']:.2f}")
        print(f"95th percentile: {stats['q95']:.2f}")

        return stats

    def propagate_by_knn(self, anchors, candidates, k=5, distance_factor=1.5):
        """
        Propagate labels using K-Nearest Neighbors approach.

        Args:
            anchors: High-confidence detections
            candidates: Low-confidence detections to evaluate
            k: Number of nearest neighbors to consider
            distance_factor: Multiplier for adaptive threshold
                           (distance < factor * median_anchor_distance)

        Returns:
            accepted: Candidates that are similar to anchors
            rejected: Candidates that are dissimilar
        """
        if len(anchors) == 0:
            return [], candidates

        # Compute anchor similarity statistics
        stats = self.compute_similarity_statistics(anchors)

        # Adaptive distance threshold based on anchor distribution
        if stats:
            distance_threshold = stats['median'] * distance_factor
        else:
            distance_threshold = float('inf')

        print(f"\n=== KNN Propagation (k={k}) ===")
        print(f"Distance threshold: {distance_threshold:.2f}")

        # Build KNN index on anchors
        anchor_features = np.stack([a['feature'] for a in anchors])
        nbrs = NearestNeighbors(n_neighbors=min(k, len(anchors)),
                                metric='euclidean').fit(anchor_features)

        accepted = []
        rejected = []

        candidate_features = np.stack([c['feature'] for c in candidates])

        # Query for each candidate
        distances, indices = nbrs.kneighbors(candidate_features)

        for i, candidate in enumerate(candidates):
            # Mean distance to k nearest anchors
            mean_dist = np.mean(distances[i])
            min_dist = np.min(distances[i])

            candidate['nearest_anchor_dist'] = min_dist
            candidate['mean_knn_dist'] = mean_dist

            # Decision: accept if similar enough to anchors
            if min_dist < distance_threshold:
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        print(f"Accepted: {len(accepted)}")
        print(f"Rejected: {len(rejected)}")

        return accepted, rejected

    def propagate_by_density(self, anchors, candidates, sigma=1.0):
        """
        Propagate based on density estimation in feature space.

        Candidates in high-density regions (near many anchors) are accepted.

        Args:
            anchors: High-confidence detections
            candidates: Low-confidence detections
            sigma: Bandwidth for density estimation

        Returns:
            accepted: Candidates in dense regions
            rejected: Candidates in sparse regions
        """
        if len(anchors) == 0:
            return [], candidates

        anchor_features = np.stack([a['feature'] for a in anchors])

        print(f"\n=== Density-Based Propagation (sigma={sigma}) ===")

        # Compute density for each candidate
        accepted = []
        rejected = []
        densities = []

        for candidate in candidates:
            cand_feat = candidate['feature'].reshape(1, -1)

            # Compute density as sum of Gaussian kernels
            distances = np.linalg.norm(anchor_features - cand_feat, axis=1)
            density = np.sum(np.exp(-distances ** 2 / (2 * sigma ** 2)))

            candidate['density'] = density
            densities.append(density)

        # Adaptive threshold: median density
        density_threshold = np.median(densities)

        print(f"Density threshold (median): {density_threshold:.4f}")

        for candidate in candidates:
            if candidate['density'] >= density_threshold:
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        print(f"Accepted: {len(accepted)}")
        print(f"Rejected: {len(rejected)}")

        return accepted, rejected

    def propagate_by_voting(self, anchors, candidates, k=10, vote_threshold=0.5):
        """
        Propagate using majority voting from k nearest anchors.

        Args:
            anchors: High-confidence detections
            candidates: Low-confidence detections
            k: Number of nearest neighbors to vote
            vote_threshold: Fraction of votes needed to accept

        Returns:
            accepted: Candidates with enough votes
            rejected: Candidates without enough votes
        """
        if len(anchors) == 0:
            return [], candidates

        print(f"\n=== Voting-Based Propagation (k={k}, threshold={vote_threshold}) ===")

        # Compute anchor similarity stats
        stats = self.compute_similarity_statistics(anchors)

        anchor_features = np.stack([a['feature'] for a in anchors])
        nbrs = NearestNeighbors(n_neighbors=min(k, len(anchors)),
                                metric='euclidean').fit(anchor_features)

        accepted = []
        rejected = []

        candidate_features = np.stack([c['feature'] for c in candidates])
        distances, indices = nbrs.kneighbors(candidate_features)

        # Use anchor distribution to determine voting weights
        distance_scale = stats['median'] if stats else 1.0

        for i, candidate in enumerate(candidates):
            # Weight votes by inverse distance
            weights = np.exp(-distances[i] / distance_scale)
            normalized_weights = weights / np.sum(weights)

            # Vote score is weighted sum
            vote_score = np.sum(normalized_weights)

            candidate['vote_score'] = vote_score

            if vote_score >= vote_threshold:
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        print(f"Accepted: {len(accepted)}")
        print(f"Rejected: {len(rejected)}")

        return accepted, rejected

    def run(self, image_dir, infer_dir, out_dir,
            method='knn', top_percentile=90, **method_kwargs):
        """
        Run similarity-based propagation.

        Args:
            image_dir: Directory with images
            infer_dir: Directory with detection .txt files
            out_dir: Output directory
            method: 'knn', 'density', or 'voting'
            top_percentile: Percentile for anchor selection
            **method_kwargs: Additional arguments for propagation method
        """
        os.makedirs(out_dir, exist_ok=True)

        print("=" * 70)
        print("SIMILARITY-BASED DETECTION PROPAGATION")
        print("=" * 70)
        print(f"Method: {method}")
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

        # Analyze distribution
        self.analyze_confidence_distribution(all_detections)

        # Automatic anchor selection
        anchors, candidates, _ = self.automatic_anchor_selection(
            all_detections, top_percentile
        )

        # Propagate labels based on similarity
        if method == 'knn':
            accepted, rejected = self.propagate_by_knn(
                anchors, candidates, **method_kwargs
            )
        elif method == 'density':
            accepted, rejected = self.propagate_by_density(
                anchors, candidates, **method_kwargs
            )
        elif method == 'voting':
            accepted, rejected = self.propagate_by_voting(
                anchors, candidates, **method_kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Final detections = anchors + accepted candidates
        final_detections = anchors + accepted

        print(f"\n=== Final Results ===")
        print(f"Original detections: {len(all_detections)}")
        print(f"Final detections: {len(final_detections)}")
        print(f"Kept: {len(final_detections)} ({100 * len(final_detections) / len(all_detections):.1f}%)")
        print(f"Removed: {len(rejected)} ({100 * len(rejected) / len(all_detections):.1f}%)")

        # Save results
        self._save_results(final_detections, image_names, out_dir)

        # Save analysis
        self._save_analysis(all_detections, anchors, accepted, rejected, out_dir)

        return {
            'all_detections': all_detections,
            'anchors': anchors,
            'accepted': accepted,
            'rejected': rejected,
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

    def _save_analysis(self, all_dets, anchors, accepted, rejected, out_dir):
        """Save analysis plots and statistics"""
        analysis_dir = os.path.join(out_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Score distribution plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        scores_all = [d['score'] for d in all_dets]
        scores_anchors = [d['score'] for d in anchors]
        scores_accepted = [d['score'] for d in accepted]
        scores_rejected = [d['score'] for d in rejected]

        plt.hist(scores_all, bins=50, alpha=0.3, label='All', color='gray')
        plt.hist(scores_anchors, bins=50, alpha=0.5, label='Anchors', color='green')
        plt.hist(scores_accepted, bins=50, alpha=0.5, label='Accepted', color='blue')
        plt.hist(scores_rejected, bins=50, alpha=0.5, label='Rejected', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Detection Score Distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        categories = ['All', 'Anchors', 'Accepted', 'Rejected', 'Final']
        counts = [len(all_dets), len(anchors), len(accepted),
                  len(rejected), len(anchors) + len(accepted)]
        colors = ['gray', 'green', 'blue', 'red', 'purple']
        plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Detection Counts')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'distribution.png'), dpi=150)
        plt.close()

        print(f"Analysis saved to: {analysis_dir}")


def run_similarity_propagation(model_name, data_root, results_inf_root,
                               method='knn', top_percentile=90, **method_kwargs):
    """
    Run similarity-based propagation.

    Args:
        model_name: Model name
        data_root: Root data directory
        results_inf_root: Root results directory
        method: 'knn' (recommended), 'density', or 'voting'
        top_percentile: Percentile for anchor selection (default: 90 = top 10%)
        **method_kwargs: Method-specific parameters:
            - knn: k=5, distance_factor=1.5
            - density: sigma=1.0
            - voting: k=10, vote_threshold=0.5
    """
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_root}/{model_name}/"
    out_dir = f"{results_inf_root}/{model_name}/simprop_{method}/"

    detector = SimProp()
    results = detector.run(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        method=method,
        top_percentile=top_percentile,
        **method_kwargs
    )

    return results


if __name__ == "__main__":
    from constants import results_inf_root, data_root, MODEL

    # Method 1: KNN-based (Recommended - Simple and effective)
    results = run_similarity_propagation(
        model_name=MODEL,
        data_root=data_root,
        results_inf_root=results_inf_root,
        method='knn',
        top_percentile=90,  # Top 10% as anchors
        k=5,  # Consider 5 nearest neighbors
        distance_factor=1.5  # Accept if distance < 1.5 * median_anchor_distance
    )

    # Method 2: Density-based (More conservative)
    results = run_similarity_propagation(
        model_name=MODEL,
        data_root=data_root,
        results_inf_root=results_inf_root,
        method='density',
        top_percentile=90,
        sigma=1.0
    )

    # Method 3: Voting-based (Most sophisticated)
    results = run_similarity_propagation(
        model_name=MODEL,
        data_root=data_root,
        results_inf_root=results_inf_root,
        method='voting',
        top_percentile=90,
        k=10,
        vote_threshold=0.5
    )