import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import ot  # Python Optimal Transport: pip install POT
from torchvision.models import ResNet50_Weights
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')



class TS:
    def __init__(self):
        pass

    def load_detections(self, txt_file, image_dir, infer_dir, use_histogram=False):
        """Load detections from YOLO format text file"""
        img_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(infer_dir, txt_file)

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            test_path = os.path.join(image_dir, img_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path is None:
            print(f"Warning: Image not found for {img_name}")
            return []

        if not os.path.exists(txt_path):
            return []

        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
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
                features = self.extract_features(crop, use_histogram)

                # Calculate bbox center for spatial consistency
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'bbox_orig': (x, y, w, h),
                    'score': score,
                    'class': int(class_id),
                    'feature': features,
                    'center': np.array([center_x / img_width, center_y / img_height])  # normalized
                }
                detections.append(detection)

        return detections

    def compute_adaptive_thresholds(self, detections, tp_quantile=0.8, fp_quantile=0.2):
        """
        Compute per-class adaptive thresholds based on score distribution.

        Args:
            detections: List of detection dicts
            tp_quantile: Quantile for high-confidence threshold (0.8 = top 20%)
            fp_quantile: Quantile for low-confidence threshold (0.2 = bottom 20%)

        Returns:
            dict: {class_id: (tp_threshold, fp_threshold)}
        """
        class_scores = defaultdict(list)
        for det in detections:
            class_scores[det['class']].append(det['score'])

        thresholds = {}
        for class_id, scores in class_scores.items():
            scores = np.array(scores)
            tp_thresh = np.quantile(scores, tp_quantile)
            fp_thresh = np.quantile(scores, fp_quantile)
            thresholds[class_id] = (tp_thresh, fp_thresh)

        return thresholds

    def kmeans_clustering(self, detections, num_clusters):
        """Apply K-means clustering"""
        if len(detections) < num_clusters:
            return detections, [d['feature'] for d in detections]

        features = torch.stack([d['feature'] for d in detections])

        kmeans = KMeans(n_clusters=num_clusters, max_iter=500, n_init=10, random_state=1234)
        kmeans.fit(features.numpy())

        clustered_dets = []
        clustered_feats = []
        for i in range(num_clusters):
            det = {
                'bbox': (0, 0, 0, 0),
                'bbox_orig': (0, 0, 0, 0),
                'score': 1.0,
                'class': detections[0]['class'],
                'feature': torch.from_numpy(kmeans.cluster_centers_[i]),
                'center': np.array([0.5, 0.5]),
                'is_cluster_center': True
            }
            clustered_dets.append(det)
            clustered_feats.append(det['feature'])

        return clustered_dets, clustered_feats

    def cal_min_dist_stats(self, detections):
        """Calculate distance statistics"""
        if len(detections) < 2:
            return float('inf'), float('inf')

        features = torch.stack([d['feature'] for d in detections])
        distances = torch.cdist(features, features, p=2)
        distances.fill_diagonal_(float('inf'))

        min_distances, _ = torch.min(distances, dim=1)
        min_distances, _ = torch.sort(min_distances)

        min_dis = float(min_distances[0])
        dist_avg = float(torch.sum(min_distances) / len(detections))

        return dist_avg, min_dis

    def _save_results(self, final_detections, image_detections, out_dir):
        """Save refined detection results"""
        for det in final_detections:
            img_name = det.get('img_name')
            if img_name:
                if img_name not in image_detections:
                    image_detections[img_name] = []
                image_detections[img_name].append(det)

        for img_name, dets in image_detections.items():
            out_path = os.path.join(out_dir, f"{img_name}.txt")
            with open(out_path, 'w') as f:
                for det in dets:
                    x, y, w, h = det['bbox_orig']
                    score = det.get('propagated_tp_score', det['score'])
                    class_id = det['class']
                    f.write(f"{class_id}, {int(x)}, {int(y)}, {int(w)}, {int(h)}, {score}\n")

        print(f"Results saved to {out_dir}")


# ============================================================================
# OPTION A: ADAPTIVE TSBP
# ============================================================================

class AdaptiveTSBP(TS):
    """
    Adaptive TSBP with quantile-based thresholds, confidence propagation, and spatial consistency constraints.
    """

    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize ResNet50 feature extractor"""
        resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet_50.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.resnet.to(self.device)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_pil, use_histogram=False):
        """Extract deep features from image crop using ResNet50"""
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet(image_tensor).squeeze().cpu()

        if use_histogram:
            image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist = hist * 15
            features = torch.from_numpy(np.concatenate((hist, features.numpy())))

        return features


    def compute_spatial_distance(self, det1, det2, sigma_spatial=0.2):
        """
        Compute spatial distance weight between two detections.

        Args:
            det1, det2: Detection dicts with 'center' field
            sigma_spatial: Bandwidth for spatial Gaussian kernel

        Returns:
            float: Spatial similarity weight in [0, 1]
        """
        spatial_dist = np.linalg.norm(det1['center'] - det2['center'])
        weight = np.exp(-spatial_dist ** 2 / (2 * sigma_spatial ** 2))
        return weight

    def refine_confidence(self, candidate, matched_det, feature_dist, lambda_weight=0.5, sigma_dist=100.0):
        """
        Refine candidate confidence based on matched detection.

        Args:
            candidate: Candidate detection dict
            matched_det: Matched high-confidence detection dict
            feature_dist: Feature distance between candidate and match
            lambda_weight: Weight for original score (vs propagated)
            sigma_dist: Bandwidth for distance-based weight

        Returns:
            float: Refined confidence score
        """
        original_score = candidate['score']
        matched_score = matched_det['score']

        # Distance-based weight (closer = higher weight)
        dist_weight = np.exp(-feature_dist / sigma_dist)

        # Propagated score
        propagated_score = dist_weight * matched_score

        # Combine original and propagated
        refined_score = lambda_weight * original_score + (1 - lambda_weight) * propagated_score

        return min(1.0, refined_score)  # Cap at 1.0


    def run_tsbp(self, image_dir, infer_dir, out_dir,
                 tp_quantile=0.8, fp_quantile=0.2,
                 start_tp_num=25, start_fp_num=25,
                 lambda_weight=0.5, sigma_spatial=0.2,
                 use_histogram=False):
        """
        Run Adaptive TSBP algorithm.

        Args:
            image_dir: Directory containing original images
            infer_dir: Directory containing detection .txt files
            out_dir: Output directory for refined detections
            tp_quantile: Quantile for high-confidence threshold (0.8 = top 20%)
            fp_quantile: Quantile for low-confidence threshold (0.2 = bottom 20%)
            start_tp_num: Number of clusters for TP K-means
            start_fp_num: Number of clusters for FP K-means
            lambda_weight: Weight for original vs propagated confidence
            sigma_spatial: Spatial consistency bandwidth
            use_histogram: Whether to use color histogram features
        """
        os.makedirs(out_dir, exist_ok=True)

        txt_files = [f for f in os.listdir(infer_dir) if f.endswith('.txt')]

        if not txt_files:
            print(f"No .txt files found in {infer_dir}")
            return

        print(f"Processing {len(txt_files)} detection files...")

        # Collect all detections
        all_detections = []
        image_detections = {}

        for txt_file in txt_files:
            img_name = os.path.splitext(txt_file)[0]

            detections = self.load_detections(txt_file, image_dir, infer_dir, use_histogram)

            if not detections:
                image_detections[img_name] = []
                continue

            for det in detections:
                det['img_name'] = img_name

            all_detections.extend(detections)
            image_detections[img_name] = []

        if len(all_detections) == 0:
            print("No detections found.")
            return

        # Compute adaptive thresholds per class
        print("\n=== Computing Adaptive Thresholds ===")
        class_thresholds = self.compute_adaptive_thresholds(all_detections, tp_quantile, fp_quantile)

        for class_id, (tp_th, fp_th) in class_thresholds.items():
            print(f"Class {class_id}: TP threshold = {tp_th:.3f}, FP threshold = {fp_th:.3f}")

        # Separate by adaptive thresholds
        all_tp_orig = []
        all_fp_orig = []
        all_candidates = []

        for det in all_detections:
            tp_thresh, fp_thresh = class_thresholds[det['class']]
            if det['score'] >= tp_thresh:
                all_tp_orig.append(det)
            elif det['score'] < fp_thresh:
                all_fp_orig.append(det)
            else:
                all_candidates.append(det)

        print(f"\nInitial counts - TP: {len(all_tp_orig)}, FP: {len(all_fp_orig)}, Candidates: {len(all_candidates)}")

        if len(all_tp_orig) == 0 or len(all_candidates) == 0:
            print("Insufficient detections for TSBP. Outputting high-confidence detections.")
            self._save_results(all_tp_orig, image_detections, out_dir)
            return

        # K-means clustering
        boxes_tp, tp_feats = self.kmeans_clustering(all_tp_orig, start_tp_num)
        boxes_fp, fp_feats = self.kmeans_clustering(all_fp_orig, start_fp_num) if len(all_fp_orig) > 0 else ([], [])

        # Distance constraints
        dist_avg_tp, glob_dis_tp = self.cal_min_dist_stats(all_tp_orig)
        dist_avg_fp, glob_dis_fp = self.cal_min_dist_stats(all_fp_orig) if len(all_fp_orig) > 0 else (
        float('inf'), float('inf'))

        print(f"Distance constraints - TP: {dist_avg_tp:.2f}, FP: {dist_avg_fp:.2f}")

        # Remove overlapping clusters
        boxes_tp_sub = []
        boxes_fp_sub = []
        del_tp_idx = set()
        del_fp_idx = set()

        for i, tp_feat in enumerate(tp_feats):
            for j, fp_feat in enumerate(fp_feats):
                if torch.dist(tp_feat, fp_feat) < glob_dis_tp:
                    del_tp_idx.add(i)
                    del_fp_idx.add(j)

        for idx in sorted(del_tp_idx, reverse=True):
            boxes_tp_sub.append(boxes_tp.pop(idx))
        for idx in sorted(del_fp_idx, reverse=True):
            boxes_fp_sub.append(boxes_fp.pop(idx))

        # Multi-round matching with spatial consistency
        thresh_dist_tp = dist_avg_tp
        thresh_dist_fp = dist_avg_fp
        use_strict = True
        round_num = 0

        while len(all_candidates) > 0:
            round_num += 1
            print(f"\n=== Round {round_num} ===")

            len_cand = len(all_candidates)
            len_tp = len(boxes_tp)
            len_fp = len(boxes_fp)
            full_len = len_cand + len_tp + len_fp

            if len_tp + len_fp == 0:
                print("No confirmed boxes left for matching")
                break

            print(f"Candidates: {len_cand}, TP: {len_tp}, FP: {len_fp}")

            # EMD distributions
            P = np.array([1.0 if i < len_cand else 0.0 for i in range(full_len)])
            Q = np.array([0.0 if i < len_cand else 1.0 for i in range(full_len)])

            # Distance matrix with spatial weighting
            Q_dets = boxes_tp + boxes_fp
            D = np.zeros((full_len, full_len))

            for i in range(len_cand):
                for j in range(len_tp + len_fp):
                    # Feature distance
                    feat_dist = torch.dist(all_candidates[i]['feature'], Q_dets[j]['feature']).item()

                    # Spatial weight (only for non-cluster centers)
                    if not Q_dets[j].get('is_cluster_center', False):
                        spatial_weight = self.compute_spatial_distance(all_candidates[i], Q_dets[j], sigma_spatial)
                    else:
                        spatial_weight = 1.0  # No spatial penalty for cluster centers

                    # Combined distance (lower spatial weight = higher distance)
                    combined_dist = feat_dist / (spatial_weight + 1e-6)

                    D[i][len_cand + j] = combined_dist
                    D[len_cand + j][i] = combined_dist

            # Run EMD
            P_norm = P / P.sum() if P.sum() > 0 else P
            Q_norm = Q / Q.sum() if Q.sum() > 0 else Q
            flow = ot.emd(P_norm, Q_norm, D)
            flow = flow * min(P.sum(), Q.sum())

            # Collect matches
            tp_matched = []
            fp_matched = []

            for i in range(len_cand):
                for j in range(len_tp + len_fp):
                    if flow[i][len_cand + j] > 0:
                        cand = all_candidates[i]
                        dist = D[i][len_cand + j]
                        matched_det = Q_dets[j]

                        if j < len_tp:
                            tp_matched.append({
                                'cand': cand,
                                'dist': dist,
                                'matched_det': matched_det
                            })
                        else:
                            fp_matched.append({
                                'cand': cand,
                                'dist': dist,
                                'matched_det': matched_det
                            })

            tp_matched.sort(key=lambda x: x['dist'])
            fp_matched.sort(key=lambda x: x['dist'])

            print(f"Matched - TP: {len(tp_matched)}, FP: {len(fp_matched)}")

            # Add candidates with refined confidence
            added_cands = []
            tp_add = 0
            fp_add = 0

            for match in tp_matched:
                if match['dist'] <= thresh_dist_tp:
                    cand = match['cand']

                    # Refine confidence score
                    if not match['matched_det'].get('is_cluster_center', False):
                        cand['score'] = self.refine_confidence(cand, match['matched_det'], match['dist'], lambda_weight)

                    all_tp_orig.append(cand)
                    boxes_tp.append(cand)
                    added_cands.append(cand)
                    tp_add += 1

            for match in fp_matched:
                if match['dist'] <= thresh_dist_fp:
                    cand = match['cand']
                    boxes_fp.append(cand)
                    added_cands.append(cand)
                    fp_add += 1

            # Remove added candidates
            for cand in added_cands:
                all_candidates = [c for c in all_candidates if c['bbox'] != cand['bbox']]

            print(f"Added - TP: {tp_add}, FP: {fp_add}")

            # Termination check
            if tp_add == 0 and fp_add == 0:
                if use_strict:
                    print("Relaxing constraints and adding separated clusters")
                    boxes_tp.extend(boxes_tp_sub)
                    boxes_fp.extend(boxes_fp_sub)
                    thresh_dist_tp = float('inf')
                    thresh_dist_fp = float('inf')
                    use_strict = False
                else:
                    print("No more candidates can be matched. Stopping.")
                    break

        print(f"\n=== Adaptive TSBP Complete ===")
        print(f"Final TP count: {len(all_tp_orig)}")
        print(f"Remaining candidates: {len(all_candidates)}")

        self._save_results(all_tp_orig, image_detections, out_dir)


# ============================================================================
# OPTION B: HIERARCHICAL FEATURE TSBP
# ============================================================================

class HierarchicalFeatureTSBP(TS):
    """
    Hierarchical Feature TSBP with multi-scale features, learned weighting, and graph-based propagation.
    """

    def __init__(self, device=None, num_scales=3):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_scales = num_scales
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize multi-scale ResNet50 feature extractor"""
        resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Extract features from multiple layers
        self.layer1 = nn.Sequential(*list(resnet_50.children())[:5])  # conv2_x
        self.layer2 = nn.Sequential(*list(resnet_50.children())[5:6])  # conv3_x
        self.layer3 = nn.Sequential(*list(resnet_50.children())[6:7])  # conv4_x
        self.layer4 = nn.Sequential(*list(resnet_50.children())[7:8])  # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer1.to(self.device)
        self.layer2.to(self.device)
        self.layer3.to(self.device)
        self.layer4.to(self.device)
        self.avgpool.to(self.device)

        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_multiscale_features(self, image_pil, use_histogram=False):
        """Extract multi-scale deep features from image"""
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.layer1(image_tensor)

            feat2 = self.layer2(x)
            feat2_pooled = self.avgpool(feat2).squeeze().cpu()

            feat3 = self.layer3(feat2)
            feat3_pooled = self.avgpool(feat3).squeeze().cpu()

            feat4 = self.layer4(feat3)
            feat4_pooled = self.avgpool(feat4).squeeze().cpu()

        features_list = [feat2_pooled, feat3_pooled, feat4_pooled]

        if use_histogram:
            image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_tensor = torch.from_numpy(hist).float()
            features_list.append(hist_tensor)

        return features_list

    def compute_attention_weights(self, features_list):
        """
        Compute attention weights for feature fusion.
        Simple variance-based attention: higher variance = more informative
        """
        weights = []
        for feat in features_list:
            # Variance as importance measure
            var = torch.var(feat)
            weights.append(var)

        weights = torch.tensor(weights)
        weights = F.softmax(weights, dim=0)
        return weights

    def fuse_features(self, features_list):
        """
        Fuse multi-scale features with attention weighting.
        Returns normalized fused feature vector.
        """
        # Normalize each feature to unit norm
        normalized_feats = [F.normalize(f.unsqueeze(0), dim=1).squeeze() for f in features_list]

        # Compute attention weights
        weights = self.compute_attention_weights(features_list)

        # Weighted concatenation (could also do weighted sum with projection)
        # For simplicity, we concatenate and weight the components
        weighted_feats = [w * f for w, f in zip(weights, normalized_feats)]
        fused = torch.cat(weighted_feats)

        # Final normalization
        fused = F.normalize(fused.unsqueeze(0), dim=1).squeeze()

        return fused


    def extract_features(self, image_pil, use_histogram=False):
        features_list = self.extract_multiscale_features(image_pil, use_histogram)
        fused_feature = self.fuse_features(features_list)
        return fused_feature

    def build_similarity_graph(self, detections, k_neighbors=10, sigma=100):
        """
        Build k-NN similarity graph for label propagation.

        Args:
            detections: List of detection dicts
            k_neighbors: Number of nearest neighbors per node

        Returns:
            Sparse similarity matrix (n x n)
        """
        n = len(detections)
        features = torch.stack([d['feature'] for d in detections])

        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)

        # Build k-NN graph
        row_idx = []
        col_idx = []
        data = []

        for i in range(n):
            # Get k nearest neighbors (excluding self)
            dists = distances[i]
            dists[i] = float('inf')  # Exclude self

            k_nearest = min(k_neighbors, n - 1)
            topk_dist, topk_idx = torch.topk(dists, k_nearest, largest=False)

            for j, dist in zip(topk_idx, topk_dist):
                # Gaussian kernel for similarity
                sim = torch.exp(-dist ** 2 / (2 * sigma ** 2))  # sigma=100

                row_idx.append(i)
                col_idx.append(j.item())
                data.append(sim.item())

        # Create sparse matrix
        similarity_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        # Symmetrize
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        # Row normalize
        row_sums = np.array(similarity_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        D_inv = csr_matrix(np.diag(1.0 / row_sums))
        similarity_matrix = D_inv @ similarity_matrix

        return similarity_matrix

    def graph_label_propagation(self, similarity_matrix, initial_labels, alpha=0.8, max_iter=30, tol=1e-3):
        """
        Perform label propagation on similarity graph.

        Args:
            similarity_matrix: Sparse n x n similarity matrix
            initial_labels: n x c matrix (n=nodes, c=classes), rows sum to 1 or 0
            alpha: Propagation factor (0.8 = 80% from neighbors, 20% from initial)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Final label distribution matrix (n x c)
        """
        Y = initial_labels.copy()
        Y_init = initial_labels.copy()

        for iteration in range(max_iter):
            Y_prev = Y.copy()

            # Propagation: Y = alpha * S @ Y + (1 - alpha) * Y_init
            Y = alpha * (similarity_matrix @ Y) + (1 - alpha) * Y_init

            # Check convergence
            diff = np.linalg.norm(Y - Y_prev)
            if diff < tol:
                print(f"Label propagation converged at iteration {iteration + 1}")
                break

        return Y

    def run_tsbp(self, image_dir, infer_dir, out_dir,
                 tp_quantile=0.8, fp_quantile=0.2,
                 k_neighbors=10, propagation_alpha=0.8,
                 use_histogram=False):
        """
        Run Hierarchical Feature TSBP with graph-based propagation.

        Args:
            image_dir: Directory containing original images
            infer_dir: Directory containing detection .txt files
            out_dir: Output directory for refined detections
            tp_threshold: Confidence threshold for high-confidence detections
            fp_threshold: Confidence threshold for low-confidence detections
            k_neighbors: Number of neighbors for k-NN graph
            propagation_alpha: Label propagation strength
            use_histogram: Whether to use color histogram features
        """
        os.makedirs(out_dir, exist_ok=True)

        txt_files = [f for f in os.listdir(infer_dir) if f.endswith('.txt')]

        if not txt_files:
            print(f"No .txt files found in {infer_dir}")
            return

        print(f"Processing {len(txt_files)} detection files...")

        # Collect all detections
        all_detections = []
        image_detections = {}

        for txt_file in txt_files:
            img_name = os.path.splitext(txt_file)[0]

            detections = self.load_detections(txt_file, image_dir, infer_dir, use_histogram)

            if not detections:
                image_detections[img_name] = []
                continue

            for det in detections:
                det['img_name'] = img_name

            all_detections.extend(detections)
            image_detections[img_name] = []

        if len(all_detections) == 0:
            print("No detections found.")
            return

        print(f"Total detections: {len(all_detections)}")

        # Build similarity graph
        print("\n=== Building Similarity Graph ===")
        similarity_matrix = self.build_similarity_graph(all_detections, k_neighbors)
        print(f"Graph built: {similarity_matrix.shape[0]} nodes, {similarity_matrix.nnz} edges")

        # Initialize labels: 1 for TP, -1 for FP, 0 for unlabeled
        n_detections = len(all_detections)
        initial_labels = np.zeros((n_detections, 2))  # [TP_score, FP_score]

        # Compute adaptive thresholds per class
        print("\n=== Computing Adaptive Thresholds ===")
        class_thresholds = self.compute_adaptive_thresholds(
            all_detections, tp_quantile, fp_quantile
        )

        for class_id, (tp_th, fp_th) in class_thresholds.items():
            print(f"Class {class_id}: TP threshold = {tp_th:.3f}, FP threshold = {fp_th:.3f}")


        for i, det in enumerate(all_detections):
            tp_thresh, fp_thresh = class_thresholds[det['class']]
            if det['score'] >= tp_thresh:
                initial_labels[i, 0] = 1.0  # TP
                det['initial_label'] = 'TP'
            elif det['score'] < fp_thresh:
                initial_labels[i, 1] = 1.0  # FP
                det['initial_label'] = 'FP'
            else:
                det['initial_label'] = 'unlabeled'

        tp_count = np.sum(initial_labels[:, 0])
        fp_count = np.sum(initial_labels[:, 1])
        unlabeled_count = n_detections - tp_count - fp_count

        print(f"Initial labels - TP: {int(tp_count)}, FP: {int(fp_count)}, Unlabeled: {int(unlabeled_count)}")

        if tp_count == 0:
            print("No high-confidence detections. Cannot propagate.")
            return

        # Label propagation
        print("\n=== Running Label Propagation ===")
        final_labels = self.graph_label_propagation(
            similarity_matrix, initial_labels,
            alpha=propagation_alpha, max_iter=30
        )

        # Assign labels based on propagated scores
        final_tp = []
        for i, det in enumerate(all_detections):
            tp_score = final_labels[i, 0]
            fp_score = final_labels[i, 1]

            # Confidence from propagation
            det['propagated_tp_score'] = tp_score
            det['propagated_fp_score'] = fp_score

            # Accept as TP if TP score > FP score and TP score > threshold
            if tp_score > fp_score and tp_score > 0.5:
                final_tp.append(det)

        print(f"\n=== Hierarchical Feature TSBP Complete ===")
        print(f"Final TP count: {len(final_tp)}")
        print(f"Propagated from {int(tp_count)} to {len(final_tp)} detections")

        self._save_results(final_tp, image_detections, out_dir)


# ============================================================================
# OPTION C: TSBP++
# ============================================================================

class TSBPPlusPlus(TS):
    """
    TSBP++: Unified framework combining adaptive thresholds, multi-scale features,
    uncertainty-aware matching, and class-specific propagation.
    """

    def __init__(self, device=None, num_scales=3):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_scales = num_scales
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize multi-scale ResNet50 feature extractor"""
        resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.layer1 = nn.Sequential(*list(resnet_50.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet_50.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet_50.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet_50.children())[7:8])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer1.to(self.device)
        self.layer2.to(self.device)
        self.layer3.to(self.device)
        self.layer4.to(self.device)
        self.avgpool.to(self.device)

        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_multiscale_features(self, image_pil, use_histogram=False):
        """Extract multi-scale deep features"""
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.layer1(image_tensor)
            feat2 = self.layer2(x)
            feat2_pooled = self.avgpool(feat2).squeeze().cpu()

            feat3 = self.layer3(feat2)
            feat3_pooled = self.avgpool(feat3).squeeze().cpu()

            feat4 = self.layer4(feat3)
            feat4_pooled = self.avgpool(feat4).squeeze().cpu()

        features_list = [feat2_pooled, feat3_pooled, feat4_pooled]

        if use_histogram:
            image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_tensor = torch.from_numpy(hist).float()
            features_list.append(hist_tensor)

        return features_list

    def fuse_features(self, features_list):
        """Fuse multi-scale features with attention"""
        normalized_feats = [F.normalize(f.unsqueeze(0), dim=1).squeeze() for f in features_list]

        weights = []
        for feat in features_list:
            var = torch.var(feat)
            weights.append(var)

        weights = torch.tensor(weights)
        weights = F.softmax(weights, dim=0)

        weighted_feats = [w * f for w, f in zip(weights, normalized_feats)]
        fused = torch.cat(weighted_feats)
        fused = F.normalize(fused.unsqueeze(0), dim=1).squeeze()

        return fused

    def extract_features(self, image_pil, use_histogram=False):
        features_list = self.extract_multiscale_features(image_pil, use_histogram)
        fused_feature = self.fuse_features(features_list)

        return fused_feature


    def compute_uncertainty(self, detection, all_detections, k=5):
        """
        Compute epistemic and aleatoric uncertainty for a detection.

        Args:
            detection: Detection dict
            all_detections: All detections for k-NN
            k: Number of neighbors

        Returns:
            (epistemic_uncertainty, aleatoric_uncertainty)
        """
        # Aleatoric: inverse of confidence
        aleatoric = 1.0 - detection['score']

        # Epistemic: variance in k-NN features
        det_feat = detection['feature']

        # Find k nearest neighbors
        distances = []
        neighbor_feats = []
        for other_det in all_detections:
            if other_det['bbox'] == detection['bbox']:
                continue
            dist = torch.dist(det_feat, other_det['feature']).item()
            distances.append(dist)
            neighbor_feats.append(other_det['feature'])

        if len(distances) < k:
            epistemic = 1.0  # High uncertainty if few neighbors
        else:
            # Get k nearest
            sorted_idx = np.argsort(distances)[:k]
            k_nearest_feats = [neighbor_feats[i] for i in sorted_idx]

            # Compute variance
            feat_matrix = torch.stack(k_nearest_feats)
            epistemic = torch.var(feat_matrix, dim=0).mean().item()

        return epistemic, aleatoric


    def run_class_specific_emd(self, candidates, confirmed_boxes, class_id, thresh_dist, uncertainty_weight=0.5):
        """
        Run EMD matching for a specific class with uncertainty weighting.

        Args:
            candidates: Candidate detections for this class
            confirmed_boxes: Confirmed boxes for this class
            class_id: Class ID
            thresh_dist: Distance threshold
            uncertainty_weight: Weight for uncertainty in distance computation

        Returns:
            List of matched candidates with refined scores
        """
        if len(candidates) == 0 or len(confirmed_boxes) == 0:
            return []

        len_cand = len(candidates)
        len_conf = len(confirmed_boxes)
        full_len = len_cand + len_conf

        # Compute uncertainties
        all_dets = candidates + confirmed_boxes
        uncertainties = []
        for det in candidates:
            epistemic, aleatoric = self.compute_uncertainty(det, all_dets, k=5)
            combined_uncertainty = epistemic + aleatoric
            uncertainties.append(combined_uncertainty)

        # EMD distributions
        P = np.array([1.0 if i < len_cand else 0.0 for i in range(full_len)])
        Q = np.array([0.0 if i < len_cand else 1.0 for i in range(full_len)])

        # Distance matrix with uncertainty weighting
        D = np.zeros((full_len, full_len))

        for i in range(len_cand):
            for j in range(len_conf):
                feat_dist = torch.dist(candidates[i]['feature'], confirmed_boxes[j]['feature']).item()

                # Uncertainty penalty: higher uncertainty = higher effective distance
                uncertainty_penalty = 1.0 + uncertainty_weight * uncertainties[i]
                weighted_dist = feat_dist * uncertainty_penalty

                D[i][len_cand + j] = weighted_dist
                D[len_cand + j][i] = weighted_dist

        # Run EMD
        P_norm = P / P.sum() if P.sum() > 0 else P
        Q_norm = Q / Q.sum() if Q.sum() > 0 else Q
        flow = ot.emd(P_norm, Q_norm, D)
        flow = flow * min(P.sum(), Q.sum())

        # Collect matches
        matched = []
        for i in range(len_cand):
            for j in range(len_conf):
                if flow[i][len_cand + j] > 0:
                    dist = D[i][len_cand + j]
                    if dist <= thresh_dist:
                        cand = candidates[i]
                        matched_det = confirmed_boxes[j]

                        # Refine confidence
                        dist_weight = np.exp(-dist / 100.0)
                        propagated_score = dist_weight * matched_det['score']
                        refined_score = 0.5 * cand['score'] + 0.5 * propagated_score
                        cand['score'] = min(1.0, refined_score)

                        matched.append(cand)
                        break  # Match to only one confirmed box

        return matched

    def run_tsbp(self, image_dir, infer_dir, out_dir,
                 tp_quantile=0.8, fp_quantile=0.2,
                 start_tp_num=25, start_fp_num=25,
                 uncertainty_weight=0.5,
                 max_rounds=10, convergence_patience=2,
                 use_histogram=False):
        """
        Run TSBP++ with all improvements.

        Args:
            image_dir: Directory containing original images
            infer_dir: Directory containing detection .txt files
            out_dir: Output directory for refined detections
            tp_quantile: Quantile for high-confidence threshold
            fp_quantile: Quantile for low-confidence threshold
            start_tp_num: Number of TP clusters
            start_fp_num: Number of FP clusters
            uncertainty_weight: Weight for uncertainty in matching
            max_rounds: Maximum propagation rounds
            convergence_patience: Stop if no improvement for this many rounds
            use_histogram: Whether to use color histogram features
        """
        os.makedirs(out_dir, exist_ok=True)

        txt_files = [f for f in os.listdir(infer_dir) if f.endswith('.txt')]

        if not txt_files:
            print(f"No .txt files found in {infer_dir}")
            return

        print(f"Processing {len(txt_files)} detection files...")

        # Collect all detections
        all_detections = []
        image_detections = {}

        for txt_file in txt_files:
            img_name = os.path.splitext(txt_file)[0]

            detections = self.load_detections(txt_file, image_dir, infer_dir, use_histogram)

            if not detections:
                image_detections[img_name] = []
                continue

            for det in detections:
                det['img_name'] = img_name

            all_detections.extend(detections)
            image_detections[img_name] = []

        if len(all_detections) == 0:
            print("No detections found.")
            return

        # Adaptive thresholds
        print("\n=== Computing Adaptive Thresholds ===")
        class_thresholds = self.compute_adaptive_thresholds(
            all_detections, tp_quantile, fp_quantile
        )

        for class_id, (tp_th, fp_th) in class_thresholds.items():
            print(f"Class {class_id}: TP={tp_th:.3f}, FP={fp_th:.3f}")

        # Separate by class and threshold
        class_detections = defaultdict(lambda: {'tp': [], 'fp': [], 'candidates': []})

        for det in all_detections:
            tp_thresh, fp_thresh = class_thresholds[det['class']]
            class_id = det['class']

            if det['score'] >= tp_thresh:
                class_detections[class_id]['tp'].append(det)
            elif det['score'] < fp_thresh:
                class_detections[class_id]['fp'].append(det)
            else:
                class_detections[class_id]['candidates'].append(det)

        # Print per-class statistics
        print("\n=== Per-Class Statistics ===")
        for class_id in sorted(class_detections.keys()):
            tp_count = len(class_detections[class_id]['tp'])
            fp_count = len(class_detections[class_id]['fp'])
            cand_count = len(class_detections[class_id]['candidates'])
            print(f"Class {class_id}: TP={tp_count}, FP={fp_count}, Candidates={cand_count}")

        # Process each class separately
        all_final_tp = []

        for class_id in sorted(class_detections.keys()):
            print(f"\n{'=' * 60}")
            print(f"Processing Class {class_id}")
            print(f"{'=' * 60}")

            tp_orig = class_detections[class_id]['tp']
            fp_orig = class_detections[class_id]['fp']
            candidates = class_detections[class_id]['candidates']

            if len(tp_orig) == 0 or len(candidates) == 0:
                print(f"Skipping class {class_id}: insufficient detections")
                all_final_tp.extend(tp_orig)
                continue

            # K-means clustering
            boxes_tp, _ = self.kmeans_clustering(tp_orig, min(start_tp_num, len(tp_orig)))
            boxes_fp, _ = self.kmeans_clustering(fp_orig, min(start_fp_num, len(fp_orig))) if len(fp_orig) > 0 else (
            [], [])

            # Distance constraints
            dist_avg_tp, _ = self.cal_min_dist_stats(tp_orig)
            dist_avg_fp, _ = self.cal_min_dist_stats(fp_orig) if len(fp_orig) > 0 else (float('inf'), float('inf'))

            print(f"Distance thresholds - TP: {dist_avg_tp:.2f}, FP: {dist_avg_fp:.2f}")

            # Multi-round class-specific EMD with convergence detection
            confirmed_tp = list(tp_orig)
            remaining_candidates = list(candidates)
            no_improvement_count = 0
            prev_tp_count = len(confirmed_tp)

            for round_num in range(1, max_rounds + 1):
                print(f"\n--- Round {round_num} ---")
                print(f"Confirmed TP: {len(confirmed_tp)}, Candidates: {len(remaining_candidates)}")

                if len(remaining_candidates) == 0:
                    print("No more candidates")
                    break

                # Run class-specific EMD
                matched = self.run_class_specific_emd(
                    remaining_candidates, confirmed_tp, class_id,
                    thresh_dist=dist_avg_tp, uncertainty_weight=uncertainty_weight
                )

                print(f"Matched {len(matched)} candidates")

                # Update
                confirmed_tp.extend(matched)
                for m in matched:
                    remaining_candidates = [c for c in remaining_candidates if c['bbox'] != m['bbox']]

                # Convergence check
                current_tp_count = len(confirmed_tp)
                improvement = current_tp_count - prev_tp_count

                if improvement == 0:
                    no_improvement_count += 1
                    print(f"No improvement (patience: {no_improvement_count}/{convergence_patience})")

                    if no_improvement_count >= convergence_patience:
                        print("Convergence reached. Stopping.")
                        break
                else:
                    no_improvement_count = 0

                prev_tp_count = current_tp_count

            print(f"\nClass {class_id} final: {len(confirmed_tp)} detections (from {len(tp_orig)})")
            all_final_tp.extend(confirmed_tp)

        print(f"\n{'=' * 60}")
        print(f"TSBP++ Complete")
        print(f"{'=' * 60}")
        print(f"Total final detections: {len(all_final_tp)}")

        self._save_results(all_final_tp, image_detections, out_dir)


# https://arxiv.org/abs/2409.16678
# https://github.com/jwhgdeu/TSBP
# IMPROVED TSBP - Conservative enhancements focused on robustness and precision

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import ot  # Python Optimal Transport: pip install POT
from torchvision.models import ResNet50_Weights


class ImprovedTSBPDetector:
    """
    Improved TSBP with conservative enhancements:

    1. Score-weighted EMD: Weight the transport by confidence scores to prioritize
       matching high-confidence candidates first

    2. Per-class processing: Handle each class independently to prevent cross-class
       interference and allow class-specific thresholds

    3. IoU-based filtering: Add spatial consistency by preferring matches that don't
       violate basic spatial logic (e.g., very distant boxes shouldn't match)

    4. Adaptive distance thresholds: Compute thresholds based on actual feature distance
       distribution rather than fixed multipliers

    5. Iterative refinement with early stopping: Stop when adding candidates no longer
       improves the quality of the confirmed set
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize ResNet50 feature extractor"""
        resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet_50.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.resnet.to(self.device)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_pil):
        """Extract deep features from image crop using ResNet50"""
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet(image_tensor).squeeze().cpu()
        return features

    def calc_hist(self, img_bgr):
        """Calculate color histogram (8x8x8 bins)"""
        hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_features_with_hist(self, image_pil):
        """Extract combined features: ResNet50 + color histogram"""
        deep_features = self.extract_features(image_pil)
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        hist = self.calc_hist(image_bgr) * 15
        features = torch.from_numpy(np.concatenate((hist, deep_features.numpy())))
        return features

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes (x1, y1, x2, y2)"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0

        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def load_detections(self, txt_path, img_path, use_histogram=False):
        """Load detections from YOLO format text file"""
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
                if use_histogram:
                    features = self.extract_features_with_hist(crop)
                else:
                    features = self.extract_features(crop)

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'bbox_orig': (x, y, w, h),
                    'score': score,
                    'class': int(class_id),
                    'feature': features
                }
                detections.append(detection)

        return detections

    def cal_min_dist_stats(self, detections):
        """Calculate average minimum distance between detections"""
        if len(detections) < 2:
            return float('inf'), float('inf')

        features = torch.stack([d['feature'] for d in detections])
        distances = torch.cdist(features, features, p=2)
        distances.fill_diagonal_(float('inf'))

        min_distances, _ = torch.min(distances, dim=1)
        min_distances, _ = torch.sort(min_distances)

        min_dis = float(min_distances[0])
        dist_avg = float(torch.sum(min_distances) / len(detections))

        return dist_avg, min_dis

    def kmeans_clustering(self, detections, num_clusters):
        """Apply K-means clustering to detections"""
        if len(detections) < num_clusters:
            return detections, [d['feature'] for d in detections]

        features = torch.stack([d['feature'] for d in detections])

        kmeans = KMeans(n_clusters=num_clusters, max_iter=500, n_init=10, random_state=1234)
        kmeans.fit(features.numpy())

        clustered_dets = []
        clustered_feats = []
        for i in range(num_clusters):
            det = {
                'bbox': (0, 0, 0, 0),
                'bbox_orig': (0, 0, 0, 0),
                'score': 1.0,
                'class': detections[0]['class'],
                'feature': torch.from_numpy(kmeans.cluster_centers_[i]),
                'is_cluster_center': True
            }
            clustered_dets.append(det)
            clustered_feats.append(det['feature'])

        return clustered_dets, clustered_feats

    def compute_adaptive_thresholds(self, tp_dets, fp_dets, tp_quantile=0.7, fp_quantile=0.7):
        """
        Compute adaptive distance thresholds based on feature distance distribution.
        Uses percentile of minimum distances within TP and FP sets.
        """
        # For TP: use stricter threshold (lower percentile of distances)
        if len(tp_dets) >= 2:
            tp_features = torch.stack([d['feature'] for d in tp_dets])
            tp_distances = torch.cdist(tp_features, tp_features, p=2)
            tp_distances.fill_diagonal_(float('inf'))
            tp_min_dists = torch.min(tp_distances, dim=1)[0]
            thresh_tp = float(torch.quantile(tp_min_dists, tp_quantile))
        else:
            thresh_tp = float('inf')

        # For FP: use more relaxed threshold
        if len(fp_dets) >= 2:
            fp_features = torch.stack([d['feature'] for d in fp_dets])
            fp_distances = torch.cdist(fp_features, fp_features, p=2)
            fp_distances.fill_diagonal_(float('inf'))
            fp_min_dists = torch.min(fp_distances, dim=1)[0]
            thresh_fp = float(torch.quantile(fp_min_dists, fp_quantile))
        else:
            thresh_fp = float('inf')

        return thresh_tp, thresh_fp

    def run_tsbp(self, image_dir, infer_dir, out_dir,
                 tp_threshold=0.50, fp_threshold=0.30,
                 start_tp_num=25, start_fp_num=25,
                 use_histogram=False,
                 adaptive_thresh=True,
                 tp_quantile=0.7,
                 fp_quantile=0.7,
                 max_iou_for_match=0.3,
                 score_weight_emd=True):
        """
        Run improved TSBP algorithm.

        Args:
            adaptive_thresh: Use adaptive distance thresholds based on distribution
            tp_quantile: Quantile for TP distance threshold (lower = stricter)
            fp_quantile: Quantile for FP distance threshold
            max_iou_for_match: Maximum IoU between candidate and confirmed boxes
                              (prevents matching boxes that heavily overlap)
            score_weight_emd: Weight EMD by detection confidence scores
        """
        os.makedirs(out_dir, exist_ok=True)

        txt_files = [f for f in os.listdir(infer_dir) if f.endswith('.txt')]
        if not txt_files:
            print(f"No .txt files found in {infer_dir}")
            return

        print(f"Processing {len(txt_files)} detection files...")

        # Collect detections per class
        class_detections = {}
        image_detections = {}

        for txt_file in txt_files:
            img_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(infer_dir, txt_file)

            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                test_path = os.path.join(image_dir, img_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            if img_path is None:
                continue

            detections = self.load_detections(txt_path, img_path, use_histogram)
            image_detections[img_name] = detections

            for det in detections:
                det['img_name'] = img_name
                class_id = det['class']
                if class_id not in class_detections:
                    class_detections[class_id] = []
                class_detections[class_id].append(det)

        # Process each class independently
        all_final_detections = []

        for class_id, all_dets in class_detections.items():
            print(f"\n{'=' * 60}")
            print(f"Processing class {class_id}: {len(all_dets)} detections")
            print(f"{'=' * 60}")

            # Split into TP, FP, and candidates based on scores
            tp_orig = [d for d in all_dets if d['score'] >= tp_threshold]
            fp_orig = [d for d in all_dets if d['score'] < fp_threshold]
            candidates = [d for d in all_dets if fp_threshold <= d['score'] < tp_threshold]

            print(f"Initial split - TP: {len(tp_orig)}, Candidates: {len(candidates)}, FP: {len(fp_orig)}")

            if not tp_orig:
                print("No high-confidence detections, skipping class")
                all_final_detections.extend(tp_orig)
                continue

            # Cluster TP and FP
            boxes_tp, _ = self.kmeans_clustering(tp_orig, min(start_tp_num, len(tp_orig)))
            boxes_fp, _ = self.kmeans_clustering(fp_orig, min(start_fp_num, len(fp_orig))) if fp_orig else ([], [])

            # Separate cluster centers from real boxes
            boxes_tp_real = [d for d in boxes_tp if not d.get('is_cluster_center', False)]
            boxes_tp_centers = [d for d in boxes_tp if d.get('is_cluster_center', False)]
            boxes_fp_centers = [d for d in boxes_fp if d.get('is_cluster_center', False)]

            # Add original TP to confirmed
            confirmed_tp = list(tp_orig)

            # Compute adaptive thresholds if enabled
            if adaptive_thresh:
                thresh_dist_tp, thresh_dist_fp = self.compute_adaptive_thresholds(
                    tp_orig, fp_orig, tp_quantile, fp_quantile
                )
                print(f"Adaptive thresholds - TP: {thresh_dist_tp:.2f}, FP: {thresh_dist_fp:.2f}")
            else:
                # Use original fixed method
                dist_avg_tp, min_dis_tp = self.cal_min_dist_stats(tp_orig)
                dist_avg_fp, min_dis_fp = self.cal_min_dist_stats(fp_orig) if fp_orig else (float('inf'), float('inf'))
                thresh_dist_tp = dist_avg_tp
                thresh_dist_fp = dist_avg_fp
                print(f"Fixed thresholds - TP: {thresh_dist_tp:.2f}, FP: {thresh_dist_fp:.2f}")

            # Iterative propagation
            use_strict = True
            max_iterations = 20

            for iteration in range(max_iterations):
                if not candidates:
                    print("No more candidates")
                    break

                print(f"\nIteration {iteration + 1}")

                len_cand = len(candidates)
                len_tp = len(boxes_tp)
                len_fp = len(boxes_fp)
                full_len = len_cand + len_tp + len_fp

                if full_len == 0 or len_cand == 0:
                    break

                print(f"Candidates: {len_cand}, TP: {len_tp}, FP: {len_fp}")

                # EMD with optional score weighting
                if score_weight_emd:
                    # Weight by scores - higher confidence candidates get more weight
                    cand_scores = np.array([c['score'] for c in candidates])
                    cand_scores = cand_scores / cand_scores.sum()
                    P = np.zeros(full_len)
                    P[:len_cand] = cand_scores
                else:
                    P = np.array([1.0 if i < len_cand else 0.0 for i in range(full_len)])

                Q = np.array([0.0 if i < len_cand else 1.0 for i in range(full_len)])

                # Distance matrix
                Q_dets = boxes_tp + boxes_fp
                D = np.zeros((full_len, full_len))

                for i in range(len_cand):
                    for j in range(len_tp + len_fp):
                        dist = torch.dist(candidates[i]['feature'], Q_dets[j]['feature']).item()

                        # Add spatial penalty for boxes with high IoU
                        # (prevents matching boxes that heavily overlap)
                        if not Q_dets[j].get('is_cluster_center', False):
                            iou = self.compute_iou(candidates[i]['bbox'], Q_dets[j]['bbox'])
                            if iou > max_iou_for_match:
                                dist *= 2.0  # Penalize matching of overlapping boxes

                        D[i][len_cand + j] = dist
                        D[len_cand + j][i] = dist

                # Normalize and compute EMD
                P_norm = P / P.sum() if P.sum() > 0 else P
                Q_norm = Q / Q.sum() if Q.sum() > 0 else Q
                flow = ot.emd(P_norm, Q_norm, D)
                flow = flow * min(P.sum(), Q.sum())

                # Collect matches
                tp_matched = []
                fp_matched = []

                for i in range(len_cand):
                    for j in range(len_tp + len_fp):
                        if flow[i][len_cand + j] > 0:
                            cand = candidates[i]
                            dist = D[i][len_cand + j]

                            if j < len_tp:
                                tp_matched.append({'cand': cand, 'dist': dist})
                            else:
                                fp_matched.append({'cand': cand, 'dist': dist})

                tp_matched.sort(key=lambda x: x['dist'])
                fp_matched.sort(key=lambda x: x['dist'])

                print(f"Matched - TP: {len(tp_matched)}, FP: {len(fp_matched)}")
                if tp_matched:
                    print(f"  TP dist range: [{tp_matched[0]['dist']:.2f}, {tp_matched[-1]['dist']:.2f}]")
                if fp_matched:
                    print(f"  FP dist range: [{fp_matched[0]['dist']:.2f}, {fp_matched[-1]['dist']:.2f}]")

                # Add candidates under threshold
                added_cands = []
                tp_add = 0
                fp_add = 0

                for match in tp_matched:
                    if match['dist'] <= thresh_dist_tp:
                        cand = match['cand']
                        confirmed_tp.append(cand)
                        boxes_tp.append(cand)
                        added_cands.append(cand)
                        tp_add += 1

                for match in fp_matched:
                    if match['dist'] <= thresh_dist_fp:
                        cand = match['cand']
                        boxes_fp.append(cand)
                        added_cands.append(cand)
                        fp_add += 1

                # Remove added candidates
                for cand in added_cands:
                    candidates = [c for c in candidates if c['bbox'] != cand['bbox']]

                print(f"Added - TP: {tp_add}, FP: {fp_add}")

                # Check termination
                if tp_add == 0 and fp_add == 0:
                    if use_strict:
                        print("Relaxing constraints")
                        boxes_tp.extend(boxes_tp_centers)
                        boxes_fp.extend(boxes_fp_centers)
                        thresh_dist_tp = float('inf')
                        thresh_dist_fp = float('inf')
                        use_strict = False
                    else:
                        print("No progress, stopping")
                        break

            print(f"\nClass {class_id} final: {len(confirmed_tp)} detections")
            all_final_detections.extend(confirmed_tp)

        print(f"\n{'=' * 60}")
        print(f"TSBP Complete - Total detections: {len(all_final_detections)}")
        print(f"{'=' * 60}")

        self._save_results(all_final_detections, image_detections, out_dir)

    def _save_results(self, final_detections, image_detections, out_dir):
        """Save refined detection results to text files"""
        # Group by image
        image_results = {}
        for det in final_detections:
            img_name = det.get('img_name')
            if img_name:
                if img_name not in image_results:
                    image_results[img_name] = []
                image_results[img_name].append(det)

        # Write files
        for img_name, dets in image_results.items():
            out_path = os.path.join(out_dir, f"{img_name}.txt")
            with open(out_path, 'w') as f:
                for det in dets:
                    x, y, w, h = det['bbox_orig']
                    score = det['score']
                    class_id = det['class']
                    f.write(f"{class_id}, {int(x)}, {int(y)}, {int(w)}, {int(h)}, {score}\n")

        print(f"Results saved to {out_dir}")


def run_tsbp_pipeline(model_name, data_root, results_inf_all_root,
                      tp_threshold=0.50, fp_threshold=0.30,
                      start_tp_num=25, start_fp_num=25,
                      use_histogram=False,
                      adaptive_thresh=True,
                      tp_quantile=0.7,
                      fp_quantile=0.7,
                      max_iou_for_match=0.3,
                      score_weight_emd=True):
    """
    Run improved TSBP pipeline.

    Key improvements over baseline:
    - Per-class processing to avoid cross-class interference
    - Adaptive distance thresholds based on feature distribution
    - IoU-based spatial filtering to prevent matching overlapping boxes
    - Score-weighted EMD to prioritize high-confidence candidates
    """
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_all_root}/{model_name}/"
    out_dir = f"{results_inf_all_root}/{model_name}/tsbp_improved/"

    print("=" * 60)
    print("Improved TSBP: Conservative Enhancements")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Adaptive thresholds: {adaptive_thresh}")
    print(f"Score-weighted EMD: {score_weight_emd}")
    print(f"Max IoU for match: {max_iou_for_match}")
    print("=" * 60)

    detector = ImprovedTSBPDetector()
    detector.run_tsbp(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        tp_threshold=tp_threshold,
        fp_threshold=fp_threshold,
        start_tp_num=start_tp_num,
        start_fp_num=start_fp_num,
        use_histogram=use_histogram,
        adaptive_thresh=adaptive_thresh,
        tp_quantile=tp_quantile,
        fp_quantile=fp_quantile,
        max_iou_for_match=max_iou_for_match,
        score_weight_emd=score_weight_emd
    )



def run_adaptive_tsbp(model_name, data_root, results_inf_all_root,
                      tp_quantile=0.8, fp_quantile=0.2,
                      start_tp_num=25, start_fp_num=25,
                      lambda_weight=0.5, sigma_spatial=0.2,
                      use_histogram=False):
    """Run Adaptive TSBP pipeline"""
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_all_root}/{model_name}/"
    out_dir = f"{results_inf_all_root}/{model_name}/tsbp_adaptive/"

    print("=" * 60)
    print("Adaptive TSBP: Quantile-based Thresholds + Confidence Propagation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"TP quantile: {tp_quantile}, FP quantile: {fp_quantile}")
    print("=" * 60)

    detector = AdaptiveTSBP()
    detector.run_tsbp(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        tp_quantile=tp_quantile,
        fp_quantile=fp_quantile,
        start_tp_num=start_tp_num,
        start_fp_num=start_fp_num,
        lambda_weight=lambda_weight,
        sigma_spatial=sigma_spatial,
        use_histogram=use_histogram
    )


def run_hierarchical_tsbp(model_name, data_root, results_inf_all_root,
                          tp_quantile=0.8, fp_quantile=0.2,
                          k_neighbors=10, propagation_alpha=0.8,
                          use_histogram=False):
    """Run Hierarchical Feature TSBP pipeline"""
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_all_root}/{model_name}/"
    out_dir = f"{results_inf_all_root}/{model_name}/tsbp_hierarchical/"

    print("=" * 60)
    print("Hierarchical Feature TSBP: Multi-scale Features + Graph Propagation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"k-neighbors: {k_neighbors}, propagation alpha: {propagation_alpha}")
    print("=" * 60)

    detector = HierarchicalFeatureTSBP()
    detector.run_tsbp(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        tp_quantile=tp_quantile,
        fp_quantile=fp_quantile,
        k_neighbors=k_neighbors,
        propagation_alpha=propagation_alpha,
        use_histogram=use_histogram
    )


def run_tsbp_plusplus(model_name, data_root, results_inf_all_root,
                      tp_quantile=0.8, fp_quantile=0.2,
                      start_tp_num=25, start_fp_num=25,
                      uncertainty_weight=0.5,
                      max_rounds=10, convergence_patience=2,
                      use_histogram=False):
    """Run TSBP++ pipeline"""
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_all_root}/{model_name}/"
    out_dir = f"{results_inf_all_root}/{model_name}/tsbp_plusplus/"

    print("=" * 60)
    print("TSBP++: Unified Adaptive Multi-scale Framework")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"TP quantile: {tp_quantile}, FP quantile: {fp_quantile}")
    print(f"Uncertainty weight: {uncertainty_weight}")
    print(f"Max rounds: {max_rounds}, patience: {convergence_patience}")
    print("=" * 60)

    detector = TSBPPlusPlus()
    detector.run_tsbp(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        tp_quantile=tp_quantile,
        fp_quantile=fp_quantile,
        start_tp_num=start_tp_num,
        start_fp_num=start_fp_num,
        uncertainty_weight=uncertainty_weight,
        max_rounds=max_rounds,
        convergence_patience=convergence_patience,
        use_histogram=use_histogram
    )


def main_one_model():
    from constants import results_inf_all_root, data_root, MODEL

    # Option A: Adaptive TSBP
    run_adaptive_tsbp(
        model_name=MODEL,
        data_root=data_root,
        results_inf_all_root=results_inf_all_root,
        tp_quantile=0.8,
        fp_quantile=0.2,
        lambda_weight=0.5,
        sigma_spatial=0.1,
        use_histogram=False
    )

    # Option B: Hierarchical Feature TSBP
    run_hierarchical_tsbp(
        model_name=MODEL,
        data_root=data_root,
        results_inf_all_root=results_inf_all_root,
        tp_quantile=0.8,
        fp_quantile=0.2,
        k_neighbors=10,
        propagation_alpha=0.8,
        use_histogram=False
    )

    # Option C: TSBP++
    run_tsbp_plusplus(
        model_name=MODEL,
        data_root=data_root,
        results_inf_all_root=results_inf_all_root,
        tp_quantile=0.8,
        fp_quantile=0.2,
        uncertainty_weight=0.5,
        max_rounds=10,
        convergence_patience=2,
        use_histogram=False
    )


def main_all_models():
    from constants import results_inf_all_root, data_root, ALL_MODELS

    for MODEL in ALL_MODELS:
        # # Option A: Adaptive TSBP
        run_adaptive_tsbp(
            model_name=MODEL,
            data_root=data_root,
            results_inf_all_root=results_inf_all_root,
            tp_quantile=0.5,
            fp_quantile=0.4,
            lambda_weight=0.75,
            sigma_spatial=0.5,
            use_histogram=False
        )

        # # Option B: Hierarchical Feature TSBP
        run_hierarchical_tsbp(
            model_name=MODEL,
            data_root=data_root,
            results_inf_all_root=results_inf_all_root,
            tp_quantile=0.5,
            fp_quantile=0.4,
            k_neighbors=20,
            propagation_alpha=0.8,
            use_histogram=False
        )

        # # Option C: TSBP++
        run_tsbp_plusplus(
            model_name=MODEL,
            data_root=data_root,
            results_inf_all_root=results_inf_all_root,
            tp_quantile=0.5,
            fp_quantile=0.4,
            start_tp_num=25,
            start_fp_num=25,
            uncertainty_weight=0.5,
            max_rounds=10,
            convergence_patience=2,
            use_histogram=False
        )

        run_tsbp_pipeline(
            model_name=MODEL,
            data_root=data_root,
            results_inf_all_root=results_inf_all_root,
            tp_threshold=0.50,
            fp_threshold=0.30,
            start_tp_num=25,
            start_fp_num=25,
            use_histogram=False,
            adaptive_thresh=True,
            tp_quantile=0.6,
            fp_quantile=0.9,
            max_iou_for_match=0.5,
            score_weight_emd=True
        )


if __name__ == "__main__":
    # main_one_model()
    main_all_models()

