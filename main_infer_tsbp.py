# https://arxiv.org/abs/2409.16678
# https://github.com/jwhgdeu/TSBP
# Instead of applying a single global confidence threshold to keep or discard predicted boxes, TSBP:
# 1. takes all detector output boxes from test images,
# 2. extracts features for each box (crop → feature extractor π, e.g., ResNet-50),
# 3. designates high-confidence boxes (per class) as confirmed seeds,
# 4. iteratively matches candidate (low-confidence) boxes to confirmed boxes using feature distances (EMD / assignment), and
# 5. propagates class labels from confirmed boxes to similar candidates — first under a strict distance constraint, then relaxing it.

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


class TSBPDetector:
    """Test-time Self-guided Bounding-box Propagation for Object Detection"""

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize ResNet50 feature extractor"""
        resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT) # weights=ResNet50_Weights.IMAGENET1K_V1 / weights=ResNet50_Weights.DEFAULT to get the most up-to-date weights
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
        hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_features_with_hist(self, image_pil):
        """Extract combined features: ResNet50 + color histogram"""
        # Deep features
        deep_features = self.extract_features(image_pil)

        # Color histogram
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        hist = self.calc_hist(image_bgr) * 15

        # Concatenate
        features = torch.from_numpy(np.concatenate((hist, deep_features.numpy())))
        return features

    def load_detections(self, txt_path, img_path, use_histogram=False):
        """
        Load detections from YOLO format text file.

        Format per line: class_id, x, y, w, h, score
        Returns list of detection dicts with bbox, score, and features
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
                if use_histogram:
                    features = self.extract_features_with_hist(crop)
                else:
                    features = self.extract_features(crop)

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'bbox_orig': (x, y, w, h),  # Keep original format
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

        # Create representative detections from cluster centers
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

    def run_tsbp(self, image_dir, infer_dir, out_dir,
                 tp_threshold=0.50, fp_threshold=0.30,
                 start_tp_num=25, start_fp_num=25,
                 use_histogram=False):
        """
        Run TSBP algorithm on detection results.

        Args:
            image_dir: Directory containing original images
            infer_dir: Directory containing detection .txt files
            out_dir: Output directory for refined detections
            tp_threshold: Confidence threshold for high-confidence (TP) detections
            fp_threshold: Confidence threshold below which detections are FP
            start_tp_num: Number of clusters for TP K-means
            start_fp_num: Number of clusters for FP K-means
            use_histogram: Whether to use color histogram features
        """
        os.makedirs(out_dir, exist_ok=True)

        # Get all detection files
        txt_files = [f for f in os.listdir(infer_dir) if f.endswith('.txt')]

        if not txt_files:
            print(f"No .txt files found in {infer_dir}")
            return

        print(f"Processing {len(txt_files)} detection files...")

        # Collect all detections from all images
        all_tp_orig = []
        all_fp_orig = []
        all_candidates = []
        image_detections = {}  # Store detections per image for final output

        for txt_file in txt_files:
            img_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(infer_dir, txt_file)

            # Try different image extensions
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                test_path = os.path.join(image_dir, img_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            if img_path is None:
                print(f"Warning: Image not found for {img_name}")
                continue

            # Load detections
            detections = self.load_detections(txt_path, img_path, use_histogram)

            if not detections:
                image_detections[img_name] = []
                continue

            # Add image name to each detection
            for det in detections:
                det['img_name'] = img_name

            # Separate by confidence
            for det in detections:
                if det['score'] >= tp_threshold:
                    all_tp_orig.append(det)
                elif det['score'] < fp_threshold:
                    all_fp_orig.append(det)
                else:
                    all_candidates.append(det)

            image_detections[img_name] = []

        print(f"Initial counts - TP: {len(all_tp_orig)}, FP: {len(all_fp_orig)}, "
              f"Candidates: {len(all_candidates)}")

        if len(all_tp_orig) == 0 or len(all_candidates) == 0:
            print("Insufficient detections for TSBP. Outputting original high-confidence detections.")
            self._save_results(all_tp_orig, image_detections, out_dir)
            return

        # Apply K-means to get representative samples
        boxes_tp, tp_feats = self.kmeans_clustering(all_tp_orig, start_tp_num)
        boxes_fp, fp_feats = self.kmeans_clustering(all_fp_orig, start_fp_num) if len(all_fp_orig) > 0 else ([], [])

        # Calculate distance constraints
        dist_avg_tp, glob_dis_tp = self.cal_min_dist_stats(all_tp_orig)
        dist_avg_fp, glob_dis_fp = self.cal_min_dist_stats(all_fp_orig) if len(all_fp_orig) > 0 else (float('inf'), float('inf'))

        print(f"Distance constraints - TP: {dist_avg_tp:.2f}, FP: {dist_avg_fp:.2f}")

        # Remove overlapping clusters between TP and FP
        boxes_tp_sub = []
        boxes_fp_sub = []
        del_tp_idx = set()
        del_fp_idx = set()

        for i, tp_feat in enumerate(tp_feats):
            for j, fp_feat in enumerate(fp_feats):
                if torch.dist(tp_feat, fp_feat) < glob_dis_tp:
                    del_tp_idx.add(i)
                    del_fp_idx.add(j)

        # Separate deleted clusters
        for idx in sorted(del_tp_idx, reverse=True):
            boxes_tp_sub.append(boxes_tp.pop(idx))
        for idx in sorted(del_fp_idx, reverse=True):
            boxes_fp_sub.append(boxes_fp.pop(idx))

        # Multi-round EMD matching
        thresh_dist_tp = dist_avg_tp
        thresh_dist_fp = dist_avg_fp
        use_strict = True
        round_num = 0

        while len(all_candidates) > 0:
            round_num += 1
            print(f"\n=== Round {round_num} ===")

            # Prepare for EMD
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

            # Distance matrix
            Q_dets = boxes_tp + boxes_fp
            D = np.zeros((full_len, full_len))

            for i in range(len_cand):
                for j in range(len_tp + len_fp):
                    dist = torch.dist(all_candidates[i]['feature'], Q_dets[j]['feature']).item()
                    D[i][len_cand + j] = dist
                    D[len_cand + j][i] = dist

            # Run EMD using POT
            # Normalize distributions
            P_norm = P / P.sum() if P.sum() > 0 else P
            Q_norm = Q / Q.sum() if Q.sum() > 0 else Q

            # Compute optimal transport plan
            # ot.emd returns the transport matrix (flow)
            flow = ot.emd(P_norm, Q_norm, D)

            # Scale flow back to original distribution scale
            flow = flow * min(P.sum(), Q.sum())

            # Collect matched candidates
            tp_matched = []
            fp_matched = []

            for i in range(len_cand):
                for j in range(len_tp + len_fp):
                    if flow[i][len_cand + j] > 0:
                        cand = all_candidates[i]
                        dist = D[i][len_cand + j]

                        if j < len_tp:  # Matched with TP
                            tp_matched.append({'cand': cand, 'dist': dist})
                        else:  # Matched with FP
                            fp_matched.append({'cand': cand, 'dist': dist})

            # Sort by distance
            tp_matched.sort(key=lambda x: x['dist'])
            fp_matched.sort(key=lambda x: x['dist'])

            print(f"Matched - TP: {len(tp_matched)}, FP: {len(fp_matched)}")
            if tp_matched:
                print(f"  TP dist range: [{tp_matched[0]['dist']:.2f}, {tp_matched[-1]['dist']:.2f}]")
            if fp_matched:
                print(f"  FP dist range: [{fp_matched[0]['dist']:.2f}, {fp_matched[-1]['dist']:.2f}]")

            # Add candidates to confirmed sets
            added_cands = []
            tp_add = 0
            fp_add = 0

            for match in tp_matched:
                if match['dist'] <= thresh_dist_tp:
                    cand = match['cand']
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

            # Check termination or relaxation
            if tp_add == 0 and fp_add == 0:
                if use_strict:
                    # Add back separated clusters and relax constraints
                    print("Relaxing constraints and adding separated clusters")
                    boxes_tp.extend(boxes_tp_sub)
                    boxes_fp.extend(boxes_fp_sub)
                    thresh_dist_tp = float('inf')
                    thresh_dist_fp = float('inf')
                    use_strict = False
                else:
                    print("No more candidates can be matched. Stopping.")
                    break

        print(f"\n=== TSBP Complete ===")
        print(f"Final TP count: {len(all_tp_orig)}")
        print(f"Remaining candidates: {len(all_candidates)}")

        # Save results
        self._save_results(all_tp_orig, image_detections, out_dir)

    def _save_results(self, final_detections, image_detections, out_dir):
        """Save refined detection results to text files"""
        # Group by image
        for det in final_detections:
            img_name = det.get('img_name')
            if img_name:
                if img_name not in image_detections:
                    image_detections[img_name] = []
                image_detections[img_name].append(det)

        # Write files
        for img_name, dets in image_detections.items():
            out_path = os.path.join(out_dir, f"{img_name}.txt")
            with open(out_path, 'w') as f:
                for det in dets:
                    x, y, w, h = det['bbox_orig']
                    score = det['score']
                    class_id = det['class']
                    f.write(f"{class_id}, {int(x)}, {int(y)}, {int(w)}, {int(h)}, {score}\n")

        print(f"Results saved to {out_dir}")


def run_tsbp_pipeline(model_name, data_root, results_inf_root,
                      tp_threshold=0.50, fp_threshold=0.30,
                      start_tp_num=25, start_fp_num=25,
                      use_histogram=False):
    """
    Convenience function to run TSBP pipeline.

    Args:
        model_name: Name of the model (for folder structure)
        data_root: Root directory containing images
        results_inf_root: Root directory containing inference results
        tp_threshold: Confidence threshold for high-confidence detections
        fp_threshold: Confidence threshold for low-confidence detections
        start_tp_num: Number of TP clusters
        start_fp_num: Number of FP clusters
        use_histogram: Whether to use color histogram features
    """
    image_dir = f"{data_root}/images/test/"
    infer_dir = f"{results_inf_root}/{model_name}/"
    out_dir = f"{results_inf_root}/{model_name}/tsbp/"

    print("=" * 60)
    print("TSBP: Test-time Self-guided Bounding-box Propagation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Image directory: {image_dir}")
    print(f"Inference directory: {infer_dir}")
    print(f"Output directory: {out_dir}")
    print(f"TP threshold: {tp_threshold}")
    print(f"FP threshold: {fp_threshold}")
    print("=" * 60)

    detector = TSBPDetector()
    detector.run_tsbp(
        image_dir=image_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        tp_threshold=tp_threshold,
        fp_threshold=fp_threshold,
        start_tp_num=start_tp_num,
        start_fp_num=start_fp_num,
        use_histogram=use_histogram
    )


if __name__ == "__main__":
    # Example usage
    from constants import results_inf_root, data_root, MODEL

    run_tsbp_pipeline(
        model_name=MODEL,
        data_root=data_root,
        results_inf_root=results_inf_root,
        tp_threshold=0.50, # accepted
        fp_threshold=0.30, # discarded, between are candidates?
        start_tp_num=25,
        start_fp_num=25,
        use_histogram=True  # Set to True for cell detection tasks
    )