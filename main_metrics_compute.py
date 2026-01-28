from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# paths
gt_json = "./data/nuclei/ground_truth_labels_test.json"                     # ground-truth COCO ann file
pred_json = "./results_data_nuclei/runs_test/yolow/val3/predictions.json"   # predictions saved by ultralytics (COCO format)

# load
cocoGt = COCO(gt_json)
cocoDt = cocoGt.loadRes(pred_json)


cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
cocoEval.params.iouThrs = np.array([0.5])  # single IoU threshold
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
map50 = cocoEval.stats[0]   # a float in [0,1]
print("mAP@50:", map50)


cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
map50_95 = cocoEval.stats[0]   # AP IoU=0.5:0.95
print("mAP@50-95:", map50_95)


# prepare evaluator: single IoU threshold (change to np.array([0.5]) or others)
cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
cocoEval.params.iouThrs = np.array([0.5])   # single IoU (use [0.5] for AP@50)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()  # prints the usual COCO table

# ---------- extract precision / recall arrays ----------
# After evaluate(), cocoEval.eval contains:
#   'precision' : array (T, R, K, A, M)
#   'recall'    : array (T, K, A, M)
# Where:
#   T = number of IoU thresholds (we used 1)
#   R = number of recall thresholds (default 101)
#   K = number of categories
#   A = number of area ranges
#   M = number of maxDets settings

# get the precision array
precision = cocoEval.eval['precision']  # shape [T, R, K, A, M]
recall_arr = cocoEval.params.recThrs    # length R, the recall thresholds used

# choose IoU index (we used only one IoU = 0.5 so index 0)
t = 0

# average precision across categories, area ranges and maxDets
# precision[t, :, :, :, :] has shape [R, K, A, M]
p = precision[t]  # shape (R, K, A, M)

# mask out missing values: COCO uses -1 to mark missing entries
valid_mask = p > -1
# sum of valid entries per recall bin
valid_counts = valid_mask.sum(axis=(1,2,3)).astype(np.float32)  # shape (R,)
# sum of precision values (with -1 filtered via mask)
p_sum = (np.where(valid_mask, p, 0.0)).sum(axis=(1,2,3))         # shape (R,)

# avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    p_mean_per_rec = np.where(valid_counts > 0, p_sum / valid_counts, -1.0)  # shape (R,)

# Now compute F1 at each recall point:
# For each recall threshold r_i, the corresponding precision is p_mean_per_rec[i]
r = recall_arr  # vector of recall thresholds
# keep only valid precision points
valid_idx = p_mean_per_rec > -1

# compute F1 safely
eps = 1e-8
p_valid = p_mean_per_rec[valid_idx]
r_valid = r[valid_idx]
f1_per_point = 2 * p_valid * r_valid / (p_valid + r_valid + eps)

# get best F1 and corresponding precision/recall
if f1_per_point.size > 0:
    best_idx = np.nanargmax(f1_per_point)
    best_f1 = f1_per_point[best_idx]
    best_precision = p_valid[best_idx]
    best_recall = r_valid[best_idx]
else:
    best_f1 = np.nan
    best_precision = np.nan
    best_recall = np.nan

# also compute simple summaries
# mean precision across recall points (ignoring -1)
mean_precision = np.mean(p_valid) if p_valid.size > 0 else np.nan

# COCO's eval['recall'] gives one recall value per IoU x category x area x maxDet.
recall_array = cocoEval.eval['recall']  # shape (T, K, A, M)
recall_t = recall_array[t]  # (K, A, M)
recall_mask = recall_t > -1
recall_mean = np.mean(recall_t[recall_mask]) if recall_mask.sum() > 0 else np.nan

# print results
print(f"\nSummary (IoU = {cocoEval.params.iouThrs[t]:.2f}):")
print(f"Mean precision (averaged across recall points, cats, areas, maxDets): {mean_precision:.4f}")
print(f"Mean recall (averaged across cats, areas, maxDets): {recall_mean:.4f}")
print(f"Best F1 on the PR curve: {best_f1:.4f}")
print(f"  -> precision = {best_precision:.4f}, recall = {best_recall:.4f} at that point")
