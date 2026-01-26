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
