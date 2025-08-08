from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Change this to the correct filename
ground_truth_file = r"D:\final\urban_dataset_yolo\val_annotations.json"
prediction_file = r"D:\final\urban_dataset_yolo\predictions_coco_format1.json"

# Load data
coco_gt = COCO(ground_truth_file)
coco_dt = coco_gt.loadRes(prediction_file)

# Evaluate
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
