import json
import os
from collections import defaultdict
from PIL import Image

# ---------------- CONFIG ----------------
yolo_path = r"D:\final\urban_dataset_yolo\predictions_coco_format.json"
rcnn_path = r"D:\final\urban_dataset_yolo\faster_rcnn_rare_preds.json"
image_dir = r"D:\final\urban_dataset_yolo\val-images-300"
output_path = r"D:\final\urban_dataset_yolo\ensemble_coco_new.json"

rare_class_ids = [8, 9, 10, 11, 12]

# ---------------- LOAD ----------------
with open(yolo_path) as f:
    yolo_data = json.load(f)

with open(rcnn_path) as f:
    rcnn_data = json.load(f)

categories = yolo_data["categories"]  # Assuming categories same for both
category_ids = {cat["id"] for cat in categories}

# ---------------- GROUP ----------------
def group_by_image(preds):
    grouped = defaultdict(list)
    for ann in preds:
        grouped[ann["image_id"]].append(ann)
    return grouped

yolo_by_img = group_by_image(yolo_data["annotations"])
rcnn_by_img = group_by_image(rcnn_data)

# ---------------- ENSEMBLE ----------------
ensemble_annotations = []
images = []
ann_id = 1
image_id = 1

filename_to_id = {}

# Load image metadata from image_dir
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
for img_file in image_files:
    file_path = os.path.join(image_dir, img_file)
    with Image.open(file_path) as img:
        width, height = img.size

    filename_to_id[img_file] = image_id
    images.append({
        "id": image_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    # YOLO: exclude rare classes
    for ann in yolo_by_img.get(image_id, []):
        if ann["category_id"] not in rare_class_ids:
            ensemble_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0
            })
            ann_id += 1

    # R-CNN: only rare classes already
    for ann in rcnn_by_img.get(image_id, []):
        if ann["category_id"] in category_ids:
            ensemble_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0
            })
            ann_id += 1

    image_id += 1

# ---------------- OUTPUT ----------------
coco_output = {
    "images": images,
    "annotations": ensemble_annotations,
    "categories": categories
}

with open(output_path, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"✅ COCO-style ensemble saved to: {output_path}")
