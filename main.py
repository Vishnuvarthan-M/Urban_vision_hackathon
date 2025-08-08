import os
import json
from ultralytics import YOLO
from PIL import Image

# ---------------- CONFIGURATION ----------------
model_path = r"D:\final\urban_dataset_yolo\runs\detect\train\weights\best.pt"
image_dir = r"D:\final\urban_dataset_yolo\val-images-300"
output_json = "predictions_coco_format.json"

# Categories (use the exact ones from your task)
categories = [
    {"id": 1, "name": "Hatchback"},
    {"id": 2, "name": "Sedan"},
    {"id": 3, "name": "SUV"},
    {"id": 4, "name": "MUV"},
    {"id": 5, "name": "Bus"},
    {"id": 6, "name": "Truck"},
    {"id": 7, "name": "Three-wheeler"},
    {"id": 8, "name": "Two-wheeler"},
    {"id": 9, "name": "LCV"},
    {"id": 10, "name": "Mini-bus"},
    {"id": 11, "name": "Mini-truck"},
    {"id": 12, "name": "tempo-traveller"},
    {"id": 13, "name": "bicycle"},
    {"id": 14, "name": "Van"},
    {"id": 15, "name": "Others"}
]

# ---------------- LOAD MODEL ----------------
model = YOLO(model_path)

# Get all image filenames
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Predict without stream=True (avoids bug)
results = list(model.predict(
    source=image_dir,
    save=False,
    save_txt=False,
    save_conf=True
))

# ---------------- CONVERT TO COCO ----------------
images = []
annotations = []
annotation_id = 1
image_id = 1

for img_file, result in zip(image_files, results):
    img_path = os.path.join(image_dir, img_file)
    with Image.open(img_path) as img:
        width, height = img.size

    images.append({
        "id": image_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    for box in result.boxes:
        cls_id = int(box.cls[0]) + 1  # YOLO is 0-based, COCO is 1-based
        x, y, w, h = box.xywh[0].tolist()

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cls_id,
            "bbox": [x - w / 2, y - h / 2, w, h],
            "area": w * h,
            "iscrowd": 0,
            "score": float(box.conf[0])
        })

        annotation_id += 1

    image_id += 1

# ---------------- SAVE JSON ----------------
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"✅ Saved COCO-format predictions to {output_json}")
