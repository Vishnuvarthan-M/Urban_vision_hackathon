import os
import json

# Paths
json_path = "urbanvision_coco.json"
images_dir = r"D:\final\urban_dataset_yolo\images"   # where your images will go
labels_dir = r"D:\final\urban_dataset_yolo\labels"   # where your txt files will be saved
os.makedirs(labels_dir, exist_ok=True)

# Load COCO JSON
with open(json_path, 'r') as f:
    coco = json.load(f)

# Build image id to file name map
img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

# Write YOLO .txt label files
for ann in coco['annotations']:
    image_id = ann['image_id']
    file_name = img_id_to_filename[image_id]
    width, height = img_id_to_size[image_id]

    class_id = ann['category_id'] - 1  # YOLO expects 0-based class indices
    x, y, w, h = ann['bbox']

    # Convert to YOLO format
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w /= width
    h /= height

    # Save to corresponding .txt file
    label_file = os.path.join(labels_dir, file_name.replace('.png', '.txt'))
    with open(label_file, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ YOLO .txt label files created.")
