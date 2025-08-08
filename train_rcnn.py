import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import os

from engine import train_one_epoch, evaluate  # from torchvision references

# Define rare classes and category IDs
RARE_CLASSES = ["Van", "Mini-bus", "Tempo-Traveler", "Bicycle", "LCV"]
RARE_CLASS_IDS = [8, 9, 10, 11, 12]  # Adjust if needed

# Custom COCO Dataset Wrapper
class CocoDetectionRCNN(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

    # Load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')

        boxes, labels, areas = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 1 or h <= 1:
                continue  # ❗ Skip invalid boxes
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(w * h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

    # If all annotations were skipped, return dummy box (to avoid crashing)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        target = {
        'boxes': boxes,
        'labels': labels,
        'image_id': torch.tensor([img_id]),
        'area': areas,
        'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


    def __len__(self):
        return len(self.ids)

# Transformations
transform = T.Compose([T.ToTensor()])

# Dataset and Dataloader
dataset = CocoDetectionRCNN(
    root=r"D:\final\urban_dataset_yolo\images\train",
    annFile=r"D:\final\urban_dataset_yolo\urbanvision_coco_rare_mapped_cleaned.json",
    transforms=transform
)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
model = fasterrcnn_resnet50_fpn(num_classes=len(RARE_CLASS_IDS) + 1)  # +1 for background
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
for epoch in range(10):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
    torch.save(model.state_dict(), f"faster_rcnn_rare_epoch{epoch}.pth")
