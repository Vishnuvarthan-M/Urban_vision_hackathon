# 🚦 Urban Vision Hackathon – Winning Solution 🏆

**1st Place – IISc Bangalore | ₹1,00,000 Cash Prize | June 20–23, 2025**

Our team developed a **hybrid AI-based vehicle detection system** for the **Urban Vision Hackathon** hosted at **Indian Institute of Science (IISc) Bangalore**.  
We combined **YOLOv8** and **Faster R-CNN** in a **model ensembling pipeline** to detect both **common** and **rare vehicle categories** with high accuracy, achieving top performance among 20 finalist teams.

---

## 🏆 Achievement
- **Winner** – Urban Vision Hackathon 2025 at IISc Bangalore
- **₹1,00,000 cash prize**
- Selected for a **3-month research internship** at IISc Bangalore
- Outperformed competing models with **rare class detection boost** using hybrid architecture

---

## 📌 Problem Statement
Traffic authorities face challenges in detecting **rare vehicle types** in urban road surveillance data. Standard models perform well on common classes but struggle with rare ones like:
- **Mini-bus**
- **LCV (Light Commercial Vehicle)**
- **Van**
- **Bicycle**
- **Tempo Traveller**

Our goal was to **increase detection accuracy** for rare classes without compromising performance on common classes.

---

## 🚀 Solution Overview
We implemented a **two-stage ensemble approach**:

1. **YOLOv8** – Trained on **all vehicle classes** for **real-time detection**.
2. **Faster R-CNN** – Trained **only on rare vehicle classes** using a custom COCO-format dataset.
3. **Ensemble Logic** – Selects **best bounding box per class** across both models based on:
   - Confidence score
   - IoU-based deduplication

---

## 📊 Results
| Model         | mAP@50 | mAP@75 | mAP@50-95 | Rare Class Accuracy |
|---------------|--------|--------|-----------|---------------------|
| YOLOv8        | High   | High   | Good      | Moderate            |
| Faster R-CNN  | Moderate | Moderate | Good  | **High**            |
| **Ensemble**  | **High** | **High** | **Best** | **Best**           |

---

## 🗂 Dataset
- **Source:** Provided by Bangalore Traffic Police (Confidential dataset – not publicly available)
- Format: COCO JSON with `images`, `annotations`, and `categories`
- Split: `train / val / test`

---

## 📦 Project Structure

UrbanVision-Winning-Solution/
│── data/ # Dataset (not included in repo)
│── yolov8_model/ # YOLOv8 training & weights
│── faster_rcnn_model/ # Faster R-CNN training & weights
│── ensemble/ # Ensemble scripts & visualizations
│── results/ # Output predictions & charts
│── requirements.txt # Python dependencies
│── README.md # Project documentation
└── main.py # Entry point for running ensemble inference
📌 Technologies Used
Python 3.10+

YOLOv8 – Ultralytics

Faster R-CNN – PyTorch

OpenCV – Image processing


