from ultralytics import YOLO

# Load the model
model = YOLO(r"runs/detect/train/weights/best.pt")

# Run prediction on one image and save results
results = model.predict(source=r"D:\final\urban_dataset_yolo\val-images-300", save=True)


print("✅ Prediction complete. Output saved in: runs/detect/predict")
