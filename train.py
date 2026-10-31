
from ultralytics import YOLO
from pathlib import Path
import torch

# Dataset configuration
data_file = Path("data.yaml").resolve()

if not data_file.exists():
    raise FileNotFoundError("data.yaml not found!")

# Check GPU availability
device = 0 if torch.cuda.is_available() else "cpu"
print("Training on:", device)

# Load pretrained segmentation model
model = YOLO("yolov8n-seg.pt")

# Train road flood detection model
model.train(
    data=str(data_file),
    epochs=50,
    imgsz=640,
    batch=4 if device != "cpu" else 2,
    device=device,
    workers=0,
    project=str(Path("runs").resolve()),
    name="road_flood_model",
    exist_ok=True,
    patience=15
)

print("Training completed!")
print("Model saved in:")
print("runs/road_flood_model/weights/best.pt")
