# 🌊 Road Flood Detection

An automated computer vision project that detects and segments flooded road areas using a custom-trained **YOLOv8 Nano Segmentation model**.

## 🚀 Overview

This project uses deep learning-based instance segmentation to identify and outline water accumulation on roads from images and videos. It aims to support road safety, flood monitoring, and disaster management.

## ✨ Key Features

- Flood detection and segmentation using YOLOv8.
- Image and video-based inference.
- Visual segmentation masks highlighting flooded regions.
- Lightweight YOLOv8 Nano architecture.

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** PyTorch, Ultralytics YOLOv8
- **Libraries:** OpenCV, NumPy, PyYAML

## 📂 Repository Structure

```text
Road-Flood-Detection/
│
├── data.yaml
├── train.py
├── test_flood.py
├── .gitignore
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Road-Flood-Detection.git
cd Road-Flood-Detection
pip install ultralytics opencv-python numpy pyyaml
```

## 💻 Usage

**Train the model:**
```bash
python train.py
```

**Run flood detection:**
```bash
python test_flood.py
```

## 📊 Dataset & Model

- **Model:** YOLOv8 Nano Segmentation (`yolov8n-seg.pt`)
- **Task:** Instance Segmentation
- **Dataset:** Custom annotated flood dataset compatible with YOLO format.
- **Configuration:** `data.yaml`

## 📸 Results

The model generates segmentation masks highlighting flooded regions in road images and videos.

*Add screenshots of the original input and segmentation output here.*

## 🔮 Future Scope

- Real-time CCTV-based flood monitoring.
- GPS-based flood location identification.
- Automated road hazard alerts.

## 👥 Contributors

- Pratiksha Baskaran
- Jayshri MS
- Monish RV
