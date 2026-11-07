
---

# Road Flood Detection 

An automated computer vision project designed to detect and segment flooded areas on roads using a custom-trained **YOLOv8 nano segmentation model (`yolov8n-seg.pt`)**. This system is built to assist in real-time disaster monitoring, road safety, and automated hazard analysis.

---

## 🚀 Project Overview

Traditional flood monitoring relies heavily on manual inspections or stationary infrastructure, which can be slow during emergency responses. This project utilizes deep learning-based **instance and semantic segmentation** via Ultralytics YOLOv8 to accurately outline water bodies and flooded roadways from images or video feeds.

### Key Features:

* **Custom Segmentation:** Detects and delineates water pixels on roads using fine-tuned YOLOv8 segmentation weights.
* **Automated Inference Scripts:** Includes dedicated scripts for training and testing/validation.
* **Lightweight & Efficient:** Optimized using the YOLOv8n architecture for rapid processing.

---

## 📂 Repository Structure

```text
Road-Flood-Detection/
│
├── data.yaml          # Dataset configuration file for YOLO training
├── train.py           # Script for training the YOLOv8 segmentation model
├── test_flood.py      # Script for running flood detection/inference
├── .gitignore         # Specifies intentionally untracked files (datasets & heavy weights)
└── README.md          # Project documentation

```





## 🛠️ Tech Stack & Prerequisites

* **Language:** Python
* **Framework:** Ultralytics YOLOv8 (PyTorch)
* **Libraries:** OpenCV, NumPy, PyYAML

## 💻 Usage Instructions

### 1. Training the Model

To train the YOLOv8 segmentation model using your custom dataset configuration (`data.yaml`):


python train.py


### 2. Running Flood Detection / Testing

To run inference and test the model on validation/test images or video streams:

```bash
python test_flood.py

```

---

## 📊 Dataset

The model is configured to train on annotated image datasets (compatible with Roboflow export formats) mapped via `data.yaml`. Each image undergoes processing to detect pixel-level water accumulation hazards on roadways.

---
