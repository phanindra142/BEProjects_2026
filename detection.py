# ------------------------------------------------------------
# Basketball Player Detection - YOLOv12 Training Script
# Author: Mathesh Vishnu
# ------------------------------------------------------------
# This script trains a YOLOv12 model on a custom basketball 
# dataset. It uses an optimized training setup for improved 
# convergence, speed, and accuracy.
# ------------------------------------------------------------

from ultralytics import YOLO
import cv2

# ------------------------------------------------------------
# Load Base Model
# ------------------------------------------------------------
# Using YOLOv12s (small variant) for faster training & inference
# Move model to Apple Silicon GPU (Metal Performance Shaders - MPS)
# for accelerated performance on macOS.
# ------------------------------------------------------------
model = YOLO("yolo12s.pt")
model.to("mps")

# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------
# - data.yaml: Path to dataset config file containing class names
# - epochs: Total number of epochs for training (150)
# - imgsz: Image size for training (1024x1024)
# - batch: Mini-batch size (4)
# - workers: Number of dataloader workers (4)
# - optimizer: AdamW for stability with weight decay
# - patience: Early stopping if no improvement for 15 epochs
# - warmup_epochs: Gradual learning rate warmup for stability
# - lr0: Initial learning rate
# - weight_decay: Regularization to prevent overfitting
# - close_mosaic: Disable mosaic augmentation in last 10 epochs
# - dropout: Dropout regularization in backbone/neck
# ------------------------------------------------------------
results = model.train(
    data="/Users/vmathesh/basketball/Basketball Players.v1-yolov12-basketball-detection.yolov12/data.yaml",
    epochs=150,
    imgsz=1024,
    batch=4,
    workers=4,
    optimizer="AdamW",
    patience=15,
    warmup_epochs=3.0,
    lr0=0.0005,
    weight_decay=0.0005,
    close_mosaic=10,
    dropout=0.15,
    verbose=True
)

# ------------------------------------------------------------
# Notes:
# ------------------------------------------------------------
# ✅ YOLOv12s chosen for balance of accuracy & inference speed.
# ✅ High-resolution images (1024) used for better small object detection.
# ✅ AdamW + Weight Decay improves generalization over SGD.
# ✅ Early stopping ensures efficiency during long training runs.
# ✅ Dropout + Mosaic augments model robustness against overfitting.
#
# Final trained weights will be saved under:
#   runs/detect/train*/weights/best.pt
# ------------------------------------------------------------
