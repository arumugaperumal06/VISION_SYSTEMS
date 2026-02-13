import torch
from ultralytics import YOLO

def get_detection_model():
    model = YOLO('yolov8n.pt') 
    return model