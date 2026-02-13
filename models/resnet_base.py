import torch
import torch.nn as nn
from torchvision import models

def get_model():
    # Base: ResNet-18 [cite: 62]
    model = models.resnet18(weights='DEFAULT')
    # Custom classifier head [cite: 64]
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return model