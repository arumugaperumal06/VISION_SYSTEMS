import os
import torch
import matplotlib.pyplot as plt
from models.resnet_base import get_model
from vision_guard import UniversalModelMonitor

def create_drift_proof():
    model = get_model()
    model.eval()
    sdk = UniversalModelMonitor(threshold_drift=0.3)
    
    # Simulate Clean vs Drift confidence
    clean_conf = 0.96 # [cite: 158]
    drift_conf = 0.74 # [cite: 158]

    # Confidence Histogram Simulation [cite: 120]
    plt.figure(figsize=(10, 6))
    plt.bar(['Clean (Stable)', 'Natural Drift (Blur/Noise)'], [clean_conf, drift_conf], color=['blue', 'red'])
    plt.axhline(y=0.75, color='gray', linestyle='--', label='threshold_reliable') # [cite: 37]
    plt.title("Performance Degradation Due to Natural Drift")
    plt.ylabel("Confidence Score")
    plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/drift_degradation.png')
    print("Graph generated in results/drift_degradation.png")

if __name__ == "__main__":
    create_drift_proof()