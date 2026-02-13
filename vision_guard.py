# vision_guard.py
from monitor import UniversalModelMonitor

# vision_guard.py

class VisionGuard:
    def __init__(self, model_type="YOLO"):
        # If it's ResNet, make the requirement for Global Adaptation harder (0.85 instead of 0.7)
        rel_threshold = 0.85 if model_type == "ResNet" else 0.7
        self.monitor = UniversalModelMonitor(model_type=model_type, reliability_threshold=rel_threshold)

   # vision_guard.py

    def get_adaptation_decision(self):
        metrics = self.monitor.get_metrics()
        
        # 1. Check for Problem (Local)
        if metrics['drift_error'] > self.monitor.t_drift:
            decision = "LOCAL (Correction)"
        
        # 2. Check for Excellence (Global)
        elif metrics['reliability'] > self.monitor.t_reliable:
            # This is where 'No Drift' occurs but 'Adaptation' is suggested
            decision = "GLOBAL (FL Share)"
        
        # 3. Standard Operation
        else:
            decision = "STABLE"
            
        return metrics, decision
            
    