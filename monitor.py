# monitor.py
import random

class UniversalModelMonitor:
    def __init__(self, model_type="YOLO", drift_threshold=0.3, reliability_threshold=0.7):
        self.model_type = model_type
        self.t_drift = drift_threshold
        self.t_reliable = reliability_threshold

    def diagnose_drift(self, drift_score, confidence):
        if drift_score <= self.t_drift:
            return "NONE"
        if confidence < 0.6:
            return "DATA DRIFT (Low Light)"
        elif drift_score > 0.45:
            return "CONCEPT DRIFT (New Class)"
        else:
            return "FEATURE DRIFT (Blur/Noise)"

    def get_metrics(self):
        if self.model_type == "YOLO":
            mean_conf = round(random.uniform(0.50, 0.90), 2)
            entropy = round(random.uniform(0.1, 0.45), 2)
        else: # ResNet
            # UPDATED: Widened the range so ResNet can occasionally 'drift'
            mean_conf = round(random.uniform(0.55, 0.98), 2) 
            entropy = round(random.uniform(0.05, 0.40), 2)
        
        drift_score = round(((1 - mean_conf) * 0.5) + (entropy * 0.5), 2)
        reliability_score = round(1 - drift_score, 2)
        drift_type = self.diagnose_drift(drift_score, mean_conf)
        
        return {
            "model": self.model_type, "confidence": mean_conf,
            "drift_error": drift_score, "reliability": reliability_score,
            "drift_type": drift_type
        }