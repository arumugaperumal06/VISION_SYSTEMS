# main.py
from vision_guard import VisionGuard
import time

def run_demo(selected_model):
    guard = VisionGuard(model_type=selected_model)
    
    print(f"\n--- [SYST] Vision Health & Drift Diagnostics ---")
    header = f"{'MODEL':<7} | {'CONF':<5} | {'ERROR':<5} | {'DECISION':<18} | {'DRIFT TYPE'}"
    print(header)
    print("-" * len(header) + "-----------------------")

    try:
        while True:
            metrics, decision = guard.get_adaptation_decision()
            
            output = (f"{metrics['model']:<7} | "
                      f"{metrics['confidence']:<5} | "
                      f"{metrics['drift_error']:<5} | "
                      f"{decision:<18} | "
                      f"{metrics['drift_type']}")
            
            print(output)
            time.sleep(1.2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    run_demo(selected_model="ResNet")
