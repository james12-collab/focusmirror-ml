import sys 
import subprocess 
def run_step(name, script_path): 
    print(f"\n==========================================") 
    print(f"=== STEP: {name}") 
    print(f"==========================================") 
    res = subprocess.run([sys.executable, script_path]) 
    if res.returncode != 0: 
        sys.exit(1) 
def main(): 
    print("=== FocusMirror Complete ML Pipeline Runner ===") 
    run_step("1. Seed Baseline Real Sessions", "src/seed_real_sessions.py") 
    run_step("2. Dataset Inspection", "src/inspect_data.py") 
    run_step("3. Proxy Label Generation", "src/labeler.py") 
    run_step("4. Synthetic Bootstrap Generation", "src/synthesizer.py") 
    run_step("5. Dataset Validation", "src/validator.py") 
    run_step("6. Logistic Regression Training and Cross-Validation", "src/train.py")
    run_step("7. Neural Network Training and Evaluation", "train_neural_network.py")
    run_step("8. Standalone Inference Test", "src/predict.py") 
    print("\n[SUCCESS] Entire FocusMirror ML Pipeline completed verified end-to-end!") 
if __name__ == "__main__": 
    main() 
