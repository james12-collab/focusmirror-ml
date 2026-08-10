import os 
import sys 
import joblib 
import pandas as pd 
 
FEATURE_COLS = ["score", "duration_min", "xp_earned"] 
 
def load_model(model_path="models/logistic_regression_pipeline.joblib"): 
    if not os.path.exists(model_path): 
        print(f"Error: Trained model not found at '{model_path}'. Run src/train.py first.") 
        sys.exit(1) 
    return joblib.load(model_path) 
 
def predict_session(model, session_data: dict): 
    df = pd.DataFrame([session_data])[FEATURE_COLS] 
    pred_class = model.predict(df)[0] 
    pred_prob = model.predict_proba(df)[0, 1] 
    label_str = "Fatigue / Burnout Risk Detected (1)" if pred_class == 1 else "Normal / Healthy Focus (0)" 
    return pred_class, pred_prob, label_str 
 
def run_test_predictions(): 
    print("--- FocusMirror Standalone Inference Test ---") 
    model = load_model() 
    print("Successfully loaded model from models/logistic_regression_pipeline.joblib\n") 
    session_1 = {"score": 92, "duration_min": 25, "xp_earned": 60} 
    _, prob_1, label_1 = predict_session(model, session_1) 
    print(f"New Session A: {session_1}") 
    print(f"  -> Prediction:  {label_1}") 
    print(f"  -> Probability: {prob_1 * 100:.1f}%% burnout probability\n") 
    session_2 = {"score": 38, "duration_min": 110, "xp_earned": 15} 
    _, prob_2, label_2 = predict_session(model, session_2) 
    print(f"New Session B: {session_2}") 
    print(f"  -> Prediction:  {label_2}") 
    print(f"  -> Probability: {prob_2 * 100:.1f}%% burnout probability\n") 
 
if __name__ == "__main__": 
    run_test_predictions() 
