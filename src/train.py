import os 
import sys 
import joblib 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix 
 
FEATURE_COLS = ["score", "duration_min", "xp_earned"] 
TARGET_COL = "target_label" 
 
def train_and_evaluate(csv_path="data/bootstrapped_sessions.csv", model_path="models/logistic_regression_pipeline.joblib"): 
    print("--- FocusMirror ML Model Trainer & Evaluator ---") 
    if not os.path.exists(csv_path): 
        print(f"Error: Dataset file '{csv_path}' not found.") 
        sys.exit(1) 
    df = pd.read_csv(csv_path) 
    X = df[FEATURE_COLS] 
    y = df[TARGET_COL] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y) 
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}") 
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", random_state=42))]) 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc") 
    print(f"\nStratified 5-Fold Cross-Validation ROC-AUC Scores: {np.round(cv_scores, 3)}") 
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.3f} (+- {cv_scores.std():.3f})") 
    pipeline.fit(X_train, y_train) 
    y_pred = pipeline.predict(X_test) 
    y_prob = pipeline.predict_proba(X_test)[:, 1] 
    acc = accuracy_score(y_test, y_pred) 
    prec = precision_score(y_test, y_pred, zero_division=0) 
    rec = recall_score(y_test, y_pred, zero_division=0) 
    f1 = f1_score(y_test, y_pred, zero_division=0) 
    auc = roc_auc_score(y_test, y_prob) 
    cm = confusion_matrix(y_test, y_pred) 
    print("\n--- Holdout Test Set Evaluation Metrics ---") 
    print(f"  Accuracy:  {acc:.4f}") 
    print(f"  Precision: {prec:.4f}") 
    print(f"  Recall:    {rec:.4f}") 
    print(f"  F1-Score:  {f1:.4f}") 
    print(f"  ROC-AUC:   {auc:.4f}") 
    print("\n--- Confusion Matrix ---") 
    print(f"                Predicted Normal (0)   Predicted Burnout (1)") 
    print(f"Actual Normal        {cm[0, 0]}                      {cm[0, 1]}") 
    print(f"Actual Burnout       {cm[1, 0]}                      {cm[1, 1]}") 
    print("\n--- What These Metrics Mean for FocusMirror ---") 
    print("  * False Positives: Recommends an unnecessary break (low cost).") 
    print("  * False Negatives: Misses cognitive fatigue (HIGH RISK).") 
    print("  * Conclusion: High RECALL is critical to protect student mental health.") 
    os.makedirs(os.path.dirname(model_path), exist_ok=True) 
    joblib.dump(pipeline, model_path) 
    print(f"\nTrained pipeline saved to: {os.path.abspath(model_path)}") 
 
if __name__ == "__main__": 
    train_and_evaluate() 
