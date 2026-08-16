import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

FEATURE_COLS = ["score", "duration_min", "xp_earned"]
TARGET_COL = "target_label"


def train_and_evaluate(
    csv_path="data/bootstrapped_sessions.csv",
    model_path="models/neural_network_pipeline.joblib",
):
    print("--- FocusMirror Neural Network Trainer & Evaluator ---")

    if not os.path.exists(csv_path):
        print(f"Error: Dataset file '{csv_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        print(f"Error: Dataset is missing columns: {sorted(missing)}")
        sys.exit(1)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "neural_network",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="lbfgs",
                    alpha=0.001,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
    )
    print(f"\n5-Fold CV ROC-AUC scores: {np.round(cv_scores, 3)}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Holdout Test Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nNeural-network model saved to: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    train_and_evaluate()
