import os
import pandas as pd

def create_baseline_real_sessions(output_path="data/real_sessions.csv"):
    print("--- FocusMirror Baseline Real Sessions Generator ---")
    
    # 3 baseline real sessions (typical productive study sessions, not burnout)
    baseline_sessions = [
        {"score": 85, "duration_min": 25, "xp_earned": 50, "method": "Pomodoro", "data_source": "real"},
        {"score": 90, "duration_min": 45, "xp_earned": 90, "method": "Dashboard Tracking", "data_source": "real"},
        {"score": 78, "duration_min": 30, "xp_earned": 60, "method": "Pomodoro", "data_source": "real"}
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(baseline_sessions)
    df.to_csv(output_path, index=False)
    
    print(f"Baseline real sessions generated: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Output location: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_baseline_real_sessions()