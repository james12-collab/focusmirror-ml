import os 
import sys 
import numpy as np 
import pandas as pd 
 
def generate_bootstrap_data(input_path="data/labeled_sessions.csv", output_path="data/bootstrapped_sessions.csv", seed=42): 
    print("--- FocusMirror Synthetic Bootstrap Generator ---") 
    if not os.path.exists(input_path): 
        print(f"Error: Input file '{input_path}' not found.") 
        sys.exit(1) 
    np.random.seed(seed) 
    real_df = pd.read_csv(input_path) 
    n_normal = 20 
    normal_scores = np.random.randint(70, 99, size=n_normal) 
    normal_durations = np.random.randint(20, 55, size=n_normal) 
    normal_xp = np.random.randint(40, 100, size=n_normal) 
    normal_methods = np.random.choice(["Pomodoro", "Dashboard Tracking"], size=n_normal) 
    synth_normal = pd.DataFrame({"score": normal_scores, "duration_min": normal_durations, "xp_earned": normal_xp, "method": normal_methods, "data_source": "synthetic", "target_label": 0}) 
    n_burnout = 20 
    burnout_scores = np.concatenate([np.random.randint(25, 49, size=10), np.random.randint(40, 60, size=10)]) 
    burnout_durations = np.concatenate([np.random.randint(15, 60, size=10), np.random.randint(95, 150, size=10)]) 
    burnout_xp = np.random.randint(10, 45, size=n_burnout) 
    burnout_methods = np.random.choice(["Pomodoro", "Dashboard Tracking"], size=n_burnout) 
    synth_burnout = pd.DataFrame({"score": burnout_scores, "duration_min": burnout_durations, "xp_earned": burnout_xp, "method": burnout_methods, "data_source": "synthetic", "target_label": 1}) 
    combined_df = pd.concat([real_df, synth_normal, synth_burnout], ignore_index=True) 
    print(f"Total samples: {len(combined_df)}") 
    print(f"Real samples: {len(combined_df[combined_df['data_source'] == 'real'])}") 
    print(f"Synthetic samples: {len(combined_df[combined_df['data_source'] == 'synthetic'])}\n") 
    print("Class distribution by data_source:") 
    print(pd.crosstab(combined_df["data_source"], combined_df["target_label"])) 
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    combined_df.to_csv(output_path, index=False) 
    print(f"\nBootstrapped dataset saved to: {os.path.abspath(output_path)}") 
 
if __name__ == "__main__": 
    generate_bootstrap_data() 
