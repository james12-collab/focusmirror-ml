import os 
import sys 
import pandas as pd 
 
def validate_dataset_ready_for_training(csv_path="data/bootstrapped_sessions.csv"): 
    print(f"--- Pre-Training Dataset Validation: {csv_path} ---") 
    if not os.path.exists(csv_path): 
        print(f"Error: Dataset file '{csv_path}' not found.") 
        sys.exit(1) 
    df = pd.read_csv(csv_path) 
    total_samples = len(df) 
    missing_count = df.isnull().sum().sum() 
    print(f"Total samples: {total_samples}") 
    print(f"Total missing cells: {missing_count}") 
    print(f"Duplicate rows: {df.duplicated().sum()}") 
    if total_samples == 0 or missing_count != 0: 
        print("FAIL: Dataset invalid.") 
        sys.exit(1) 
    print(f"Real samples: {len(df[df['data_source'] == 'real'])}") 
    print(f"Synthetic samples: {len(df[df['data_source'] == 'synthetic'])}") 
    classes = df["target_label"].unique() 
    print(f"Target class distribution: {df['target_label'].value_counts().to_dict()}") 
    if len(classes) != 2: 
        print("FAIL: Only one class exists in target_label!") 
        sys.exit(1) 
    print("Class breakdown by data_source:") 
    print(pd.crosstab(df["data_source"], df["target_label"]).to_string()) 
    print("\nSUCCESS: Dataset passed all validation checks and is ready for model training!") 
    return True 
 
if __name__ == "__main__": 
    validate_dataset_ready_for_training() 
