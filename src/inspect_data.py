import os 
import sys 
import pandas as pd 
 
def inspect_dataset(csv_path="data/real_sessions.csv"): 
    print(f"--- Dataset Inspection: {csv_path} ---") 
    if not os.path.exists(csv_path): 
        print(f"Error: Dataset file not found at '{csv_path}'.") 
        sys.exit(1) 
    df = pd.read_csv(csv_path) 
    print(f"Number of samples: {len(df)}") 
    print(f"Number of columns: {len(df.columns)}") 
    print(f"Column names & types:\n{df.dtypes}\n") 
    missing = df.isnull().sum() 
    print("Missing values per column:") 
    if missing.sum() != 0: 
        print(missing) 
    else: 
        print("  None detected\n") 
    print(f"Duplicate samples: {df.duplicated().sum()}\n") 
    if "data_source" in df.columns: 
        print("Data source distribution:") 
        print(df["data_source"].value_counts().to_string()) 
        print() 
    num_cols = df.select_dtypes(include=["number"]).columns 
    print("Basic statistical summary (Numerical Features):") 
    print(df[num_cols].describe().to_string()) 
 
if __name__ == "__main__": 
    inspect_dataset() 
