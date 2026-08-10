import os 
import sys 
import pandas as pd 
 
def assign_proxy_label(row): 
    score = row.get('score', 100) 
    duration = row.get('duration_min', 0) 
    if score < 50 or (duration > 90 and score < 65): 
        return 1 
    return 0 
 
def generate_labels(input_path='data/real_sessions.csv', output_path='data/labeled_sessions.csv'): 
    print('--- FocusMirror Proxy Label Generator ---') 
    if not os.path.exists(input_path): 
        print(f'Error: Input file {input_path} not found.') 
        sys.exit(1) 
    df = pd.read_csv(input_path) 
    df['target_label'] = df.apply(assign_proxy_label, axis=1) 
    print('Scientific Terminology Legend:') 
    print('  label = 0 -> non-burnout / proxy-normal') 
    print('  label = 1 -> burnout / proxy-burnout\n') 
    print('Label Distribution:') 
    counts = df['target_label'].value_counts().to_dict() 
    for lbl, cnt in sorted(counts.items()): 
        name = 'proxy-normal (0)' if lbl == 0 else 'proxy-burnout (1)' 
        print(f'  {name}: {cnt} sample(s)') 
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    df.to_csv(output_path, index=False) 
    print(f'\nLabeled dataset saved to: {os.path.abspath(output_path)}') 
 
if __name__ == '__main__': 
    generate_labels() 
