import os
import sys
import pandas as pd
from supabase_client import get_supabase_client, fetch_all_sessions

REQUIRED_FIELDS = ["score", "duration_min", "xp_earned", "method"]

def export_sessions_to_csv(output_path: str = "data/real_sessions.csv"):
    print("--- FocusMirror Session Exporter ---")
    
    # 1. Initialize Supabase Client
    try:
        client = get_supabase_client()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your environment variables before running:")
        print("  set SUPABASE_URL=https://your-project.supabase.co")
        print("  set SUPABASE_KEY=your-anon-or-service-key")
        sys.exit(1)
        
    # 2. Retrieve Sessions
    print("Retrieving sessions from public.sessions...")
    try:
        raw_data = fetch_all_sessions(client)
    except Exception as e:
        print(f"Database query failed: {e}")
        sys.exit(1)
        
    total_retrieved = len(raw_data)
    print(f"Sessions retrieved: {total_retrieved}")
    
    if total_retrieved == 0:
        print("Warning: 0 sessions retrieved from database. Check RLS policies or data.")
        return
        
    # 3. Validate Fields & Filter
    accepted = []
    rejected_count = 0
    missing_fields_summary = set()
    
    for row in raw_data:
        missing = [f for f in REQUIRED_FIELDS if f not in row or row[f] is None]
        if missing:
            rejected_count += 1
            for mf in missing:
                missing_fields_summary.add(mf)
        else:
            # Enforce data_source = 'real' for genuine observations
            row_clean = {k: row[k] for k in REQUIRED_FIELDS}
            row_clean["data_source"] = "real"
            accepted.append(row_clean)
            
    print(f"Sessions accepted: {len(accepted)}")
    print(f"Sessions rejected: {rejected_count}")
    if missing_fields_summary:
        print(f"Missing fields observed: {list(missing_fields_summary)}")
        
    # 4. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(accepted)
    df.to_csv(output_path, index=False)
    print(f"Output location: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    export_sessions_to_csv()