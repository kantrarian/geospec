
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

def run_backfill(days_back=30):
    print("="*60)
    print(f"STARTING DASHBOARD BACKFILL ({days_back} days)")
    print("="*60)
    
    # Regions to backfill (new historical regions)
    regions = ['kaikoura', 'anchorage', 'kumamoto', 'hualien']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    current_date = start_date
    project_root = Path(__file__).parent
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\nProcessing Date: {date_str}")
        
        for region in regions:
            print(f"  > Backfilling {region}...", end="", flush=True)
            
            # Construct command
            # python -m monitoring.src.run_ensemble_daily --region <region> --date <date>
            cmd = [
                sys.executable, "-m", "monitoring.src.run_ensemble_daily",
                "--region", region,
                "--date", date_str
            ]
            
            try:
                # Run silently unless error
                result = subprocess.run(
                    cmd, 
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(" DONE")
                else:
                    print(" FAILED")
                    print(f"    Error: {result.stderr}")
                    
            except Exception as e:
                print(f" ERROR: {e}")
                
        current_date += timedelta(days=1)
        
    print("\n" + "="*60)
    print("REGENERATING DASHBOARD CSV")
    print("="*60)
    
    try:
        cmd = [sys.executable, "monitoring/generate_dashboard_csv.py"]
        subprocess.run(cmd, cwd=project_root, check=True)
        print("Success!")
    except Exception as e:
        print(f"Failed to regenerate CSV: {e}")

if __name__ == "__main__":
    run_backfill()
