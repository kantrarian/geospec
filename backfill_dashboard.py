
import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


ALL_REGIONS = [
    'ridgecrest', 'socal_saf_mojave', 'socal_saf_coachella',
    'norcal_hayward', 'cascadia', 'tokyo_kanto',
    'istanbul_marmara', 'turkey_kahramanmaras', 'campi_flegrei',
    'kaikoura', 'anchorage', 'kumamoto', 'hualien', 'mexico_guerrero',
]


def find_earliest_date(ensemble_dir: Path) -> str:
    """Find the earliest existing ensemble date."""
    files = sorted(ensemble_dir.glob('ensemble_2*.json'))
    if not files:
        return None
    return files[0].stem.replace('ensemble_', '')


def run_backfill(start_date: str, end_date: str, regions: list, project_root: Path):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1

    print("=" * 60)
    print(f"DASHBOARD BACKFILL: {start_date} to {end_date} ({total_days} days)")
    print(f"Regions: {len(regions)}")
    print("=" * 60)

    current = start
    completed = 0

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        print(f"\n[{completed + 1}/{total_days}] {date_str}")

        for region in regions:
            print(f"  > {region}...", end="", flush=True)

            cmd = [
                sys.executable, "-m", "monitoring.src.run_ensemble_daily",
                "--region", region,
                "--date", date_str,
                "--quiet"
            ]

            try:
                result = subprocess.run(
                    cmd, cwd=project_root,
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print(" OK")
                else:
                    print(" FAIL")
                    if result.stderr:
                        for line in result.stderr.strip().split('\n')[-2:]:
                            print(f"    {line}")
            except subprocess.TimeoutExpired:
                print(" TIMEOUT")
            except Exception as e:
                print(f" ERROR: {e}")

        completed += 1
        current += timedelta(days=1)

    print("\n" + "=" * 60)
    print("REGENERATING DASHBOARD CSV")
    print("=" * 60)

    try:
        cmd = [sys.executable, "monitoring/generate_dashboard_csv.py"]
        subprocess.run(cmd, cwd=project_root, check=True)
        print("Done!")
    except Exception as e:
        print(f"Failed to regenerate CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Backfill dashboard data')
    parser.add_argument(
        '--extend-back', type=int, default=30,
        help='Days to extend before earliest existing data (default: 30)'
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help='Explicit start date YYYY-MM-DD (overrides --extend-back)'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='Explicit end date YYYY-MM-DD (default: earliest existing date - 1)'
    )
    parser.add_argument(
        '--regions', type=str, nargs='+', default=None,
        help='Specific regions (default: all)'
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent
    ensemble_dir = project_root / 'monitoring' / 'data' / 'ensemble_results'

    regions = args.regions or ALL_REGIONS
    earliest = find_earliest_date(ensemble_dir)

    if args.start:
        start_date = args.start
    elif earliest:
        earliest_dt = datetime.strptime(earliest, '%Y-%m-%d')
        start_date = (earliest_dt - timedelta(days=args.extend_back)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=args.extend_back)).strftime('%Y-%m-%d')

    if args.end:
        end_date = args.end
    elif earliest:
        end_date = (datetime.strptime(earliest, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Earliest existing data: {earliest or 'none'}")
    print(f"Will backfill: {start_date} to {end_date}")

    run_backfill(start_date, end_date, regions, project_root)


if __name__ == "__main__":
    main()
