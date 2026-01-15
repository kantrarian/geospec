#!/usr/bin/env python3
"""
rtcm_health_check.py
Analyze RTCM capture health: throughput, reconnects, gaps.

Produces:
- Per-station hourly stats (bytes, files)
- Reconnect analysis from logs
- health.json summary for dashboard

Usage:
    python rtcm_health_check.py
    python rtcm_health_check.py --station COSO00USA0
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def safe_print(msg: str) -> None:
    """Print safely on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def analyze_rtcm_files(rtcm_dir: Path, station: Optional[str] = None) -> Dict:
    """Analyze RTCM file sizes and coverage."""
    results = {}

    # Find all station directories
    if station:
        station_dirs = [rtcm_dir / station] if (rtcm_dir / station).exists() else []
    else:
        station_dirs = [d for d in rtcm_dir.iterdir() if d.is_dir()]

    for station_dir in station_dirs:
        station_name = station_dir.name
        results[station_name] = {
            'total_bytes': 0,
            'total_files': 0,
            'hourly_stats': [],
            'zero_byte_hours': 0,
            'min_hourly_bytes': None,
            'max_hourly_bytes': 0,
            'days_covered': set(),
        }

        # Find all date directories
        for date_dir in sorted(station_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            date_str = date_dir.name
            results[station_name]['days_covered'].add(date_str)

            # Analyze each hourly file
            for rtcm_file in sorted(date_dir.glob("*.rtcm3")):
                file_size = rtcm_file.stat().st_size
                results[station_name]['total_bytes'] += file_size
                results[station_name]['total_files'] += 1

                # Extract hour from filename (e.g., COSO00USA0_2026-01-11_1500Z.rtcm3)
                match = re.search(r'_(\d{2})00Z\.rtcm3$', rtcm_file.name)
                hour = match.group(1) if match else "??"

                results[station_name]['hourly_stats'].append({
                    'date': date_str,
                    'hour': hour,
                    'bytes': file_size,
                    'file': rtcm_file.name,
                })

                if file_size == 0:
                    results[station_name]['zero_byte_hours'] += 1

                if results[station_name]['min_hourly_bytes'] is None:
                    results[station_name]['min_hourly_bytes'] = file_size
                else:
                    results[station_name]['min_hourly_bytes'] = min(
                        results[station_name]['min_hourly_bytes'], file_size
                    )
                results[station_name]['max_hourly_bytes'] = max(
                    results[station_name]['max_hourly_bytes'], file_size
                )

        # Convert set to list for JSON
        results[station_name]['days_covered'] = sorted(results[station_name]['days_covered'])

    return results


def analyze_logs(log_dir: Path, station: Optional[str] = None) -> Dict:
    """Analyze capture logs for reconnects and errors."""
    results = {}

    if station:
        log_files = list(log_dir.glob(f"{station}*.log"))
    else:
        log_files = list(log_dir.glob("*.log"))

    for log_file in log_files:
        station_name = log_file.stem.replace('.log', '')
        if station_name.endswith('.err'):
            continue

        results[station_name] = {
            'reconnects': 0,
            'errors': [],
            'first_connect': None,
            'last_activity': None,
            'connect_times': [],
        }

        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    # Look for connection messages (handle both patterns)
                    if 'Connected (' in line and ('session' in line or 'reconnect' in line):
                        results[station_name]['reconnects'] += 1
                        # Extract timestamp
                        match = re.search(r'\[([\d\-T:+.]+)\]', line)
                        if match:
                            ts = match.group(1)
                            results[station_name]['connect_times'].append(ts)
                            if results[station_name]['first_connect'] is None:
                                results[station_name]['first_connect'] = ts
                            results[station_name]['last_activity'] = ts

                    # Look for errors
                    if 'Stream error' in line or 'ERROR' in line:
                        results[station_name]['errors'].append(line.strip()[:100])

                    # Track file writes as activity
                    if 'Writing to:' in line:
                        match = re.search(r'\[([\d\-T:+.]+)\]', line)
                        if match:
                            results[station_name]['last_activity'] = match.group(1)

        except Exception as e:
            results[station_name]['errors'].append(f"Log parse error: {e}")

    return results


def compute_health_summary(rtcm_stats: Dict, log_stats: Dict) -> Dict:
    """Compute overall health summary."""
    summary = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'stations': {},
        'overall_status': 'OK',
    }

    issues = []

    for station, stats in rtcm_stats.items():
        log_info = log_stats.get(station, {})

        # Compute metrics
        total_hours = stats['total_files']
        avg_bytes_per_hour = stats['total_bytes'] / total_hours if total_hours > 0 else 0
        reconnects = log_info.get('reconnects', 0)

        # Compute median for relative threshold
        hourly_bytes = [h['bytes'] for h in stats['hourly_stats'] if h['bytes'] > 0]
        median_bytes = sorted(hourly_bytes)[len(hourly_bytes)//2] if hourly_bytes else 0

        # Health assessment
        status = 'OK'
        station_issues = []

        if stats['zero_byte_hours'] > 0:
            status = 'DEGRADED'
            station_issues.append(f"{stats['zero_byte_hours']} zero-byte hours")

        if reconnects > total_hours * 2:  # More than 2 reconnects per hour average
            status = 'DEGRADED'
            station_issues.append(f"High reconnect rate: {reconnects}/{total_hours}h")

        # Relative threshold: warn if min < 25% of median (station-specific)
        if stats['min_hourly_bytes'] is not None and median_bytes > 0:
            if stats['min_hourly_bytes'] < median_bytes * 0.25:
                status = 'WARNING' if status == 'OK' else status
                station_issues.append(f"Low min hourly: {stats['min_hourly_bytes']:,} (<25% of median {median_bytes:,})")
        elif stats['min_hourly_bytes'] is not None and stats['min_hourly_bytes'] < 1000:
            # Absolute floor for "dead stream"
            status = 'DEGRADED'
            station_issues.append(f"Near-zero hourly bytes: {stats['min_hourly_bytes']}")

        summary['stations'][station] = {
            'status': status,
            'total_bytes': stats['total_bytes'],
            'total_hours': total_hours,
            'avg_bytes_per_hour': int(avg_bytes_per_hour),
            'min_hourly_bytes': stats['min_hourly_bytes'],
            'max_hourly_bytes': stats['max_hourly_bytes'],
            'zero_byte_hours': stats['zero_byte_hours'],
            'reconnects': reconnects,
            'days_covered': stats['days_covered'],
            'issues': station_issues,
        }

        if status != 'OK':
            issues.append(f"{station}: {status}")

    if issues:
        summary['overall_status'] = 'DEGRADED' if 'DEGRADED' in str(issues) else 'WARNING'

    return summary


def print_report(rtcm_stats: Dict, log_stats: Dict, health: Dict) -> None:
    """Print human-readable report."""
    safe_print("=" * 70)
    safe_print("  RTCM Capture Health Report")
    safe_print("=" * 70)
    safe_print(f"  Generated: {health['timestamp']}")
    safe_print(f"  Overall Status: {health['overall_status']}")
    safe_print("")

    for station, stats in health['stations'].items():
        safe_print(f"\n  {station}")
        safe_print("-" * 50)
        safe_print(f"    Status:           {stats['status']}")
        safe_print(f"    Total bytes:      {stats['total_bytes']:,}")
        safe_print(f"    Total hours:      {stats['total_hours']}")
        safe_print(f"    Avg bytes/hour:   {stats['avg_bytes_per_hour']:,}")
        safe_print(f"    Min hourly:       {stats['min_hourly_bytes']:,}" if stats['min_hourly_bytes'] else "    Min hourly:       N/A")
        safe_print(f"    Max hourly:       {stats['max_hourly_bytes']:,}")
        safe_print(f"    Zero-byte hours:  {stats['zero_byte_hours']}")
        safe_print(f"    Reconnects:       {stats['reconnects']}")
        safe_print(f"    Days covered:     {', '.join(stats['days_covered'])}")

        if stats['issues']:
            safe_print(f"    Issues:           {'; '.join(stats['issues'])}")

        # Show hourly breakdown if available
        if station in rtcm_stats and rtcm_stats[station]['hourly_stats']:
            safe_print("\n    Hourly breakdown (last 10):")
            for h in rtcm_stats[station]['hourly_stats'][-10:]:
                safe_print(f"      {h['date']} {h['hour']}:00Z  {h['bytes']:>10,} bytes")

    safe_print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RTCM capture health check")
    parser.add_argument('--station', help="Check specific station only")
    parser.add_argument('--rtcm-dir', default='monitoring/data/rtcm',
                        help="RTCM data directory")
    parser.add_argument('--log-dir', default='monitoring/logs/rtcm',
                        help="Log directory")
    parser.add_argument('--output', help="Write health.json to this path")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent.parent
    rtcm_dir = script_dir / args.rtcm_dir
    log_dir = script_dir / args.log_dir

    if not rtcm_dir.exists():
        safe_print(f"RTCM directory not found: {rtcm_dir}")
        safe_print("Run capture first with: python -m monitoring.src.capture_rtcm --mount COSO00USA0 --hours 1")
        return 1

    # Analyze
    rtcm_stats = analyze_rtcm_files(rtcm_dir, args.station)
    log_stats = analyze_logs(log_dir, args.station) if log_dir.exists() else {}
    health = compute_health_summary(rtcm_stats, log_stats)

    # Print report
    print_report(rtcm_stats, log_stats, health)

    # Save health.json
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = rtcm_dir / 'health.json'

    with open(output_path, 'w') as f:
        json.dump(health, f, indent=2)
    safe_print(f"\nHealth summary saved to: {output_path}")

    # Append to health_history.csv for dashboard trending
    history_path = rtcm_dir / 'health_history.csv'
    write_header = not history_path.exists()

    with open(history_path, 'a', newline='') as f:
        import csv
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'station', 'status', 'total_bytes', 'hours',
                           'avg_bytes_hr', 'reconnects', 'zero_hours'])

        ts = health['timestamp']
        for station, stats in health['stations'].items():
            writer.writerow([
                ts,
                station,
                stats['status'],
                stats['total_bytes'],
                stats['total_hours'],
                stats['avg_bytes_per_hour'],
                stats['reconnects'],
                stats['zero_byte_hours'],
            ])

    safe_print(f"Health history appended to: {history_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
