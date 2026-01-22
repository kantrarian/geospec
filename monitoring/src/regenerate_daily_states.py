#!/usr/bin/env python3
"""
regenerate_daily_states.py - Regenerate daily_states.csv from ensemble JSON files

This script reads all ensemble_*.json files and regenerates daily_states.csv
with consistent data and calibration metadata.

Usage:
    python regenerate_daily_states.py
    python regenerate_daily_states.py --dry-run

Author: R.J. Mathews / Claude
Date: January 2026
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibration_metadata(data_dir: Path) -> Dict[str, str]:
    """Load calibration dates from baseline files."""
    metadata = {
        'lambda_geo_calibration': 'unknown',
        'thd_calibration': 'unknown',
    }

    # Lambda_geo baselines
    lg_baseline_file = data_dir / 'baselines' / 'lambda_geo_baselines.json'
    if lg_baseline_file.exists():
        with open(lg_baseline_file, 'r') as f:
            lg_data = json.load(f)
            ts = lg_data.get('calibration_timestamp', '')
            if ts:
                metadata['lambda_geo_calibration'] = ts[:10]  # Just date part

    # THD baselines (find most recent)
    thd_files = list((data_dir / 'baselines').glob('thd_baselines_*.json'))
    if thd_files:
        thd_files.sort(reverse=True)
        try:
            with open(thd_files[0], 'r') as f:
                thd_data = json.load(f)
                ts = thd_data.get('metadata', {}).get('generated', '')
                if ts:
                    metadata['thd_calibration'] = ts[:10]
        except json.JSONDecodeError as e:
            # Extract date from filename if JSON is malformed
            filename = thd_files[0].stem  # thd_baselines_20260112
            if '_' in filename:
                date_part = filename.split('_')[-1]
                if len(date_part) == 8:
                    metadata['thd_calibration'] = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            logger.warning(f"THD baseline file malformed, using filename date: {e}")

    return metadata


def extract_row_from_ensemble(
    ensemble_data: Dict,
    region: str,
    calibration_date: str,
    regenerated_at: str,
) -> Optional[Dict]:
    """Extract a single row from ensemble JSON data."""

    regions = ensemble_data.get('regions', {})
    if region not in regions:
        return None

    r = regions[region]
    components = r.get('components', {})

    # Extract component values
    lg_ratio = ''
    thd_val = ''
    fc_l2l1 = ''

    if 'lambda_geo' in components:
        comp = components['lambda_geo']
        if comp.get('available'):
            raw = comp.get('raw_value', 0)
            if raw > 0:
                lg_ratio = f"{raw:.3f}"

    if 'seismic_thd' in components:
        comp = components['seismic_thd']
        if comp.get('available'):
            raw = comp.get('raw_value', 0)
            if raw > 0:
                thd_val = f"{raw:.4f}"

    if 'fault_correlation' in components:
        comp = components['fault_correlation']
        if comp.get('available'):
            raw = comp.get('raw_value', 0)
            if raw > 0:
                fc_l2l1 = f"{raw:.4f}"

    # Determine status
    status = 'REGENERATED'

    # Build notes
    notes_list = []
    orig_notes = r.get('notes', '')
    if orig_notes and 'capped' in orig_notes.lower():
        notes_list.append('tier_capped')
    if r.get('tier') == -1:
        notes_list.append('degraded')
    notes = ','.join(notes_list) if notes_list else ''

    return {
        'date': ensemble_data.get('date', ''),
        'region': region,
        'tier': r.get('tier', 0),
        'risk': f"{r.get('combined_risk', 0):.4f}",
        'methods': r.get('methods_available', 0),
        'confidence': f"{r.get('confidence', 0):.2f}",
        'lg_ratio': lg_ratio,
        'thd': thd_val,
        'fc_l2l1': fc_l2l1,
        'status': status,
        'notes': notes,
        'calibration_date': calibration_date,
        'regenerated_at': regenerated_at,
    }


def regenerate_daily_states(
    data_dir: Path,
    output_file: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Regenerate daily_states.csv from ensemble JSON files.

    Args:
        data_dir: Path to monitoring/data directory
        output_file: Output CSV path (default: data_dir/ensemble_results/daily_states.csv)
        dry_run: If True, don't write file, just return data

    Returns:
        List of row dicts
    """
    ensemble_dir = data_dir / 'ensemble_results'
    if output_file is None:
        output_file = ensemble_dir / 'daily_states.csv'

    # Load calibration metadata
    cal_meta = load_calibration_metadata(data_dir)
    calibration_date = cal_meta['lambda_geo_calibration']
    regenerated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

    logger.info(f"Calibration date: {calibration_date}")
    logger.info(f"Regenerating at: {regenerated_at}")

    # Find all ensemble JSON files
    json_files = sorted(ensemble_dir.glob('ensemble_*.json'))
    logger.info(f"Found {len(json_files)} ensemble JSON files")

    if not json_files:
        logger.error("No ensemble JSON files found!")
        return []

    # Process each file
    all_rows = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {json_file}: {e}")
            continue

        date = data.get('date', '')
        regions = data.get('regions', {})

        for region in regions.keys():
            row = extract_row_from_ensemble(
                data, region, calibration_date, regenerated_at
            )
            if row:
                all_rows.append(row)

    # Sort by date, then region
    all_rows.sort(key=lambda x: (x['date'], x['region']))

    logger.info(f"Generated {len(all_rows)} rows from {len(json_files)} files")

    # Write CSV
    if not dry_run:
        # Backup existing file
        if output_file.exists():
            backup_file = output_file.with_suffix('.csv.backup')
            output_file.rename(backup_file)
            logger.info(f"Backed up existing file to {backup_file}")

        fieldnames = [
            'date', 'region', 'tier', 'risk', 'methods', 'confidence',
            'lg_ratio', 'thd', 'fc_l2l1', 'status', 'notes',
            'calibration_date', 'regenerated_at'
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        logger.info(f"Wrote {len(all_rows)} rows to {output_file}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate daily_states.csv from ensemble JSON files'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview without writing file'
    )
    parser.add_argument(
        '--data-dir', type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Path to monitoring/data directory'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("REGENERATE DAILY_STATES.CSV FROM ENSEMBLE JSON FILES")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("DRY RUN - No files will be modified\n")

    rows = regenerate_daily_states(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )

    # Print summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    if rows:
        dates = sorted(set(r['date'] for r in rows))
        regions = sorted(set(r['region'] for r in rows))

        print(f"  Total rows: {len(rows)}")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Regions: {len(regions)}")
        print(f"  Calibration date: {rows[0]['calibration_date']}")

        # Tier distribution
        tier_counts = {}
        for r in rows:
            t = r['tier']
            tier_counts[t] = tier_counts.get(t, 0) + 1

        print(f"\n  Tier distribution:")
        for tier in sorted(tier_counts.keys()):
            tier_name = {0: 'NORMAL', 1: 'WATCH', 2: 'ELEVATED', 3: 'CRITICAL', -1: 'DEGRADED'}.get(tier, f'T{tier}')
            print(f"    {tier_name}: {tier_counts[tier]}")

    print("-" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
