#!/usr/bin/env python
"""
run_thd_recal.py — INCIDENT 2026-07-31 (D1) Action-2: weekly THD baseline recalibration.

Extends R3's rolling-recalibration principle to the seismic_thd component baselines (cayley 2026-07-31).
Runs the R3-consistent rolling recal — **90-day window ending today-30d** (matching the production lambda_geo
R3 recal in run_and_publish.ps1) — for the calibratable stations and writes a dated
`data/baselines/thd_baselines_<YYYYMMDD>.json` in the flat format that
`station_baselines._load_newest_baseline_file()` consumes newest-first. This closes the stale-frozen-baseline
class that produced the IU.COLA z=26 artifact: baselines refresh weekly instead of being frozen from a one-off
2026-01 calibration.

Cadence: intended to run WEEKLY (scheduler, or the daily run with --if-due, which no-ops unless the newest
baseline file is older than RECAL_INTERVAL_DAYS). Actual execution fetches ~90 days of waveforms per station.

Usage:
    python run_thd_recal.py --if-due          # recal only if the newest baseline file is >7 days old
    python run_thd_recal.py --force           # recal now regardless of cadence
    python run_thd_recal.py --dry-run         # print the plan (stations, window) without fetching
    python run_thd_recal.py --stations IU.COLA IU.ANTO   # subset
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from station_baselines import STATION_BASELINES  # noqa: E402

logger = logging.getLogger(__name__)

# R3-consistent parameters (cayley Action 2): mirror the PRODUCTION lambda_geo R3 recal in run_and_publish.ps1
# (90-day window ending today-30d, refreshed weekly), NOT the RollingBaseline default lag -- consistency with
# what R3 actually runs in production.
LOOKBACK_DAYS = 90
EXCLUDE_RECENT_DAYS = 30         # 30-day lag, matching the production lambda_geo recal (--end-date today-30d)
RECAL_INTERVAL_DAYS = 7          # weekly cadence
BASELINE_DIR = Path(__file__).resolve().parent.parent / "data" / "baselines"


def _calibratable_stations():
    """Stations with a real prior calibration window (skip UNCALIBRATED manual estimates)."""
    return [k for k, b in STATION_BASELINES.items()
            if b.calibration_period and b.calibration_period != "UNCALIBRATED"]


def _newest_baseline_age_days():
    """Age (days) of the newest dated thd_baselines_*.json by filename date; None if none / unparseable."""
    files = sorted(BASELINE_DIR.glob("thd_baselines_*.json"), key=lambda p: p.name, reverse=True)
    for f in files:
        try:
            datestr = f.stem.split("thd_baselines_")[-1][:8]
            d = datetime.strptime(datestr, "%Y%m%d")
            return (datetime.now().replace(tzinfo=None) - d).days
        except Exception:
            continue
    return None


def run_recal(stations, end_date=None, dry_run=False):
    """Recalibrate `stations` on the R3 rolling window and write a dated flat baseline file. Returns the
    output path (or None on dry-run)."""
    from calibrate_thd_baselines import calibrate_station
    end_date = end_date or datetime.now().replace(tzinfo=None)
    window_end = end_date - timedelta(days=EXCLUDE_RECENT_DAYS)
    window_start = window_end - timedelta(days=LOOKBACK_DAYS)
    logger.info(f"THD rolling recal: {len(stations)} stations, window {window_start.date()}..{window_end.date()} "
                f"(lookback {LOOKBACK_DAYS}d, exclude-recent {EXCLUDE_RECENT_DAYS}d)")
    if dry_run:
        for s in stations:
            print(f"  would recal {s} over {window_start.date()}..{window_end.date()}")
        return None

    out = {}
    for key in stations:
        net, sta = key.split(".", 1)
        try:
            # Pass the R3-lagged window_end (= end_date - EXCLUDE_RECENT_DAYS) EXPLICITLY. calibrate_station only
            # self-applies exclude_recent_days when end_date is None; passing end_date=end_date here silently
            # bypassed the 30-day R3 lag (window ended today, contaminating the baseline with the recent window).
            # (INCIDENT 2026-07-31 D1 lag-fix, grassmann 2026-08-07, cayley-confirmed option-1.)
            r = calibrate_station(network=net, station=sta, days_back=LOOKBACK_DAYS,
                                  exclude_recent_days=EXCLUDE_RECENT_DAYS, end_date=window_end)
        except Exception as e:
            logger.error(f"recal {key} failed: {e}")
            continue
        if r.get("mean_thd") is None:
            logger.warning(f"recal {key} produced no baseline ({r.get('error', 'n/a')}); skipping")
            continue
        out[key] = {
            "station": key,
            "mean_thd": r["mean_thd"],
            "std_thd": r["std_thd"],
            "n_samples": r.get("n_samples", 0),
            "calibration_period": r["calibration_period"],
            "notes": f"Rolling recal {LOOKBACK_DAYS}d/{EXCLUDE_RECENT_DAYS}d (incident 2026-07-31)",
        }
    if not out:
        logger.error("recal produced no station baselines; NOT writing (keeping prior file)")
        return None
    path = BASELINE_DIR / f"thd_baselines_{end_date.strftime('%Y%m%d')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {len(out)} rolling THD baselines to {path.name} (loaded newest-first next run)")
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Weekly R3-consistent THD baseline recalibration (incident 2026-07-31)")
    ap.add_argument("--if-due", action="store_true", help="recal only if newest baseline > weekly cadence old")
    ap.add_argument("--force", action="store_true", help="recal now regardless of cadence")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without fetching")
    ap.add_argument("--stations", nargs="*", help="subset of station keys (default: all calibratable)")
    args = ap.parse_args()

    if args.if_due and not args.force:
        age = _newest_baseline_age_days()
        if age is not None and age < RECAL_INTERVAL_DAYS:
            logger.info(f"THD recal skipped: newest baseline is {age}d old (< {RECAL_INTERVAL_DAYS}d cadence)")
            return
    stations = args.stations or _calibratable_stations()
    run_recal(stations, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
