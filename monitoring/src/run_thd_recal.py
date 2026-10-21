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
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from station_baselines import STATION_BASELINES  # noqa: E402
from station_baselines import _baseline_from_entry  # noqa: E402 (shared loader parser)

logger = logging.getLogger(__name__)

# R3-consistent parameters (cayley Action 2): mirror the PRODUCTION lambda_geo R3 recal in run_and_publish.ps1
# (90-day window ending today-30d, refreshed weekly), NOT the RollingBaseline default lag -- consistency with
# what R3 actually runs in production.
LOOKBACK_DAYS = 90
EXCLUDE_RECENT_DAYS = 30         # 30-day lag, matching the production lambda_geo recal (--end-date today-30d)
RECAL_INTERVAL_DAYS = 7          # weekly cadence
BASELINE_DIR = Path(__file__).resolve().parent.parent / "data" / "baselines"


# Configured THD stations with NO real prior calibration window that the OWNER approved for calibration -- each one is
# its own acquisition (LOOKBACK_DAYS station-days at the native rate). Explicit and attributed, never derived from the
# region map (deriving would silently start acquisitions for every configured fallback). Owner 2026-10-10, routed
# agent-framework 643dd746: "brk replaces bks. release the cafe channel." (cayley, codex b7a29458 correction 2).
# Owner 2026-10-14, routed agent-framework 68e91628: "yes, add SNZO to the calibration list" -- IU.SNZO's only prior
# record is the UNCALIBRATED n=0 default, so it is an addition; a bound station, calibrated by its daily operator.
CALIBRATION_ADDITIONS = {
    "IV.CAFE": "owner 2026-10-10 'release the cafe channel' (643dd746): declared IV.CAFE..HHZ@100Hz, calibration-first",
    "BK.BRK": "owner 2026-10-10 'brk replaces bks' (643dd746): norcal_hayward's THD station, its own baseline",
    "IU.SNZO": "owner 2026-10-14 'yes, add SNZO to the calibration list' (68e91628): kaikoura's bound station, BHZ 40 Hz",
}

# Calibrated stations that NO configured region reads: their weekly recal is no longer COLLECTED. Collection-cost
# maintenance, not a qualification change (codex 72777c8b: "a small explicit collection-list change from configured
# stations plus approved calibration additions, retaining historical baselines and any declared research consumers";
# owner 2026-10-15 in cayley's session: "start the collection-list trim while waiting"). Explicit and attributed, like
# the additions; a test keeps every collected station configured or approved, and every retired one unconfigured.
# Nothing is deleted: each retired station's newest recalibrated record is RETAINED in every new file with its own
# calibration_date (so the ordinary age rules expire it), dated files and built-in records stay, and `--stations` can
# still name a retired station explicitly.
COLLECTION_RETIRED = {
    "BK.BKS": "owner 2026-10-10 'brk replaces bks' (643dd746): norcal_hayward reads BK.BRK since GeoSpec ac79f2ba; "
              "BKS's whole feed stopped upstream 2026-10-02",
    "GE.ARPR": "configured by no region (istanbul_marmara and turkey_kahramanmaras read IU.ANTO); 20 Hz; its built-in "
               "record is an UNCALIBRATED manual estimate",
    "GE.CSS": "configured by no region; its built-in record is an UNCALIBRATED manual estimate",
}


def _calibratable_stations():
    """Stations with a real prior calibration window (skip UNCALIBRATED manual estimates) that are not
    COLLECTION_RETIRED, plus the owner-approved CALIBRATION_ADDITIONS (in their declared order, never duplicated).
    A station both retired and added is a contradiction and refuses."""
    both = sorted(set(COLLECTION_RETIRED) & set(CALIBRATION_ADDITIONS))
    if both:
        raise ValueError("stations both retired from collection and added for calibration: %s" % both)
    real = [k for k, b in STATION_BASELINES.items()
            if b.calibration_period and b.calibration_period != "UNCALIBRATED" and k not in COLLECTION_RETIRED]
    return real + [k for k in CALIBRATION_ADDITIONS if k not in real]


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


def _prior_effective_entries():
    """The station entries of the newest loadable dated baseline file (the file the runtime loader would select),
    parsed with the loader's own parser so a malformed row never counts. {} when none."""
    import json as _json
    for f in sorted(BASELINE_DIR.glob("thd_baselines_*.json"), key=lambda p: p.name, reverse=True):
        try:
            with open(f, encoding="utf-8") as fh:
                data = _json.load(fh)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("baselines"), list):
            entries = data["baselines"]
        elif isinstance(data, dict):
            entries = [v for v in data.values() if isinstance(v, dict) and "station" in v]
        else:
            entries = []
        out = {}
        for e in entries:
            try:
                parsed = _baseline_from_entry(e, f.name)
            except Exception:
                continue
            out[parsed.station] = dict(e, calibration_date=parsed.calibration_date)
        if out:
            return out
    return {}


def run_recal(stations, end_date=None, dry_run=False):
    """Recalibrate `stations` on the R3 rolling window and write a dated flat baseline file. Returns the
    output path (or None on dry-run).

    thd-bound-station-operator-v1 (grassmann 2026-10-04; codex review db9a28ff finding 3): (a) a BOUND station's entry
    carries the operator record (identity + bound channel/location/response/rate/gap/estimator/normalization) so a reader
    can tell which operator produced it; every entry carries an explicit `calibration_date`; (b) a station whose recal
    FAILS or is EMPTY is no longer dropped from the new file: its prior effective record is PRESERVED unchanged (its own
    calibration_date / period / identity) and the failed attempt is recorded under `_recal_attempts`, so the ordinary
    eligibility age/QA rules expire it instead of a silent reversion to the n=0 default. With no prior record an
    explicit unavailable attempt is recorded and nothing is fabricated."""
    from calibrate_thd_baselines import calibrate_station
    try:
        from thd_bound_station_operator import is_bound, operator_record
    except ImportError:
        is_bound, operator_record = (lambda n, s: False), None
    end_date = end_date or datetime.now().replace(tzinfo=None)
    window_end = end_date - timedelta(days=EXCLUDE_RECENT_DAYS)
    window_start = window_end - timedelta(days=LOOKBACK_DAYS)
    logger.info(f"THD rolling recal: {len(stations)} stations, window {window_start.date()}..{window_end.date()} "
                f"(lookback {LOOKBACK_DAYS}d, exclude-recent {EXCLUDE_RECENT_DAYS}d)")
    if dry_run:
        for s in stations:
            print(f"  would recal {s} over {window_start.date()}..{window_end.date()}")
        for s in COLLECTION_RETIRED:
            if s not in stations:
                print(f"  not collected {s} (newest recalibrated record retained, if any)")
        return None

    out = {}
    prior = _prior_effective_entries()
    attempts = []
    attempted_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _preserve(key, outcome, reason):
        if key in prior:
            kept = dict(prior[key])
            kept.setdefault("calibration_date", None)
            kept["notes"] = (kept.get("notes") or "") + " | PRESERVED unchanged after failed recal %s (%s)" % (attempted_utc, outcome)
            out[key] = kept
            attempts.append(dict(station=key, attempted_utc=attempted_utc, outcome=outcome, reason=reason,
                                 disposition="PRIOR_RECORD_PRESERVED", preserved_calibration_date=kept.get("calibration_date"),
                                 preserved_period=kept.get("calibration_period")))
        else:
            attempts.append(dict(station=key, attempted_utc=attempted_utc, outcome=outcome, reason=reason,
                                 disposition="UNAVAILABLE_NO_PRIOR_RECORD"))

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
            _preserve(key, "ERROR", f"{type(e).__name__}: {e}"[:200])
            continue
        if r.get("mean_thd") is None:
            logger.warning(f"recal {key} produced no baseline ({r.get('error', 'n/a')}); preserving prior record if any")
            _preserve(key, "EMPTY", str(r.get("error", "n/a"))[:200])
            continue
        entry = {
            "station": key,
            "mean_thd": r["mean_thd"],
            "std_thd": r["std_thd"],
            "n_samples": r.get("n_samples", 0),
            "calibration_period": r["calibration_period"],
            "calibration_date": end_date.strftime("%Y-%m-%d"),
            "notes": f"Rolling recal {LOOKBACK_DAYS}d/{EXCLUDE_RECENT_DAYS}d (incident 2026-07-31)",
        }
        if is_bound(net, sta) and operator_record is not None:
            entry["operator"] = operator_record(key)
        # the calibration convention, from the producer itself (never from an identity that may fail to form)
        try:
            import thd_daily_measurement as TDM
            import thd_bound_station_operator as _OP
            # a bound station is calibrated by its DAILY operator (not the fetch operator record above)
            entry["calibration_convention"] = (_OP.DAILY_OPERATOR["operator_version"] if is_bound(net, sta)
                                               else TDM.MEASUREMENT_VERSION)
        except ImportError:
            pass
        # thd-station-channel-v1: a declared-channel station's baseline names the exact selector it was measured on
        try:
            import thd_provider_routing as _TPR
            _declared = _TPR.declared_channel(net, sta)
            if _declared is not None:
                entry["selector"] = _declared["selector"]
        except ImportError:
            pass
        if isinstance(r.get("estimator"), dict):   # thd-robust-estimator-v1: what center/scale mean, from the producer
            entry["estimator"] = r["estimator"]
        if r.get("daily_receipts") is not None:   # grassmann 641c01b7: every calibration day with its input receipt
            entry["calibration_days"] = r["daily_receipts"]
        try:   # thd-daily-measurement-v1: which measurement produced these moments (recording only)
            import thd_daily_measurement as TDM
            from seismic_thd import SeismicTHDAnalyzer as _A
            entry["measurement"] = TDM.measurement_record(net, sta, _A(n_harmonics=5, freq_tolerance=0.1, window_hours=24))
        except Exception as _e:  # noqa: BLE001 -- a recording failure never drops a baseline
            entry["measurement"] = {"identity": "UNIDENTIFIED", "error_class": type(_e).__name__}
        out[key] = entry
        attempts.append(dict(station=key, attempted_utc=attempted_utc, outcome="VALUE", disposition="RECALIBRATED"))
    # COLLECTION_RETIRED (cayley 2026-10-15): not fetched. The newest recalibrated record is carried forward as the
    # loader parsed it (its own calibration_date and period), so readers keep it and its age expires it; nothing is
    # deleted. A retired station named explicitly in `stations` was fetched above instead.
    for key in COLLECTION_RETIRED:
        if key in stations or key in out or key not in prior:
            continue
        out[key] = dict(prior[key])
        attempts.append(dict(station=key, attempted_utc=attempted_utc, outcome="NOT_COLLECTED",
                             reason=COLLECTION_RETIRED[key], disposition="PRIOR_RECORD_RETAINED",
                             retained_calibration_date=prior[key].get("calibration_date"),
                             retained_period=prior[key].get("calibration_period")))
    if not any(a.get("disposition") == "RECALIBRATED" for a in attempts):
        logger.error("recal produced no station baselines; NOT writing (keeping prior file)")
        return None
    out["_recal_attempts"] = attempts
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
