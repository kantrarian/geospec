#!/usr/bin/env python3
"""thd_bootstrap.py - explicit, isolated bootstrap of a THD station baseline (grassmann 2026-10-03; codex
MEASUREMENT_SUPPORT_AND_CONTEXT_PLAN_CODEX_20261003.md section 2).

The deadlock: run_thd_recal._calibratable_stations() skips every station whose calibration_period is
'UNCALIBRATED' (IU.SNZO, AK.SSL), so the weekly rolling recalibration can never commission such a station, and the
ensemble keeps scoring it against a hand-typed default (n_samples=0). This module is the one explicit way out:

  * it takes DAY OBSERVATIONS (one per UTC day) that carry what the ordinary fetch path discards -- location code,
    trace count, gap seconds, filled samples, native rate, coverage, response availability, source and digest;
  * it QUALIFIES each day against the registered R3 window (run_thd_recal.LOOKBACK_DAYS / EXCLUDE_RECENT_DAYS),
    the expected channel/rate/epoch, a contiguity requirement (no interpolated or filled samples count as a valid
    day), an optional response requirement, and exact-day de-duplication (a repeated report is not a new day);
  * it ESTIMATES THD with the production SeismicTHDAnalyzer on demeaned/linearly-detrended samples, exactly as
    calibrate_thd_baselines.compute_daily_thd does for the weekly recal;
  * it refuses honestly below QA_THRESHOLDS['min_days'] qualifying days (60; stricter than calibrate_station's 10,
    because a bootstrap has no prior window to fall back on), and otherwise returns the same robust statistics the
    weekly recal writes (median as mean_thd, MAD*1.4826 as std_thd) plus baseline_qa, with full provenance;
  * it composes a CANDIDATE dated baseline file whose name deliberately does NOT match the production loader's
    glob (`thd_baselines_*.json`), and refuses to write inside the production baselines directory. Landing the
    candidate (rename + place on the host) is a reviewed host action, never a side effect of running this tool.

Once a dated file carrying a real calibration_period for the station is loaded newest-first by station_baselines,
_calibratable_stations() includes the station with NO code change, and the ordinary weekly recal takes over.

Nothing here fetches from the network. Retained raw waveforms come from the fault-correlation seismic cache
(monitoring/data/seismic_cache/<region>/<YYYYMMDD>/*_waveforms.pkl, a dict {'NET.STA': obspy.Stream}); a bounded
FDSN acquisition for missing days is a separate, named request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_qa import QA_THRESHOLDS, compute_baseline_qa  # noqa: E402
from run_thd_recal import BASELINE_DIR, EXCLUDE_RECENT_DAYS, LOOKBACK_DAYS  # noqa: E402

SCHEMA = "geospec.thd-bootstrap.v1"
EXPECTED_CHANNEL = "BHZ"
MIN_HOURS = 12.0                                  # the analyzer's own floor (SeismicTHDAnalyzer.compute_thd)
MIN_BOOTSTRAP_DAYS = int(QA_THRESHOLDS["min_days"])  # 60
REFUSAL_CODES = ("DAY_OUTSIDE_WINDOW", "DAY_TOO_RECENT", "CHANNEL_MISMATCH", "EPOCH_MISMATCH", "RATE_MISMATCH",
                 "NO_DATA", "COVERAGE_SHORT", "GAP_FILLED", "RESPONSE_MISSING", "DUPLICATE_DAY", "ESTIMATOR_ZERO")


class BootstrapRefused(Exception):
    """Named refusal; the message starts with the code."""


@dataclass
class DayObservation:
    """One UTC day of one station, with the acquisition facts the ordinary path throws away."""
    day: str                       # 'YYYY-MM-DD'
    station: str                   # 'NET.STA'
    location: str                  # location code actually used ('' allowed)
    channel: str
    sampling_rate: float
    start_utc: str
    end_utc: str
    npts: int                      # samples actually present (not counting fills)
    n_traces: int                  # traces for this location before any merge
    gap_seconds: float             # total gap duration inside [start, end]
    filled_samples: int            # samples that are/would be interpolated or filled
    response_available: Optional[bool]   # None = not measured
    source: str                    # 'seismic_cache' | 'fdsn' | 'fixture'
    source_ref: str                # path / URL / fixture name
    source_sha256: Optional[str] = None
    thd: Optional[float] = None
    p1: Optional[float] = None
    f1: Optional[float] = None

    def coverage_hours(self) -> float:
        return (self.npts / self.sampling_rate) / 3600.0 if self.sampling_rate else 0.0


def registered_window(today: date) -> Tuple[date, date]:
    """The R3 window the weekly recal uses: LOOKBACK_DAYS ending EXCLUDE_RECENT_DAYS before `today`."""
    end = today - timedelta(days=EXCLUDE_RECENT_DAYS)
    return end - timedelta(days=LOOKBACK_DAYS), end


def qualify(obs: DayObservation, window: Tuple[date, date], *, today: date, expected_rate: Optional[float] = None,
            epoch: Optional[Tuple[Optional[date], Optional[date]]] = None, require_response: bool = False,
            seen_days: Optional[set] = None) -> List[str]:
    """Every reason this observation is NOT a valid bootstrap day (empty list = qualifies). Order is fixed."""
    reasons: List[str] = []
    d = date.fromisoformat(obs.day)
    if not (window[0] <= d <= window[1]):
        reasons.append("DAY_OUTSIDE_WINDOW")
    if d > today - timedelta(days=EXCLUDE_RECENT_DAYS):
        reasons.append("DAY_TOO_RECENT")
    if obs.channel != EXPECTED_CHANNEL:
        reasons.append("CHANNEL_MISMATCH")
    if epoch is not None:
        e0, e1 = epoch
        if (e0 is not None and d < e0) or (e1 is not None and d > e1):
            reasons.append("EPOCH_MISMATCH")
    if expected_rate is not None and abs(float(obs.sampling_rate) - float(expected_rate)) > 1e-6:
        reasons.append("RATE_MISMATCH")
    if obs.npts <= 0 or obs.sampling_rate <= 0:
        reasons.append("NO_DATA")
    elif obs.coverage_hours() < MIN_HOURS:
        reasons.append("COVERAGE_SHORT")
    if obs.filled_samples > 0 or obs.gap_seconds != 0:
        reasons.append("GAP_FILLED")      # a gap, an overlap or any filled sample never makes a valid day;
                                          # abutting traces (gap_seconds == 0) merge without interpolation
    if require_response and not obs.response_available:
        reasons.append("RESPONSE_MISSING")
    if seen_days is not None:
        if obs.day in seen_days:
            reasons.append("DUPLICATE_DAY")
        else:
            seen_days.add(obs.day)
    return reasons


def _detrend(x: np.ndarray) -> np.ndarray:
    """demean + linear detrend, as fetch_continuous_data_for_thd does before compute_thd."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    t = np.arange(x.size, dtype=np.float64)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def estimate(obs: DayObservation, data: np.ndarray, analyzer=None) -> DayObservation:
    """Production estimator on one day: SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)."""
    if analyzer is None:
        from seismic_thd import SeismicTHDAnalyzer
        analyzer = SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)
    thd, p1, _harm, f1 = analyzer.compute_thd(_detrend(data), float(obs.sampling_rate))
    obs.thd, obs.p1, obs.f1 = float(thd), float(p1), float(f1)
    return obs


def observe_cached_day(path: str, station: str, day: str, *, prefer_location: str = "00",
                       analyzer=None, estimate_now: bool = True) -> DayObservation:
    """Build a DayObservation from one retained fault-correlation cache pickle ({'NET.STA': Stream}).

    One location code is used per day (prefer_location, else the lexically first present); both are recorded in
    source_ref. Contiguity is measured on the chosen location's traces before any merge: n_traces, get_gaps(),
    masked (filled) samples. Nothing is written."""
    with open(path, "rb") as fh:
        blob = fh.read()
    obj = pickle.loads(blob)
    st = obj.get(station) if isinstance(obj, dict) else None
    if st is None or len(st) == 0:
        return DayObservation(day, station, "", "", 0.0, "", "", 0, 0, 0.0, 0, None, "seismic_cache", path,
                              hashlib.sha256(blob).hexdigest())
    from obspy import Stream
    locs = sorted({tr.stats.location for tr in st})
    loc = prefer_location if prefer_location in locs else locs[0]
    sub = Stream(sorted([tr for tr in st if tr.stats.location == loc], key=lambda t: t.stats.starttime))
    gaps = sub.get_gaps()
    gap_seconds = float(sum(abs(g[6]) for g in gaps))      # gaps AND overlaps (negative) both count
    filled = int(sum(int(np.ma.count_masked(tr.data)) if np.ma.isMaskedArray(tr.data) else 0 for tr in sub))
    first, last = sub[0], sub[-1]
    data = None
    if filled == 0 and gap_seconds == 0:
        if len(sub) == 1:
            data = np.asarray(first.data)
        else:   # abutting traces: merge WITHOUT a fill value; any masked sample after the merge is a fill
            merged = sub.copy().merge(method=1, fill_value=None)
            if len(merged) == 1 and not np.ma.isMaskedArray(merged[0].data):
                data = np.asarray(merged[0].data)
            else:
                filled = int(np.ma.count_masked(merged[0].data)) if np.ma.isMaskedArray(merged[0].data) else -1
    obs = DayObservation(day=day, station=station, location=loc, channel=first.stats.channel,
                         sampling_rate=float(first.stats.sampling_rate), start_utc=str(first.stats.starttime),
                         end_utc=str(last.stats.endtime), npts=int(sum(tr.stats.npts for tr in sub)),
                         n_traces=len(sub), gap_seconds=gap_seconds, filled_samples=filled, response_available=None,
                         source="seismic_cache", source_ref=f"{path}#{station}#loc={loc}#locs_present={','.join(locs)}",
                         source_sha256=hashlib.sha256(blob).hexdigest())
    if filled != 0:
        obs.filled_samples = abs(filled)
    if estimate_now and data is not None:
        estimate(obs, data, analyzer=analyzer)
    return obs


def cache_days(cache_dir: str, pattern_suffix: str = "_waveforms.pkl") -> List[Tuple[str, str]]:
    """(day, path) for every YYYYMMDD subdirectory with a waveform pickle; the three segment files per day are
    byte-identical in the kaikoura cache, so the lexically first one is used."""
    out = []
    for name in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, name)
        if not (os.path.isdir(d) and len(name) == 8 and name.isdigit()):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(pattern_suffix))
        if files:
            out.append((f"{name[:4]}-{name[4:6]}-{name[6:]}", os.path.join(d, files[0])))
    return out


def bootstrap(station: str, observations: Iterable[DayObservation], *, today: date,
              min_days: int = MIN_BOOTSTRAP_DAYS, expected_rate: Optional[float] = None,
              epoch: Optional[Tuple[Optional[date], Optional[date]]] = None,
              require_response: bool = False) -> Dict:
    """Qualify, then either refuse (ok=False, refusal=INSUFFICIENT_DAYS, every refusal counted) or return the
    baseline entry with provenance and QA. Statistics mirror calibrate_thd_baselines.calibrate_station."""
    window = registered_window(today)
    seen: set = set()
    qualified: List[DayObservation] = []
    per_day: List[Dict] = []
    refused: Dict[str, int] = {}
    for obs in observations:
        if obs.station != station:
            raise BootstrapRefused(f"STATION_MISMATCH: observation for {obs.station}, bootstrap for {station}")
        reasons = qualify(obs, window, today=today, expected_rate=expected_rate, epoch=epoch,
                          require_response=require_response, seen_days=seen)
        if not reasons and not (obs.thd is not None and obs.thd > 0 and obs.p1 is not None and obs.p1 > 0):
            reasons = ["ESTIMATOR_ZERO"]
        per_day.append(dict(day=obs.day, reasons=reasons, thd=obs.thd, location=obs.location,
                            coverage_hours=round(obs.coverage_hours(), 3), n_traces=obs.n_traces,
                            gap_seconds=obs.gap_seconds, filled_samples=obs.filled_samples, source=obs.source,
                            source_sha256=obs.source_sha256))
        if reasons:
            for r in reasons:
                refused[r] = refused.get(r, 0) + 1
        else:
            qualified.append(obs)
    n_requested = (window[1] - window[0]).days + 1
    base = dict(schema=SCHEMA, station=station, today=today.isoformat(),
                window_registered=dict(start=window[0].isoformat(), end=window[1].isoformat(),
                                       lookback_days=LOOKBACK_DAYS, exclude_recent_days=EXCLUDE_RECENT_DAYS,
                                       days_requested=n_requested),
                n_observations=len(per_day), n_qualified=len(qualified), min_days=min_days, refused=refused,
                per_day=per_day)
    if len(qualified) < min_days:
        return dict(base, ok=False, refusal="INSUFFICIENT_DAYS",
                    detail=f"{len(qualified)} qualifying days < {min_days}")
    values = sorted((o.day, float(o.thd)) for o in qualified)
    thd = np.array([v for _, v in values])
    median = float(np.median(thd))
    mad = float(np.median(np.abs(thd - median)))
    rate = int(round(float(np.median([o.sampling_rate for o in qualified]))))
    qa = compute_baseline_qa(station, values, n_requested, sample_rate_hz=rate)
    sources = sorted({o.source for o in qualified})
    entry = {
        "station": station,
        "mean_thd": round(median, 6),               # median, as calibrate_station writes it
        "std_thd": round(mad * 1.4826, 6),          # MAD-sigma, as calibrate_station writes it
        "n_samples": len(values),
        "calibration_period": f"{values[0][0]} to {values[-1][0]}",   # the days actually used, not the request
        "notes": (f"BOOTSTRAP (thd_bootstrap.py, grassmann 2026-10-03): {len(values)} contiguous days from "
                  f"{'/'.join(sources)} inside the registered window {window[0]}..{window[1]}; QA {qa.quality_grade}"),
        "mean_thd_classic": round(float(np.mean(thd)), 6),
        "std_thd_classic": round(float(np.std(thd)), 6),
        "bootstrap": dict(method="thd_bootstrap.v1", window_registered=base["window_registered"],
                          days_qualified=len(values), days_refused=refused, sources=sources,
                          locations=sorted({o.location for o in qualified}), channel=EXPECTED_CHANNEL,
                          sampling_rate_hz=rate, response_measured=any(o.response_available is not None
                                                                        for o in qualified),
                          generated_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")),
        "qa": qa.to_dict(),
    }
    return dict(base, ok=True, entry=entry, daily_values=values)


LOADER_PREFIX = "thd_baselines_"     # station_baselines._load_newest_baseline_file globs thd_baselines_*.json


def compose_candidate_file(entry: Dict, base_entries: Dict[str, Dict], out_path: str,
                           *, forbid_dir: Path = BASELINE_DIR) -> str:
    """Write base_entries + the bootstrap entry as a CANDIDATE file. Refuses: a name the production loader would
    pick up, a location inside the production baselines directory, or an existing file."""
    out = Path(out_path)
    if out.name.startswith(LOADER_PREFIX):
        raise BootstrapRefused("CANDIDATE_NAME_MATCHES_LOADER_GLOB: rename happens only at the reviewed landing")
    try:
        inside = out.resolve().parent == Path(forbid_dir).resolve()
    except OSError:
        inside = False
    if inside:
        raise BootstrapRefused("CANDIDATE_MUST_NOT_LAND_IN_PRODUCTION_BASELINES")
    if out.exists():
        raise BootstrapRefused("CANDIDATE_EXISTS: refusing to overwrite")
    merged = {k: dict(v) for k, v in base_entries.items()}
    merged[entry["station"]] = {k: entry[k] for k in ("station", "mean_thd", "std_thd", "n_samples",
                                                       "calibration_period", "notes")}
    merged[entry["station"]]["bootstrap"] = entry["bootstrap"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return str(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--station", required=True, help="e.g. IU.SNZO")
    ap.add_argument("--cache-dir", required=True, help="seismic_cache/<region> directory with YYYYMMDD subdirs")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD clock read; default = UTC today")
    ap.add_argument("--out-dir", required=True, help="evidence directory OUTSIDE the production baselines dir")
    ap.add_argument("--base-file", default=None, help="existing dated thd_baselines file whose entries the "
                                                    "candidate carries forward (read-only)")
    ap.add_argument("--expected-rate", type=float, default=None)
    ap.add_argument("--epoch-start", default=None)
    ap.add_argument("--prefer-location", default="00")
    ap.add_argument("--require-response", action="store_true")
    args = ap.parse_args(argv)
    today = date.fromisoformat(args.today) if args.today else datetime.utcnow().date()
    window = registered_window(today)
    obs = []
    outside = []           # cached days outside the registered window: counted, never opened, never scored
    for day, path in cache_days(args.cache_dir):
        d = date.fromisoformat(day)
        if window[0] <= d <= window[1]:
            obs.append(observe_cached_day(path, args.station, day, prefer_location=args.prefer_location))
        else:
            outside.append(day)
    epoch = (date.fromisoformat(args.epoch_start), None) if args.epoch_start else None
    res = bootstrap(args.station, obs, today=today, expected_rate=args.expected_rate, epoch=epoch,
                    require_response=args.require_response)
    missing = []
    present = {o.day for o in obs}
    d = window[0]
    while d <= window[1]:
        if d.isoformat() not in present:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    res["cache"] = dict(cache_dir=os.path.abspath(args.cache_dir), days_cached_total=len(obs) + len(outside),
                        days_cached_in_window=len(obs), days_cached_outside_window=len(outside),
                        outside_window_range=[outside[0], outside[-1]] if outside else None,
                        window_days_missing_from_cache=missing)
    os.makedirs(args.out_dir, exist_ok=True)
    rpath = os.path.join(args.out_dir, f"thd_bootstrap_{args.station}_{today.strftime('%Y%m%d')}.json")
    with open(rpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    print("THD_BOOTSTRAP", "OK" if res["ok"] else "REFUSED", args.station, "qualified=%d/%d refused=%s" %
          (res["n_qualified"], res["n_observations"], json.dumps(res["refused"], sort_keys=True)))
    if not res["ok"]:
        print("  ", res["refusal"], res["detail"])
        return 2
    e = res["entry"]
    print("   entry: mean_thd=%s std_thd=%s n=%s period=%s qa=%s issues=%s" % (
        e["mean_thd"], e["std_thd"], e["n_samples"], e["calibration_period"], e["qa"]["quality_grade"],
        e["qa"]["issues"]))
    base_entries = {}
    if args.base_file:
        with open(args.base_file, encoding="utf-8") as fh:
            data = json.load(fh)
        base_entries = {k: v for k, v in data.items() if isinstance(v, dict) and "station" in v}
    cpath = compose_candidate_file(e, base_entries, os.path.join(
        args.out_dir, f"CANDIDATE_{LOADER_PREFIX}{today.strftime('%Y%m%d')}.json"))
    print("   candidate:", cpath, "(rename to %s%s.json and place in data/baselines ONLY at the reviewed landing)"
          % (LOADER_PREFIX, today.strftime('%Y%m%d')))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
