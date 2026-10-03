#!/usr/bin/env python3
"""thd_bootstrap.py - explicit, isolated bootstrap of a THD station baseline (grassmann 2026-10-03, v2 after codex
MEASUREMENT_SUPPORT_PACKET_REVIEW_CODEX_20261003 findings 1-3; codex plan section 2).

The deadlock: run_thd_recal._calibratable_stations() skips every station whose calibration_period is 'UNCALIBRATED'
(IU.SNZO, AK.SSL), so the weekly rolling recalibration can never commission such a station. This module is the one
explicit way out, and it never installs anything:

  * DECLARED OPERATOR. A bootstrap sample reproduces the WEEKLY RECAL operator (calibrate_thd_baselines.compute_daily_thd
    -> fetch_continuous_data_for_thd -> SeismicTHDAnalyzer.compute_thd): the UTC day D 00:00:00 to D+1 01:00:00 (25 h)
    at the NATIVE rate, demeaned and linearly detrended, five harmonics, tolerance 0.1, 24 h window. The daily ensemble
    operator ([target - 25 h, target], resample_poly to 1 Hz, compute_thd_with_noise) is DIFFERENT; the same bytes are
    also pushed through it and retained as a diagnostic (thd_daily_1hz), never as the sample. Neither operator is changed.
  * BOUND SUPPORT. Every DayObservation carries the actual NSLC and location, timezone-aware start/end of the samples
    actually used, npts, native rate, trace pieces, gap seconds, filled samples, the declared target window, the source
    and a sha256 of the support bytes. qualify() refuses by name when any of these disagree: a label whose bytes lie
    outside its window, empty or naive clocks, a non-finite rate, an npts/rate/span inconsistency, a location other
    than the one bound for the bootstrap (no silent fallback), a station or channel mismatch, an epoch mismatch, any
    gap/overlap/fill, a repeated day label, the same support bytes under two labels, and support newer than the
    registered 30-day exclusion (checked on the bytes' own timestamps, never on a directory name).
  * STITCHED, NOT INTERPOLATED. Retained fault-correlation cache days (24 h, 07:00 to 07:00 UTC) are stitched into the
    declared 25 h window from adjacent days. Pieces must abut exactly (one sample apart); an exact duplicate boundary
    sample is dropped only when its values are identical; any gap or non-identical overlap is a discontinuity, and the
    day refuses. Missing support is reported as exact raw intervals so a bounded acquisition can be sized precisely.
  * DIAGNOSTIC != ELIGIBLE. bootstrap() always returns diagnostics (counts, robust statistics, per-day manifest) and
    separately decides candidate_eligible: the registered floor (QA_THRESHOLDS min_days = 60) cannot be lowered, every
    value must be finite, dispersion must be finite and positive, and a QA grade of 'fail' refuses. Coverage below the
    QA threshold is a retained warning, not silently equated with the day count. A candidate entry retains its QA, its
    operator and a sha256-bound per-day manifest.
  * PRESERVED SIBLINGS. A candidate file is composed only from a hash-bound snapshot of the effective base set (the
    newest loadable dated file, exactly as station_baselines loads it); every untouched entry keeps its numbers, its
    window AND its effective calibration_date as an explicit per-entry field. station_baselines must honour that field
    (one-line backward-compatible change in the same branch; the loader falls back to the file-name date for legacy
    entries). A stale or empty base snapshot refuses. The candidate file name cannot match the loader's glob, cannot sit
    in the production baselines directory, is created exclusively, and is strict JSON (allow_nan=False).

Nothing here fetches from the network, and nothing is written into the production baselines directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_qa import QA_THRESHOLDS, compute_baseline_qa  # noqa: E402
from run_thd_recal import BASELINE_DIR, EXCLUDE_RECENT_DAYS, LOOKBACK_DAYS  # noqa: E402
from station_baselines import calibration_date_from_name  # noqa: E402

SCHEMA = "geospec.thd-bootstrap.v2"
EXPECTED_CHANNEL = "BHZ"
MIN_HOURS = 12.0                                      # the analyzer's own floor
MIN_BOOTSTRAP_DAYS = int(QA_THRESHOLDS["min_days"])   # 60: the registered floor; cannot be lowered here
WINDOW_HOURS = 25                                     # compute_daily_thd: day 00:00 + 25 h
LOADER_PREFIX = "thd_baselines_"                      # station_baselines globs thd_baselines_*.json
OPERATOR_WEEKLY = dict(
    name="weekly_recal", source="calibrate_thd_baselines.compute_daily_thd -> fetch_continuous_data_for_thd",
    window="UTC day D 00:00:00 to D+1 01:00:00 (25 h)", rate="native (no resampling)",
    preprocessing="demean + linear detrend", estimator="SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, "
    "window_hours=24).compute_thd", statistics="median -> mean_thd; MAD*1.4826 -> std_thd (calibrate_station)")
OPERATOR_DAILY = dict(
    name="daily_ensemble", source="ensemble.compute_thd_risk", window="[target - 25 h, target]",
    rate="resample_poly to 1 Hz", preprocessing="demean + linear detrend", estimator="compute_thd_with_noise",
    role="DIAGNOSTIC_ONLY: not the bootstrap sample")
REFUSAL_CODES = ("NSLC_MISMATCH", "LOCATION_MISMATCH", "CHANNEL_MISMATCH", "TIMESTAMPS_INVALID", "WINDOW_MISMATCH",
                 "SUPPORT_OUTSIDE_WINDOW", "DAY_OUTSIDE_WINDOW", "DAY_TOO_RECENT", "EPOCH_MISMATCH", "RATE_INVALID",
                 "RATE_MISMATCH", "NPTS_SPAN_INCONSISTENT", "NO_DATA", "COVERAGE_SHORT", "GAP_FILLED",
                 "SUPPORT_INCOMPLETE", "RESPONSE_MISSING", "DUPLICATE_DAY", "DUPLICATE_SUPPORT", "ESTIMATOR_ZERO")
ELIGIBILITY_CODES = ("INSUFFICIENT_DAYS", "NONFINITE_VALUES", "ZERO_DISPERSION", "QA_FAIL")


class BootstrapRefused(Exception):
    """Named refusal; the message starts with the code."""


# ----------------------------------------------------------------------------- time helpers
def parse_utc(s: str) -> Optional[datetime]:
    """Aware UTC datetime from an ISO string ('Z' or an explicit offset); None for empty/naive/unparseable."""
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        return None
    return d.astimezone(timezone.utc)


def iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def target_window(day: str) -> Tuple[datetime, datetime]:
    """The weekly operator's support for day D: [D 00:00:00Z, D+1 01:00:00Z)."""
    d0 = datetime.combine(date.fromisoformat(day), datetime.min.time(), tzinfo=timezone.utc)
    return d0, d0 + timedelta(hours=WINDOW_HOURS)


def registered_window(today: date) -> Tuple[date, date]:
    """The R3 window the weekly recal uses: LOOKBACK_DAYS ending EXCLUDE_RECENT_DAYS before `today`."""
    end = today - timedelta(days=EXCLUDE_RECENT_DAYS)
    return end - timedelta(days=LOOKBACK_DAYS), end


# ----------------------------------------------------------------------------- the observation
@dataclass
class DayObservation:
    """One target day of one station, with the support facts the ordinary fetch throws away."""
    day: str                       # label 'YYYY-MM-DD'; the declared window is target_window(day)
    station: str                   # 'NET.STA' the bootstrap is for
    network: str                   # NSLC actually read
    station_code: str
    location: str
    channel: str
    sampling_rate: float
    start_utc: str                 # first sample actually used (aware UTC ISO)
    end_utc: str                   # last sample actually used
    npts: int
    n_traces: int                  # trace pieces stitched (1 = single contiguous trace)
    gap_seconds: float             # total |gap or overlap| seconds inside the support (0 required)
    filled_samples: int            # samples filled/interpolated (0 required)
    response_available: Optional[bool]
    source: str                    # 'seismic_cache' | 'fdsn' | 'fixture'
    source_ref: str
    support_sha256: Optional[str] = None     # sha256 of the int32 support bytes actually used
    window_start_utc: str = ""     # declared operator window (must equal target_window(day))
    window_end_utc: str = ""
    operator: str = OPERATOR_WEEKLY["name"]
    thd: Optional[float] = None
    p1: Optional[float] = None
    f1: Optional[float] = None
    thd_daily_1hz: Optional[float] = None    # diagnostic: same bytes through the daily 1-Hz operator
    missing_support: List[List[str]] = field(default_factory=list)   # raw [start, end) intervals not available
    discontinuities: List[List[str]] = field(default_factory=list)   # [prev_end, next_start] joins that did not abut

    def coverage_hours(self) -> float:
        return (self.npts / self.sampling_rate) / 3600.0 if self.sampling_rate and self.sampling_rate > 0 else 0.0


def qualify(obs: DayObservation, window: Tuple[date, date], *, today: date, station: str, expected_location: str,
            expected_rate: Optional[float] = None, epoch: Optional[Tuple[Optional[date], Optional[date]]] = None,
            require_response: bool = False, seen_days: Optional[set] = None,
            seen_support: Optional[dict] = None) -> List[str]:
    """Every reason this observation is NOT a valid bootstrap day (empty list = qualifies). Order is fixed."""
    reasons: List[str] = []
    if obs.station != station or f"{obs.network}.{obs.station_code}" != station:
        reasons.append("NSLC_MISMATCH")
    if obs.location != expected_location:
        reasons.append("LOCATION_MISMATCH")
    if obs.channel != EXPECTED_CHANNEL:
        reasons.append("CHANNEL_MISMATCH")
    d = date.fromisoformat(obs.day)
    ws, we = target_window(obs.day)
    start, end = parse_utc(obs.start_utc), parse_utc(obs.end_utc)
    rate_ok = isinstance(obs.sampling_rate, (int, float)) and math.isfinite(obs.sampling_rate) and obs.sampling_rate > 0
    if start is None or end is None or end <= start:
        reasons.append("TIMESTAMPS_INVALID")
    if parse_utc(obs.window_start_utc) != ws or parse_utc(obs.window_end_utc) != we:
        reasons.append("WINDOW_MISMATCH")
    if start is not None and end is not None and rate_ok:
        dt = timedelta(seconds=1.0 / obs.sampling_rate)
        if start < ws - dt / 2 or end > we + dt / 2 or end <= start:
            reasons.append("SUPPORT_OUTSIDE_WINDOW")
        if end.date() > today - timedelta(days=EXCLUDE_RECENT_DAYS):
            reasons.append("DAY_TOO_RECENT")        # on the bytes' own clock, not the label
    if not (window[0] <= d <= window[1]):
        reasons.append("DAY_OUTSIDE_WINDOW")
    if epoch is not None:
        e0, e1 = epoch
        if (e0 is not None and d < e0) or (e1 is not None and d > e1):
            reasons.append("EPOCH_MISMATCH")
    if not rate_ok:
        reasons.append("RATE_INVALID")
    elif expected_rate is not None and abs(float(obs.sampling_rate) - float(expected_rate)) > 1e-6:
        reasons.append("RATE_MISMATCH")
    if rate_ok and start is not None and end is not None and end > start:
        expected_npts = int(round((end - start).total_seconds() * obs.sampling_rate)) + 1
        if abs(int(obs.npts) - expected_npts) > 1:
            reasons.append("NPTS_SPAN_INCONSISTENT")
    if obs.npts <= 0 or not rate_ok:
        reasons.append("NO_DATA")
    elif obs.coverage_hours() < MIN_HOURS:
        reasons.append("COVERAGE_SHORT")
    if obs.filled_samples > 0 or obs.gap_seconds != 0:
        reasons.append("GAP_FILLED")
    if obs.missing_support:
        reasons.append("SUPPORT_INCOMPLETE")   # the declared window is not fully covered by retained bytes
    if require_response and not obs.response_available:
        reasons.append("RESPONSE_MISSING")
    if seen_days is not None:
        if obs.day in seen_days:
            reasons.append("DUPLICATE_DAY")
        else:
            seen_days.add(obs.day)
    if seen_support is not None and obs.support_sha256:
        prior = seen_support.get(obs.support_sha256)
        if prior is not None and prior != obs.day:
            reasons.append("DUPLICATE_SUPPORT")
        else:
            seen_support.setdefault(obs.support_sha256, obs.day)
    return reasons


# ----------------------------------------------------------------------------- operators
def _detrend(x: np.ndarray) -> np.ndarray:
    """demean + linear detrend, as fetch_continuous_data_for_thd does (obspy detrend('demean'), detrend('linear'))."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    t = np.arange(x.size, dtype=np.float64)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def _analyzer(analyzer=None):
    if analyzer is not None:
        return analyzer
    from seismic_thd import SeismicTHDAnalyzer
    return SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)


def weekly_operator(data: np.ndarray, rate: float, analyzer=None) -> Tuple[float, float, float]:
    """The declared operator on prepared support: (thd, p1, f1)."""
    thd, p1, _harm, f1 = _analyzer(analyzer).compute_thd(_detrend(data), float(rate))
    return float(thd), float(p1), float(f1)


def daily_operator_1hz(data: np.ndarray, rate: float, analyzer=None) -> Optional[float]:
    """DIAGNOSTIC: the daily ensemble path on the same bytes (resample_poly to 1 Hz, compute_thd_with_noise)."""
    try:
        from math import gcd
        from scipy.signal import resample_poly
    except ImportError:
        return None
    x = _detrend(data)
    if rate > 1.5:
        up, down = 100, int(rate * 100)
        g = gcd(up, down)
        x = resample_poly(x, up // g, down // g)
        rate = 1.0
    out = _analyzer(analyzer).compute_thd_with_noise(x, float(rate))
    return float(out[0])


def estimate(obs: DayObservation, data: np.ndarray, analyzer=None, *, daily_diagnostic: bool = True) -> DayObservation:
    obs.thd, obs.p1, obs.f1 = weekly_operator(data, obs.sampling_rate, analyzer)
    if daily_diagnostic:
        obs.thd_daily_1hz = daily_operator_1hz(data, obs.sampling_rate, analyzer)
    return obs


# ----------------------------------------------------------------------------- retained cache -> stitched window
def _cache_file(cache_dir: str, day: date, suffix: str = "_waveforms.pkl") -> Optional[str]:
    d = os.path.join(cache_dir, day.strftime("%Y%m%d"))
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(suffix))
    return os.path.join(d, files[0]) if files else None


def _load_traces(path: str, station: str, location: str):
    """Traces of one station+location from a cache pickle ({'NET.STA': Stream}); [] when absent. No fallback."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    st = obj.get(station) if isinstance(obj, dict) else None
    if st is None:
        return [], set()
    locs = {tr.stats.location for tr in st}
    return sorted([tr for tr in st if tr.stats.location == location], key=lambda t: t.stats.starttime), locs


def stitch_cached_window(cache_dir: str, day: str, station: str, location: str, *, analyzer=None,
                         estimate_now: bool = True) -> DayObservation:
    """Build the declared 25 h support for `day` from the retained cache days D-1 and D (each 07:00->07:00 UTC).

    Pieces are joined only where they abut exactly (one sample apart) or where a single boundary sample is an exact
    duplicate (same time, identical value; the duplicate is dropped). Any other gap or overlap is a discontinuity: the
    observation records it (gap_seconds > 0, n_traces > 1) and qualify() refuses it. Samples outside the window are
    discarded. Missing support is recorded as exact raw intervals. Nothing is written."""
    ws, we = target_window(day)
    d = date.fromisoformat(day)
    net, sta = station.split(".", 1)
    paths = [p for p in (_cache_file(cache_dir, d - timedelta(days=1)), _cache_file(cache_dir, d)) if p]
    pieces, locs_seen, refs = [], set(), []
    for p in paths:
        trs, locs = _load_traces(p, station, location)
        locs_seen |= locs
        refs.append(os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p))
        pieces.extend(trs)
    pieces.sort(key=lambda t: t.stats.starttime)
    base = DayObservation(day=day, station=station, network=net, station_code=sta, location=location,
                          channel=EXPECTED_CHANNEL, sampling_rate=0.0, start_utc="", end_utc="", npts=0, n_traces=0,
                          gap_seconds=0.0, filled_samples=0, response_available=None, source="seismic_cache",
                          source_ref="%s#%s#loc=%s#locs_present=%s" % ("+".join(refs) or "NONE", station, location,
                                                                      ",".join(sorted(locs_seen)) or "NONE"),
                          window_start_utc=iso(ws), window_end_utc=iso(we))
    if not pieces:
        base.missing_support = [[iso(ws), iso(we)]]
        return base
    rate = float(pieces[0].stats.sampling_rate)
    base.channel = pieces[0].stats.channel
    base.sampling_rate = rate
    dt = 1.0 / rate
    # walk the pieces: build (t0, data) chunks that are exactly contiguous
    chunks = []          # list of [t0 (UTCDateTime), np.ndarray]
    gap_total = 0.0
    for tr in pieces:
        data = np.asarray(tr.data)
        if np.ma.isMaskedArray(tr.data):
            base.filled_samples += int(np.ma.count_masked(tr.data))
            data = np.asarray(np.ma.filled(tr.data, 0))
        if not chunks:
            chunks.append([tr.stats.starttime, data]); continue
        t0_prev, d_prev = chunks[-1]
        prev_end = t0_prev + (d_prev.size - 1) * dt
        delta = float(tr.stats.starttime - prev_end)
        if abs(delta - dt) <= dt / 4:                                  # exact abut
            chunks[-1][1] = np.concatenate([d_prev, data])
        elif abs(delta) <= dt / 4 and data.size and d_prev.size and data[0] == d_prev[-1]:   # identical duplicate sample
            chunks[-1][1] = np.concatenate([d_prev, data[1:]])
        else:                                                          # gap, overlap or non-identical duplicate
            gap_total += abs(delta - dt)
            base.discontinuities.append([iso(prev_end.datetime.replace(tzinfo=timezone.utc)),
                                         iso(tr.stats.starttime.datetime.replace(tzinfo=timezone.utc))])
            chunks.append([tr.stats.starttime, data])
    # choose the single chunk that covers the window best; cut it to [ws, we)
    from obspy import UTCDateTime
    uws, uwe = UTCDateTime(ws), UTCDateTime(we)
    best, best_cov = chunks[0], -math.inf
    for t0, data in chunks:
        t1 = t0 + (data.size - 1) * dt
        cov = float(min(t1, uwe) - max(t0, uws))
        if cov > best_cov:
            best, best_cov = (t0, data), cov
    t0, data = best
    i0 = max(0, int(math.ceil(float(uws - t0) * rate - 1e-6)))
    i1 = min(data.size, int(math.floor(float(uwe - t0) * rate - 1e-6)) + 1)   # last sample strictly before `we`
    sel = data[i0:i1] if i1 > i0 else data[:0]
    s0 = t0 + i0 * dt
    base.n_traces = len(chunks)
    base.gap_seconds = round(gap_total, 6) if len(chunks) > 1 else 0.0
    if sel.size == 0:
        base.missing_support = [[iso(ws), iso(we)]]
        return base
    s1 = s0 + (sel.size - 1) * dt
    base.start_utc, base.end_utc, base.npts = iso(s0.datetime.replace(tzinfo=timezone.utc)), \
        iso(s1.datetime.replace(tzinfo=timezone.utc)), int(sel.size)
    base.support_sha256 = hashlib.sha256(np.ascontiguousarray(sel.astype(np.int32)).tobytes()).hexdigest()
    # missing support = the window minus the UNION of every chunk's span (not just the chunk used for estimation), so
    # a failed join reports only the uncovered edges; the join itself is listed under discontinuities. A sub-sample
    # phase offset at an edge is not missing support; a whole sample or more is.
    spans = []
    for ct0, cdata in chunks:
        ct1 = ct0 + (cdata.size - 1) * dt
        a, b = max(float(ct0 - uws), 0.0), min(float(ct1 - uws) + dt, float(uwe - uws))
        if b > a:
            spans.append((a, b))
    spans.sort()
    missing, cursor = [], 0.0
    for a, b in spans:
        if a - cursor >= dt:
            missing.append([iso(ws + timedelta(seconds=cursor)), iso(ws + timedelta(seconds=a))])
        cursor = max(cursor, b)
    if float(uwe - uws) - cursor > 2 * dt:
        missing.append([iso(ws + timedelta(seconds=cursor)), iso(we)])
    base.missing_support = missing
    if estimate_now and base.gap_seconds == 0 and base.filled_samples == 0 and not missing:
        estimate(base, sel, analyzer=analyzer)
    return base


def _merge_intervals(pairs) -> List[List[datetime]]:
    iv = sorted((a, b) for a, b in pairs if a and b and b > a)
    merged: List[List[datetime]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _sized(merged: List[List[datetime]], rate: float) -> Dict:
    seconds = sum((b - a).total_seconds() for a, b in merged)
    samples = int(round(seconds * rate))
    return dict(intervals=[[iso(a), iso(b)] for a, b in merged], n_intervals=len(merged), total_seconds=round(seconds, 3),
                total_days=round(seconds / 86400, 3), samples_at_rate=samples, bytes_raw_int32=samples * 4,
                miniseed_estimate_bytes=[int(samples * 1.2), int(samples * 1.6)])


def acquisition_spec(observations: Iterable[DayObservation], per_day_reasons: Dict[str, List[str]], rate: float) -> Dict:
    """Exact support still needed, two ways: STRICT = the union of the full declared windows of every target day that
    did not qualify (one contiguous fetch per window; no splicing with cached bytes); MINIMAL = only the absent edges
    and the join discontinuities (splicing newly fetched bytes into cached bytes, which must then be re-validated by
    the same contiguity rules). Both are sized in int32 samples with a miniSEED estimate."""
    strict, minimal = [], []
    for o in observations:
        reasons = per_day_reasons.get(o.day, [])
        if not reasons:
            continue
        ws, we = target_window(o.day)
        strict.append((ws, we))
        for a, b in o.missing_support:
            minimal.append((parse_utc(a), parse_utc(b)))
        for a, b in o.discontinuities:
            minimal.append((parse_utc(a), parse_utc(b)))
    return dict(strict_full_windows=_sized(_merge_intervals(strict), rate),
                minimal_splice=_sized(_merge_intervals(minimal), rate),
                estimate_basis="Steim2 ~1.2-1.6 bytes/sample for broadband counts; an estimate, not a measurement",
                note="STRICT is the complete specification for one bounded fetch; MINIMAL assumes sample-exact splicing "
                     "into the retained cache and re-qualification afterwards")


def missing_support_report(observations: Iterable[DayObservation], rate: float) -> Dict:
    """Absent raw intervals only (merged, sized). See acquisition_spec for the complete request."""
    return _sized(_merge_intervals((parse_utc(a), parse_utc(b)) for o in observations for a, b in o.missing_support), rate)


# ----------------------------------------------------------------------------- bootstrap
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _jsonable(o):
    """Plain-Python copy (numpy scalars -> Python scalars) so strict JSON (allow_nan=False) can judge every value."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def bootstrap(station: str, observations: Iterable[DayObservation], *, today: date, expected_location: str,
              expected_rate: Optional[float] = None, epoch: Optional[Tuple[Optional[date], Optional[date]]] = None,
              require_response: bool = False, min_days: Optional[int] = None) -> Dict:
    """Qualify every observation, compute diagnostics ALWAYS, and decide candidate eligibility SEPARATELY.
    `min_days` can only raise the registered floor; it can never lower it."""
    window = registered_window(today)
    floor = MIN_BOOTSTRAP_DAYS
    eff_min = max(floor, int(min_days or 0))
    seen_days: set = set(); seen_support: dict = {}
    qualified: List[DayObservation] = []; per_day: List[Dict] = []; refused: Dict[str, int] = {}
    for obs in observations:
        reasons = qualify(obs, window, today=today, station=station, expected_location=expected_location,
                          expected_rate=expected_rate, epoch=epoch, require_response=require_response,
                          seen_days=seen_days, seen_support=seen_support)
        if not reasons and not (_finite(obs.thd) and obs.thd > 0 and _finite(obs.p1) and obs.p1 > 0):
            reasons = ["ESTIMATOR_ZERO"]
        per_day.append(dict(day=obs.day, reasons=reasons, thd=obs.thd, thd_daily_1hz=obs.thd_daily_1hz,
                            nslc=f"{obs.network}.{obs.station_code}.{obs.location}.{obs.channel}",
                            sampling_rate=obs.sampling_rate, start_utc=obs.start_utc, end_utc=obs.end_utc,
                            npts=obs.npts, coverage_hours=round(obs.coverage_hours(), 3), n_traces=obs.n_traces,
                            gap_seconds=obs.gap_seconds, filled_samples=obs.filled_samples, source=obs.source,
                            source_ref=obs.source_ref, support_sha256=obs.support_sha256,
                            missing_support=obs.missing_support, discontinuities=obs.discontinuities))
        for r in reasons:
            refused[r] = refused.get(r, 0) + 1
        if not reasons:
            qualified.append(obs)
    n_requested = (window[1] - window[0]).days + 1
    values = sorted((o.day, float(o.thd)) for o in qualified)
    thd = np.array([v for _, v in values], dtype=np.float64)
    finite_mask = np.isfinite(thd) if thd.size else np.array([], dtype=bool)
    diag: Dict = dict(n_qualified=len(values), n_nonfinite=int((~finite_mask).sum()) if thd.size else 0)
    if thd.size and finite_mask.all():
        med = float(np.median(thd)); mad = float(np.median(np.abs(thd - med)))
        diag.update(median=round(med, 6), mad_sigma=round(mad * 1.4826, 6), mean=round(float(np.mean(thd)), 6),
                    std=round(float(np.std(thd)), 6), min=round(float(thd.min()), 6), max=round(float(thd.max()), 6),
                    first_day=values[0][0], last_day=values[-1][0])
    manifest = [dict(day=o.day, nslc=f"{o.network}.{o.station_code}.{o.location}.{o.channel}", rate=o.sampling_rate,
                     start_utc=o.start_utc, end_utc=o.end_utc, npts=o.npts, source=o.source, source_ref=o.source_ref,
                     support_sha256=o.support_sha256, thd=o.thd, thd_daily_1hz=o.thd_daily_1hz) for o in qualified]
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    rate = int(round(float(np.median([o.sampling_rate for o in qualified])))) if qualified else None
    qa = compute_baseline_qa(station, values, n_requested, sample_rate_hz=rate or 0) if values and finite_mask.all() \
        else None
    elig: List[str] = []
    if len(values) < eff_min:
        elig.append("INSUFFICIENT_DAYS")
    if not thd.size or not finite_mask.all():
        elig.append("NONFINITE_VALUES") if thd.size else None
    if thd.size and finite_mask.all() and (diag["mad_sigma"] <= 0 or diag["std"] <= 0 or not math.isfinite(diag["std"])):
        elig.append("ZERO_DISPERSION")
    if qa is not None and qa.quality_grade == "fail":
        elig.append("QA_FAIL")
    coverage = dict(days_qualified=len(values), days_requested=n_requested,
                    coverage_pct=round(100.0 * len(values) / n_requested, 2),
                    qa_min_coverage_pct=QA_THRESHOLDS["min_coverage_pct"],
                    below_qa_threshold=100.0 * len(values) / n_requested < QA_THRESHOLDS["min_coverage_pct"],
                    policy="coverage below the QA threshold is retained as a QA warning on the entry; it does not by "
                           "itself refuse; a QA grade of 'fail' refuses; the registered day floor is separate")
    result: Dict = dict(schema=SCHEMA, station=station, today=today.isoformat(), expected_location=expected_location,
                        operator=OPERATOR_WEEKLY, daily_operator_diagnostic=OPERATOR_DAILY,
                        window_registered=dict(start=window[0].isoformat(), end=window[1].isoformat(),
                                               lookback_days=LOOKBACK_DAYS, exclude_recent_days=EXCLUDE_RECENT_DAYS,
                                               days_requested=n_requested),
                        min_days_registered=floor, min_days_effective=eff_min, n_observations=len(per_day),
                        n_qualified=len(values), refused=refused, per_day=per_day,
                        diagnostic_complete=True, diagnostics=diag, coverage_policy=coverage,
                        qa=qa.to_dict() if qa else None, manifest=manifest, manifest_sha256=manifest_sha,
                        candidate_eligible=not elig, eligibility_refusals=elig)
    result = _jsonable(result)
    if elig:
        result["detail"] = "; ".join(elig)
        return result
    result["entry"] = {
        "station": station,
        "mean_thd": diag["median"],                 # median, as calibrate_station writes it
        "std_thd": diag["mad_sigma"],               # MAD-sigma, as calibrate_station writes it
        "n_samples": len(values),
        "calibration_period": f"{values[0][0]} to {values[-1][0]}",   # the days actually used
        "calibration_date": today.isoformat(),
        "notes": (f"BOOTSTRAP (thd_bootstrap v2, grassmann): {len(values)} contiguous {WINDOW_HOURS} h days via the "
                  f"weekly_recal operator inside the registered window {window[0]}..{window[1]}; QA {qa.quality_grade}"),
        "mean_thd_classic": diag["mean"], "std_thd_classic": diag["std"],
        "operator": OPERATOR_WEEKLY, "coverage_policy": coverage, "qa": qa.to_dict(),
        "manifest_sha256": manifest_sha, "manifest": manifest,
        "bootstrap": dict(method=SCHEMA, window_registered=result["window_registered"], days_qualified=len(values),
                          days_refused=refused, location=expected_location, channel=EXPECTED_CHANNEL,
                          sampling_rate_hz=rate, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
    }
    return _jsonable(result)


# ----------------------------------------------------------------------------- base snapshot + candidate file
def snapshot_effective_base(bdir) -> Dict:
    """The newest LOADABLE dated file, selected exactly as station_baselines._load_newest_baseline_file does, hash-bound.
    Refuses when no file yields entries (EMPTY_BASE_SNAPSHOT)."""
    bdir = Path(bdir)
    files = sorted(bdir.glob(LOADER_PREFIX + "*.json"), key=lambda p: p.name, reverse=True)
    for f in files:
        raw = f.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        entries = data["baselines"] if isinstance(data, dict) and isinstance(data.get("baselines"), list) else \
            ([v for v in data.values() if isinstance(v, dict) and "station" in v] if isinstance(data, dict) else [])
        loaded = {e["station"]: dict(e) for e in entries if e.get("mean_thd") is not None and e.get("std_thd") is not None}
        if loaded:
            return dict(path=str(f), name=f.name, sha256=hashlib.sha256(raw).hexdigest(),
                        calibration_date=calibration_date_from_name(f.name), entries=loaded)
    raise BootstrapRefused("EMPTY_BASE_SNAPSHOT: no loadable dated baseline file in %s" % bdir)


def compose_candidate_file(result: Dict, base_snapshot: Dict, out_path: str, *, forbid_dir: Path = BASELINE_DIR) -> str:
    """Write base entries (numbers, windows AND effective calibration_date preserved per entry) + the eligible bootstrap
    entry as a CANDIDATE file. Refuses: an ineligible result, a stale/empty base, a name the loader would pick up, a
    location inside the production baselines directory, an existing file. Strict JSON, exclusive creation."""
    if not result.get("candidate_eligible") or "entry" not in result:
        raise BootstrapRefused("CANDIDATE_NOT_ELIGIBLE: " + "; ".join(result.get("eligibility_refusals") or ["no entry"]))
    entry = result["entry"]
    if not base_snapshot or not base_snapshot.get("entries"):
        raise BootstrapRefused("EMPTY_BASE_SNAPSHOT")
    raw = Path(base_snapshot["path"]).read_bytes() if Path(base_snapshot["path"]).exists() else b""
    if hashlib.sha256(raw).hexdigest() != base_snapshot["sha256"]:
        raise BootstrapRefused("BASE_SNAPSHOT_STALE: %s changed since the snapshot" % base_snapshot["name"])
    out = Path(out_path)
    if out.name.startswith(LOADER_PREFIX):
        raise BootstrapRefused("CANDIDATE_NAME_MATCHES_LOADER_GLOB: rename happens only at the reviewed landing")
    try:
        inside = out.resolve().parent == Path(forbid_dir).resolve()
    except OSError:
        inside = False
    if inside:
        raise BootstrapRefused("CANDIDATE_MUST_NOT_LAND_IN_PRODUCTION_BASELINES")
    merged: Dict[str, Dict] = {}
    for st, e in base_snapshot["entries"].items():
        if st == entry["station"]:
            continue
        keep = dict(e)
        keep["calibration_date"] = e.get("calibration_date") or base_snapshot["calibration_date"]
        keep["carried_from"] = dict(file=base_snapshot["name"], sha256=base_snapshot["sha256"])
        merged[st] = keep
    merged[entry["station"]] = dict(entry)
    merged["_bootstrap_candidate"] = dict(schema=SCHEMA, station=entry["station"], base_snapshot=dict(
        name=base_snapshot["name"], sha256=base_snapshot["sha256"], calibration_date=base_snapshot["calibration_date"]),
        requires="station_baselines loader honouring per-entry calibration_date (branch change); landing = reviewed "
                 "rename to %s<YYYYMMDD>.json on the host" % LOADER_PREFIX)
    body = json.dumps(_jsonable(merged), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if out.exists():
        raise BootstrapRefused("CANDIDATE_EXISTS: refusing to overwrite %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "x", encoding="utf-8", newline="\n") as fh:     # exclusive: never overwrite, even on a race
            fh.write(body)
    except FileExistsError:
        raise BootstrapRefused("CANDIDATE_EXISTS: refusing to overwrite %s" % out)
    return str(out)


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--station", required=True)
    ap.add_argument("--cache-dir", required=True, help="seismic_cache/<region> with YYYYMMDD subdirs (read-only)")
    ap.add_argument("--location", required=True, help="bound location code, e.g. 00 (no fallback)")
    ap.add_argument("--today", default=None)
    ap.add_argument("--out-dir", required=True, help="evidence directory OUTSIDE the production baselines dir")
    ap.add_argument("--base-dir", default=str(BASELINE_DIR), help="directory holding the effective dated baselines "
                                                                  "(read-only snapshot source)")
    ap.add_argument("--expected-rate", type=float, default=None)
    ap.add_argument("--epoch-start", default=None)
    ap.add_argument("--require-response", action="store_true")
    args = ap.parse_args(argv)
    today = date.fromisoformat(args.today) if args.today else datetime.now(timezone.utc).date()
    window = registered_window(today)
    obs = []
    d = window[0]
    while d <= window[1]:
        obs.append(stitch_cached_window(args.cache_dir, d.isoformat(), args.station, args.location))
        d += timedelta(days=1)
    epoch = (date.fromisoformat(args.epoch_start), None) if args.epoch_start else None
    res = bootstrap(args.station, obs, today=today, expected_location=args.location,
                    expected_rate=args.expected_rate, epoch=epoch, require_response=args.require_response)
    rate = args.expected_rate or next((o.sampling_rate for o in obs if o.sampling_rate), 40.0)
    res["missing_support"] = missing_support_report(obs, rate)
    res["acquisition_spec"] = acquisition_spec(obs, {p["day"]: p["reasons"] for p in res["per_day"]}, rate)
    res["cache"] = dict(cache_dir=os.path.abspath(args.cache_dir), target_days=len(obs))
    os.makedirs(args.out_dir, exist_ok=True)
    rpath = os.path.join(args.out_dir, f"thd_bootstrap_{args.station}_{today.strftime('%Y%m%d')}.json")
    with open(rpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print("THD_BOOTSTRAP", "ELIGIBLE" if res["candidate_eligible"] else "NOT_ELIGIBLE", args.station,
          "qualified=%d/%d refused=%s" % (res["n_qualified"], res["n_observations"],
                                          json.dumps(res["refused"], sort_keys=True)))
    print("   diagnostics:", json.dumps(res["diagnostics"], sort_keys=True))
    for key in ("strict_full_windows", "minimal_splice"):
        s = res["acquisition_spec"][key]
        print("   acquisition %-20s intervals=%d days=%s raw_int32=%d B miniSEED~%s B" % (
            key, s["n_intervals"], s["total_days"], s["bytes_raw_int32"], s["miniseed_estimate_bytes"]))
    if not res["candidate_eligible"]:
        print("   eligibility refusals:", res["eligibility_refusals"])
        return 2
    base = snapshot_effective_base(args.base_dir)
    cpath = compose_candidate_file(res, base, os.path.join(
        args.out_dir, f"CANDIDATE_{LOADER_PREFIX}{today.strftime('%Y%m%d')}.json"))
    print("   candidate:", cpath, "(base snapshot %s %s)" % (base["name"], base["sha256"][:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
