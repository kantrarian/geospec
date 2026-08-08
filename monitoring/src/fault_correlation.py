"""
fault_correlation.py
Fault Correlation Dynamics Analysis for Earthquake Precursor Detection.

Physical Basis:
- Normal state: Fault segments are correlated (stress distributed)
- Pre-earthquake: Segments decouple as stress concentrates
- Observable: Eigenvalue ratios DROP before rupture

This module computes:
1. Correlation matrix between fault segment activities
2. Eigenvalue spectrum (especially λ2/λ1 ratio)
3. Participation ratio (stress distribution measure)
4. Decorrelation detection (precursor signal)

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import hashlib
import os
import re
import logging
import json

from fault_segments import FaultSegment, get_segments_for_region
from seismic_data import SeismicDataFetcher, compute_segment_activity_index
import seismic_data as SD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# D2 RE-BAND rev-2 seams (INCIDENT 2026-07-31): fail-closed observability gate,
# exact-UTC-grid alignment, externally-pinned calibration capsule, station-topology
# contract. None of these lift any of the five D2 freezes; they make the honest path
# expressible so the freeze can eventually be lifted with real calibration values.
# =============================================================================

_MIN_PRE_ENVELOPE_RATE_HZ = 25.0          # provenance: envelope taken at native rate
_MIN_EFFECTIVE_SAMPLES = 8
_DEGENERACY_REL_STD = 1e-6                 # scale-independent constant/near-constant floor
_COVERAGE_TOL = 1e-3
_GRID_TOL = 1e-6
_CALIBRATION_SCHEMA = "geospec-d2-calibration-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CalibrationUnavailable(Exception):
    """Raised when a calibration capsule cannot be honestly admitted (missing, pin
    mismatch, schema/binding/date/lag violation). Fail-closed: no capsule -> no score."""

    def __init__(self, reasons):
        self.reasons = list(reasons) if isinstance(reasons, (list, tuple)) else [str(reasons)]
        super().__init__("; ".join(self.reasons))


def observability_gate(series_by_segment):
    """Fail-closed BEFORE correlation. Returns (ok, reasons). Requires: >= 2 segments,
    finite samples, minimum effective length, scale-independent non-degeneracy (constant /
    near-constant fails -- no std->1.0 pass-through), and PROVENANCE (pre_envelope_rate_hz
    >= 25 Hz + rev-2 band/processing version -- the labels are non-forgeable via the
    seismic_data identity chain)."""
    reasons = []
    n_valid = 0
    if len(series_by_segment) < 2:
        reasons.append(f"need >= 2 observable segments (got {len(series_by_segment)})")
    band_tag = getattr(SD, "_FAULT_CORR_BAND_TAG", "1-10Hz")
    version = SD.PROCESSING_VERSION
    for name, s in series_by_segment.items():
        v = np.asarray(getattr(s, "values", s), dtype=float).ravel()
        if v.size < _MIN_EFFECTIVE_SAMPLES:
            reasons.append(f"{name}: insufficient effective samples ({v.size})")
            continue
        if not np.all(np.isfinite(v)):
            reasons.append(f"{name}: non-finite samples")
            continue
        pre_rate = float(getattr(s, "pre_envelope_rate_hz", 0.0))
        if pre_rate < _MIN_PRE_ENVELOPE_RATE_HZ:
            reasons.append(f"{name}: pre_envelope_rate_hz {pre_rate} < "
                           f"{_MIN_PRE_ENVELOPE_RATE_HZ} (anti-alias erased provenance)")
            continue
        if getattr(s, "band_tag", None) != band_tag:
            reasons.append(f"{name}: band_tag {getattr(s, 'band_tag', None)} != {band_tag}")
            continue
        if getattr(s, "processing_version", None) != version:
            reasons.append(f"{name}: processing_version "
                           f"{getattr(s, 'processing_version', None)} != {version}")
            continue
        mean = float(np.mean(v))
        std = float(np.std(v))
        rel = std / (abs(mean) + 1e-30)
        if std <= 0.0 or rel < _DEGENERACY_REL_STD:
            reasons.append(f"{name}: degenerate (std={std:.3e}, rel={rel:.3e})")
            continue
        n_valid += 1
    if n_valid < 2:
        if not reasons:
            reasons.append("fewer than 2 observable segments after QC")
        return False, reasons
    return (len(reasons) == 0), reasons


def capsule_registry_entry(region, registry_path):
    """Read the EXTERNAL capsule registry (region-keyed JSON) and return
    {capsule_path, expected_sha256, topology_version}. The registered digest is the ONLY
    production source of expected_sha256 -- a loader that hashes the capsule it is about to
    load is self-attestation. Missing registry/region raises CalibrationUnavailable."""
    if not registry_path or not os.path.exists(registry_path):
        raise CalibrationUnavailable([f"capsule registry not found: {registry_path}"])
    try:
        with open(registry_path, encoding="utf-8") as fh:
            registry = json.load(fh)
    except Exception as exc:
        raise CalibrationUnavailable([f"capsule registry unreadable: {exc}"])
    entry = registry.get(region) if isinstance(registry, dict) else None
    if not isinstance(entry, dict):
        raise CalibrationUnavailable([f"no registry entry for region {region}"])
    for k in ("capsule_path", "expected_sha256", "topology_version"):
        if k not in entry:
            raise CalibrationUnavailable([f"registry entry for {region} missing {k}"])
    return {
        "capsule_path": entry["capsule_path"],
        "expected_sha256": entry["expected_sha256"],
        "topology_version": entry["topology_version"],
    }


def _parse_date(s):
    """Strict YYYY-MM-DD -> date, else None."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def align_activity_series(series, *, max_gap_seconds, min_coverage):
    """Exact-UTC-grid alignment. Returns (A | None, names, qc). Fails closed (A=None) on:
    naive/aware-non-UTC start; non-finite/non-positive/unequal dt; off-grid start (half-sample
    phase offset -- NEVER interpolate); a gap lying outside the typed sample carrier
    [start, start+n*dt] (finite-positive duration + aware-UTC start required); declared coverage
    inconsistent with the in-span gaps + sample count; empty/short overlap; coverage <
    min_coverage; or a gap > max_gap_seconds. A short IN-carrier gap keeps ok but records qc."""
    names = list(series.keys())
    reasons = []
    if len(names) < 2:
        return None, names, [f"need >= 2 series (got {len(names)})"]

    dts = []
    starts = {}
    sizes = {}
    for name in names:
        s = series[name]
        st = getattr(s, "start_utc", None)
        if (not isinstance(st, datetime) or st.tzinfo is None
                or st.utcoffset() != timedelta(0)):
            reasons.append(f"{name}: start_utc must be AWARE UTC (got {st!r})")
            continue
        dt = float(getattr(s, "dt_seconds", float("nan")))
        if not np.isfinite(dt) or dt <= 0.0:
            reasons.append(f"{name}: dt_seconds must be finite positive (got {dt})")
            continue
        v = np.asarray(getattr(s, "values", s), dtype=float).ravel()
        n = v.size
        starts[name] = st
        sizes[name] = n
        dts.append(dt)
        span = n * dt
        carrier_end = st + timedelta(seconds=span)
        cov = float(getattr(s, "coverage", 1.0))
        in_span_gap = 0.0
        for g in list(getattr(s, "gaps", [])):
            try:
                giso, gsec = g[0], float(g[1])
            except Exception:
                reasons.append(f"{name}: malformed gap {g!r}")
                continue
            if not np.isfinite(gsec) or gsec <= 0:
                reasons.append(f"{name}: gap duration must be finite positive (got {gsec})")
                continue
            try:
                gstart = SD._parse_iso_utc(giso)
            except Exception:
                reasons.append(f"{name}: unparseable gap start {giso!r}")
                continue
            # codex 0409/0426 #2: aware-UTC gap start + the gap must lie WHOLLY inside the typed
            # sample carrier [start, start + n*dt]. Same-day metadata outside the carrier is
            # REJECTED (never recorded as mere QC) -- the honest carrier is n*dt seconds, not a day.
            if gstart.tzinfo is None or gstart.utcoffset() != timedelta(0):
                reasons.append(f"{name}: gap start {giso} must be aware UTC")
                continue
            gend = gstart + timedelta(seconds=gsec)
            if not (st <= gstart < gend <= carrier_end):
                reasons.append(f"{name}: gap {giso} (+{gsec}s) outside the sample carrier "
                               f"[start, start+n*dt]")
                continue
            if gsec > float(max_gap_seconds):
                reasons.append(f"{name}: gap {gsec}s exceeds max_gap {max_gap_seconds}s")
            in_span_gap += gsec
        expected_cov = max(0.0, 1.0 - in_span_gap / span) if span > 0 else 0.0
        if abs(cov - expected_cov) > _COVERAGE_TOL:
            reasons.append(f"{name}: declared coverage {cov} inconsistent with gaps "
                           f"(expected {expected_cov:.4f})")
        if cov < float(min_coverage):
            reasons.append(f"{name}: coverage {cov} < min_coverage {min_coverage}")

    if reasons:
        return None, names, reasons

    if len({round(d, 9) for d in dts}) != 1:
        return None, names, [f"dt mismatch across series: {dts}"]
    dt = dts[0]
    base = min(starts.values())
    for name in names:
        off = (starts[name] - base).total_seconds() / dt
        if abs(off - round(off)) > _GRID_TOL:
            return None, names, [f"{name}: start not on the common {dt}s grid (offset {off})"]

    ov_start = max(starts.values())
    ov_end = min(starts[name] + timedelta(seconds=sizes[name] * dt) for name in names)
    overlap_seconds = (ov_end - ov_start).total_seconds()
    if overlap_seconds <= 0:
        return None, names, ["empty overlap after UTC alignment"]
    n_ov = int(round(overlap_seconds / dt))
    if n_ov < 1:
        return None, names, ["overlap shorter than one sample"]

    rows = []
    qc = []
    for name in names:
        s = series[name]
        offset = int(round((ov_start - starts[name]).total_seconds() / dt))
        v = np.asarray(getattr(s, "values", s), dtype=float).ravel()
        row = v[offset:offset + n_ov]
        if row.size != n_ov:
            return None, names, [f"{name}: overlap slice short ({row.size} != {n_ov})"]
        rows.append(row)
        for g in getattr(s, "gaps", []):
            qc.append(f"{name}: gap {g[0]} ({g[1]}s)")

    return np.vstack(rows), names, qc


def load_calibration_capsule(region, scored_day, *, band_tag, processing_version,
                             topology_version, capsule_dir, expected_sha256,
                             embargo_days=30):
    """Load + admit an externally-pinned calibration capsule. The capsule file's EXACT
    bytes must sha256 to expected_sha256 (the registered pin) -- a self-consistent mutated
    capsule fails. EXACT schema (no missing/extra keys); band/processing/topology bindings;
    date well-formedness; window start<=end; threshold finite in (0,1); scored_day <=
    valid_through; and a LEAKAGE/LAG embargo (>= embargo_days, default 30, per the corrected
    R3 lag). Any violation raises CalibrationUnavailable."""
    path = os.path.join(capsule_dir, f"{region}.json")
    if not os.path.exists(path):
        raise CalibrationUnavailable([f"no capsule for region {region} in {capsule_dir}"])
    with open(path, "rb") as fh:
        raw = fh.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha256:
        raise CalibrationUnavailable(
            [f"capsule sha256 {actual_sha} != registered pin {expected_sha256}"])
    try:
        cap = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise CalibrationUnavailable([f"capsule not valid JSON: {exc}"])

    expected_keys = {"schema", "region", "band_tag", "processing_version", "topology_version",
                     "threshold", "calibration_window", "source_commit",
                     "input_manifest_sha256", "replay_output_sha256", "issued_utc",
                     "valid_through"}
    if not isinstance(cap, dict) or set(cap.keys()) != expected_keys:
        raise CalibrationUnavailable([
            f"capsule schema keys mismatch: "
            f"{sorted(cap.keys()) if isinstance(cap, dict) else type(cap).__name__}"])
    if cap["schema"] != _CALIBRATION_SCHEMA:
        raise CalibrationUnavailable([f"schema {cap['schema']} != {_CALIBRATION_SCHEMA}"])
    if cap["region"] != region:
        raise CalibrationUnavailable([f"capsule region {cap['region']} != {region}"])
    if cap["band_tag"] != band_tag:
        raise CalibrationUnavailable([f"capsule band_tag {cap['band_tag']} != {band_tag}"])
    if cap["processing_version"] != processing_version:
        raise CalibrationUnavailable(
            [f"capsule processing_version {cap['processing_version']} != {processing_version}"])
    if cap["topology_version"] != topology_version:
        raise CalibrationUnavailable(
            [f"capsule topology_version {cap['topology_version']} != {topology_version}"])
    for k, rx in (("source_commit", _HEX40), ("input_manifest_sha256", _HEX64),
                  ("replay_output_sha256", _HEX64)):
        if not (isinstance(cap[k], str) and rx.match(cap[k])):
            raise CalibrationUnavailable([f"{k} not {rx.pattern}"])

    thr = cap["threshold"]
    if (isinstance(thr, bool) or not isinstance(thr, (int, float))
            or not np.isfinite(thr) or not (0.0 < float(thr) < 1.0)):
        raise CalibrationUnavailable([f"threshold {thr} non-finite or outside (0,1)"])

    win = cap["calibration_window"]
    if not (isinstance(win, dict) and set(win.keys()) == {"start", "end"}):
        raise CalibrationUnavailable(["calibration_window must have exactly {start, end}"])
    w_start = _parse_date(win["start"])
    w_end = _parse_date(win["end"])
    valid_through = _parse_date(cap["valid_through"])
    scored = _parse_date(scored_day)
    if None in (w_start, w_end, valid_through, scored):
        raise CalibrationUnavailable(["malformed/unparseable date(s)"])
    try:
        SD._parse_iso_utc(cap["issued_utc"])
    except Exception:
        raise CalibrationUnavailable([f"issued_utc unparseable: {cap['issued_utc']!r}"])
    if w_start > w_end:
        raise CalibrationUnavailable(
            [f"calibration window start {win['start']} > end {win['end']}"])
    if scored > valid_through:
        raise CalibrationUnavailable(
            [f"scored day {scored_day} past valid_through {cap['valid_through']} (STALE)"])
    lag_days = (scored - w_end).days
    if lag_days < int(embargo_days):
        raise CalibrationUnavailable(
            [f"calibration ends {lag_days}d before scored day (< embargo {embargo_days}d)"])
    return cap


def validate_topology(region):
    """(ok, reasons). Rejects a region whose correlated segments SHARE a NET.STA station
    (the same sensor in two segments trivially inflates their correlation). Disjoint-station
    topologies pass. Unknown regions fail closed."""
    try:
        segments = get_segments_for_region(region)
    except Exception as exc:
        return False, [f"unknown/invalid region {region}: {exc}"]
    counts = Counter()
    for seg in segments:
        for nsta in {f"{s.network}.{s.code}" for s in seg.stations}:
            counts[nsta] += 1
    shared = sorted([nsta for nsta, c in counts.items() if c >= 2])
    if shared:
        return False, [f"shared NET.STA across correlated segments: {shared}"]
    return True, []


def _aggregate_segment_envelope(env_by_station):
    """Median-stack station EnvelopeSeries into one per-segment activity EnvelopeSeries.
    codex 0409 #1: the stations are EXACT-UTC-GRID-ALIGNED before the median (never
    index-truncated by element zero); the aggregate carries the common-intersection
    start/dt/length. Rate/phase/timezone mismatch across stations -> unavailable (None).
    Requires >= 2 stations."""
    series = {sid: s for sid, s in env_by_station.items() if s is not None}
    if len(series) < 2:
        return None
    A, names, _qc = align_activity_series(series, max_gap_seconds=float("inf"),
                                          min_coverage=0.0)
    if A is None or A.ndim != 2 or A.shape[1] < 1:
        return None
    dt = float(series[names[0]].dt_seconds)
    ov_start = max(series[n].start_utc for n in names)
    ref = series[names[0]]
    return SD.EnvelopeSeries(
        values=np.median(A, axis=0),
        start_utc=ov_start,
        dt_seconds=dt,
        coverage=min(float(series[n].coverage) for n in names),
        gaps=[],
        source_ids=[sid for n in names for sid in series[n].source_ids],
        band_tag=ref.band_tag,
        processing_version=ref.processing_version,
        pre_envelope_rate_hz=ref.pre_envelope_rate_hz,
    )


@dataclass
class CorrelationResult:
    """Container for fault correlation analysis results."""
    region: str
    date: datetime
    segment_names: List[str]
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray  # λ₁, λ₂, ... (descending)
    eigenvalue_ratios: np.ndarray  # λᵢ/λ₁ for i>1
    participation_ratio: float  # (Σλᵢ)² / Σλᵢ²
    dominant_eigenvector: np.ndarray
    is_decorrelated: bool
    decorrelation_threshold: float
    notes: str = ""
    # D2 rev-2: fail-closed data-quality gate outcome (observability + align + topology)
    data_quality_ok: bool = True
    qc_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'region': self.region,
            'date': self.date.isoformat(),
            'segment_names': self.segment_names,
            'eigenvalues': self.eigenvalues.tolist(),
            'eigenvalue_ratios': self.eigenvalue_ratios.tolist(),
            'participation_ratio': float(self.participation_ratio),
            'lambda2_lambda1': float(self.eigenvalue_ratios[0]) if len(self.eigenvalue_ratios) > 0 else 0.0,
            'is_decorrelated': bool(self.is_decorrelated),
            'notes': self.notes,
        }


class FaultCorrelationMonitor:
    """
    Monitors fault segment correlation dynamics for earthquake precursors.

    The key diagnostic is the eigenvalue spectrum of the correlation matrix:
    - High λ2/λ1: Segments coupled, stress distributed (normal)
    - Low λ2/λ1: Segments decoupled, stress concentrated (pre-earthquake)

    Attributes:
        data_fetcher: SeismicDataFetcher for waveform acquisition
        window_hours: Window size for correlation computation (default 24h)
        step_hours: Step size for time evolution (default 6h)
        decorrelation_threshold: λ2/λ1 threshold for alert (default 0.3)
    """

    def __init__(
        self,
        data_fetcher: Optional[SeismicDataFetcher] = None,
        window_hours: int = 24,
        step_hours: int = 6,
        decorrelation_threshold: float = 0.3,
    ):
        """
        Initialize the FaultCorrelationMonitor.

        Args:
            data_fetcher: SeismicDataFetcher instance (created if None)
            window_hours: Window size for correlation in hours
            step_hours: Step size for evolution in hours
            decorrelation_threshold: λ2/λ1 ratio below which to flag decorrelation
        """
        self.data_fetcher = data_fetcher or SeismicDataFetcher()
        self.window_hours = window_hours
        self.step_hours = step_hours
        self.decorrelation_threshold = decorrelation_threshold

        logger.info(f"FaultCorrelationMonitor initialized: "
                   f"window={window_hours}h, step={step_hours}h, "
                   f"threshold={decorrelation_threshold}")

    def compute_segment_activity(
        self,
        segment: FaultSegment,
        start: datetime,
        end: datetime
    ) -> Optional[np.ndarray]:
        """
        Compute aggregate seismic activity for a fault segment.

        Args:
            segment: FaultSegment object
            start: Start datetime
            end: End datetime

        Returns:
            Activity time series or None if insufficient data
        """
        envelopes = self.data_fetcher.get_segment_envelopes(segment, start, end)

        if len(envelopes) < 2:
            logger.warning(f"Insufficient stations for {segment.name}: {len(envelopes)}")
            return None

        # rev-2: get_segment_envelopes returns EnvelopeSeries; extract the value arrays for the
        # legacy median activity index (retained for the standalone validation entry points --
        # the rev-2 correlation path uses observability_gate + align_activity_series instead).
        value_arrays = {nslc: np.asarray(getattr(s, "values", s), dtype=float)
                        for nslc, s in envelopes.items()}
        activity = compute_segment_activity_index(value_arrays)

        if len(activity) == 0:
            return None

        return activity

    def compute_correlation_matrix(
        self,
        region: str,
        start: datetime,
        end: datetime
    ) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
        """
        Compute the fault-segment correlation matrix (rev-2, fail-closed).

        Returns a 3-tuple (C | None, segment_names, qc_reasons). Fails closed to
        (None, names, reasons) when the station topology is invalid, the observability gate
        rejects the segments, the UTC-grid alignment fails, or the resulting matrix is not a
        finite unit-diagonal PSD correlation matrix. There is NO std->1.0 substitution and
        NO nan_to_num masking -- a degenerate or unalignable input yields None, never a
        fabricated correlation.
        """
        ok_topo, topo_reasons = validate_topology(region)
        if not ok_topo:
            return None, [], list(topo_reasons)

        try:
            segments = get_segments_for_region(region)
        except Exception as exc:
            return None, [], [f"unknown region {region}: {exc}"]

        series_by_segment = {}
        for segment in segments:
            env = self.data_fetcher.get_segment_envelopes(segment, start, end)
            agg = _aggregate_segment_envelope(env)
            if agg is not None:
                series_by_segment[segment.name] = agg

        ok_gate, qc = observability_gate(series_by_segment)
        if not ok_gate:
            return None, list(series_by_segment.keys()), list(qc)

        A, segment_names, align_qc = align_activity_series(
            series_by_segment,
            max_gap_seconds=self.window_hours * 3600,
            min_coverage=0.5)
        if A is None:
            return None, list(segment_names), list(align_qc)

        C = np.atleast_2d(np.corrcoef(A))
        if not np.all(np.isfinite(C)):
            return None, list(segment_names), list(align_qc) + ["correlation matrix non-finite"]
        if not np.allclose(np.diag(C), 1.0, atol=1e-6):
            return None, list(segment_names), list(align_qc) + ["diagonal not unit"]
        if float(np.min(np.linalg.eigvalsh(C))) < -1e-6:
            return None, list(segment_names), list(align_qc) + ["matrix not PSD"]

        return C, list(segment_names), list(align_qc)

    def analyze_eigenvalue_spectrum(
        self,
        C: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Analyze eigenvalue spectrum of correlation matrix.

        Args:
            C: Correlation matrix

        Returns:
            Tuple of (eigenvalues, ratios, participation_ratio, dominant_eigenvector)
        """
        # Compute eigenvalues (symmetric matrix -> use eigh for stability)
        eigenvalues, eigenvectors = eigh(C)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Ensure positive (numerical noise can make small negative)
        eigenvalues = np.maximum(eigenvalues, 0)

        # Eigenvalue ratios (λᵢ/λ₁)
        if eigenvalues[0] > 1e-10:
            ratios = eigenvalues[1:] / eigenvalues[0]
        else:
            ratios = np.zeros(len(eigenvalues) - 1)

        # Participation ratio: (Σλᵢ)² / Σλᵢ²
        # Measures how distributed the stress is
        # PR = n means uniform distribution, PR = 1 means concentrated
        sum_lambda = eigenvalues.sum()
        sum_lambda_sq = (eigenvalues ** 2).sum()
        if sum_lambda_sq > 1e-10:
            participation_ratio = (sum_lambda ** 2) / sum_lambda_sq
        else:
            participation_ratio = 1.0

        # Dominant eigenvector
        dominant_eigenvector = eigenvectors[:, 0]

        return eigenvalues, ratios, participation_ratio, dominant_eigenvector

    def detect_decorrelation(
        self,
        baseline_ratios: np.ndarray,
        current_ratios: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Detect decorrelation by comparing current to baseline.

        Args:
            baseline_ratios: Baseline eigenvalue ratios
            current_ratios: Current eigenvalue ratios
            threshold: Decorrelation threshold (uses instance default if None)

        Returns:
            Tuple of (is_decorrelated, drop_factor)
        """
        threshold = threshold or self.decorrelation_threshold

        if len(baseline_ratios) == 0 or len(current_ratios) == 0:
            return False, 1.0

        # Compare λ2/λ1 ratios
        baseline_ratio = baseline_ratios[0]  # λ2/λ1
        current_ratio = current_ratios[0]

        if baseline_ratio < 1e-10:
            return False, 1.0

        drop_factor = current_ratio / baseline_ratio

        # Decorrelated if ratio dropped significantly
        is_decorrelated = current_ratio < threshold

        return is_decorrelated, drop_factor

    def _resolve_threshold(self, calibration) -> float:
        """The verdict threshold comes FROM the admitted capsule when present; otherwise the
        instance default (kept for the legacy validation entry points)."""
        if isinstance(calibration, dict) and "threshold" in calibration:
            try:
                return float(calibration["threshold"])
            except Exception:
                pass
        return self.decorrelation_threshold

    def analyze_region(
        self,
        region: str,
        target_date: datetime,
        *,
        calibration: Optional[dict] = None,
        baseline_days: int = 30,
        gap_days: int = 7,
        **kwargs
    ) -> CorrelationResult:
        """
        Perform full correlation analysis for a region (rev-2).

        The decorrelation verdict threshold comes FROM the admitted calibration capsule
        (kwarg `calibration`), not a hardcoded constant. When the fail-closed data-quality
        gate rejects the region (compute_correlation_matrix -> None), the result carries
        data_quality_ok=False, is_decorrelated=False, participation_ratio=0.0, and the qc
        reasons; no risk values are minted.
        """
        logger.info(f"Analyzing {region} for {target_date.date()}")

        current_start = target_date - timedelta(hours=self.window_hours)
        current_end = target_date

        C_current, segment_names, qc = self.compute_correlation_matrix(
            region, current_start, current_end
        )
        threshold = self._resolve_threshold(calibration)

        if C_current is None:
            return CorrelationResult(
                region=region,
                date=target_date,
                segment_names=list(segment_names or []),
                correlation_matrix=np.array([]),
                eigenvalues=np.array([]),
                eigenvalue_ratios=np.array([]),
                participation_ratio=0.0,
                dominant_eigenvector=np.array([]),
                is_decorrelated=False,
                decorrelation_threshold=threshold,
                notes="data quality gate failed",
                data_quality_ok=False,
                qc_reasons=list(qc) if qc else ["insufficient/invalid correlation data"],
            )

        eigenvalues, ratios, participation_ratio, dominant_eigenvector = \
            self.analyze_eigenvalue_spectrum(C_current)

        l2_l1 = float(ratios[0]) if len(ratios) > 0 else 0.0
        is_decorrelated = bool(l2_l1 < threshold)

        result = CorrelationResult(
            region=region,
            date=target_date,
            segment_names=list(segment_names),
            correlation_matrix=C_current,
            eigenvalues=eigenvalues,
            eigenvalue_ratios=ratios,
            participation_ratio=participation_ratio,
            dominant_eigenvector=dominant_eigenvector,
            is_decorrelated=is_decorrelated,
            decorrelation_threshold=threshold,
            notes="",
            data_quality_ok=True,
            qc_reasons=list(qc),
        )

        logger.info(f"  Segments: {len(segment_names)}")
        logger.info(f"  L2/L1: {l2_l1:.4f}" if len(ratios) > 0 else "  L2/L1: N/A")
        logger.info(f"  Participation ratio: {participation_ratio:.2f}")
        logger.info(f"  Decorrelated: {is_decorrelated} (threshold {threshold})")

        return result

    def compute_evolution(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CorrelationResult]:
        """
        Compute time evolution of correlation metrics.

        Args:
            region: Region name
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of CorrelationResult for each time step
        """
        results = []
        current = start_date + timedelta(hours=self.window_hours)

        while current <= end_date:
            result = self.analyze_region(region, current)
            results.append(result)
            current += timedelta(hours=self.step_hours)

        return results


def run_ridgecrest_validation():
    """
    Validate fault correlation on 2019 Ridgecrest sequence.

    Expected: λ2/λ1 should DROP before the M6.4 foreshock and M7.1 mainshock.
    """
    print("=" * 60)
    print("Ridgecrest Fault Correlation Validation")
    print("=" * 60)

    monitor = FaultCorrelationMonitor(
        window_hours=24,
        step_hours=12,
        decorrelation_threshold=0.3
    )

    # Key dates:
    # M6.4 foreshock: 2019-07-04 17:33:49 UTC
    # M7.1 mainshock: 2019-07-06 03:19:53 UTC

    # Analyze period before foreshock
    print("\n--- Pre-Foreshock Period ---")
    pre_foreshock = datetime(2019, 7, 4, 12, 0)  # 5 hours before M6.4
    result = monitor.analyze_region('ridgecrest', pre_foreshock)

    if result.segment_names:
        print(f"Segments: {result.segment_names}")
        print(f"L2/L1 ratio: {result.eigenvalue_ratios[0]:.4f}" if len(result.eigenvalue_ratios) > 0 else "N/A")
        print(f"Participation ratio: {result.participation_ratio:.2f}")
        print(f"Decorrelated: {result.is_decorrelated}")
    else:
        print("Could not retrieve sufficient data for Ridgecrest")
        print("NOTE: CI network data requires SCEDC access which may be rate-limited")

    return result


if __name__ == "__main__":
    result = run_ridgecrest_validation()
