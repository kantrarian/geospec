#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 RE-BAND red-KATs (cayley, 2026-08-07) under codex D2 §5 review `66b7f79` (WORKS-WITH-FIX,
five freezes STAY). Step 1 of the convergent release sequence: these bars freeze the five
locking-KAT groups; grassmann implements UNEDITED; codex one bounded verification; the sealed
five-region calibration capsule and any freeze lift are SEPARATE, owner-visible steps.

Hermetic: numpy/scipy only; obspy is stubbed (codex probe pattern) — the envelope contract is a
PURE-ARRAY CORE with a thin obspy shell, so the physics is testable without waveforms.

CONTRACT (grassmann implements to THIS, unedited — the decouple)
=================================================================
seismic_data.py rev-2:
* class DataUnavailable(Exception) with `.reasons: list[str]` (machine-readable).
* @dataclass EnvelopeSeries: values (np.ndarray at out_rate), start_utc (aware datetime),
  dt_seconds (float), coverage (float 0..1), gaps (list), source_ids (list[str]), band_tag (str),
  processing_version (str), pre_envelope_rate_hz (float).
* PROCESSING_VERSION: str constant (bumped for the rev-2 path; part of cache identity).
* compute_band_envelope_from_array(data, rate_hz, *, freqmin=1.0, freqmax=10.0, out_rate_hz=1.0,
  min_rate_hz=25.0, start_utc, source_id) -> EnvelopeSeries
    - raises DataUnavailable (reason naming the rate/Nyquist) if rate_hz < min_rate_hz;
    - bandpass at NATIVE rate -> analytic (Hilbert) envelope while the carrier still exists ->
      low-pass + resample the ENVELOPE to out_rate_hz; pre_envelope_rate_hz records the native rate.
* get_segment_envelopes / process_waveforms route ALL fault-corr envelope production through the
  core above (codex finding 1 repair; source-level check D2R-1e; codex verifies integration).
* _cache_path(self, region, start: datetime, end: datetime, data_type: str, identity: str = "")
  -> Path — the key binds exact start AND end (sub-day) and the identity string (band tag +
  PROCESSING_VERSION + source/rate/digest material). Same-day different windows never collide.

fault_correlation.py rev-2:
* class CalibrationUnavailable(Exception) with `.reasons: list[str]`.
* observability_gate(series_by_segment: dict[str, EnvelopeSeries-like]) -> (ok, reasons)
    - fail-closed BEFORE correlation: finite values; >= 2 segments with data; minimum effective
      samples; per-segment robust variance/dynamic range above degeneracy; PROVENANCE: every input
      carries pre_envelope_rate_hz >= 25 and the rev-2 band_tag/processing_version (an anti-alias-
      erased 1 Hz-provenance series is inadmissible by construction). Machine-readable reasons.
* align_activity_series(series: dict[str, EnvelopeSeries], *, max_gap_seconds, min_coverage)
  -> (A | None, names, qc: list[str])
    - EXACT UTC-grid intersection alignment (never element-zero truncation); empty/short overlap,
      coverage < min_coverage, or any gap > max_gap_seconds -> (None, names, reasons);
      a short permitted gap keeps ok but RECORDS a qc flag.
* compute_correlation_matrix -> (C | None, segment_names, qc_reasons) — 3-tuple. NO std->1.0
  substitution, NO nan_to_num repair: a degenerate/invalid matrix (non-finite, non-unit diagonal,
  non-PSD beyond tolerance) is (None, names, reasons), never "repaired".
* CorrelationResult gains data_quality_ok: bool and qc_reasons: list. A gate/alignment/matrix
  failure yields data_quality_ok=False, is_decorrelated=False, NO risk-bearing values.
* load_calibration_capsule(region, scored_day: str, *, band_tag, processing_version,
  topology_version, capsule_dir, embargo_days=14) -> dict
    - loads <capsule_dir>/<region>.json; raises CalibrationUnavailable on: missing file; region /
      band_tag / processing_version / topology_version mismatch; LEAKAGE (manifest
      calibration_window.end within embargo_days of scored_day or later); STALE
      (valid_through < scored_day). Returns the capsule dict (incl. `threshold`).
* analyze_region(region, target_date, *, calibration: dict, ...) — the decorrelation threshold
  comes from the CAPSULE, never a fixed constant; same matrix + different capsule threshold flips
  the verdict (D2R-4e). Gate-failed inputs -> data_quality_ok=False (never is_decorrelated=True).
* validate_topology(region) -> (ok, reasons) — rejects a segment set where any NET.STA appears in
  two or more of the region's correlated segments (or the implementation may instead cross-fit:
  see D2R-5d, codex's either/or). Current japan_tohoku and ridgecrest topologies REJECT (that is
  the frozen reality: IU.MAJO+PS.TSK shared; CI.WBS+CI.LRL shared).

ensemble.py rev-2:
* compute_fault_correlation_risk returns available=False (zero effective weight) whenever
  data_quality_ok is False, the capsule is unavailable, or the topology contract fails.
* tokyo_kanto: the silent cross-geography remap tokyo_kanto -> japan_tohoku is REMOVED; the
  component is UNAVAILABLE for tokyo_kanto (reason naming the carrier/topology) until true Kanto
  segments/stations exist. No component may be scored under a region label whose fault carrier
  fails the geographic contract.
* component_frozen(region, component) -> bool — freeze keyed on the RESOLVED CARRIER, not the raw
  string: True for BOTH 'tokyo_kanto' and 'japan_tohoku' (same carrier), for BOTH
  'socal_saf_coachella' and 'socal_coachella'; False for 'kumamoto'. (Recon finding: the current
  `(self.region, cname) in FROZEN_COMPONENTS` check is bypassable by constructing with the
  FC-canonical key.)

RED AS AUTHORED (rev-2 seams absent from the landed modules).
"""
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import hilbert as sp_hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def raises_unavailable(fn, exc_names=("DataUnavailable", "CalibrationUnavailable")):
    try:
        fn()
        return False, None
    except Exception as exc:
        return type(exc).__name__ in exc_names, exc


def _install_obspy_stub():
    """codex probe pattern: satisfy seismic_data's obspy imports without obspy."""
    if "obspy" in sys.modules:
        return
    obspy = types.ModuleType("obspy")
    obspy.Stream = type("Stream", (), {})
    obspy.Trace = type("Trace", (), {})
    obspy.UTCDateTime = object
    fdsn = types.ModuleType("obspy.clients.fdsn")
    fdsn.Client = type("Client", (), {"__init__": lambda self, *a, **k: None})
    clients = types.ModuleType("obspy.clients")
    clients.fdsn = fdsn
    obspy.clients = clients
    sig = types.ModuleType("obspy.signal")
    sigfilter = types.ModuleType("obspy.signal.filter")
    sigfilter.envelope = lambda a: np.abs(sp_hilbert(a))

    def _bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
        b, a = sp_signal.butter(corners, [freqmin / (df / 2.0), freqmax / (df / 2.0)], btype="band")
        return sp_signal.filtfilt(b, a, data) if zerophase else sp_signal.lfilter(b, a, data)

    sigfilter.bandpass = _bandpass
    sig.filter = sigfilter
    obspy.signal = sig
    for name, mod in (("obspy", obspy), ("obspy.clients", clients), ("obspy.clients.fdsn", fdsn),
                      ("obspy.signal", sig), ("obspy.signal.filter", sigfilter)):
        sys.modules[name] = mod


# ---------------------------------------------------------------- fixtures --
FS = 40.0
T_SEC = 600
UTC0 = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)


def am_fixture():
    """5 Hz carrier, known slow AM (0.02 Hz) — the modulation envelope is the ground truth."""
    t = np.arange(int(FS * T_SEC)) / FS
    m = 1.0 + 0.6 * np.sin(2 * np.pi * 0.02 * t)
    x = m * np.sin(2 * np.pi * 5.0 * t)
    m_1hz = m[:: int(FS)]                     # truth sampled at 1 Hz
    return x, m_1hz


def old_path_envelope(x):
    """The CURRENT (broken) order: bandpass 1-10 @ native -> decimate to 1 Hz -> envelope."""
    b, a = sp_signal.butter(4, [1.0 / (FS / 2), 10.0 / (FS / 2)], btype="band")
    y = sp_signal.filtfilt(b, a, x)
    y = sp_signal.decimate(y, 10)
    y = sp_signal.decimate(y, 4)
    return np.abs(sp_hilbert(y))


def _trim_corr(a, b, trim=30):
    n = min(a.size, b.size)
    a, b = a[:n][trim:n - trim], b[:n][trim:n - trim]
    return float(np.corrcoef(a, b)[0, 1])


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


def _mk_series(SD, values, start_utc, *, rate_pre=40.0, coverage=1.0, gaps=(),
               source_ids=("XX.AAA",), band_tag=None, version=None):
    return SD.EnvelopeSeries(values=np.asarray(values, dtype=float), start_utc=start_utc,
                             dt_seconds=1.0, coverage=coverage, gaps=list(gaps),
                             source_ids=list(source_ids),
                             band_tag=(band_tag or getattr(SD, "_FAULT_CORR_BAND_TAG", "1-10Hz")),
                             processing_version=(version or SD.PROCESSING_VERSION),
                             pre_envelope_rate_hz=rate_pre)


def main():
    _install_obspy_stub()
    import seismic_data as SD
    import fault_correlation as FC
    import ensemble as EN

    need_sd = ("DataUnavailable", "EnvelopeSeries", "PROCESSING_VERSION",
               "compute_band_envelope_from_array")
    if not all(hasattr(SD, n) for n in need_sd):
        check("D2R-0a seismic_data rev-2 seams present (EnvelopeSeries core + DataUnavailable)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return
    need_fc = ("CalibrationUnavailable", "observability_gate", "align_activity_series",
               "load_calibration_capsule", "validate_topology")
    if not all(hasattr(FC, n) for n in need_fc):
        check("D2R-0b fault_correlation rev-2 seams present (gate/align/capsule/topology)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return
    if not hasattr(EN, "component_frozen"):
        check("D2R-0c ensemble rev-2 seams present (carrier-keyed component_frozen)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return

    # =====================================================================
    # GROUP 1 — envelope order (codex finding 1)
    # =====================================================================
    x, m_true = am_fixture()
    es = SD.compute_band_envelope_from_array(x, FS, start_utc=UTC0, source_id="XX.AAA")
    r_new = _trim_corr(es.values, m_true)
    rms_ratio = _rms(es.values) / _rms(m_true)
    check("D2R-1a AM modulation SURVIVES the rev-2 path (r >= 0.95, amplitude within 2x)",
          r_new >= 0.95 and 0.5 <= rms_ratio <= 2.0, f"r={r_new:.3f} rms_ratio={rms_ratio:.3f}")
    ok_low, exc = raises_unavailable(
        lambda: SD.compute_band_envelope_from_array(x[: int(20 * T_SEC)], 20.0,
                                                    start_utc=UTC0, source_id="XX.AAA"))
    check("D2R-1b a 20 Hz trace fails DataUnavailable (never coerced through the band)",
          ok_low, f"got {type(exc).__name__ if exc else 'no raise'}")
    check("D2R-1c output is a 1 Hz ENVELOPE while the pre-envelope waveform stayed at native rate",
          abs(1.0 / es.dt_seconds - 1.0) < 1e-9 and es.pre_envelope_rate_hz == FS
          and abs(es.values.size - T_SEC) <= 2,
          f"dt={es.dt_seconds} pre_rate={es.pre_envelope_rate_hz} n={es.values.size}")
    env_old = old_path_envelope(x)
    r_old = _trim_corr(env_old, m_true)
    rms_old = _rms(env_old) / _rms(m_true)
    check("D2R-1d REGRESSION LOCK: the old filter->decimate-to-1Hz->envelope order FAILS the "
          "same criterion (the bar discriminates)",
          not (r_old >= 0.95 and 0.5 <= rms_old <= 2.0) and rms_old < 0.05,
          f"r_old={r_old:.3f} rms_old={rms_old:.4f}")
    src_sd = open(os.path.join(HERE, "seismic_data.py"), encoding="utf-8").read()
    seg_env_src = src_sd[src_sd.find("def get_segment_envelopes"):]
    seg_env_src = seg_env_src[: seg_env_src.find("\n    def ", 10)]
    check("D2R-1e get_segment_envelopes routes through compute_band_envelope_from_array",
          "compute_band_envelope_from_array" in seg_env_src)

    # =====================================================================
    # GROUP 2 — fail-closed observability gate (codex finding 2)
    # =====================================================================
    good = _mk_series(SD, np.random.default_rng(1).normal(1.0, 0.3, 600), UTC0)

    def gate_fails(desc, series_by_seg, must_mention=None):
        ok, reasons = FC.observability_gate(series_by_seg)
        cond = (ok is False and isinstance(reasons, list) and len(reasons) > 0
                and (must_mention is None
                     or any(must_mention in str(r).lower() for r in reasons)))
        check(desc, cond, f"ok={ok} reasons={reasons}")

    gate_fails("D2R-2a constant segments fail the gate (no std->1.0 pass-through)",
               {"a": _mk_series(SD, np.full(600, 3.7), UTC0), "b": good})
    gate_fails("D2R-2b near-constant (1e-14 jitter) fails the gate",
               {"a": _mk_series(SD, 3.7 + 1e-14 * np.random.default_rng(2).normal(size=600), UTC0),
                "b": good})
    gate_fails("D2R-2c all-NaN fails the gate",
               {"a": _mk_series(SD, np.full(600, np.nan), UTC0), "b": good})
    gate_fails("D2R-2d a single valid segment fails the gate (need >= 2)",
               {"a": good})
    gate_fails("D2R-2e anti-alias-erased PROVENANCE fails the gate (pre_envelope_rate 1 Hz)",
               {"a": _mk_series(SD, np.abs(np.random.default_rng(3).normal(0, 4e-3, 600)),
                                UTC0, rate_pre=1.0), "b": good})
    ok_good, reasons_good = FC.observability_gate({"a": good,
                                                   "b": _mk_series(SD, np.random.default_rng(4)
                                                                   .normal(1.0, 0.3, 600), UTC0)})
    check("D2R-2f two healthy segments PASS the gate", ok_good is True, f"reasons={reasons_good}")

    # codex's exact reproduction, end-to-end: zeros must never become max risk
    monitor = FC.FaultCorrelationMonitor(data_fetcher=object())
    monitor.compute_correlation_matrix = lambda *a, **k: (None, ["flat_a", "flat_b"],
                                                          ["degenerate: constant rows"])
    capsule = {"region": "synthetic", "threshold": 0.3, "band_tag": "1-10Hz",
               "processing_version": SD.PROCESSING_VERSION, "topology_version": "t1"}
    res = monitor.analyze_region("synthetic", datetime(2026, 8, 7), calibration=capsule)
    check("D2R-2g gate-failed region: data_quality_ok False, NEVER is_decorrelated, no risk values",
          getattr(res, "data_quality_ok", None) is False and res.is_decorrelated is False
          and res.participation_ratio == 0.0 and len(getattr(res, "qc_reasons", [])) > 0,
          f"dq={getattr(res, 'data_quality_ok', 'MISSING')} decorr={res.is_decorrelated}")
    ens = EN.GeoSpecEnsemble.__new__(EN.GeoSpecEnsemble)
    ens.region = "kumamoto"
    ens.fault_corr_monitor = monitor
    try:
        mr = ens.compute_fault_correlation_risk(datetime(2026, 8, 7))[0]
        check("D2R-2h ensemble: gate-failed fault_correlation is available=False (zero weight)",
              mr.available is False, f"available={mr.available} note={getattr(mr, 'notes', '')}")
    except Exception as exc:
        check("D2R-2h ensemble: gate-failed fault_correlation is available=False (zero weight)",
              False, f"RAISED {exc}")

    # =====================================================================
    # GROUP 3 — UTC identity, alignment, gaps, cache binding (codex finding 3)
    # =====================================================================
    f = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    f.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    d0, d6 = UTC0, UTC0 + timedelta(hours=6)
    p_a = f._cache_path("ridgecrest", d0, d0 + timedelta(hours=24), "seg_env")
    p_b = f._cache_path("ridgecrest", d6, d6 + timedelta(hours=24), "seg_env")
    check("D2R-3a six-hour-offset windows get DISTINCT cache identities", p_a != p_b,
          f"{p_a} == {p_b}")
    p_c = f._cache_path("ridgecrest", d0, d0 + timedelta(hours=24), "seg_env",
                        identity="band=1-10Hz;v=OLD")
    p_d = f._cache_path("ridgecrest", d0, d0 + timedelta(hours=24), "seg_env",
                        identity="band=1-10Hz;v=NEW")
    check("D2R-3b band/processing identity is bound into the cache key", p_c != p_d)

    rng = np.random.default_rng(7)
    vals = rng.normal(0.0, 1.0, 1000)
    sA = _mk_series(SD, vals, UTC0)
    sB = _mk_series(SD, vals, UTC0 + timedelta(seconds=360))     # SAME bytes, shifted UTC start
    A, names, qc = FC.align_activity_series({"a": sA, "b": sB},
                                            max_gap_seconds=600, min_coverage=0.9)
    if A is None:
        check("D2R-3c UTC alignment: shifted packets never become zero-lag identical", True)
    else:
        r_pair = float(np.corrcoef(A[0], A[1])[0, 1])
        check("D2R-3c UTC alignment: shifted packets never become zero-lag identical "
              "(element-zero truncation would give r=1.0)",
              A.shape[1] == 640 and abs(r_pair) < 0.5, f"n={A.shape[1]} r={r_pair:.3f}")
    A2, _, qc2 = FC.align_activity_series(
        {"a": _mk_series(SD, vals, UTC0, coverage=0.5), "b": sA},
        max_gap_seconds=600, min_coverage=0.9)
    check("D2R-3d insufficient coverage fails closed with a machine-readable reason",
          A2 is None and len(qc2) > 0, f"qc={qc2}")
    A3, _, qc3 = FC.align_activity_series(
        {"a": _mk_series(SD, vals, UTC0, gaps=[("2026-08-01T02:00:00Z", 3600)]), "b": sA},
        max_gap_seconds=600, min_coverage=0.9)
    check("D2R-3e a gap beyond max_gap fails closed", A3 is None and len(qc3) > 0, f"qc={qc3}")
    A4, _, qc4 = FC.align_activity_series(
        {"a": _mk_series(SD, vals, UTC0, gaps=[("2026-08-01T02:00:00Z", 60)]), "b": sA},
        max_gap_seconds=600, min_coverage=0.9)
    check("D2R-3f a short permitted gap stays available AND records its QC flag",
          A4 is not None and len(qc4) > 0, f"qc={qc4}")

    # =====================================================================
    # GROUP 4 — calibration capsule, no fixed threshold (codex finding 4)
    # =====================================================================
    cap_dir = tempfile.mkdtemp()
    manifest = {"region": "ridgecrest", "band_tag": "1-10Hz",
                "processing_version": SD.PROCESSING_VERSION, "topology_version": "t1",
                "threshold": 0.30,
                "calibration_window": {"start": "2026-05-01", "end": "2026-07-01"},
                "valid_through": "2026-12-31"}
    with open(os.path.join(cap_dir, "ridgecrest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    kw = dict(band_tag="1-10Hz", processing_version=SD.PROCESSING_VERSION,
              topology_version="t1", capsule_dir=cap_dir, embargo_days=14)
    cap = FC.load_calibration_capsule("ridgecrest", "2026-08-07", **kw)
    check("D2R-4a a matching capsule loads and carries the per-region threshold",
          isinstance(cap, dict) and cap.get("threshold") == 0.30)
    ok_m, _ = raises_unavailable(lambda: FC.load_calibration_capsule("kumamoto", "2026-08-07", **kw))
    check("D2R-4b missing capsule -> CalibrationUnavailable (missing/stale never defaults)", ok_m)
    for field_, bad in (("band_tag", "0.01-1Hz"), ("processing_version", "STALE-V0"),
                        ("topology_version", "t0")):
        bad_kw = dict(kw)
        bad_kw[field_] = bad
        ok_x, _ = raises_unavailable(
            lambda b=bad_kw: FC.load_calibration_capsule("ridgecrest", "2026-08-07", **b))
        check(f"D2R-4c capsule {field_} mismatch rejects", ok_x)
    ok_leak, _ = raises_unavailable(
        lambda: FC.load_calibration_capsule("ridgecrest", "2026-07-05", **kw))
    check("D2R-4d LEAKAGE: scored day within embargo of the calibration window rejects "
          "(the scored day never enters calibration)", ok_leak)
    ok_stale, _ = raises_unavailable(
        lambda: FC.load_calibration_capsule("ridgecrest", "2027-06-01", **kw))
    check("D2R-4d2 STALE: scored day past valid_through rejects", ok_stale)
    # threshold follows the capsule (same matrix, different capsule -> different verdict)
    C_fixed = np.array([[1.0, 0.5], [0.5, 1.0]])                # lambda2/lambda1 = 1/3
    mon2 = FC.FaultCorrelationMonitor(data_fetcher=object())
    mon2.compute_correlation_matrix = lambda *a, **k: (C_fixed, ["s1", "s2"], [])
    res_lo = mon2.analyze_region("ridgecrest", datetime(2026, 8, 7),
                                 calibration={**manifest, "threshold": 0.30})
    res_hi = mon2.analyze_region("ridgecrest", datetime(2026, 8, 7),
                                 calibration={**manifest, "threshold": 0.40})
    check("D2R-4e the verdict threshold comes FROM THE CAPSULE (0.30 -> no; 0.40 -> yes on the "
          "same matrix); no fixed 0.3 constant decides",
          res_lo.is_decorrelated is False and res_hi.is_decorrelated is True
          and getattr(res_lo, "data_quality_ok", None) is True,
          f"lo={res_lo.is_decorrelated} hi={res_hi.is_decorrelated}")

    # =====================================================================
    # GROUP 5 — topology contract + honest carriers (codex finding 5)
    # =====================================================================
    ok_t, reasons_t = FC.validate_topology("japan_tohoku")
    check("D2R-5a japan_tohoku topology REJECTS (IU.MAJO + PS.TSK shared across correlated "
          "segments — the frozen reality is pinned)", ok_t is False and len(reasons_t) > 0,
          f"ok={ok_t} reasons={reasons_t}")
    ok_r, reasons_r = FC.validate_topology("ridgecrest")
    check("D2R-5b ridgecrest topology REJECTS (CI.WBS + CI.LRL shared)",
          ok_r is False and len(reasons_r) > 0, f"ok={ok_r}")
    ok_k, reasons_k = FC.validate_topology("kumamoto")
    check("D2R-5c a station-disjoint topology PASSES", ok_k is True, f"reasons={reasons_k}")
    check("D2R-5d the silent tokyo_kanto -> japan_tohoku cross-geography remap is REMOVED",
          EN.REGION_KEY_MAP.get("tokyo_kanto") != "japan_tohoku",
          f"map={EN.REGION_KEY_MAP.get('tokyo_kanto')}")
    ens2 = EN.GeoSpecEnsemble.__new__(EN.GeoSpecEnsemble)
    ens2.region = "tokyo_kanto"
    ens2.fault_corr_monitor = FC.FaultCorrelationMonitor(data_fetcher=object())
    try:
        mr2 = ens2.compute_fault_correlation_risk(datetime(2026, 8, 7))[0]
        check("D2R-5e tokyo_kanto fault_correlation is UNAVAILABLE until true Kanto segments "
              "exist (never scored on the Tohoku carrier)", mr2.available is False,
              f"available={mr2.available}")
    except Exception as exc:
        check("D2R-5e tokyo_kanto fault_correlation is UNAVAILABLE until true Kanto segments "
              "exist (never scored on the Tohoku carrier)", False, f"RAISED {exc}")
    frozen_pairs = (("tokyo_kanto", True), ("japan_tohoku", True),
                    ("socal_saf_coachella", True), ("socal_coachella", True),
                    ("ridgecrest", True), ("kumamoto", False))
    ok_frozen = all(EN.component_frozen(r, "fault_correlation") is want for r, want in frozen_pairs)
    check("D2R-5f freeze is CARRIER-KEYED: the FC-canonical key cannot bypass it "
          "(recon: `(self.region, c) in FROZEN_COMPONENTS` was string-keyed)", ok_frozen,
          f"{[(r, EN.component_frozen(r, 'fault_correlation')) for r, _ in frozen_pairs]}")


main()
print()
if FAILS:
    print(f"D2 RE-BAND RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 RE-BAND RED-KATs PASS (envelope-order + fail-closed QC + UTC/cache + capsule + topology)")
