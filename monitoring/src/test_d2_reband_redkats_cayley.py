#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 RE-BAND red-KATs — REV 3 (cayley, 2026-08-08) under codex reviews `66b7f79` (§5), `1cdbb81`
(rev-1 contract), and 0207 (`c07ab86`, WORKS-WITH-THREE-CLOSURE-DELTAS — the final bar-only
convergence turn). Five freezes STAY; grassmann implements UNEDITED after codex passes THIS rev.

REV 3 closure deltas (codex 0207):
  #1 BLOCKER  D2R-1f now proves the shell USES the core's return (unique EnvelopeSeries sentinel;
      `out["XX.AAA"] is sentinel`) and trips AssertionError if any legacy
      process_waveforms/compute_envelope helper runs; D2R-4p/4q/4r freeze the REGISTRY seam: the
      production default (no injected capsule_loader) resolves capsule_registry_entry() and the
      REGISTERED digest reaches expected_sha256 — a capsule mutated under the unchanged registered
      digest fails closed (a default that re-hashes the capsule it loads is self-attestation).
  #2 HIGH     D2R-3a is now the ALL-TEN-field dataclasses.replace matrix with positional args held
      identical; D2R-3b tampers a stored values element under an UNCHANGED output_sha256 and
      requires rejection (the cached payload is JSON-inspectable per the storage contract).
  #3 MODERATE D2R-3p uses a wholly in-carrier gap (coverage-consistency isolated); D2R-3q
      aware-non-UTC (+09:00) start rejects; D2R-3r/3s zero and NaN dt reject.

REV 2 repairs (codex 2324):
  #1 BLOCKER  production COMPOSITION frozen, not helpers/source-text: D2R-1f spies on the core
      through the ACTUAL get_segment_envelopes shell (core called once per stream; NO DSP on the
      stream object — the old waveform->1Hz->envelope path cannot run); D2R-3g asserts the shell's
      cache identity is DERIVED (rate/digest/band/version/nslc), not a caller literal; D2R-4g runs
      GeoSpecEnsemble.compute_fault_correlation_risk with a capsule-loader spy (capsule threshold
      reaches analyze_region; missing capsule => available=False).
  #2 HIGH     canonical EnvelopeCacheIdentity + verify-on-load cache: identity derived from exact
      interval, NSLC, actual native rate, band, processing version, topology version, input-byte
      digest; stored WITH the payload + output digest; load takes an EXPECTED identity and rejects
      mismatch (an old-order payload relabelled rev-2 is rejected). This locks the provenance
      chain, so the caller-labelled EnvelopeSeries fields of D2R-2e are backed by a non-forgeable
      carrier; no absolute variance floor is needed.
  #3 HIGH     calibration capsule is externally pinned + exact-schema: loader takes
      expected_sha256 (registry pin) and refuses a self-consistent mutated capsule; embargo
      DEFAULT 30 DAYS (R3 lag consistency: the corrected R3 recal is a 90-day window ending
      today-30, run_thd_recal.py); malformed dates / start>end / nonfinite-or-out-of-range
      threshold / outside validity all reject; 29-day lag rejects, 30-day lag passes.
  #4 MODERATE exact-UTC-grid lock: aware UTC starts; finite positive EQUAL dt; starts on one
      exact common grid (offset an integer multiple of dt); gaps inside the carrier interval;
      coverage consistent with declared gaps + sample count; every mismatch fails unavailable —
      NEVER interpolated.

CONTRACT (grassmann implements to THIS, unedited — the decouple)
=================================================================
seismic_data.py rev-2:
* class DataUnavailable(Exception) with `.reasons: list[str]`.
* @dataclass EnvelopeSeries: values (np.ndarray at out_rate), start_utc (AWARE UTC datetime),
  dt_seconds (float), coverage (float 0..1), gaps (list of (iso_utc, seconds)), source_ids
  (list[str]), band_tag (str), processing_version (str), pre_envelope_rate_hz (float).
* PROCESSING_VERSION: str constant (bumped for rev-2; part of every cache identity).
* compute_band_envelope_from_array(data, rate_hz, *, freqmin=1.0, freqmax=10.0, out_rate_hz=1.0,
  min_rate_hz=25.0, start_utc, source_id) -> EnvelopeSeries
    - raises DataUnavailable (reason naming rate/Nyquist) if rate_hz < min_rate_hz;
    - bandpass at NATIVE rate -> Hilbert envelope while the carrier exists -> low-pass + resample
      the ENVELOPE to out_rate_hz; pre_envelope_rate_hz = the native rate. ALL DSP LIVES HERE.
* @dataclass EnvelopeCacheIdentity: region, segment, source_id (NSLC[.channel]), start_utc,
  end_utc, native_rate_hz, band_tag, processing_version, topology_version, input_sha256.
* build_envelope_cache_identity(...) -> EnvelopeCacheIdentity — derives native_rate_hz from the
  ACTUAL trace and input_sha256 from the exact input bytes (sha256 of the raw sample array bytes).
* _cache_path(self, region, start, end, data_type, identity: EnvelopeCacheIdentity) -> Path —
  ALL TEN identity fields participate in the key (any single field change, positional args held
  identical, yields a different path — D2R-3a matrix via dataclasses.replace).
* _save_cached_series(self, path, identity, series) — stores a JSON payload with top-level keys
  {"identity": {…all ten fields, datetimes ISO}, "output_sha256": sha256 over the canonical JSON
  of the stored values list, "series": {…EnvelopeSeries fields incl. "values": list}}.
  _load_cached_series(self, path, expected_identity) -> EnvelopeSeries | None — returns the
  payload ONLY IF the stored identity equals the expected identity AND the stored output digest
  RECOMPUTES over the stored values; a tampered values element under an unchanged output_sha256
  returns None (D2R-3b), as does any identity mismatch/relabel (fail-closed recompute).
* get_segment_envelopes (the SHELL): per stream it may merge/detrend/copy the stream object but
  performs NO filter/decimate/resample on it; it extracts (raw array, actual native rate, nslc),
  builds the DERIVED EnvelopeCacheIdentity, consults the validated cache, calls
  compute_band_envelope_from_array exactly once per cache-missed stream, and RETURNS THE CORE'S
  EnvelopeSeries OBJECTS as its outputs (out[nslc] IS the core's return — D2R-1f sentinel); any
  legacy process_waveforms/compute_envelope helpers are never invoked on the fault-corr path.

fault_correlation.py rev-2:
* class CalibrationUnavailable(Exception) with `.reasons: list[str]`.
* observability_gate(series_by_segment) -> (ok, reasons) — fail-closed BEFORE correlation:
  finite; >= 2 segments; minimum effective samples; scale-independent degeneracy (constant /
  near-constant); PROVENANCE: pre_envelope_rate_hz >= 25 and rev-2 band/processing version
  (backed by the identity chain above — labels alone cannot be minted outside the core+cache).
* capsule_registry_entry(region, registry_path) -> {"capsule_path", "expected_sha256",
  "topology_version"} — reads the EXTERNAL registry JSON (region-keyed); missing registry/region
  raises CalibrationUnavailable. The registered digest is the ONLY source of expected_sha256 in
  production — a loader that hashes the capsule it is about to load is self-attestation.
* align_activity_series(series, *, max_gap_seconds, min_coverage) -> (A | None, names, qc)
    - requires: AWARE start_utc with EXACTLY UTC offset on every series (naive OR aware-non-UTC
      => unavailable); finite positive dt_seconds (zero/NaN => unavailable) EQUAL across series
      (mismatch => unavailable); starts lying on
      ONE exact common grid (start offsets integer multiples of dt — a half-sample phase offset
      => unavailable; NEVER interpolate); gaps must lie inside their carrier interval
      [start, start + n*dt] (else unavailable); declared coverage must be consistent with the
      declared gaps and sample count (contradiction => unavailable);
    - exact UTC-grid intersection alignment; empty/short overlap, coverage < min_coverage, or a
      gap > max_gap_seconds -> (None, names, reasons); a short permitted gap keeps ok but RECORDS
      its qc flag.
* compute_correlation_matrix -> (C | None, segment_names, qc_reasons) — NO std->1.0, NO
  nan_to_num; invalid matrix (non-finite / non-unit diagonal / non-PSD beyond tol) is (None, ...).
* CorrelationResult gains data_quality_ok: bool and qc_reasons: list.
* load_calibration_capsule(region, scored_day, *, band_tag, processing_version, topology_version,
  capsule_dir, expected_sha256, embargo_days=30) -> dict
    - the capsule file's EXACT BYTES must sha256 to expected_sha256 (the externally registered
      pin) — a self-consistent mutated capsule fails; EXACT schema (no missing/extra keys):
      {schema: "geospec-d2-calibration-v1", region, band_tag, processing_version,
       topology_version, threshold, calibration_window: {start, end}, source_commit (40-hex),
       input_manifest_sha256 (64-hex), replay_output_sha256 (64-hex), issued_utc, valid_through};
    - rejects: any binding mismatch; malformed/unparseable dates; window start > end; threshold
      non-finite or outside (0, 1); scored_day > valid_through (STALE); LEAKAGE/LAG:
      (scored_day - calibration_window.end).days < embargo_days (default 30, per the corrected
      R3 lag) — 29-day lag rejects, 30-day lag passes.
* analyze_region(region, target_date, *, calibration: dict, ...) — threshold FROM the capsule.
* validate_topology(region) -> (ok, reasons) — rejects shared NET.STA across correlated segments
  (current japan_tohoku + ridgecrest reject; disjoint topologies pass).

ensemble.py rev-2:
* GeoSpecEnsemble carries an injectable `capsule_loader` seam AND a `capsule_registry_path`
  attribute (production default = the repo capsule registry). When `capsule_loader` is NOT
  injected, compute_fault_correlation_risk builds the production default: it resolves
  `capsule_registry_entry(region, registry_path=self.capsule_registry_path)` and calls
  `load_calibration_capsule(region, day, band_tag=<module band>, processing_version=
  <SD.PROCESSING_VERSION>, topology_version=<registry entry's>, capsule_dir=
  dirname(entry.capsule_path), expected_sha256=<entry's REGISTERED digest>)` — never a digest
  computed from the capsule bytes themselves. It passes the returned capsule into analyze_region
  as `calibration`, and returns available=False (zero effective weight) when the loader raises
  CalibrationUnavailable, when data_quality_ok is False, or when the topology contract fails.
* tokyo_kanto: the silent cross-geography remap is REMOVED; unavailable until true Kanto
  segments exist.
* component_frozen(region, component) -> bool — CARRIER-keyed (both 'tokyo_kanto' and
  'japan_tohoku' frozen; both coachella keys; kumamoto not).

RED AS AUTHORED (rev-2 seams absent from the landed modules).
"""
import hashlib
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
    for name, mod in (("obspy", obspy), ("obspy.clients", clients), ("obspy.clients.fdsn", fdsn),
                      ("obspy.signal", sig), ("obspy.signal.filter", sigfilter)):
        sys.modules[name] = mod


class _RecordingTraceStats:
    def __init__(self, rate):
        self.sampling_rate = rate


class _RecordingTrace:
    def __init__(self, data, rate):
        self.data = data
        self.stats = _RecordingTraceStats(rate)


class _RecordingStream:
    """Stream double: records every method call; DSP methods must never be invoked by the shell."""

    def __init__(self, data, rate):
        self.calls = []
        self._trace = _RecordingTrace(data, rate)

    def __getattr__(self, name):
        def _rec(*a, **k):
            self.calls.append(name)
            return self
        return _rec

    def __getitem__(self, i):
        return self._trace

    def __len__(self):
        return 1


# ---------------------------------------------------------------- fixtures --
FS = 40.0
T_SEC = 600
UTC0 = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
SCORED_DAY = "2026-08-07"
COMMIT40 = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


def am_fixture():
    """5 Hz carrier, known slow AM (0.02 Hz) — the modulation envelope is the ground truth."""
    t = np.arange(int(FS * T_SEC)) / FS
    m = 1.0 + 0.6 * np.sin(2 * np.pi * 0.02 * t)
    x = m * np.sin(2 * np.pi * 5.0 * t)
    m_1hz = m[:: int(FS)]
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
               source_ids=("XX.AAA",), band_tag=None, version=None, dt_seconds=1.0):
    return SD.EnvelopeSeries(values=np.asarray(values, dtype=float), start_utc=start_utc,
                             dt_seconds=dt_seconds, coverage=coverage, gaps=list(gaps),
                             source_ids=list(source_ids),
                             band_tag=(band_tag or getattr(SD, "_FAULT_CORR_BAND_TAG", "1-10Hz")),
                             processing_version=(version or SD.PROCESSING_VERSION),
                             pre_envelope_rate_hz=rate_pre)


def _capsule_dict(SD, **overrides):
    cap = {"schema": "geospec-d2-calibration-v1", "region": "ridgecrest",
           "band_tag": "1-10Hz", "processing_version": SD.PROCESSING_VERSION,
           "topology_version": "t1", "threshold": 0.30,
           "calibration_window": {"start": "2026-04-01", "end": "2026-07-01"},
           "source_commit": COMMIT40,
           "input_manifest_sha256": "a" * 64, "replay_output_sha256": "b" * 64,
           "issued_utc": "2026-08-07T00:00:00Z", "valid_through": "2026-12-31"}
    cap.update(overrides)
    return cap


def _write_capsule(cap_dir, cap, region=None):
    p = os.path.join(cap_dir, f"{region or cap['region']}.json")
    body = json.dumps(cap, sort_keys=True).encode()
    with open(p, "wb") as fh:
        fh.write(body)
    return hashlib.sha256(body).hexdigest()


def main():
    _install_obspy_stub()
    import seismic_data as SD
    import fault_correlation as FC
    import ensemble as EN

    need_sd = ("DataUnavailable", "EnvelopeSeries", "PROCESSING_VERSION",
               "compute_band_envelope_from_array", "EnvelopeCacheIdentity",
               "build_envelope_cache_identity")
    if not all(hasattr(SD, n) for n in need_sd):
        check("D2R-0a seismic_data rev-2 seams present (EnvelopeSeries core + cache identity)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return
    need_sd_m = ("_cache_path", "_save_cached_series", "_load_cached_series",
                 "get_segment_envelopes")
    if not all(hasattr(SD.SeismicDataFetcher, n) for n in need_sd_m):
        check("D2R-0a2 fetcher rev-2 cache seams present (verify-on-load cached series)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return
    need_fc = ("CalibrationUnavailable", "observability_gate", "align_activity_series",
               "load_calibration_capsule", "validate_topology", "capsule_registry_entry")
    if not all(hasattr(FC, n) for n in need_fc):
        check("D2R-0b fault_correlation rev-2 seams present (gate/align/capsule/topology)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return
    if not hasattr(EN, "component_frozen"):
        check("D2R-0c ensemble rev-2 seams present (carrier-keyed component_frozen)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return

    # =====================================================================
    # GROUP 1 — envelope order (codex finding 1) + PRODUCTION COMPOSITION
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

    # D2R-1f (codex 2324 #1): spy on the core THROUGH the actual shell — composition, not source text
    from fault_segments import FaultSegment, SeismicStation
    seg = FaultSegment(name="syn_seg", region="syn_region",
                       stations=[SeismicStation("XX", "AAA", 0.0, 0.0)],
                       polygon=[(0, 0), (0, 1), (1, 1), (1, 0)], strike=0.0, dip=90.0, rake=0.0)
    fetcher = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    fetcher.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    stream = _RecordingStream(x, FS)
    fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": stream}
    core_calls = []
    real_core = SD.compute_band_envelope_from_array
    sentinel = real_core(x, FS, start_utc=UTC0, source_id="XX.AAA")   # unique object identity

    def spy_core(data, rate_hz, **kw):
        core_calls.append((np.asarray(data).size, float(rate_hz)))
        return sentinel

    def _legacy_boom(*a, **k):
        raise AssertionError("legacy waveform->1Hz->envelope path invoked")

    patched_legacy = []
    for legacy in ("process_waveforms", "compute_envelope"):
        if hasattr(SD.SeismicDataFetcher, legacy):
            patched_legacy.append((legacy, getattr(SD.SeismicDataFetcher, legacy)))
            setattr(SD.SeismicDataFetcher, legacy, _legacy_boom)
    SD.compute_band_envelope_from_array = spy_core
    core_err = None
    try:
        out = fetcher.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=T_SEC),
                                            use_cache=False)
    except Exception as e:
        out, core_err = None, e
    finally:
        SD.compute_band_envelope_from_array = real_core
        for legacy, orig in patched_legacy:
            setattr(SD.SeismicDataFetcher, legacy, orig)
    dsp_on_stream = [c for c in stream.calls if c in ("filter", "decimate", "resample")]
    check("D2R-1f COMPOSITION: the ACTUAL shell routes each stream through the core exactly once "
          "at the native rate, RETURNS the core's own EnvelopeSeries (sentinel identity), never "
          "invokes a legacy DSP helper, and performs no filter/decimate/resample on the stream",
          out is not None and len(core_calls) == 1 and core_calls[0][1] == FS
          and core_calls[0][0] == x.size and not dsp_on_stream
          and isinstance(out, dict) and out.get("XX.AAA") is sentinel,
          f"core_calls={core_calls} dsp_on_stream={dsp_on_stream} "
          f"identity_ok={isinstance(out, dict) and out.get('XX.AAA') is sentinel} "
          f"err={core_err}")

    # =====================================================================
    # GROUP 2 — fail-closed observability gate (codex finding 2)
    # =====================================================================
    good = _mk_series(SD, np.random.default_rng(1).normal(1.0, 0.3, 600), UTC0)

    def gate_fails(desc, series_by_seg):
        ok, reasons = FC.observability_gate(series_by_seg)
        check(desc, ok is False and isinstance(reasons, list) and len(reasons) > 0,
              f"ok={ok} reasons={reasons}")

    gate_fails("D2R-2a constant segments fail the gate (no std->1.0 pass-through)",
               {"a": _mk_series(SD, np.full(600, 3.7), UTC0), "b": good})
    gate_fails("D2R-2b near-constant (1e-14 jitter) fails the gate",
               {"a": _mk_series(SD, 3.7 + 1e-14 * np.random.default_rng(2).normal(size=600), UTC0),
                "b": good})
    gate_fails("D2R-2c all-NaN fails the gate",
               {"a": _mk_series(SD, np.full(600, np.nan), UTC0), "b": good})
    gate_fails("D2R-2d a single valid segment fails the gate (need >= 2)", {"a": good})
    gate_fails("D2R-2e anti-alias-erased PROVENANCE fails the gate (pre_envelope_rate 1 Hz; the "
               "label itself is non-forgeable via the D2R-3h/3i identity chain)",
               {"a": _mk_series(SD, np.abs(np.random.default_rng(3).normal(0, 4e-3, 600)),
                                UTC0, rate_pre=1.0), "b": good})
    ok_good, reasons_good = FC.observability_gate(
        {"a": good, "b": _mk_series(SD, np.random.default_rng(4).normal(1.0, 0.3, 600), UTC0)})
    check("D2R-2f two healthy segments PASS the gate", ok_good is True, f"reasons={reasons_good}")

    monitor = FC.FaultCorrelationMonitor(data_fetcher=object())
    monitor.compute_correlation_matrix = lambda *a, **k: (None, ["flat_a", "flat_b"],
                                                          ["degenerate: constant rows"])
    res = monitor.analyze_region("synthetic", datetime(2026, 8, 7),
                                 calibration=_capsule_dict(SD, region="synthetic"))
    check("D2R-2g gate-failed region: data_quality_ok False, NEVER is_decorrelated, no risk values",
          getattr(res, "data_quality_ok", None) is False and res.is_decorrelated is False
          and res.participation_ratio == 0.0 and len(getattr(res, "qc_reasons", [])) > 0,
          f"dq={getattr(res, 'data_quality_ok', 'MISSING')} decorr={res.is_decorrelated}")

    # =====================================================================
    # GROUP 3 — identity, alignment, gaps, verify-on-load cache (findings 3 + codex-2324 #2/#4)
    # =====================================================================
    ident_kw = dict(region="ridgecrest", segment="syn_seg", source_id="XX.AAA",
                    start_utc=UTC0, end_utc=UTC0 + timedelta(hours=24),
                    band_tag="1-10Hz", processing_version=SD.PROCESSING_VERSION,
                    topology_version="t1")
    ident = SD.build_envelope_cache_identity(native_rate_hz=40.0,
                                             input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                             **ident_kw)
    f2 = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    f2.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    p_base = f2._cache_path("ridgecrest", ident_kw["start_utc"], ident_kw["end_utc"],
                            "seg_env", ident)
    # codex 0207 delta 2: ALL TEN fields, dataclasses.replace variants, positional args HELD
    # IDENTICAL for every call — only the identity may cause the key to move.
    import dataclasses as _dc
    field_variants = {
        "region": {"region": "kaikoura"},
        "segment": {"segment": "other_seg"},
        "source_id": {"source_id": "YY.BBB"},
        "start_utc": {"start_utc": UTC0 + timedelta(hours=6)},
        "end_utc": {"end_utc": ident_kw["end_utc"] + timedelta(hours=6)},
        "native_rate_hz": {"native_rate_hz": 100.0},
        "band_tag": {"band_tag": "0.01-1Hz"},
        "processing_version": {"processing_version": "OLD-V0"},
        "topology_version": {"topology_version": "t0"},
        "input_sha256": {"input_sha256": "0" * 64},
    }
    changed = {}
    for fname, kwv in field_variants.items():
        v = _dc.replace(ident, **kwv)
        changed[fname] = (f2._cache_path("ridgecrest", ident_kw["start_utc"],
                                         ident_kw["end_utc"], "seg_env", v) != p_base)
    check("D2R-3a ALL TEN identity fields participate in the key (dataclasses.replace variants; "
          "positional args identical for every call)", all(changed.values()),
          f"unmoved={[k for k, ok in changed.items() if not ok]}")

    # verify-on-load: an old-order payload relabelled as rev-2 is REJECTED (codex 2324 #2)
    series_ok = _mk_series(SD, np.random.default_rng(9).normal(1.0, 0.3, 600), UTC0)
    f2._save_cached_series(p_base, _dc.replace(ident, processing_version="OLD-V0"), series_ok)
    loaded = f2._load_cached_series(p_base, ident)                              # expected rev-2
    check("D2R-3c verify-on-load: a payload stored under an OLD identity, requested as rev-2, "
          "is REJECTED (None -> fail-closed recompute)", loaded is None, f"loaded={loaded!r}")
    f2._save_cached_series(p_base, ident, series_ok)
    loaded_ok = f2._load_cached_series(p_base, ident)
    check("D2R-3d verify-on-load: the matching identity round-trips the payload",
          loaded_ok is not None
          and np.array_equal(np.asarray(loaded_ok.values), np.asarray(series_ok.values)))
    # codex 0207 delta 2: OUTPUT-DIGEST verification — tamper one stored values element while
    # leaving output_sha256 untouched; the load must recompute and refuse.
    try:
        with open(p_base, encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["series"]["values"][7] = float(payload["series"]["values"][7]) + 1.0
        with open(p_base, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tampered = f2._load_cached_series(p_base, ident)
        check("D2R-3b OUTPUT DIGEST: a tampered stored value under an UNCHANGED output_sha256 is "
              "REJECTED (digest recomputed on load)", tampered is None, f"loaded={tampered!r}")
    except Exception as exc:
        check("D2R-3b OUTPUT DIGEST: a tampered stored value under an UNCHANGED output_sha256 is "
              "REJECTED (digest recomputed on load)", False,
              f"payload not JSON-inspectable per contract: {type(exc).__name__}: {exc}")

    # D2R-3g (codex 2324 #1): the SHELL derives the identity — not a caller literal
    f3 = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    f3.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    stream3 = _RecordingStream(x, FS)
    f3.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": stream3}
    seen = {}
    orig_load = SD.SeismicDataFetcher._load_cached_series
    orig_save = SD.SeismicDataFetcher._save_cached_series

    def spy_load(self, path, expected_identity):
        seen["load_identity"] = expected_identity
        return None

    def spy_save(self, path, identity, series):
        seen["save_identity"] = identity

    SD.SeismicDataFetcher._load_cached_series = spy_load
    SD.SeismicDataFetcher._save_cached_series = spy_save
    try:
        f3.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=T_SEC), use_cache=True)
    except Exception as e:
        seen["error"] = str(e)
    finally:
        SD.SeismicDataFetcher._load_cached_series = orig_load
        SD.SeismicDataFetcher._save_cached_series = orig_save
    got = seen.get("save_identity") or seen.get("load_identity")
    check("D2R-3g the shell's cache identity is DERIVED from the actual trace (rate 40, input "
          "digest of the real samples, nslc, rev-2 version) — never a caller literal",
          got is not None and getattr(got, "native_rate_hz", None) == FS
          and getattr(got, "input_sha256", "") == hashlib.sha256(x.tobytes()).hexdigest()
          and getattr(got, "source_id", "") == "XX.AAA"
          and getattr(got, "processing_version", "") == SD.PROCESSING_VERSION,
          f"seen={ {k: getattr(v, '__dict__', v) for k, v in seen.items()} }")

    # UTC alignment + the codex-2324 #4 mismatch matrix
    rng = np.random.default_rng(7)
    vals = rng.normal(0.0, 1.0, 1000)
    sA = _mk_series(SD, vals, UTC0)
    sB = _mk_series(SD, vals, UTC0 + timedelta(seconds=360))
    A, names, qc = FC.align_activity_series({"a": sA, "b": sB},
                                            max_gap_seconds=600, min_coverage=0.9)
    if A is None:
        check("D2R-3h UTC alignment: shifted packets never become zero-lag identical", True)
    else:
        r_pair = float(np.corrcoef(A[0], A[1])[0, 1])
        check("D2R-3h UTC alignment: shifted packets never become zero-lag identical "
              "(element-zero truncation would give r=1.0)",
              A.shape[1] == 640 and abs(r_pair) < 0.5, f"n={A.shape[1]} r={r_pair:.3f}")

    def align_fails(desc, series_map):
        Ax, _, qcx = FC.align_activity_series(series_map, max_gap_seconds=600, min_coverage=0.9)
        check(desc, Ax is None and len(qcx) > 0, f"A={'ok' if Ax is not None else None} qc={qcx}")

    align_fails("D2R-3i insufficient coverage fails closed",
                {"a": _mk_series(SD, vals, UTC0, coverage=0.5), "b": sA})
    align_fails("D2R-3j a gap beyond max_gap fails closed",
                {"a": _mk_series(SD, vals, UTC0, gaps=[("2026-08-01T02:00:00Z", 3600)]), "b": sA})
    A4, _, qc4 = FC.align_activity_series(
        {"a": _mk_series(SD, vals, UTC0, gaps=[("2026-08-01T02:00:00Z", 60)]), "b": sA},
        max_gap_seconds=600, min_coverage=0.9)
    check("D2R-3k a short permitted gap stays available AND records its QC flag",
          A4 is not None and len(qc4) > 0, f"qc={qc4}")
    align_fails("D2R-3l mismatched sample intervals fail unavailable (dt 1.0 vs 2.0 -- never "
                "interpolated)", {"a": sA, "b": _mk_series(SD, vals[:500], UTC0, dt_seconds=2.0)})
    align_fails("D2R-3m a half-sample phase offset fails unavailable (starts must lie on ONE "
                "exact common grid)",
                {"a": sA, "b": _mk_series(SD, vals, UTC0 + timedelta(seconds=360.5))})
    align_fails("D2R-3n a NAIVE (timezone-less) start fails unavailable",
                {"a": sA, "b": _mk_series(SD, vals, datetime(2026, 8, 1, 0, 6, 0))})
    align_fails("D2R-3o a gap OUTSIDE the carrier interval fails unavailable",
                {"a": _mk_series(SD, vals, UTC0, gaps=[("2026-09-15T00:00:00Z", 60)]), "b": sA})
    align_fails("D2R-3p declared coverage contradicting the declared gaps fails unavailable "
                "(a wholly IN-carrier 100 s gap at sample 300 with coverage=1.0 on the 1000 s "
                "carrier -- isolates the consistency check, codex 0207 delta 3)",
                {"a": _mk_series(SD, vals, UTC0, coverage=1.0,
                                 gaps=[("2026-08-01T00:05:00Z", 100)]), "b": sA})
    align_fails("D2R-3q an AWARE NON-UTC start (+09:00) fails unavailable (UTC offset must be "
                "exactly zero)",
                {"a": sA, "b": _mk_series(SD, vals, datetime(2026, 8, 1, 9, 6, 0,
                                          tzinfo=timezone(timedelta(hours=9))))})
    align_fails("D2R-3r dt_seconds = 0 fails unavailable",
                {"a": sA, "b": _mk_series(SD, vals, UTC0, dt_seconds=0.0)})
    align_fails("D2R-3s dt_seconds = NaN fails unavailable",
                {"a": sA, "b": _mk_series(SD, vals, UTC0, dt_seconds=float("nan"))})

    # =====================================================================
    # GROUP 4 — externally pinned calibration capsule (finding 4 + codex-2324 #3)
    # =====================================================================
    cap_dir = tempfile.mkdtemp()
    cap = _capsule_dict(SD)
    pin = _write_capsule(cap_dir, cap)
    kw = dict(band_tag="1-10Hz", processing_version=SD.PROCESSING_VERSION,
              topology_version="t1", capsule_dir=cap_dir, expected_sha256=pin)
    loaded_cap = FC.load_calibration_capsule("ridgecrest", SCORED_DAY, **kw)
    check("D2R-4a a correctly pinned capsule loads (30+ day lag; DEFAULT embargo) and carries "
          "the per-region threshold", isinstance(loaded_cap, dict)
          and loaded_cap.get("threshold") == 0.30)
    ok_m, _ = raises_unavailable(lambda: FC.load_calibration_capsule("kumamoto", SCORED_DAY, **kw))
    check("D2R-4b missing capsule -> CalibrationUnavailable", ok_m)
    # SELF-CONSISTENT MUTATION fails the EXTERNAL pin (codex 2324 #3)
    mut_dir = tempfile.mkdtemp()
    _write_capsule(mut_dir, _capsule_dict(SD, threshold=0.999999))
    ok_pin, _ = raises_unavailable(
        lambda: FC.load_calibration_capsule("ridgecrest", SCORED_DAY,
                                            **{**kw, "capsule_dir": mut_dir}))
    check("D2R-4c a self-consistent MUTATED capsule fails the external sha256 pin "
          "(self-attestation is not admission)", ok_pin)
    for field_, bad in (("band_tag", "0.01-1Hz"), ("processing_version", "STALE-V0"),
                        ("topology_version", "t0")):
        bad_kw = dict(kw)
        bad_kw[field_] = bad
        ok_x, _ = raises_unavailable(
            lambda b=bad_kw: FC.load_calibration_capsule("ridgecrest", SCORED_DAY, **b))
        check(f"D2R-4d capsule {field_} mismatch rejects", ok_x)

    def bad_capsule_rejects(desc, **overrides):
        d = tempfile.mkdtemp()
        c = _capsule_dict(SD, **overrides)
        p2 = _write_capsule(d, c)
        ok_b, _ = raises_unavailable(
            lambda: FC.load_calibration_capsule("ridgecrest", SCORED_DAY,
                                                **{**kw, "capsule_dir": d,
                                                   "expected_sha256": p2}))
        check(desc, ok_b)

    bad_capsule_rejects("D2R-4e schema: an EXTRA key rejects (exact schema)", grants_lift=True)
    bad_capsule_rejects("D2R-4f malformed window date rejects",
                        calibration_window={"start": "04/01/2026", "end": "2026-07-01"})
    bad_capsule_rejects("D2R-4g window start > end rejects",
                        calibration_window={"start": "2026-07-01", "end": "2026-04-01"})
    bad_capsule_rejects("D2R-4h non-finite threshold rejects", threshold=float("nan"))
    bad_capsule_rejects("D2R-4i out-of-range threshold rejects", threshold=1.5)
    bad_capsule_rejects("D2R-4j LAG: calibration ending 29 days before the scored day rejects "
                        "(embargo default 30 d, R3-lag consistent)",
                        calibration_window={"start": "2026-04-01", "end": "2026-07-09"})
    d30 = tempfile.mkdtemp()
    c30 = _capsule_dict(SD, calibration_window={"start": "2026-04-01", "end": "2026-07-08"})
    p30 = _write_capsule(d30, c30)
    try:
        got30 = FC.load_calibration_capsule("ridgecrest", SCORED_DAY,
                                            **{**kw, "capsule_dir": d30, "expected_sha256": p30})
        check("D2R-4k LAG: a correctly pinned 30-day-lag capsule PASSES", isinstance(got30, dict))
    except Exception as exc:
        check("D2R-4k LAG: a correctly pinned 30-day-lag capsule PASSES", False, f"RAISED {exc}")
    ok_stale, _ = raises_unavailable(
        lambda: FC.load_calibration_capsule("ridgecrest", "2027-06-01", **kw))
    check("D2R-4l STALE: scored day past valid_through rejects", ok_stale)
    # threshold follows the capsule
    C_fixed = np.array([[1.0, 0.5], [0.5, 1.0]])                # lambda2/lambda1 = 1/3
    mon2 = FC.FaultCorrelationMonitor(data_fetcher=object())
    mon2.compute_correlation_matrix = lambda *a, **k: (C_fixed, ["s1", "s2"], [])
    res_lo = mon2.analyze_region("ridgecrest", datetime(2026, 8, 7),
                                 calibration={**cap, "threshold": 0.30})
    res_hi = mon2.analyze_region("ridgecrest", datetime(2026, 8, 7),
                                 calibration={**cap, "threshold": 0.40})
    check("D2R-4m the verdict threshold comes FROM THE CAPSULE (0.30 -> no; 0.40 -> yes on the "
          "same matrix)", res_lo.is_decorrelated is False and res_hi.is_decorrelated is True
          and getattr(res_lo, "data_quality_ok", None) is True,
          f"lo={res_lo.is_decorrelated} hi={res_hi.is_decorrelated}")

    # D2R-4n (codex 2324 #1): ENSEMBLE INTEGRATION — the capsule loader seam is consulted
    seen_cal = {}
    mon3 = FC.FaultCorrelationMonitor(data_fetcher=object())
    mon3.compute_correlation_matrix = lambda *a, **k: (C_fixed, ["s1", "s2"], [])
    real_analyze = mon3.analyze_region

    def spy_analyze(region, target_date, *, calibration, **k):
        seen_cal["calibration"] = calibration
        return real_analyze(region, target_date, calibration=calibration, **k)

    mon3.analyze_region = spy_analyze
    ens = EN.GeoSpecEnsemble.__new__(EN.GeoSpecEnsemble)
    ens.region = "ridgecrest"
    ens.fault_corr_monitor = mon3
    ens.capsule_loader = lambda *a, **k: dict(loaded_cap)
    try:
        mr_ok = ens.compute_fault_correlation_risk(datetime(2026, 8, 7))[0]
        check("D2R-4n ensemble consults the capsule loader and its threshold REACHES "
              "analyze_region", mr_ok.available is True
              and seen_cal.get("calibration", {}).get("threshold") == 0.30,
              f"available={mr_ok.available} cal={seen_cal.get('calibration')}")
    except Exception as exc:
        check("D2R-4n ensemble consults the capsule loader and its threshold REACHES "
              "analyze_region", False, f"RAISED {exc}")

    def _raise_cal(*a, **k):
        raise FC.CalibrationUnavailable(["no capsule registered"])

    ens.capsule_loader = _raise_cal
    try:
        mr_no = ens.compute_fault_correlation_risk(datetime(2026, 8, 7))[0]
        check("D2R-4o ensemble: CalibrationUnavailable -> available=False (zero effective weight)",
              mr_no.available is False, f"available={mr_no.available}")
    except Exception as exc:
        check("D2R-4o ensemble: CalibrationUnavailable -> available=False (zero effective weight)",
              False, f"RAISED {exc}")

    # codex 0207 delta 1: the PRODUCTION DEFAULT (no injected capsule_loader) must take
    # expected_sha256 from the EXTERNAL registry — never from the capsule bytes themselves.
    reg_home = tempfile.mkdtemp()
    reg_cap = _capsule_dict(SD)
    reg_pin = _write_capsule(reg_home, reg_cap)
    reg_path = os.path.join(reg_home, "capsule_registry.json")
    with open(reg_path, "w", encoding="utf-8") as fh:
        json.dump({"ridgecrest": {"capsule_path": os.path.join(reg_home, "ridgecrest.json"),
                                  "expected_sha256": reg_pin,
                                  "topology_version": "t1"}}, fh)
    entry = FC.capsule_registry_entry("ridgecrest", registry_path=reg_path)
    check("D2R-4p capsule_registry_entry resolves {capsule_path, expected_sha256, "
          "topology_version} from the external registry",
          entry.get("expected_sha256") == reg_pin
          and entry.get("capsule_path", "").endswith("ridgecrest.json")
          and entry.get("topology_version") == "t1", f"entry={entry}")
    seen_pin = {}
    real_loader = FC.load_calibration_capsule

    def spy_loader(region, scored_day, **kw2):
        seen_pin["expected_sha256"] = kw2.get("expected_sha256")
        return real_loader(region, scored_day, **kw2)

    mon5 = FC.FaultCorrelationMonitor(data_fetcher=object())
    mon5.compute_correlation_matrix = lambda *a, **k: (C_fixed, ["s1", "s2"], [])
    ens5 = EN.GeoSpecEnsemble.__new__(EN.GeoSpecEnsemble)
    ens5.region = "ridgecrest"
    ens5.fault_corr_monitor = mon5
    ens5.capsule_registry_path = reg_path          # production seam; NO capsule_loader assigned
    FC.load_calibration_capsule = spy_loader
    if getattr(EN, "load_calibration_capsule", None) is real_loader:
        EN.load_calibration_capsule = spy_loader   # cover a from-import in ensemble
    try:
        mr_reg = ens5.compute_fault_correlation_risk(datetime(2026, 8, 7))[0]
        check("D2R-4q PRODUCTION DEFAULT: without an injected loader, the ensemble consults the "
              "registry and the REGISTERED digest reaches expected_sha256 (available=True)",
              mr_reg.available is True and seen_pin.get("expected_sha256") == reg_pin,
              f"available={getattr(mr_reg, 'available', 'ERR')} seen={seen_pin}")
    except Exception as exc:
        check("D2R-4q PRODUCTION DEFAULT: without an injected loader, the ensemble consults the "
              "registry and the REGISTERED digest reaches expected_sha256 (available=True)",
              False, f"RAISED {exc}")
    finally:
        FC.load_calibration_capsule = real_loader
        if getattr(EN, "load_calibration_capsule", None) is spy_loader:
            EN.load_calibration_capsule = real_loader
    # mutate the capsule UNDER THE UNCHANGED registered digest -> fail closed, no self-attestation
    _write_capsule(reg_home, _capsule_dict(SD, threshold=0.999))   # same path, new bytes
    try:
        mr_mut = ens5.compute_fault_correlation_risk(datetime(2026, 8, 7))[0]
        check("D2R-4r a capsule mutated under the UNCHANGED registered digest -> available=False "
              "(the default never re-hashes the capsule to bless itself)",
              mr_mut.available is False, f"available={getattr(mr_mut, 'available', 'ERR')}")
    except Exception as exc:
        check("D2R-4r a capsule mutated under the UNCHANGED registered digest -> available=False "
              "(the default never re-hashes the capsule to bless itself)",
              False, f"RAISED {exc}")

    # =====================================================================
    # GROUP 5 — topology contract + honest carriers (finding 5; unchanged per codex 2324)
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
    ens2.capsule_loader = lambda *a, **k: dict(loaded_cap)
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
    check("D2R-5f freeze is CARRIER-KEYED: the FC-canonical key cannot bypass it", ok_frozen,
          f"{[(r, EN.component_frozen(r, 'fault_correlation')) for r, _ in frozen_pairs]}")


main()
print()
if FAILS:
    print(f"D2 RE-BAND RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 RE-BAND RED-KATs PASS (composition-frozen envelope order + non-forgeable identity "
      "chain + pinned capsule + UTC-grid lock + topology)")
