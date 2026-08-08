#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 SEGMENTED-SUPPORT red-KATs — REV 3 (cayley, 2026-08-08) — phase
`codex-d2-segmented-support-2026-08-08-v1`; codex assignment `27e227b`, bar review 1732, bounded
correction 1747, closing PASS 1812, and the LC-2b/fractional-phase RULING 1906 (`8e44378`).
Pinned start GeoSpec `54cea7b`; interim implementation `7d169ec`. Grassmann implements UNEDITED
after codex passes THIS revision. Nothing here lifts a freeze / admits calibration / supports a
claim; the historical incident/control replay through this enhancement is a LATER diagnostic run.

REV 3 (codex 1906 ruling): the rev-2 "fractional phase contributes ZERO bins" rule was MY defect —
internally inconsistent with SS-7b's fractional-span fixtures and mission-breaking on real archives
(published sessions start at microsecond-fractional instants; gap durations are not integer
seconds). FROZEN PHASE SEMANTICS now: for output grid index k the claimed timestamp is EXACTLY
t_k = session_start_utc + k/out_rate_hz; a fragment may contribute at ANY fractional phase; its
value at t_k is derived from that fragment's independently processed native carrier and evaluated
AT that same t_k through an EXPLICIT interpolation/resampling operator — the fragment start is
never rounded, shifted, or rewritten; nearest-OUTPUT-grid placement AND nearest-NATIVE-sample
selection are both FORBIDDEN; no interpolation/filter support crosses a gap. The valid mask is
TIMESTAMP-DERIVED, never length-rounded: t_k is valid exactly when
fragment_start + EDGE_TRIM_SECONDS <= t_k < fragment_exclusive_end - EDGE_TRIM_SECONDS and the raw
span meets MIN_CONTIGUOUS_SPAN_SECONDS. support_sha256 keeps binding the ORIGINAL fragment phase,
session grid, and resulting mask. The single-span sentinel/identity return is allowed ONLY for an
exact-phase (fragment_start == session_start) full-session span with the exact edge-trimmed mask
attached; a fractional-phase span MUST take the general exact-time path (SS-11d/e are the
executable guard). SS-11d is reworked from zero-contribution to ANTI-RELABEL; SS-11e adds the
published-style microfractional fixture (+0.948447 s at 40.5 Hz). The companion live-carrier bar's
LC-2b pair is updated to segmented semantics in the same commit (codex option A; LC-1/1b/3/2a
byte-semantically unchanged).

REV 2 closures (codex 1732):
  #1 BLOCKER  SS-10 drives the PRODUCTION monitor (`FaultCorrelationMonitor.
      compute_correlation_matrix`) with a synthetic fetcher returning supported series, POISONS the
      legacy aggregate path, and requires: filler-0/999 bit-identical production output;
      contradictory series -> (None, ..., reasons); and a 3-eligible-segment case where A/B pairwise
      support >= 50% but A^B^C < 50% must FAIL WITH ALL THREE NAMES (no subset search).
  #2 BLOCKER  station identity frozen as NET.STA (exactly one parseable identity per station-level
      series): two channels of the SAME NET.STA never satisfy the two-station rule (SS-11a); frame
      guards at BOTH aggregation and correlation: mismatched start_utc (incl. +0.5 s), dt_seconds,
      requested_grid_count, band_tag, processing_version each fail unavailable (SS-11b/11c); a
      fractional-phase RAW fragment contributes no bins or raises — never rounded onto the session
      grid (SS-11d).
  #3 MAJOR    support_digest signature made expressible:
      support_digest(fragments, *, session_start_utc, session_seconds, out_rate_hz, source_id) —
      canonical hash over ordered (start_iso, exclusive_end_iso, rate, npts, source_id, raw_sha256)
      + session grid identity + derived valid-mask bytes. Six independent mutations (raw bytes,
      rate, npts, gap endpoint, source ID, session phase) each change it (SS-4b), and the REAL
      cache save/load path is exercised: support_sha256 lives in the v3 identity, moves the cache
      path, and a support-mutated identity misses on load (SS-4c/4d). SS-7a now asserts the core
      receives the ORIGINAL arrays/rates/starts and that `stream.calls == []` (no detrend/trim/
      taper/merge or ANY stream-method call can slip through).
  #4 MAJOR    finite trim gets a DECLARED ERROR CONTRACT: EDGE_EFFECT_REL_TOL = 2e-4 — "valid"
      means edge discrepancy <= 2e-4 on the frozen operator-reference suite below (a KAT bound,
      NOT a universal signal theorem; codex's exact-operator execution of THIS bar's 25 Hz
      boundary-impulse reference measured 1.042e-4, so 2e-4 gives ~1.9x margin over the frozen
      operator stress — codex 1747 bounded correction). SS-9 runs the reference-continuation suite at native rates 25/40/40.5/100 Hz
      with AM, deterministic-broadband, step+chirp, and boundary-impulse stressors (span alone vs
      embedded; trimmed interiors compared). The half-open mask is frozen EXACTLY: a gapless
      86,400 s session is valid on [90, 86310) = 86,220 bins; a 240 s span on [90, 150) = 60 bins
      (the old 55..62 allowance is removed).

FROZEN CONSTANTS (operator-derived per the delegation; never from incident/control data):
  * EDGE_TRIM_SECONDS = 90 — per-stage settle = 10 cycles of the stage's lowest cutoff, doubled
    for zerophase application, summed: bandpass 1–10 Hz ord-4 zerophase 2x10/1.0 = 20 s;
    FFT-Hilbert edge ~3 longest periods = 3 s; envelope anti-alias low-pass ~0.4 Hz zerophase
    2x10/0.4 = 50 s; total 73 s worst-case -> rounded up to 90 s.
  * MIN_CONTIGUOUS_SPAN_SECONDS = 240 — 2 x EDGE_TRIM + 60 s minimum valid interior (one minute of
    1 Hz output). A shorter span contributes ZERO valid bins.
  * MIN_COMMON_SUPPORT_FRACTION = 0.50 of the requested session grid (== 43,200 @ 24 h).
  * STATION_COVERAGE_FLOOR = 0.50 (existing frozen floor; 0.50 exactly is eligible).
  * EDGE_EFFECT_REL_TOL = 2e-4 — the edge-error KAT bound defined in #4 above.

CONTRACT (grassmann implements to THIS, unedited — the decouple)
=================================================================
seismic_data.py rev-3 (segmented support):
* EnvelopeSeries gains `valid_mask` (bool, len == values) and `requested_grid_count: int`;
  `values` DENSE over the requested session grid; invalid bins carry meaningless filler;
  `coverage == count(valid_mask)/requested_grid_count` EXACTLY; `gaps` = EXACT run-length
  complement of valid_mask; station-level series carry EXACTLY ONE parseable source identity
  whose station key is `NET.STA`.
* validate_support(series) -> (ok, reasons) — mask/gaps/coverage contradiction, length mismatch,
  non-bool mask, or unparseable source identity fails; downstream paths MUST consult it.
* SEG_SUPPORT = {"edge_trim_seconds": 90, "min_contiguous_span_seconds": 240,
  "min_common_support_fraction": 0.50, "station_coverage_floor": 0.50,
  "edge_effect_rel_tol": 2e-4}.
* compute_band_envelope_supported(fragments, *, session_start_utc, session_seconds=86400,
  out_rate_hz=1.0, freqmin=1.0, freqmax=10.0, min_rate_hz=25.0, source_id) -> EnvelopeSeries
    - fragments: ascending non-overlapping (data, rate_hz, aware-UTC start) raw spans; overlap /
      naive start / rate < min_rate_hz -> DataUnavailable; a fragment may sit at ANY fractional
      phase and contributes under the FROZEN PHASE SEMANTICS above (exact-time evaluation at t_k;
      timestamp-derived mask; never rounded/shifted/relabeled; no nearest-sample selection);
    - bandpass -> Hilbert envelope -> anti-aliased resample INDEPENDENTLY per span; NO operation
      of any kind across a gap; spans < MIN_CONTIGUOUS_SPAN contribute nothing;
    - EXACT timestamp-derived validity: t_k valid iff fragment_start + 90 <= t_k <
      fragment_exclusive_end - 90 (span >= 240 s). Integer-aligned examples: gapless 86,400 s
      session -> [90, 86310), 86,220 bins; 240 s span at t0=0 -> [90, 150), 60 bins; fractional
      example: span [300.5, 900.5) -> t_k in {391..810}, 420 bins;
    - "valid" is an APPROXIMATION CONTRACT: on the operator-reference suite (SS-9) the trimmed
      interior of a span processed alone matches the same interior processed inside a longer
      carrier within EDGE_EFFECT_REL_TOL (max abs diff / max abs reference), at native rates
      {25, 40, 40.5, 100} Hz under AM / deterministic-broadband / step+chirp / boundary-impulse
      stressors; on a gapless full-session fragment the valid bins equal the pinned rev-2 core's
      corresponding samples.
* support_digest(fragments, *, session_start_utc, session_seconds, out_rate_hz, source_id) -> hex
    - canonical SHA-256 over the ordered fragment records (start_iso, exclusive_end_iso, rate,
      npts, source_id, raw_sha256) + the session grid identity (start_iso, seconds, out_rate)
      + the derived valid-mask bytes. Any change to raw bytes, rate, npts, a gap endpoint, the
      source ID, the session phase, or the mask changes the digest.
* build_envelope_cache_identity v3: gains `support_sha256` (an 11th field; all ten v2 fields keep
  participating); `_cache_path` moves when it changes; `_save_cached_series`/`_load_cached_series`
  store and verify it — a support-mutated expected identity MISSES on load.
* get_segment_envelopes (SHELL): builds one fragment per contiguous raw trace, passing the
  ORIGINAL arrays/rates/starts to compute_band_envelope_supported exactly once per stream and
  returning ITS EnvelopeSeries objects. The shell makes NO method call on the stream object
  (`stream.calls == []` on the recording double): no merge/detrend/trim/taper/filter/decimate/
  resample — trace data/stats are READ only.

fault_correlation.py rev-3:
* station_eligible(series) -> bool: validate_support ok AND coverage >= 0.50 (0.50 eligible).
* aggregate_segment_supported(station_series_list) -> EnvelopeSeries | None
    - counts DISTINCT `NET.STA` station keys: a per-bin aggregate value is VALID only where >= 2
      DISTINCT NET.STA are concurrently valid (two channels/locations of one NET.STA are ONE
      station); median at each valid bin over the stations valid there; truthful derived
      mask/gaps/coverage (never gaps=[] merely because aggregation succeeded); station order
      irrelevant; None when no bin has 2-distinct-station support;
    - FRAME GUARD: mismatched start_utc / dt_seconds / requested_grid_count / band_tag /
      processing_version across inputs -> None (unavailable), NEVER index-aligned.
* compute_correlation_matrix_supported(segment_series, names) -> (C | None, names, qc)
    - AND-mask common support of ALL independently eligible segments; >= 2 segments AND common
      samples >= MIN_COMMON_SUPPORT_FRACTION x requested_grid_count; eligible set determined by
      eligibility ALONE (a failing common support FAILS with all eligible names — no subset
      search); correlations computed ONLY over common-support bins (invalid-bin filler mutations
      leave C/names/qc bit-identical); same FRAME GUARD as aggregation.
* FaultCorrelationMonitor.compute_correlation_matrix (the PRODUCTION path) routes through
  aggregate_segment_supported + compute_correlation_matrix_supported for supported inputs — the
  legacy index-truncation aggregate (`_aggregate_segment_envelope`/`compute_segment_activity_index`)
  is never invoked on this path (poisoned in SS-10).
* ALL existing gates unchanged (topology, provenance, non-degeneracy, PSD/unit-diagonal,
  calibration binding, registry, owner gates).

Fixture policy (codex 1712): observed archive support patterns calibrate fixture SHAPE only
(23/67-gap days, zero-eligible day); synthetic signals; no measured amplitudes or ratios.

RED AS AUTHORED (rev-3 seams absent from the pinned `54cea7b` modules).
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import hilbert as sp_hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def _install_obspy_stub():
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


# ---------------------------------------------------------------- fixtures --
FS = 40.0
UTC0 = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
SESSION = 86400


def am_sig(seconds, fs, seed=0):
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    ph = np.random.default_rng(seed).uniform(0, 6.28)
    return (1.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t + ph)) * np.sin(2 * np.pi * 5.0 * t)


def broadband_sig(seconds, fs, seed=1):
    return np.random.default_rng(seed).normal(0.0, 1.0, int(round(seconds * fs)))


def step_chirp_sig(seconds, fs, seed=2):
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = sp_signal.chirp(t, f0=1.0, f1=10.0, t1=seconds) * 0.7
    x[n // 2:] += 1.5                                   # amplitude step at midpoint
    return x


def impulse_sig(seconds, fs, at_second, seed=3):
    x = broadband_sig(seconds, fs, seed) * 0.05
    x[int(round(at_second * fs))] = 50.0
    return x


def gap_day_fragments(rng, n_gaps, gap_seconds=120, fs=FS):
    usable = SESSION - n_gaps * gap_seconds
    span = usable / (n_gaps + 1)
    frags, cursor = [], 0.0
    for _ in range(n_gaps + 1):
        frags.append((am_sig(span, fs, seed=int(cursor) % 97), fs,
                      UTC0 + timedelta(seconds=cursor)))
        cursor += span + gap_seconds
    return frags


def _mask_series(SD, mask, start=UTC0, values=None, source="XX.AAA..BHZ", dt=1.0,
                 band="1-10Hz", version=None, grid=None):
    n = mask.size
    vals = values if values is not None else np.where(mask, 1.0, 0.0)
    gaps = []
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gaps.append(((start + timedelta(seconds=i * dt)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         (j - i) * dt))
            i = j
        else:
            i += 1
    return SD.EnvelopeSeries(values=np.asarray(vals, dtype=float), start_utc=start,
                             dt_seconds=dt, coverage=float(mask.sum()) / n, gaps=gaps,
                             source_ids=[source], band_tag=band,
                             processing_version=(version or SD.PROCESSING_VERSION),
                             pre_envelope_rate_hz=40.0, valid_mask=mask.copy(),
                             requested_grid_count=(grid or n))


class _FragTrace:
    def __init__(self, data, rate, start):
        self.data = data
        self.stats = types.SimpleNamespace(sampling_rate=rate, starttime=start)


class _FragStream:
    def __init__(self, frags):
        self.calls = []
        self._traces = [_FragTrace(d, r, s) for d, r, s in frags]

    def __getattr__(self, name):
        def _rec(*a, **k):
            self.calls.append(name)
            return self
        return _rec

    def __getitem__(self, i):
        return self._traces[i]

    def __len__(self):
        return len(self._traces)

    def __iter__(self):
        return iter(self._traces)


def main():
    _install_obspy_stub()
    import seismic_data as SD
    import fault_correlation as FC

    # ---- gates -----------------------------------------------------------
    need_sd = ("compute_band_envelope_supported", "validate_support", "support_digest",
               "SEG_SUPPORT")
    if not all(hasattr(SD, n) for n in need_sd):
        check("SS-0a seismic_data rev-3 seams present", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")
        return
    try:
        _p = SD.EnvelopeSeries(values=np.zeros(4), start_utc=UTC0, dt_seconds=1.0, coverage=1.0,
                               gaps=[], source_ids=["XX.AAA..BHZ"], band_tag="1-10Hz",
                               processing_version=SD.PROCESSING_VERSION,
                               pre_envelope_rate_hz=40.0,
                               valid_mask=np.ones(4, dtype=bool), requested_grid_count=4)
        ok_fields = hasattr(_p, "valid_mask") and hasattr(_p, "requested_grid_count")
    except Exception:
        ok_fields = False
    if not ok_fields:
        check("SS-0a2 EnvelopeSeries carries valid_mask + requested_grid_count", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")
        return
    need_fc = ("station_eligible", "aggregate_segment_supported",
               "compute_correlation_matrix_supported")
    if not all(hasattr(FC, n) for n in need_fc):
        check("SS-0b fault_correlation rev-3 seams present", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")
        return

    K = SD.SEG_SUPPORT
    check("SS-0c frozen constants exact (operator-derived; derivation in this bar's docstring)",
          K.get("edge_trim_seconds") == 90 and K.get("min_contiguous_span_seconds") == 240
          and K.get("min_common_support_fraction") == 0.50
          and K.get("station_coverage_floor") == 0.50
          and K.get("edge_effect_rel_tol") == 2e-4
          and 0.50 * 86400 == 43200, f"SEG_SUPPORT={K}")

    rng = np.random.default_rng(11)

    # ---- LOCK 1: gapless regression + EXACT half-open mask ---------------
    day = am_sig(SESSION, FS, seed=7)
    es = SD.compute_band_envelope_supported([(day, FS, UTC0)], session_start_utc=UTC0,
                                            source_id="XX.AAA..BHZ")
    core = SD.compute_band_envelope_from_array(day, FS, start_utc=UTC0, source_id="XX.AAA..BHZ")
    exp_mask = np.zeros(SESSION, dtype=bool)
    exp_mask[90:86310] = True
    core_vals = np.asarray(core.values)
    m = min(core_vals.size, SESSION)
    check("SS-1 GAPLESS: mask is EXACTLY [90, 86310) (86,220 bins), coverage 86220/86400, and "
          "valid bins equal the pinned rev-2 core at those bins",
          es.values.size == SESSION and np.array_equal(es.valid_mask, exp_mask)
          and es.coverage == 86220 / 86400
          and np.allclose(es.values[:m][exp_mask[:m]], core_vals[:m][exp_mask[:m]],
                          rtol=1e-6, atol=1e-9),
          f"n={es.values.size} valid={int(es.valid_mask.sum())} cov={es.coverage}")

    # ---- LOCK 2: impulse adjacent to a gap cannot leak across it ---------
    span_a = am_sig(3600, FS, seed=8)
    span_b = am_sig(3600, FS, seed=9)
    frags_base = [(span_a, FS, UTC0), (span_b, FS, UTC0 + timedelta(seconds=3900))]
    spiked = span_a.copy()
    spiked[-1] = 1e6
    frags_spike = [(spiked, FS, UTC0), (span_b, FS, UTC0 + timedelta(seconds=3900))]
    sess = 7500
    e_base = SD.compute_band_envelope_supported(frags_base, session_start_utc=UTC0,
                                                session_seconds=sess, source_id="XX.AAA..BHZ")
    e_spike = SD.compute_band_envelope_supported(frags_spike, session_start_utc=UTC0,
                                                 session_seconds=sess, source_id="XX.AAA..BHZ")
    exp2 = np.zeros(sess, dtype=bool)
    exp2[90:3510] = True
    exp2[3990:7410] = True
    sel = np.zeros(sess, dtype=bool)
    sel[3990:7410] = True
    check("SS-2 GAP ISOLATION: exact per-span masks ([90,3510) + [3990,7410)); a 1e6 impulse at "
          "span-A's gap edge leaves span-B's valid bins BIT-IDENTICAL",
          np.array_equal(e_base.valid_mask, exp2) and np.array_equal(e_spike.valid_mask, exp2)
          and np.array_equal(e_base.values[sel], e_spike.values[sel]),
          f"valid={int(e_base.valid_mask.sum())} exp={int(exp2.sum())}")

    # ---- LOCK 3: minimums fail closed, EXACT bins -------------------------
    short = SD.compute_band_envelope_supported([(am_sig(239, FS), FS, UTC0)],
                                               session_start_utc=UTC0, session_seconds=600,
                                               source_id="XX.AAA..BHZ")
    okmin = SD.compute_band_envelope_supported([(am_sig(240, FS), FS, UTC0)],
                                               session_start_utc=UTC0, session_seconds=600,
                                               source_id="XX.AAA..BHZ")
    exp3 = np.zeros(600, dtype=bool)
    exp3[90:150] = True
    check("SS-3a 239 s span -> ZERO valid bins; 240 s span -> EXACTLY [90, 150) (60 bins)",
          int(short.valid_mask.sum()) == 0 and np.array_equal(okmin.valid_mask, exp3),
          f"short={int(short.valid_mask.sum())} okmin={int(okmin.valid_mask.sum())}")
    n = 1000
    m_half = np.zeros(n, dtype=bool)
    m_half[:500] = True
    m_less = m_half.copy()
    m_less[499] = False
    check("SS-3b station floor boundary: coverage 0.500 ELIGIBLE, 0.499 NOT",
          FC.station_eligible(_mask_series(SD, m_half)) is True
          and FC.station_eligible(_mask_series(SD, m_less)) is False)

    # ---- LOCK 4: contradictions; expressible digest; REAL cache load ------
    import dataclasses as _dc

    def _with(series, **kw):
        try:
            return _dc.replace(series, **kw)
        except Exception:
            for k2, v2 in kw.items():
                object.__setattr__(series, k2, v2)
            return series

    bad = _with(_mask_series(SD, m_half), coverage=0.9)
    ok4, reasons4 = SD.validate_support(bad)
    bad2 = _with(_mask_series(SD, m_half), gaps=[])
    ok4b, _ = SD.validate_support(bad2)
    good4, _ = SD.validate_support(_mask_series(SD, m_half))
    check("SS-4a mask/coverage + mask/gaps contradictions FAIL validate_support; honest passes",
          ok4 is False and ok4b is False and good4 is True, f"reasons={reasons4}")

    fA = am_sig(3600, FS, seed=12)
    fB = am_sig(3600, FS, seed=13)
    base_kw = dict(session_start_utc=UTC0, session_seconds=SESSION, out_rate_hz=1.0,
                   source_id="XX.AAA..BHZ")
    frags0 = [(fA, FS, UTC0), (fB, FS, UTC0 + timedelta(seconds=3720))]
    d0 = SD.support_digest(frags0, **base_kw)
    fA_mut = fA.copy()
    fA_mut[100] += 1.0
    variants = {
        "raw bytes": SD.support_digest([(fA_mut, FS, UTC0),
                                        (fB, FS, UTC0 + timedelta(seconds=3720))], **base_kw),
        "rate": SD.support_digest([(fA, 100.0, UTC0),
                                   (fB, FS, UTC0 + timedelta(seconds=3720))], **base_kw),
        "npts": SD.support_digest([(fA[:-1], FS, UTC0),
                                   (fB, FS, UTC0 + timedelta(seconds=3720))], **base_kw),
        "gap endpoint": SD.support_digest([(fA, FS, UTC0),
                                           (fB, FS, UTC0 + timedelta(seconds=3721))], **base_kw),
        "source id": SD.support_digest(frags0, **{**base_kw, "source_id": "YY.BBB..HHZ"}),
        "session phase": SD.support_digest(
            frags0, **{**base_kw,
                       "session_start_utc": UTC0 - timedelta(microseconds=500000)}),
    }
    moved = {k: (v != d0) for k, v in variants.items()}
    check("SS-4b support_digest: raw-bytes / rate / npts / gap-endpoint / source-id / "
          "session-phase mutations EACH change the digest",
          len(d0) == 64 and all(moved.values()),
          f"unmoved={[k for k, ok in moved.items() if not ok]}")

    ident_kw = dict(region="ridgecrest", segment="syn_seg", source_id="XX.AAA..BHZ",
                    start_utc=UTC0, end_utc=UTC0 + timedelta(seconds=SESSION),
                    band_tag="1-10Hz", processing_version=SD.PROCESSING_VERSION,
                    topology_version="t1",
                    input_sha256=hashlib.sha256(fA.tobytes()).hexdigest())
    id_base = SD.build_envelope_cache_identity(native_rate_hz=40.0, support_sha256=d0, **ident_kw)
    id_supp = SD.build_envelope_cache_identity(native_rate_hz=40.0,
                                               support_sha256=variants["gap endpoint"], **ident_kw)
    f2 = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    f2.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    p_base = f2._cache_path("ridgecrest", UTC0, UTC0 + timedelta(seconds=SESSION), "seg_env",
                            id_base)
    p_supp = f2._cache_path("ridgecrest", UTC0, UTC0 + timedelta(seconds=SESSION), "seg_env",
                            id_supp)
    check("SS-4c support_sha256 participates in the cache KEY (gap-endpoint move -> new path)",
          p_base != p_supp)
    series_ok = _mask_series(SD, np.ones(SESSION, dtype=bool)[:1000])
    f2._save_cached_series(p_base, id_base, series_ok)
    hit = f2._load_cached_series(p_base, id_base)
    miss = f2._load_cached_series(p_base, id_supp)
    check("SS-4d REAL cache load: matching support identity round-trips; support-mutated "
          "expected identity MISSES (None)", hit is not None and miss is None)

    # ---- LOCK 5/6: seam-level invariance + support rules (kept from rev 1) --
    def seg_matrix(filler, station_order):
        base = np.ones(n, dtype=bool)
        base[100:200] = False
        m2 = np.ones(n, dtype=bool)
        m2[700:820] = False
        v = rng2.normal(1.0, 0.3, (4, n))
        srcs = ["XX.S0..BHZ", "XX.S1..BHZ", "XX.S2..BHZ", "XX.S3..BHZ"]
        series = []
        for i, mask in enumerate((base, m2, base, m2)):
            vals = np.where(mask, v[i], filler)
            series.append(_mask_series(SD, mask, values=vals, source=srcs[i]))
        segA = FC.aggregate_segment_supported([series[j] for j in station_order[0]])
        segB = FC.aggregate_segment_supported([series[j] for j in station_order[1]])
        C, names, qc = FC.compute_correlation_matrix_supported([segA, segB], ["A", "B"])
        return C

    rng2 = np.random.default_rng(23)
    c_ref = seg_matrix(0.0, ([0, 1], [2, 3]))
    rng2 = np.random.default_rng(23)
    c_fill = seg_matrix(999.0, ([0, 1], [2, 3]))
    rng2 = np.random.default_rng(23)
    c_perm = seg_matrix(0.0, ([1, 0], [3, 2]))
    check("SS-5 SEAM INVARIANCE: invalid-bin filler (0 vs 999) and station order change NOTHING "
          "(bit-identical C)", c_ref is not None and np.array_equal(c_ref, c_fill)
          and np.array_equal(c_ref, c_perm))

    mA = np.zeros(n, dtype=bool)
    mA[0:600] = True
    mB = np.zeros(n, dtype=bool)
    mB[400:1000] = True
    agg = FC.aggregate_segment_supported([_mask_series(SD, mA, source="XX.P..BHZ"),
                                          _mask_series(SD, mB, source="XX.Q..BHZ")])
    check("SS-6a per-bin 2-station rule: aggregate valid EXACTLY on [400, 600); gaps truthful",
          agg is not None and int(agg.valid_mask.sum()) == 200
          and bool(agg.valid_mask[400]) and bool(agg.valid_mask[599])
          and not agg.valid_mask[399] and not agg.valid_mask[600] and len(agg.gaps) > 0)
    single = FC.aggregate_segment_supported([_mask_series(SD, mA, source="XX.P..BHZ")])
    check("SS-6b a single station has no 2-station bins -> unavailable (None)", single is None)
    sA6 = _mask_series(SD, m_half, values=np.where(m_half, rng.normal(1, .3, n), 0.0),
                       source="XX.P..BHZ")
    mC = np.zeros(n, dtype=bool)
    mC[200:700] = True
    sB6 = _mask_series(SD, mC, values=np.where(mC, rng.normal(1, .3, n), 0.0),
                       source="XX.Q..BHZ")
    C6, _, qc6 = FC.compute_correlation_matrix_supported([sA6, sB6], ["A", "B"])
    check("SS-6c AND-mask common support below the 0.50 session fraction -> (None, reasons)",
          C6 is None and len(qc6) > 0, f"qc={qc6}")

    # ---- LOCK 10 (codex #1): PRODUCTION composition through the monitor ---
    from fault_segments import FaultSegment, SeismicStation

    def _mk_seg(nm, stas):
        return FaultSegment(name=nm, region="syn_region",
                            stations=[SeismicStation("XX", s, 0.0, 0.0) for s in stas],
                            polygon=[(0, 0), (0, 1), (1, 1), (1, 0)],
                            strike=0.0, dip=90.0, rake=0.0)

    segs3 = [_mk_seg("segA", ["A1", "A2"]), _mk_seg("segB", ["B1", "B2"]),
             _mk_seg("segC", ["C1", "C2"])]

    def _seg_masks_series(seg_mask, stas, filler, seed):
        r = np.random.default_rng(seed)
        out = {}
        for s in stas:
            vals = np.where(seg_mask, r.normal(1.0, 0.3, n), filler)
            out[f"XX.{s}"] = _mask_series(SD, seg_mask, values=vals, source=f"XX.{s}..BHZ")
        return out

    def production_run(seg_masks, filler, poison_log):
        class _Fetcher:
            def get_segment_envelopes(self, segment, start, end, use_cache=True):
                idx = [s.name for s in segs3].index(segment.name)
                return _seg_masks_series(seg_masks[idx],
                                         [st.code for st in segs3[idx].stations], filler,
                                         seed=100 + idx)

        mon = FC.FaultCorrelationMonitor(data_fetcher=_Fetcher())
        real_get = FC.get_segments_for_region
        FC.get_segments_for_region = lambda region: segs3
        legacy_aggregate = FC._aggregate_segment_envelope

        def _boom_aggregate(*args, **kwargs):
            poison_log.append("_aggregate_segment_envelope")
            raise AssertionError("legacy aggregate path invoked")

        FC._aggregate_segment_envelope = _boom_aggregate
        legacy_fn = getattr(SD, "compute_segment_activity_index", None)
        if legacy_fn is not None:
            SD.compute_segment_activity_index = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("compute_segment_activity_index invoked"))
        try:
            return mon.compute_correlation_matrix("syn_region", UTC0,
                                                  UTC0 + timedelta(seconds=n))
        finally:
            FC._aggregate_segment_envelope = legacy_aggregate
            FC.get_segments_for_region = real_get
            if legacy_fn is not None:
                SD.compute_segment_activity_index = legacy_fn

    mgood = np.ones(n, dtype=bool)
    mgood[10:60] = False                                # honest 5% loss, well above floors
    poison = []
    try:
        C_a = production_run([mgood, mgood, mgood], 0.0, poison)
        C_b = production_run([mgood, mgood, mgood], 999.0, poison)
        ok10a = (C_a[0] is not None and C_b[0] is not None
                 and np.array_equal(C_a[0], C_b[0]) and C_a[1] == C_b[1] and C_a[2] == C_b[2]
                 and not poison)
        check("SS-10a PRODUCTION: FaultCorrelationMonitor.compute_correlation_matrix consumes "
              "supported series; filler 0 vs 999 -> C/names/QC BIT-IDENTICAL; legacy aggregate "
              "path NEVER invoked", ok10a, f"poisoned={poison}")
    except Exception as exc:
        check("SS-10a PRODUCTION: FaultCorrelationMonitor.compute_correlation_matrix consumes "
              "supported series; filler 0 vs 999 -> C/names/QC BIT-IDENTICAL; legacy aggregate "
              "path NEVER invoked", False, f"RAISED {type(exc).__name__}: {exc}")
    try:
        class _BadFetcher:
            def get_segment_envelopes(self, segment, start, end, use_cache=True):
                s = _mask_series(SD, mgood, source="XX.Z1..BHZ")
                s = _with(s, coverage=0.123)            # contradictory supported series
                return {"XX.Z1": s,
                        "XX.Z2": _mask_series(SD, mgood, source="XX.Z2..BHZ")}

        mon_bad = FC.FaultCorrelationMonitor(data_fetcher=_BadFetcher())
        real_get = FC.get_segments_for_region
        FC.get_segments_for_region = lambda region: segs3[:2]
        try:
            Cbad, nbad, qbad = mon_bad.compute_correlation_matrix("syn_region", UTC0,
                                                                  UTC0 + timedelta(seconds=n))
        finally:
            FC.get_segments_for_region = real_get
        check("SS-10b PRODUCTION: a contradictory supported series -> (None, ..., reasons)",
              Cbad is None and len(qbad) > 0, f"qc={qbad}")
    except Exception as exc:
        check("SS-10b PRODUCTION: a contradictory supported series -> (None, ..., reasons)",
              False, f"RAISED {type(exc).__name__}: {exc}")
    try:
        mA3 = np.zeros(n, dtype=bool)
        mA3[0:700] = True                               # 70%
        mB3 = np.zeros(n, dtype=bool)
        mB3[200:900] = True                             # 70%; A^B = [200,700) = 50%
        mC3 = np.zeros(n, dtype=bool)
        mC3[450:1000] = True                            # 55%; A^B^C = [450,700) = 25%
        poison3 = []
        C3, n3, q3 = production_run([mA3, mB3, mC3], 0.0, poison3)
        check("SS-10c PRODUCTION anti-subset-search: A/B pairwise support 50% but A^B^C 25% -> "
              "FAIL carrying ALL THREE eligible names (no passing-pair hunt)",
              C3 is None and set(n3) == {"segA", "segB", "segC"} and len(q3) > 0 and not poison3,
              f"names={n3} qc={q3}")
    except Exception as exc:
        check("SS-10c PRODUCTION anti-subset-search: A/B pairwise support 50% but A^B^C 25% -> "
              "FAIL carrying ALL THREE eligible names (no passing-pair hunt)",
              False, f"RAISED {type(exc).__name__}: {exc}")

    # ---- LOCK 11 (codex #2): station identity + frame guards --------------
    two_ch = FC.aggregate_segment_supported(
        [_mask_series(SD, mA, source="XX.AAA..BHZ"),
         _mask_series(SD, mA, source="XX.AAA..HHZ")])   # SAME NET.STA, two channels
    check("SS-11a two channels of ONE NET.STA never satisfy the two-station rule -> None",
          two_ch is None)
    frame_cases = {
        "start_utc +0.5 s": dict(start=UTC0 + timedelta(microseconds=500000)),
        "start_utc +6 min": dict(start=UTC0 + timedelta(seconds=360)),
        "dt 2.0": dict(dt=2.0),
        "grid count": dict(grid=n + 10),
        "band": dict(band="0.01-1Hz"),
        "processing version": dict(version="OLD-V0"),
    }
    agg_fails, corr_fails = [], []
    for label, kw in frame_cases.items():
        other = _mask_series(SD, mB, source="XX.Q..BHZ", **kw)
        a_res = FC.aggregate_segment_supported(
            [_mask_series(SD, mA, source="XX.P..BHZ"), other])
        if a_res is not None:
            agg_fails.append(label)
        c_res, _, qc_res = FC.compute_correlation_matrix_supported(
            [_mask_series(SD, mA, source="XX.P..BHZ"), other], ["A", "B"])
        if not (c_res is None and len(qc_res) > 0):
            corr_fails.append(label)
    check("SS-11b FRAME GUARD at aggregation: start(+0.5s/+6min)/dt/grid/band/version mismatch "
          "each -> unavailable (never index-aligned)", not agg_fails, f"passed_bad={agg_fails}")
    check("SS-11c FRAME GUARD at correlation: same six mismatches each -> (None, reasons)",
          not corr_fails, f"passed_bad={corr_fails}")
    def _cont_sig(t):
        """Shared continuous synthetic signal (5 Hz carrier, 0.02 Hz AM) sampled at times t."""
        return (1.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t + 1.234)) * np.sin(2 * np.pi * 5.0 * t)

    def _anti_relabel(rate, frac_off, lock_id):
        long_sec = 1200
        t_ref = np.arange(int(round(long_sec * rate))) / rate
        ref = SD.compute_band_envelope_supported(
            [(_cont_sig(t_ref), rate, UTC0)], session_start_utc=UTC0,
            session_seconds=long_sec, source_id="XX.AAA..BHZ")
        f_start = 300.0 + frac_off
        n_frag = int(round(600 * rate))
        t_frag = f_start + np.arange(n_frag) / rate
        frag = SD.compute_band_envelope_supported(
            [(_cont_sig(t_frag), rate, UTC0 + timedelta(seconds=f_start))],
            session_start_utc=UTC0, session_seconds=long_sec, source_id="XX.AAA..BHZ")
        exp_m = np.zeros(long_sec, dtype=bool)
        exp_m[391:811] = True                    # 300+off+90 <= t_k < 900+off-90, off in (0,1)
        mask_ok = np.array_equal(frag.valid_mask, exp_m)
        both = exp_m & ref.valid_mask
        if mask_ok and int(both.sum()) == 420:
            r = ref.values[both]
            rel = float(np.max(np.abs(frag.values[both] - r)) / max(np.max(np.abs(r)), 1e-12))
        else:
            rel = float("inf")
        tol = float(K["edge_effect_rel_tol"])
        check(f"{lock_id} ANTI-RELABEL @ +{frac_off} s, {rate} Hz: timestamp-derived mask EXACTLY "
              f"{{391..810}} (420 bins) and exact-time values match the full-carrier reference at "
              f"IDENTICAL grid instants within {tol:g} (nearest-grid rounding fails visibly)",
              mask_ok and rel <= tol,
              f"mask_ok={mask_ok} valid={int(frag.valid_mask.sum())} rel={rel:.2e}")

    _anti_relabel(40.0, 0.5, "SS-11d")           # the ambiguous half-second phase
    _anti_relabel(40.5, 0.948447, "SS-11e")      # published-style microfractional phase

    # ---- LOCK 9 (codex #4): operator-reference edge-error contract --------
    TOL = float(K["edge_effect_rel_tol"])
    stressors = {"am": am_sig, "broadband": broadband_sig, "step_chirp": step_chirp_sig,
                 "impulse": lambda sec, fs: impulse_sig(sec, fs, at_second=299.0)}
    worst = 0.0
    fail9 = []
    for rate in (25.0, 40.0, 40.5, 100.0):
        for nm, fn in stressors.items():
            long_sec = 1200
            x_long = fn(long_sec, rate)
            i0, i1 = int(round(300 * rate)), int(round(900 * rate))
            span = x_long[i0:i1]
            e_alone = SD.compute_band_envelope_supported(
                [(span, rate, UTC0 + timedelta(seconds=300))],
                session_start_utc=UTC0, session_seconds=long_sec, source_id="XX.AAA..BHZ")
            e_embed = SD.compute_band_envelope_supported(
                [(x_long, rate, UTC0)],
                session_start_utc=UTC0, session_seconds=long_sec, source_id="XX.AAA..BHZ")
            selv = np.zeros(long_sec, dtype=bool)
            selv[390:810] = True                        # span-alone trimmed interior
            both = selv & e_alone.valid_mask & e_embed.valid_mask
            if int(both.sum()) != 420:
                fail9.append(f"{nm}@{rate}:support={int(both.sum())}")
                continue
            ref = e_embed.values[both]
            rel = float(np.max(np.abs(e_alone.values[both] - ref)) / max(np.max(np.abs(ref)),
                                                                         1e-12))
            worst = max(worst, rel)
            if rel > TOL:
                fail9.append(f"{nm}@{rate}:rel={rel:.2e}")
    check("SS-9 OPERATOR-REFERENCE edge contract: span-alone vs embedded trimmed interiors agree "
          "within EDGE_EFFECT_REL_TOL=2e-4 at 25/40/40.5/100 Hz under am/broadband/step_chirp/"
          "impulse stressors (KAT bound, not a theorem)", not fail9,
          f"failures={fail9} worst={worst:.2e}")

    # ---- LOCK 7 (composition + shape fixtures, strengthened per codex #3) --
    seg7 = _mk_seg("syn_seg", ["AAA"])
    for n_gaps in (23, 67):
        frags = gap_day_fragments(np.random.default_rng(n_gaps), n_gaps)
        stream = _FragStream(frags)
        fetcher = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
        fetcher.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
        fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": stream}
        got = []
        real_sup = SD.compute_band_envelope_supported
        sentinel = real_sup(frags, session_start_utc=UTC0, source_id="XX.AAA..BHZ")

        def spy(fr, **kw):
            got.append(list(fr))
            return sentinel

        SD.compute_band_envelope_supported = spy
        err = None
        try:
            out = fetcher.get_segment_envelopes(seg7, UTC0, UTC0 + timedelta(seconds=SESSION),
                                                use_cache=False)
        except Exception as e:
            out, err = None, e
        finally:
            SD.compute_band_envelope_supported = real_sup
        frag_ok = (len(got) == 1 and len(got[0]) == n_gaps + 1
                   and all(np.shares_memory(np.asarray(g[0]), f[0]) or
                           np.array_equal(np.asarray(g[0]), f[0])
                           for g, f in zip(got[0], frags))
                   and all(float(g[1]) == float(f[1]) and g[2] == f[2]
                           for g, f in zip(got[0], frags)))
        check(f"SS-7a COMPOSITION ({n_gaps}-gap day): shell passes the ORIGINAL {n_gaps + 1} "
              "arrays/rates/starts to the core once, returns ITS series, and makes ZERO stream "
              "method calls (stream.calls == [])",
              out is not None and frag_ok and stream.calls == []
              and isinstance(out, dict) and out.get("XX.AAA") is sentinel,
              f"calls={stream.calls} frag_ok={frag_ok} err={err}")
        exp_cov_gap = 1.0 - (n_gaps * 120 + (n_gaps + 1) * 2 * 90) / SESSION
        check(f"SS-7b {n_gaps}-gap day: truthful coverage (~{exp_cov_gap:.3f}) with >= {n_gaps} "
              "gap records and NEVER gaps=[]",
              abs(sentinel.coverage - exp_cov_gap) < 0.02 and len(sentinel.gaps) >= n_gaps,
              f"cov={sentinel.coverage:.3f} gaps={len(sentinel.gaps)}")
    low = np.zeros(n, dtype=bool)
    low[:100] = True
    zero_day = [_mask_series(SD, low, source=f"XX.Z{i}..BHZ") for i in range(4)]
    check("SS-7c zero-eligible-segment day: station_eligible False for every station",
          all(FC.station_eligible(s) is False for s in zero_day))

    # ---- LOCK 8: existing suites remain green ------------------------------
    suites = [
        os.path.join(HERE, "test_d2_reband_redkats_cayley.py"),
        os.path.join(HERE, "test_d2_livecarrier_fixes_grassmann.py"),
        os.path.join(REPO, "tests", "test_publication_receipt_redkats_cayley.py"),
        os.path.join(HERE, "test_r4_receipt_gate_redkats_cayley.py"),
        os.path.join(REPO, "tests", "test_build_daily_receipt_redkats_cayley.py"),
        os.path.join(HERE, "test_r4_prospective_scorer.py"),
    ]
    for path in suites:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=600,
                           cwd=REPO)
        check(f"SS-8 regression green: {os.path.basename(path)}", r.returncode == 0,
              (r.stdout + r.stderr)[-300:])


main()
print()
if FAILS:
    print(f"D2 SEGMENTED-SUPPORT RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 SEGMENTED-SUPPORT RED-KATs PASS (valid-support carrier enforced through production)")
