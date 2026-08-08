#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 SEGMENTED-SUPPORT red-KATs (cayley, 2026-08-08) — phase
`codex-d2-segmented-support-2026-08-08-v1` (codex 1712 assignment, `27e227b`), pinned starting
implementation GeoSpec `54cea7b`. Grassmann implements UNEDITED after this bar is routed; codex one
closing verification. NOTHING here lifts a freeze, admits calibration, or supports a claim; the
historical incident/control replay through this enhancement is a LATER diagnostic run.

THE PROBLEM THIS PHASE SOLVES: fix-#2's interim any-gap-skip rule is empirically too strict for
real archives (step-4 HOLD: 23–67 gaps per station-day blocked every incident replay). The repair
is NOT "compute each side of a gap" — a dense value array plus decorative gap metadata could still
feed fabricated gap samples into a correlation. This bar freezes an EXPLICIT VALID-SUPPORT CARRIER
and makes every downstream operation honor it.

FROZEN CONSTANTS (delegated to cayley by codex 1712; derived ONLY from the fixed operator chain and
the 1 Hz output grid — never from incident/control amplitudes, ratios, labels, or region outcomes):
  * EDGE_TRIM_SECONDS = 90
      Derivation: per-stage settle = 10 cycles of the stage's LOWEST cutoff, doubled for zerophase
      (forward+backward) application, summed sequentially: bandpass 1–10 Hz Butterworth ord-4
      zerophase → 2×10/1.0 = 20 s; FFT-Hilbert edge ≈ 3 longest periods → 3/1.0 = 3 s; envelope
      anti-alias low-pass (≈0.4 Hz, zerophase) → 2×10/0.4 = 50 s; total 73 s worst-case → rounded
      up to 90 s margin. Only output bins whose complete DSP support lies inside the trimmed span
      interior are valid.
  * MIN_CONTIGUOUS_SPAN_SECONDS = 240
      Derivation: 2 × EDGE_TRIM (both edges) + 60 s minimum valid interior (one minute of 1 Hz
      output = the smallest round unit exceeding the operator support window). A raw span shorter
      than this contributes ZERO valid bins (the whole span is rejected, not partially admitted).
  * MIN_COMMON_SUPPORT_FRACTION = 0.50 of the requested session grid
      (0.50 × 86,400 = 43,200 one-second samples for the 24-hour session — codex's constant,
      expressed as the session-scaled fraction; asserted equal below).
  * STATION_COVERAGE_FLOOR = 0.50 (the existing frozen floor, unchanged.)

CONTRACT (grassmann implements to THIS, unedited — the decouple)
=================================================================
seismic_data.py rev-3 (segmented support):
* EnvelopeSeries gains `valid_mask` (bool array, SAME length as `values`) and
  `requested_grid_count: int`. `values` is DENSE over the requested session grid
  (len == requested_grid_count); invalid bins carry filler with NO scientific meaning.
  `coverage == count(valid_mask)/requested_grid_count` EXACTLY; `gaps` is the EXACT run-length
  complement of `valid_mask` on the session grid (list of (iso_utc, seconds)).
* validate_support(series) -> (ok: bool, reasons: list[str]) — any mask/gaps/coverage
  contradiction, length mismatch, or non-bool mask fails; align/observability paths MUST consult
  it (a contradiction is UNAVAILABLE downstream, never repaired).
* SEG_SUPPORT = {"edge_trim_seconds": 90, "min_contiguous_span_seconds": 240,
  "min_common_support_fraction": 0.50, "station_coverage_floor": 0.50} (module constant).
* compute_band_envelope_supported(fragments, *, session_start_utc, session_seconds=86400,
  out_rate_hz=1.0, freqmin=1.0, freqmax=10.0, min_rate_hz=25.0, source_id) -> EnvelopeSeries
    - fragments: ascending, NON-overlapping list of (data: ndarray, rate_hz: float,
      start_utc: aware-UTC datetime) contiguous raw spans (overlap/naive-start/low-rate →
      DataUnavailable, fail closed);
    - runs native-rate bandpass → Hilbert envelope → anti-aliased resample INDEPENDENTLY inside
      each span; NEVER merges/filters/Hilbert-transforms/resamples/interpolates/pads ACROSS a gap;
    - trims EDGE_TRIM_SECONDS from each span end; spans < MIN_CONTIGUOUS_SPAN_SECONDS contribute
      nothing; outputs anchor to the requested session's 1 Hz UTC grid (a span that cannot map
      without borrowing across a gap contributes nothing — phase is never erased by index
      truncation);
    - the returned series carries the true mask/gaps/coverage; on a gapless full-session fragment
      the VALID bins equal the pinned rev-2 core's output at those bins (regression equivalence).
* build_envelope_cache_identity v3: gains `support_sha256` — the digest of the ordered fragment
  timing/support structure ((start_utc_iso, exclusive_end_iso, rate, sample_count, source_id,
  raw_sha256) per fragment + the resulting valid_mask bytes), via a new seam
  support_digest(fragments) -> hex. A changed gap endpoint (any fragment bound moved by ≥ one
  sample) is a DIFFERENT identity → cache miss. All ten v2 fields keep participating.
* get_segment_envelopes (the SHELL): extracts the per-trace fragment list from each fetched
  stream (one fragment per contiguous trace; NO merge with fill, NO DSP on the stream object) and
  calls compute_band_envelope_supported exactly once per stream, RETURNING the core's own
  EnvelopeSeries objects (sentinel identity). Legacy any-gap-skip is retired on this path.

fault_correlation.py rev-3:
* station_eligible(series) -> bool: validate_support ok AND coverage >= STATION_COVERAGE_FLOOR
  (0.50 exactly is eligible).
* aggregate_segment_supported(station_series_list) -> EnvelopeSeries | None
    - a per-bin aggregate value is VALID only where >= 2 DISTINCT stations are concurrently valid;
      the median at each valid bin uses ONLY the stations valid at that bin; the derived
      mask/gaps/coverage are propagated truthfully (gaps is NEVER [] merely because aggregation
      succeeded); station ORDER never changes the result; returns None (unavailable) when no bin
      has 2-station support.
* correlation path: uses the AND-mask COMMON SUPPORT of every independently eligible fault
  segment; requires >= 2 eligible segments AND common-support samples >=
  MIN_COMMON_SUPPORT_FRACTION × requested_grid_count (= 43,200 for 24 h); the eligible-segment
  set is determined by eligibility ALONE (no result-dependent subset search); correlations are
  computed ONLY over common-support bins (invalid-bin filler mutations leave every output
  unchanged, bit-for-bit).
* ALL existing gates unchanged: topology, native-rate/Nyquist provenance, non-degeneracy, finite
  correlation, unit diagonal, PSD, calibration binding, registry, owner gates.

Fixture policy (codex 1712): observed archive support patterns calibrate fixture SHAPE only —
synthetic 23-gap and 67-gap days and a zero-eligible-segment day appear below with SYNTHETIC
signals; no measured amplitude series or expected ratios are copied into this bar.

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


def raises_unavailable(fn, exc_names=("DataUnavailable",)):
    try:
        fn()
        return False, None
    except Exception as exc:
        return type(exc).__name__ in exc_names, exc


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


def smooth_signal(rng, seconds, fs=FS):
    """Synthetic in-band signal (5 Hz carrier, slow random AM) — SHAPE fixture, no measured data."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    am = 1.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t + rng.uniform(0, 6.28))
    return am * np.sin(2 * np.pi * 5.0 * t)


def gap_day_fragments(rng, n_gaps, gap_seconds=120, fs=FS):
    """A full-day session split into n_gaps+1 contiguous spans by n_gaps equal gaps (SHAPE-only
    analog of the observed 23/67-gap archive days)."""
    usable = SESSION - n_gaps * gap_seconds
    span = usable / (n_gaps + 1)
    frags, cursor = [], 0.0
    for _ in range(n_gaps + 1):
        frags.append((smooth_signal(rng, span, fs), fs,
                      UTC0 + timedelta(seconds=cursor)))
        cursor += span + gap_seconds
    return frags


def _mask_series(SD, mask, start=UTC0, values=None, source="XX.AAA"):
    """Hand-built supported series (for eligibility/aggregation/validation locks)."""
    n = mask.size
    vals = values if values is not None else np.where(mask, 1.0, 0.0)
    gaps = []
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gaps.append(((start + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ"), j - i))
            i = j
        else:
            i += 1
    return SD.EnvelopeSeries(values=np.asarray(vals, dtype=float), start_utc=start,
                             dt_seconds=1.0, coverage=float(mask.sum()) / n, gaps=gaps,
                             source_ids=[source], band_tag="1-10Hz",
                             processing_version=SD.PROCESSING_VERSION,
                             pre_envelope_rate_hz=40.0, valid_mask=mask.copy(),
                             requested_grid_count=n)


class _FragTrace:
    def __init__(self, data, rate, start):
        self.data = data
        self.stats = types.SimpleNamespace(sampling_rate=rate, starttime=start)


class _FragStream:
    """Stream double with MULTIPLE contiguous traces (= raw spans); records method calls."""

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
        check("SS-0a seismic_data rev-3 seams present (supported core + validate_support + "
              "support_digest + SEG_SUPPORT)", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")
        return
    try:
        _probe = SD.EnvelopeSeries(values=np.zeros(4), start_utc=UTC0, dt_seconds=1.0,
                                   coverage=1.0, gaps=[], source_ids=["X"], band_tag="1-10Hz",
                                   processing_version=SD.PROCESSING_VERSION,
                                   pre_envelope_rate_hz=40.0,
                                   valid_mask=np.ones(4, dtype=bool), requested_grid_count=4)
        ok_fields = hasattr(_probe, "valid_mask") and hasattr(_probe, "requested_grid_count")
    except Exception:
        ok_fields = False
    if not ok_fields:
        check("SS-0a2 EnvelopeSeries carries valid_mask + requested_grid_count", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")
        return
    need_fc = ("station_eligible", "aggregate_segment_supported")
    if not all(hasattr(FC, n) for n in need_fc):
        check("SS-0b fault_correlation rev-3 seams present (station_eligible + "
              "aggregate_segment_supported)", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")
        return

    K = SD.SEG_SUPPORT
    check("SS-0c frozen constants exact (operator-derived; derivation in this bar's docstring)",
          K.get("edge_trim_seconds") == 90 and K.get("min_contiguous_span_seconds") == 240
          and K.get("min_common_support_fraction") == 0.50
          and K.get("station_coverage_floor") == 0.50
          and 0.50 * 86400 == 43200, f"SEG_SUPPORT={K}")

    rng = np.random.default_rng(11)

    # ---- LOCK 1: gapless regression-equivalence to the pinned core -------
    day = smooth_signal(rng, SESSION)
    es = SD.compute_band_envelope_supported([(day, FS, UTC0)], session_start_utc=UTC0,
                                            source_id="XX.AAA")
    core = SD.compute_band_envelope_from_array(day, FS, start_utc=UTC0, source_id="XX.AAA")
    vm = es.valid_mask
    core_vals = np.asarray(core.values)
    m = min(core_vals.size, SESSION)
    exp_cov = (SESSION - 2 * 90) / SESSION
    check("SS-1 GAPLESS REGRESSION: valid bins equal the pinned rev-2 core at those bins; "
          "coverage = 1 - 2*trim/session; dense grid length",
          es.values.size == SESSION and vm.size == SESSION
          and abs(es.coverage - exp_cov) < 3.0 / SESSION
          and np.allclose(es.values[:m][vm[:m]], core_vals[:m][vm[:m]], rtol=1e-6, atol=1e-9),
          f"n={es.values.size} cov={es.coverage:.5f} exp~{exp_cov:.5f}")

    # ---- LOCK 2: impulse adjacent to a gap cannot leak across it ---------
    span_a = smooth_signal(rng, 3600)
    span_b = smooth_signal(rng, 3600)
    frags_base = [(span_a, FS, UTC0), (span_b, FS, UTC0 + timedelta(seconds=3900))]
    spiked = span_a.copy()
    spiked[-1] = 1e6                                   # impulse at the gap-adjacent edge of A
    frags_spike = [(spiked, FS, UTC0), (span_b, FS, UTC0 + timedelta(seconds=3900))]
    sess = 7500
    e_base = SD.compute_band_envelope_supported(frags_base, session_start_utc=UTC0,
                                                session_seconds=sess, source_id="XX.AAA")
    e_spike = SD.compute_band_envelope_supported(frags_spike, session_start_utc=UTC0,
                                                 session_seconds=sess, source_id="XX.AAA")
    b_bins = np.zeros(sess, dtype=bool)
    b_bins[3900:] = True                               # bins belonging to span B
    sel = b_bins & e_base.valid_mask & e_spike.valid_mask
    gap_bins = np.zeros(sess, dtype=bool)
    gap_bins[3600:3900] = True
    check("SS-2 GAP ISOLATION: a 1e6 impulse at span-A's gap edge changes NOTHING in span-B's "
          "valid bins (bit-identical), and gap bins are never valid",
          sel.sum() > 0
          and np.array_equal(e_base.values[sel], e_spike.values[sel])
          and not (e_base.valid_mask & gap_bins).any(),
          f"b_bins_compared={int(sel.sum())}")

    # ---- LOCK 3: minimums fail closed (span / station floor / common support) --
    short = SD.compute_band_envelope_supported(
        [(smooth_signal(rng, 239), FS, UTC0)], session_start_utc=UTC0, session_seconds=600,
        source_id="XX.AAA")
    okmin = SD.compute_band_envelope_supported(
        [(smooth_signal(rng, 240), FS, UTC0)], session_start_utc=UTC0, session_seconds=600,
        source_id="XX.AAA")
    check("SS-3a a 239 s span contributes ZERO valid bins; a 240 s span contributes ~60",
          int(short.valid_mask.sum()) == 0 and 55 <= int(okmin.valid_mask.sum()) <= 62,
          f"short={int(short.valid_mask.sum())} okmin={int(okmin.valid_mask.sum())}")
    n = 1000
    m_half = np.zeros(n, dtype=bool)
    m_half[:500] = True                                # coverage exactly 0.50
    m_less = m_half.copy()
    m_less[499] = False                                # 0.499
    check("SS-3b station floor boundary: coverage 0.500 ELIGIBLE, 0.499 NOT",
          FC.station_eligible(_mask_series(SD, m_half)) is True
          and FC.station_eligible(_mask_series(SD, m_less)) is False)

    # ---- LOCK 4: contradictions + support-bound cache identity -----------
    import dataclasses as _dc

    def _with(series, **kw):
        try:
            return _dc.replace(series, **kw)
        except Exception:
            for k2, v2 in kw.items():
                object.__setattr__(series, k2, v2)
            return series

    bad = _with(_mask_series(SD, m_half), coverage=0.9)     # contradicts its own mask
    ok4, reasons4 = SD.validate_support(bad)
    bad2 = _with(_mask_series(SD, m_half), gaps=[])         # mask says half missing; gaps say none
    ok4b, _ = SD.validate_support(bad2)
    good4, _ = SD.validate_support(_mask_series(SD, m_half))
    check("SS-4a mask/coverage and mask/gaps contradictions FAIL validate_support; the honest "
          "series passes", ok4 is False and ok4b is False and good4 is True,
          f"reasons={reasons4}")
    fr1 = [(smooth_signal(rng, 3600), FS, UTC0),
           (smooth_signal(rng, 3600), FS, UTC0 + timedelta(seconds=3720))]
    fr2 = [(fr1[0][0], FS, UTC0),
           (fr1[1][0], FS, UTC0 + timedelta(seconds=3721))]    # ONE gap endpoint moved 1 s
    d1, d2 = SD.support_digest(fr1), SD.support_digest(fr2)
    check("SS-4b support_digest: moving one gap endpoint by 1 s changes the digest "
          "(=> cache identity miss)", d1 != d2 and len(d1) == 64)

    # ---- LOCK 5: filler / input-order / station-order invariance ---------
    def seg_matrix(filler, station_order):
        """Build 2 segments × 2 stations from fixed masks, run aggregation + correlation over
        common support; returns the correlation matrix bytes."""
        base = np.ones(n, dtype=bool)
        base[100:200] = False
        m2 = np.ones(n, dtype=bool)
        m2[700:820] = False
        v = rng2.normal(1.0, 0.3, (4, n))
        series = []
        for i, mask in enumerate((base, m2, base, m2)):
            vals = np.where(mask, v[i], filler)
            series.append(_mask_series(SD, mask, values=vals, source=f"XX.S{i}"))
        segA = FC.aggregate_segment_supported([series[j] for j in station_order[0]])
        segB = FC.aggregate_segment_supported([series[j] for j in station_order[1]])
        C, names, qc = FC.compute_correlation_matrix_supported([segA, segB], ["A", "B"])
        return C

    if hasattr(FC, "compute_correlation_matrix_supported"):
        rng2 = np.random.default_rng(23)
        c_ref = seg_matrix(0.0, ([0, 1], [2, 3]))
        rng2 = np.random.default_rng(23)
        c_fill = seg_matrix(999.0, ([0, 1], [2, 3]))
        rng2 = np.random.default_rng(23)
        c_perm = seg_matrix(0.0, ([1, 0], [3, 2]))
        check("SS-5 INVARIANCE: invalid-bin filler (0 vs 999) and station order change NOTHING "
              "in the correlation output (bit-identical)",
              c_ref is not None and np.array_equal(c_ref, c_fill)
              and np.array_equal(c_ref, c_perm))
    else:
        check("SS-5 INVARIANCE: correlation-over-common-support seam "
              "(compute_correlation_matrix_supported) present", False,
              "AWAITING grassmann's rev-3 -- red-first as authored")

    # ---- LOCK 6: per-bin two-station support + AND-mask common support ----
    mA = np.zeros(n, dtype=bool)
    mA[0:600] = True
    mB = np.zeros(n, dtype=bool)
    mB[400:1000] = True
    agg = FC.aggregate_segment_supported([_mask_series(SD, mA, source="XX.P"),
                                          _mask_series(SD, mB, source="XX.Q")])
    check("SS-6a per-bin 2-station rule: aggregate valid ONLY on the 400..599 overlap; gaps "
          "truthfully non-empty",
          agg is not None and int(agg.valid_mask.sum()) == 200
          and bool(agg.valid_mask[400]) and bool(agg.valid_mask[599])
          and not agg.valid_mask[399] and not agg.valid_mask[600] and len(agg.gaps) > 0,
          f"agg_valid={None if agg is None else int(agg.valid_mask.sum())}")
    single = FC.aggregate_segment_supported([_mask_series(SD, mA, source="XX.P")])
    check("SS-6b a single-station segment has NO 2-station bins -> unavailable (None)",
          single is None)
    if hasattr(FC, "compute_correlation_matrix_supported"):
        sA = _mask_series(SD, m_half, values=np.where(m_half, rng.normal(1, .3, n), 0.0))
        mC = np.zeros(n, dtype=bool)
        mC[200:700] = True                             # AND with m_half -> 300 common bins
        sB = _mask_series(SD, mC, values=np.where(mC, rng.normal(1, .3, n), 0.0))
        C6, _, qc6 = FC.compute_correlation_matrix_supported([sA, sB], ["A", "B"])
        # common = 300 of n=1000 requested -> 0.30 < 0.50 floor -> unavailable
        check("SS-6c AND-mask common support below the 0.50 session fraction -> (None, reasons)",
              C6 is None and len(qc6) > 0, f"qc={qc6}")

    # ---- LOCK 7: production composition on SHAPE-calibrated fixtures -----
    from fault_segments import FaultSegment, SeismicStation
    seg = FaultSegment(name="syn_seg", region="syn_region",
                       stations=[SeismicStation("XX", "AAA", 0.0, 0.0)],
                       polygon=[(0, 0), (0, 1), (1, 1), (1, 0)], strike=0.0, dip=90.0, rake=0.0)
    for n_gaps in (23, 67):
        frags = gap_day_fragments(np.random.default_rng(n_gaps), n_gaps)
        stream = _FragStream(frags)
        fetcher = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
        fetcher.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
        fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": stream}
        core_calls = []
        real_sup = SD.compute_band_envelope_supported
        sentinel = real_sup(frags, session_start_utc=UTC0, source_id="XX.AAA")

        def spy(fr, **kw):
            core_calls.append(len(fr))
            return sentinel

        SD.compute_band_envelope_supported = spy
        err = None
        try:
            out = fetcher.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=SESSION),
                                                use_cache=False)
        except Exception as e:
            out, err = None, e
        finally:
            SD.compute_band_envelope_supported = real_sup
        dsp = [c for c in stream.calls if c in ("filter", "decimate", "resample", "merge")]
        check(f"SS-7a COMPOSITION ({n_gaps}-gap day): shell passes {n_gaps + 1} fragments to the "
              "supported core once, returns ITS series, no DSP/merge on the stream",
              out is not None and core_calls == [n_gaps + 1] and not dsp
              and isinstance(out, dict) and out.get("XX.AAA") is sentinel,
              f"calls={core_calls} dsp={dsp} err={err}")
        exp_cov_gap = 1.0 - (n_gaps * 120 + (n_gaps + 1) * 2 * 90) / SESSION
        check(f"SS-7b {n_gaps}-gap day: supported core yields truthful coverage "
              f"(~{exp_cov_gap:.3f}) with {n_gaps}+ gap records and NEVER gaps=[]",
              abs(sentinel.coverage - exp_cov_gap) < 0.02 and len(sentinel.gaps) >= n_gaps,
              f"cov={sentinel.coverage:.3f} gaps={len(sentinel.gaps)}")
    low = np.zeros(n, dtype=bool)
    low[:100] = True                                   # 10% coverage stations everywhere
    zero_day = [_mask_series(SD, low, source=f"XX.Z{i}") for i in range(4)]
    check("SS-7c zero-eligible-segment day: station_eligible False for every station "
          "(eligibility gates the segment set upstream; no result-dependent subset search)",
          all(FC.station_eligible(s) is False for s in zero_day))

    # ---- LOCK 8: existing suites remain green -----------------------------
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
print("ALL D2 SEGMENTED-SUPPORT RED-KATs PASS (valid-support carrier enforced end-to-end)")
