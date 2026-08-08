#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 LIVE-CARRIER fix locks (grassmann, 2026-08-08) — reviewer-specified expectations from codex's
0409 D2 implementation verification (WORKS-WITH-THREE-NARROW-FIXES). These transcribe codex's exact
counterexamples + expected values so the three live-carrier repairs are locked; cayley + codex each
verify once. The frozen bar `test_d2_reband_redkats_cayley.py` (396151d) stays BYTE-UNEDITED.

Covered here (green after the repairs):
  LC-1 (codex #1, HIGH): station envelopes are EXACT-UTC-GRID-ALIGNED before the median. Two views of
        one carrier — station1 x[0:1000]@00:00:00, station2 x[100:1100]@00:01:40 — aggregate to
        start 00:01:40Z, n=900, values == x[100:1000] (no element-zero index truncation).
  LC-3 (codex #3, MODERATE): a non-integer native rate produces the correct 1 Hz carrier. 40.5 Hz x
        600 s -> a 1 Hz carrier of 600 samples (within one-sample tolerance), realized dt ~ 1.0 s.
  LC-2b(rev3) (codex 1906 ruling, option A — supersedes the fix-#2 any-gap-skip): a real
        MULTI-TRACE stream (contiguous raw spans) through get_segment_envelopes mints ONE truthful
        SEGMENTED envelope — no stream method calls, no cross-gap DSP, gap bins invalid, truthful
        mask/gaps/coverage. The clean contrast carries the EXACT 90 s edge-trim mask/coverage
        (never coverage=1.0/gaps=[]). Updated by cayley per codex 1906 (bar rev 3); the interim
        skip-on-gap expectation was retired by phase codex-d2-segmented-support-2026-08-08-v1.
  LC-2a (codex #2, HIGH — align half): align_activity_series rejects a gap whose end lies past the
        SAMPLE carrier [start, start+n*dt] (codex's 00:30Z-on-a-1000 s-carrier counterexample ->
        unavailable), while an honest IN-carrier gap (00:05Z, coverage 0.94) stays available + QC,
        and a zero/non-finite duration is rejected. Membership rule:
        series_start <= gap_start < gap_end <= series_start + n*dt (finite-positive duration +
        aware-UTC start). Landed after cayley's frozen-bar D2R-3k/3o fixture correction (geospec
        f6da210, codex 0426 ruling) — the frozen bar's old 02:00Z "permitted" gap that had conflicted
        was itself the defective fixture.
"""
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


UTC0 = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)


# --- doubles for the live-stream segmented KAT (LC-2b rev3) -------------------
class _Stats:
    def __init__(self, rate, start):
        self.sampling_rate = rate
        self.starttime = start


class _Trace:
    def __init__(self, data, rate, start):
        self.data = data
        self.stats = _Stats(rate, start)


class _RecStream:
    """Recording multi-trace stream double: each contiguous raw span is ONE trace; any method
    call on the stream object is recorded (the rev-3 shell must make none)."""

    def __init__(self, frags):
        self.calls = []
        self._traces = [_Trace(d, r, s) for d, r, s in frags]

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


def _mk_series(SD, values, start_utc, *, dt_seconds=1.0, rate_pre=40.0, coverage=1.0, gaps=()):
    return SD.EnvelopeSeries(values=np.asarray(values, dtype=float), start_utc=start_utc,
                             dt_seconds=dt_seconds, coverage=coverage, gaps=list(gaps),
                             source_ids=["XX.AAA"],
                             band_tag=getattr(SD, "_FAULT_CORR_BAND_TAG", "1-10Hz"),
                             processing_version=SD.PROCESSING_VERSION,
                             pre_envelope_rate_hz=rate_pre)


def main():
    _install_obspy_stub()
    import seismic_data as SD
    import fault_correlation as FC
    from fault_segments import FaultSegment, SeismicStation

    rng = np.random.default_rng(20260808)

    # =====================================================================
    # LC-1 (codex #1) — exact-UTC-grid station alignment before the median
    # =====================================================================
    x = rng.normal(0.0, 1.0, 1200)
    s1 = _mk_series(SD, x[0:1000], UTC0)                                   # 00:00:00, n=1000
    s2 = _mk_series(SD, x[100:1100], UTC0 + timedelta(seconds=100))        # 00:01:40, n=1000
    agg = FC._aggregate_segment_envelope({"XX.AAA": s1, "YY.BBB": s2})
    ok1 = (agg is not None
           and agg.start_utc == UTC0 + timedelta(seconds=100)
           and np.asarray(agg.values).size == 900
           and np.array_equal(np.asarray(agg.values), x[100:1000]))
    max_err = (float(np.max(np.abs(np.asarray(agg.values) - x[100:1000])))
               if (agg is not None and np.asarray(agg.values).size == 900) else float("nan"))
    check("LC-1 station envelopes are UTC-grid-aligned before the median (aggregate start 00:01:40Z, "
          "n=900, values == x[100:1000]; NOT element-zero index-truncated)", ok1,
          f"start={getattr(agg, 'start_utc', None)} n={np.asarray(agg.values).size if agg is not None else None} "
          f"max_err={max_err}")

    # a genuine rate/timezone mismatch across stations -> unavailable (never index-truncate)
    s3 = _mk_series(SD, x[0:1000], UTC0, dt_seconds=2.0)
    agg_bad = FC._aggregate_segment_envelope({"XX.AAA": s1, "ZZ.CCC": s3})
    check("LC-1b a dt (rate) mismatch across stations -> unavailable (None), never index-truncated",
          agg_bad is None, f"agg_bad={agg_bad!r}")

    # =====================================================================
    # LC-3 (codex #3) — a non-integer native rate produces the right 1 Hz carrier
    # =====================================================================
    fs = 40.5
    t_sec = 600
    n = int(round(fs * t_sec))
    t = np.arange(n) / fs
    m = 1.0 + 0.6 * np.sin(2 * np.pi * 0.02 * t)
    xr = m * np.sin(2 * np.pi * 5.0 * t)
    es = SD.compute_band_envelope_from_array(xr, fs, start_utc=UTC0, source_id="XX.AAA")
    realized_rate = 1.0 / es.dt_seconds
    realized_dur = np.asarray(es.values).size * es.dt_seconds
    check("LC-3 a 40.5 Hz x 600 s carrier yields a 1 Hz carrier of 600 samples within one-sample "
          "tolerance (realized dt ~ 1.0 s; 600 real seconds stay 600 declared seconds)",
          abs(np.asarray(es.values).size - t_sec) <= 1
          and abs(realized_rate - 1.0) < 0.02
          and abs(realized_dur - t_sec) <= 1.0
          and es.pre_envelope_rate_hz == fs,
          f"n={np.asarray(es.values).size} realized_rate={realized_rate:.6f} "
          f"realized_dur={realized_dur:.4f} pre_rate={es.pre_envelope_rate_hz}")

    # =====================================================================
    # LC-2b (codex #2, shell) — a raw data gap is never interpolated into a full-coverage capsule
    # =====================================================================
    seg = FaultSegment(name="syn_seg", region="syn_region",
                       stations=[SeismicStation("XX", "AAA", 0.0, 0.0)],
                       polygon=[(0, 0), (0, 1), (1, 1), (1, 0)], strike=0.0, dip=90.0, rake=0.0)
    fetcher = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    fetcher.cache_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    # rev3: a real gap = MULTIPLE contiguous traces; the shell segments and mints truthfully.
    n300 = int(round(300 * fs))
    n240 = int(round(240 * fs))
    gstream = _RecStream([(xr[:n300], fs, UTC0),
                          (xr[:n240], fs, UTC0 + timedelta(seconds=360))])   # 60 s raw gap
    fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": gstream}
    out = fetcher.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=t_sec), use_cache=False)
    minted = out.get("XX.AAA")
    exp_lc2b = np.zeros(t_sec, dtype=bool)
    exp_lc2b[90:210] = True                      # span [0,300)   -> valid [90,210)
    exp_lc2b[450:510] = True                     # span [360,600) -> valid [450,510)
    ok_lc2b = (minted is not None and gstream.calls == []
               and np.array_equal(np.asarray(minted.valid_mask), exp_lc2b)
               and minted.coverage == 180 / 600 and len(minted.gaps) == 3)
    check("LC-2b(rev3) a multi-trace gappy stream mints ONE truthful SEGMENTED envelope (mask "
          "[90,210)+[450,510), coverage 0.30, 3 gap records, ZERO stream method calls, no "
          "cross-gap DSP)", ok_lc2b,
          f"minted={'None' if minted is None else (minted.coverage, len(minted.gaps))} "
          f"calls={gstream.calls}")

    # contrast: a GAPLESS full-session span mints the EXACT edge-trimmed capsule, never 1.0/[]
    fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": _RecStream([(xr, fs, UTC0)])}
    out2 = fetcher.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=t_sec), use_cache=False)
    good = out2.get("XX.AAA")
    exp_clean = np.zeros(t_sec, dtype=bool)
    exp_clean[90:510] = True
    check("LC-2b(rev3-contrast) a gapless 600 s span carries the EXACT 90 s edge-trim mask "
          "([90,510), coverage 0.70, 2 trim-gap records) — never coverage=1.0/gaps=[]",
          good is not None and np.array_equal(np.asarray(good.valid_mask), exp_clean)
          and good.coverage == 0.70 and len(good.gaps) == 2,
          f"good={'None' if good is None else (good.coverage, len(good.gaps))}")

    # =====================================================================
    # LC-2a (codex #2, align half) — gap membership is the TYPED sample carrier, not the UTC day
    # (landed after cayley's frozen D2R-3k/3o fixture correction, geospec f6da210)
    # =====================================================================
    valsg = rng.normal(0.0, 1.0, 1000)                        # 1000 s carrier from 00:00Z
    sB = _mk_series(SD, valsg, UTC0)
    a_out = _mk_series(SD, valsg, UTC0, coverage=1.0, gaps=[("2026-08-01T00:30:00Z", 60)])
    Ao, _no, qco = FC.align_activity_series({"a": a_out, "b": sB},
                                            max_gap_seconds=600, min_coverage=0.9)
    check("LC-2a align rejects a gap whose end lies past the sample carrier [start, start+n*dt] "
          "(00:30Z on a 1000 s carrier, inside the UTC day -> UNAVAILABLE, never mere QC)",
          Ao is None and len(qco) > 0, f"A={'None' if Ao is None else 'ok'} qc={qco}")

    a_in = _mk_series(SD, valsg, UTC0, coverage=0.94, gaps=[("2026-08-01T00:05:00Z", 60)])
    Ai, _ni, qci = FC.align_activity_series({"a": a_in, "b": sB},
                                            max_gap_seconds=600, min_coverage=0.9)
    check("LC-2a(contrast) an honest IN-carrier gap (00:05Z, 60 s, coverage 0.94) stays AVAILABLE "
          "and records its QC flag", Ai is not None and len(qci) > 0,
          f"A={'ok' if Ai is not None else None} qc={qci}")

    a_zero = _mk_series(SD, valsg, UTC0, gaps=[("2026-08-01T00:05:00Z", 0)])
    Az, _nz, qcz = FC.align_activity_series({"a": a_zero, "b": sB},
                                            max_gap_seconds=600, min_coverage=0.9)
    check("LC-2a(dur) a zero-duration gap is rejected (finite positive gap duration required)",
          Az is None and len(qcz) > 0, f"A={'None' if Az is None else 'ok'} qc={qcz}")


main()
print()
if FAILS:
    print(f"D2 LIVE-CARRIER FIX FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 LIVE-CARRIER FIX LOCKS PASS (station UTC-align + non-integer-rate carrier + "
      "no-interpolation gap skip + typed-sample-carrier gap membership)")
