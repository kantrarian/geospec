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
  LC-2b (codex #2, HIGH — shell half): a raw ObsPy data gap is NEVER interpolated. A gappy stream
        through get_segment_envelopes mints NO coverage=1.0/gaps=[] capsule (station skipped).
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


# --- doubles for the live-stream gap KAT (LC-2b) ------------------------------
class _Stats:
    def __init__(self, rate):
        self.sampling_rate = rate


class _Trace:
    def __init__(self, data, rate):
        self.data = data
        self.stats = _Stats(rate)


class _GappyStream:
    """Stream double whose get_gaps() reports a real ObsPy gap; the shell must skip it and mint
    no envelope (no interpolation, no coverage=1.0/gaps=[] capsule)."""

    def __init__(self, data, rate, gaps):
        self._trace = _Trace(data, rate)
        self._gaps = gaps

    def get_gaps(self):
        return self._gaps

    def copy(self):
        return self

    def merge(self, *a, **k):
        return self

    def detrend(self, *a, **k):
        return self

    def __getitem__(self, i):
        return self._trace

    def __len__(self):
        return 1


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
    # one raw ObsPy-style gap entry -> the station must be skipped
    gappy = _GappyStream(xr, fs, gaps=[["XX", "AAA", "", "HHZ", "t0", "t1", 4.0, 160]])
    fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": gappy}
    out = fetcher.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=t_sec), use_cache=False)
    minted = out.get("XX.AAA")
    check("LC-2b a raw data gap mints NO coverage=1.0/gaps=[] capsule (gappy station skipped, never "
          "interpolated)", isinstance(out, dict) and minted is None,
          f"out_keys={list(out.keys()) if isinstance(out, dict) else out}")

    # contrast: a GAPLESS stream through the same shell DOES mint an honest full-coverage capsule
    class _CleanStream(_GappyStream):
        def get_gaps(self):
            return []

    fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA": _CleanStream(xr, fs, [])}
    out2 = fetcher.get_segment_envelopes(seg, UTC0, UTC0 + timedelta(seconds=t_sec), use_cache=False)
    good = out2.get("XX.AAA")
    check("LC-2b(contrast) a gapless stream still mints an honest envelope (coverage=1.0, gaps=[])",
          good is not None and good.coverage == 1.0 and list(good.gaps) == [],
          f"good={'None' if good is None else (good.coverage, good.gaps)}")

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
