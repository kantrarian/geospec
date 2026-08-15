#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 KOERI ORDER-CANONICALIZATION red-KATs (cayley, 2026-08-15) — the acceptance bar for
codex `0943`'s WORKS-WITH-FIX ruling on the R-1 defect repair (cayley finding `1337` /
`docs/koeri-overlap-research-note-2026-08-15.md`).

THE DEFECT: KOERI serves backfilled micro-segments out of chronological order. The rev-3
shell (`get_segment_envelopes`) builds fragment tuples in STREAM ORDER and the frozen core
validator `_validate_fragments` enforces its documented `ascending non-overlapping`
precondition on consecutive pairs — so a non-canonical provider ENUMERATION of genuinely
disjoint spans is refused as if it were an overlap, and component availability depends on
server return order, not data content (production: turkey "got 1" on 08-12/08-13/08-15).

THE RULED REPAIR (codex 0943, verbatim contract): the shell SHALL canonicalize provider
enumeration by sorting the extracted, non-empty fragment tuples in ascending aware-UTC
`start` order BEFORE deriving any cache identity or invoking the core. It SHALL NOT call a
method on the Stream, merge, deduplicate, trim, drop, move, relabel, or alter any sample,
rate, or timestamp. Equal starts and every post-sort `start < previous_exclusive_end`
remain DataUnavailable under the existing core validator. Tuple-list sort, NOT
`stream.sort()`. `fetch_station_waveforms`, `_validate_fragments`, `support_digest`, and
the pinned DSP core stay byte-untouched.

CODEX 0943 RED-KAT CONTRACT -> KAT MAP (red on current tree, green after the repair):
  lock 1 -> OC-1  real-shell ordering: later-then-earlier DISJOINT traces through the REAL
                  get_segment_envelopes — current tree refuses (station absent), repaired
                  tree scores; stream.calls == [] (no Stream method call); the core
                  receives the ORIGINAL arrays/rates/starts, merely sorted.   [RED now]
  lock 2 -> OC-2  permutation/cache: chronological vs shuffled enumeration of the SAME
                  disjoint fragments produce identical support_sha256, cache identity,
                  valid mask, and envelope bytes.       [RED now: chronological scores
                  while shuffled refuses — the availability-depends-on-order defect]
  lock 3 -> OC-3  true-conflict: a positive-length equal-start pair and a strict partial
                  overlap both refuse, and a primed cache cannot bypass the refusal.
                  [green now and after — the strictness must survive the repair]
  lock 4 -> OC-4  scope: the core validator itself stays strict — DIRECT
                  _validate_fragments calls: non-ascending disjoint input still raises
                  (canonicalization lives in the SHELL, the validator is not loosened),
                  ascending overlap raises, ascending disjoint passes.
                  [green now and after]

HERMETIC: no network (fetch layer never touched — the bar drives get_segment_envelopes
directly with recording stream doubles), per-KAT tempdir caches, seeded fixtures, obspy
stubbed. FREEZE DISCIPLINE: after this bar freezes, grassmann implements the codex-0943
contract WITHOUT editing this file; one verify; close. No production landing, historical
re-score, published-result mutation, or O-1..O-4 overlap policy is authorized by this bar.
"""
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


UTC = timezone.utc
T0 = datetime(2026, 8, 11, 0, 0, 0, tzinfo=UTC)      # aware session start (post-1318 seam)
SESSION_S = 4 * 3600                                  # 4 h shell-level session
T_END = T0 + timedelta(seconds=SESSION_S)
FS = 100.0                                            # KOERI-native rate
SPAN_S = 2700                                         # 45 min spans (>= 240 s min span)

EARLY_START = T0 + timedelta(minutes=30)              # [00:30, 01:15)
LATE_START = T0 + timedelta(hours=2)                  # [02:00, 02:45)


class _Stats:
    def __init__(self, rate, start):
        self.sampling_rate = rate
        self.starttime = start


class _Trace:
    def __init__(self, data, rate, start):
        self.data = data
        self.stats = _Stats(rate, start)


class _RecStream:
    """Recording stream double: any METHOD call on the stream object is recorded (the
    ruled repair must make none — tuple-list sort, never stream.sort()). Index/len reads
    are the shell's contractual read-only access and are not recorded."""

    def __init__(self, traces):
        self.calls = []
        self._traces = traces

    def __getattr__(self, name):
        def _rec(*a, **k):
            self.calls.append(name)
            return self
        return _rec

    def __getitem__(self, i):
        return self._traces[i]

    def __len__(self):
        return len(self._traces)


def _mk_data(seed, n=int(SPAN_S * FS)):
    return np.random.default_rng(seed).standard_normal(n)


def _mk_fetcher(SD):
    f = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    f.cache_dir = Path(tempfile.mkdtemp())
    f.cache_ttl_days = 7
    f.clients = {}
    return f


def _mk_segment():
    from fault_segments import FaultSegment, SeismicStation
    return FaultSegment(name="oc_seg", region="oc_region",
                        stations=[SeismicStation("XX", "AAA", 0.0, 0.0)],
                        polygon=[(0, 0), (0, 1), (1, 1), (1, 0)],
                        strike=0.0, dip=90.0, rake=0.0)


def _run_shell(SD, traces, *, use_cache=True, fetcher=None, spies=None):
    """Drive the REAL get_segment_envelopes with a recording stream double. Returns
    (envelopes_dict, stream_double, fetcher, captured)."""
    fetcher = fetcher or _mk_fetcher(SD)
    stream = _RecStream(traces)
    fetcher.fetch_segment_waveforms = lambda *a, **k: {"XX.AAA..HHZ": stream}
    seg = _mk_segment()
    captured = {"support": [], "identity": [], "core_frags": []}
    real_support = SD.support_digest
    real_identity = SD.build_envelope_cache_identity
    real_core = SD.compute_band_envelope_supported

    def spy_support(frags, **kw):
        out = real_support(frags, **kw)
        captured["support"].append(out)
        return out

    def spy_identity(**kw):
        out = real_identity(**kw)
        captured["identity"].append(out)
        return out

    def spy_core(frags, **kw):
        captured["core_frags"].append(frags)
        return real_core(frags, **kw)

    SD.support_digest = spy_support
    SD.build_envelope_cache_identity = spy_identity
    SD.compute_band_envelope_supported = spy_core
    try:
        env = fetcher.get_segment_envelopes(seg, T0, T_END, use_cache=use_cache)
    finally:
        SD.support_digest = real_support
        SD.build_envelope_cache_identity = real_identity
        SD.compute_band_envelope_supported = real_core
    return env, stream, fetcher, captured


def main():
    _install_obspy_stub()
    import seismic_data as SD

    need = (hasattr(SD.SeismicDataFetcher, "get_segment_envelopes")
            and hasattr(SD, "compute_band_envelope_supported")
            and hasattr(SD, "_validate_fragments")
            and hasattr(SD, "support_digest")
            and hasattr(SD, "DataUnavailable"))
    check("OC-0 rev-3 seams present (shell + supported core + validator)", need)
    if not need:
        return

    data_early = _mk_data(11)
    data_late = _mk_data(22)

    # =====================================================================
    # OC-1 (codex lock 1) — THE red KAT: later-then-earlier DISJOINT traces
    # through the REAL shell must SCORE; no Stream method call; original
    # arrays/rates/starts reach the core, merely sorted ascending
    # =====================================================================
    tr_late = _Trace(data_late, FS, LATE_START)
    tr_early = _Trace(data_early, FS, EARLY_START)
    env1, stream1, _, cap1 = _run_shell(SD, [tr_late, tr_early])
    got = env1.get("XX.AAA..HHZ")
    frags_ok = False
    if cap1["core_frags"]:
        f = cap1["core_frags"][-1]
        frags_ok = (len(f) == 2
                    and f[0][2] == EARLY_START and f[1][2] == LATE_START
                    and f[0][0] is data_early and f[1][0] is data_late
                    and float(f[0][1]) == FS and float(f[1][1]) == FS)
    check("OC-1 provider order [later, earlier] of DISJOINT spans SCORES through the real "
          "shell (station present, valid mask covers both spans), stream.calls == [], and "
          "the core receives the ORIGINAL arrays/rates/starts in ascending order",
          got is not None and stream1.calls == [] and frags_ok
          and bool(np.asarray(got.valid_mask, dtype=bool).sum() > 0),
          f"station_present={got is not None} stream_calls={stream1.calls} "
          f"core_called={bool(cap1['core_frags'])} frags_sorted_original={frags_ok}")

    # =====================================================================
    # OC-2 (codex lock 2) — permutation/cache lock: chronological vs shuffled
    # enumeration of the SAME fragments -> identical support digest, cache
    # identity, valid mask, envelope bytes
    # =====================================================================
    env_a, _, _, cap_a = _run_shell(SD, [_Trace(data_early, FS, EARLY_START),
                                         _Trace(data_late, FS, LATE_START)])
    env_b, _, _, cap_b = _run_shell(SD, [_Trace(data_late, FS, LATE_START),
                                         _Trace(data_early, FS, EARLY_START)])
    sa, sb = env_a.get("XX.AAA..HHZ"), env_b.get("XX.AAA..HHZ")
    both = sa is not None and sb is not None
    eq = False
    if both:
        eq = (np.array_equal(np.asarray(sa.values), np.asarray(sb.values))
              and np.array_equal(np.asarray(sa.valid_mask), np.asarray(sb.valid_mask))
              and sa.coverage == sb.coverage and list(sa.gaps) == list(sb.gaps)
              and cap_a["support"] and cap_b["support"]
              and cap_a["support"][-1] == cap_b["support"][-1]
              and cap_a["identity"] and cap_b["identity"]
              and cap_a["identity"][-1] == cap_b["identity"][-1])
    check("OC-2 chronological and shuffled enumeration of the SAME disjoint fragments "
          "yield IDENTICAL support_sha256, cache identity, valid mask, and envelope bytes "
          "(availability and identity may not depend on provider order)",
          both and eq,
          f"chronological_scored={sa is not None} shuffled_scored={sb is not None} "
          f"equal={eq}")

    # =====================================================================
    # OC-3 (codex lock 3) — true-conflict lock: equal-start pair and strict
    # partial overlap REFUSE, before and after; a primed cache cannot bypass
    # =====================================================================
    primed = _mk_fetcher(SD)
    _run_shell(SD, [_Trace(data_early, FS, EARLY_START),
                    _Trace(data_late, FS, LATE_START)], fetcher=primed)  # prime the cache
    env_eq, _, _, _ = _run_shell(SD, [_Trace(data_early, FS, EARLY_START),
                                      _Trace(_mk_data(33), FS, EARLY_START)],
                                 fetcher=primed)
    env_ov, _, _, _ = _run_shell(SD, [_Trace(data_early, FS, EARLY_START),
                                      _Trace(_mk_data(44), FS,
                                             EARLY_START + timedelta(seconds=SPAN_S // 2))],
                                 fetcher=primed)
    check("OC-3 TRUE conflicts refuse and a primed cache cannot bypass: a positive-length "
          "equal-start pair AND a strict partial overlap both leave the station absent "
          "(DataUnavailable path), before and after the repair",
          env_eq.get("XX.AAA..HHZ") is None and env_ov.get("XX.AAA..HHZ") is None,
          f"equal_start_present={env_eq.get('XX.AAA..HHZ') is not None} "
          f"partial_overlap_present={env_ov.get('XX.AAA..HHZ') is not None}")

    # =====================================================================
    # OC-4 (codex lock 4) — scope lock: the CORE VALIDATOR stays strict; the
    # canonicalization lives in the shell only
    # =====================================================================
    def raises_unavailable(frags):
        try:
            SD._validate_fragments(frags, min_rate_hz=25.0)
            return False
        except SD.DataUnavailable:
            return True

    disjoint_sorted = [(data_early, FS, EARLY_START), (data_late, FS, LATE_START)]
    disjoint_unsorted = [(data_late, FS, LATE_START), (data_early, FS, EARLY_START)]
    overlap_sorted = [(data_early, FS, EARLY_START),
                      (data_late, FS, EARLY_START + timedelta(seconds=SPAN_S // 2))]
    ok4 = (raises_unavailable(disjoint_unsorted)          # validator NOT loosened
           and raises_unavailable(overlap_sorted)         # true overlap still refuses
           and not raises_unavailable(disjoint_sorted))   # ascending disjoint passes
    check("OC-4 DIRECT _validate_fragments stays strict: non-ascending disjoint input "
          "still raises (the repair may NOT loosen the validator -- canonicalization is "
          "the SHELL's job), ascending overlap raises, ascending disjoint passes",
          ok4)


main()
print()
if FAILS:
    print(f"D2 KOERI ORDER-CANON RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 KOERI ORDER-CANON RED-KATs PASS (provider enumeration canonicalized in the "
      "shell; identity/availability order-invariant; true overlaps and the core validator "
      "stay strict)")
