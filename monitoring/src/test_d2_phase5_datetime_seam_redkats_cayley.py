#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 PHASE-5 DATETIME-SEAM red-KATs (cayley, 2026-08-14) — the acceptance bar for the
naive-vs-aware boundary repair ruled by codex `1318` (WORKS-WITH-FIX, placement approved)
on grassmann's first-run finding `c41b3a5` (production run `9356d07`, 2026-08-12 scored day:
all three lifted carriers available=False with "Error: can't subtract offset-naive and
offset-aware datetimes", root cause seismic_data.py:427 `_timestamp_placements`
`(start - session_start_utc)` — aware fragment start minus NAIVE runner session start).

THE SEAM UNDER TEST (codex 1318, verbatim contract): at the TOP of
`GeoSpecEnsemble.compute_fault_correlation_risk`, before any capsule or analysis use,
ALL accepted datetimes become AWARE UTC —
    naive  -> explicit legacy meaning "UTC"  (replace(tzinfo=timezone.utc))
    aware  -> an instant, CONVERTED to UTC   (astimezone(timezone.utc)), never relabelled
and the SAME normalized value feeds BOTH `_resolve_calibration_capsule(fc_region, date_utc)`
and `analyze_region(fc_region, date_utc, calibration=...)` — one scored-instant binding
across admission, fetch, cache identity, timestamp placement, and result provenance.
The rev-3 core stays STRICT; `seismic_data` and the runner are untouched.

WHY NO EXISTING BAR CAUGHT IT: every prior ensemble-level KAT (D2R-4n/4o/...) stubs
`compute_correlation_matrix` — the real placement path never ran under the ensemble entry.
THIS bar stubs ONLY the network edge (`fetch_segment_waveforms`); the REAL
`get_segment_envelopes` -> `compute_band_envelope_supported` -> `_timestamp_placements`
chain runs with AWARE-UTC fragment starts, exactly as in production (codex point 1).

HERMETIC: no network (fetch stubbed at the fetcher instance; no FDSN client is ever
constructed), fresh tempdir envelope cache per KAT, seeded rng fixtures, obspy stubbed.

CODEX 1318 ACCEPTANCE CONTRACT -> KAT MAP (red on `5ffe38f`, green after the repair):
  point 1 -> DT-1   naive runner-style target_date traverses the REAL rev-3 placement path
                    with aware-UTC fragment starts; the mixed-datetime TypeError shape must
                    be ABSENT and the component must SCORE or return only an honest
                    data-quality/calibration result.            [RED now: notes carry the
                    exact production error string]
  point 2 -> DT-2   capsule loader and analyze_region observe ONE normalized value: loader
                    day '2026-08-12', analyze target AWARE UTC, same instant, same day
                    binding for the legacy midnight input.      [RED now: analyze receives
                    the NAIVE date]
  point 3 -> DT-3a  already-aware UTC input is VALUE-PRESERVED (same instant reaches
                    analyze; day binding unchanged; component scores).  [green now — this
                    is also the greenability proof for DT-1: post-repair, DT-1's normalized
                    date follows exactly this path on the same fixture]
            DT-3b  a non-UTC aware input (+03:00) is CONVERTED to the same UTC instant —
                    never relabelled with replace(): analyze sees utcoffset 0 at the SAME
                    instant, and the capsule binds the UTC day '2026-08-12', not the
                    foreign-wall-clock day '2026-08-13'.        [RED now: the +03:00
                    datetime passes through raw; a replace() repair also stays RED]
  point 4 -> DT-4   change-scope FREEZE: post-lift FROZEN_COMPONENTS state exact (both key
                    directions), capsule threshold reaches analyze_region unchanged,
                    CalibrationUnavailable and the data-quality gate stay fail-closed.
                    [green now and after — the repair may move NONE of this]

FREEZE DISCIPLINE: after this bar freezes, grassmann implements the codex-1318 boundary
repair WITHOUT editing this file; one verify; close. The committed 2026-08-12 production
result stays an honest fail-closed/unscored result; any `--date 2026-08-12` re-score is
asylum's separate decision. No renewal/publication/claim authority is read into this bar.
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

MIXED_SHAPE = "offset-naive and offset-aware"          # the exact production defect shape
HONEST_PREFIXES = ("data quality gate failed", "calibration unavailable")


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
TZ3 = timezone(timedelta(hours=3))                     # non-UTC aware zone for DT-3b

NAIVE_MIDNIGHT = datetime(2026, 8, 12)                                  # runner convention
AWARE_MIDNIGHT = datetime(2026, 8, 12, tzinfo=UTC)                      # same legacy instant
PLUS3_INPUT = datetime(2026, 8, 13, 1, 0, tzinfo=TZ3)                   # == 2026-08-12T22:00Z
PLUS3_UTC_INSTANT = datetime(2026, 8, 12, 22, 0, tzinfo=UTC)
DAY = "2026-08-12"

FS = 25.0                       # native rate: exactly the provenance floor
FRAG_OFFSET_S = 5 * 3600        # fragment starts 5 h into the 24 h session (fractional-phase path)
FRAG_SECONDS = 13 * 3600        # 46,800 s interior clears the 0.50 common-support floor after trim
REGION = "istanbul_marmara"     # lifted carrier, identity-mapped runner->FC key, 3 segments


def _to_utc(dt):
    """Stub-side helper only (NOT the seam under test): aware-UTC image of any datetime."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


class _Stats:
    def __init__(self, rate, start):
        self.sampling_rate = rate
        self.starttime = start


class _Trace:
    def __init__(self, data, rate, start):
        self.data = data
        self.stats = _Stats(rate, start)


class _Stream:
    def __init__(self, traces):
        self._traces = traces

    def __len__(self):
        return len(self._traces)

    def __getitem__(self, i):
        return self._traces[i]


def _mk_fetcher(SD, mode="good"):
    """Hermetic fetcher: network edge stubbed, everything downstream REAL. Per segment, two
    DISTINCT NET.STA stations, one contiguous 13 h trace each at 25 Hz with an AWARE-UTC
    start (as _to_aware_utc mints from real obspy stats in production), seeded per
    (segment, station). mode='empty' returns no waveforms at all (data-quality-gate KAT)."""
    fetcher = SD.SeismicDataFetcher.__new__(SD.SeismicDataFetcher)
    fetcher.cache_dir = Path(tempfile.mkdtemp())
    fetcher.cache_ttl_days = 7
    fetcher.clients = {}
    seg_index = {}

    def fetch(segment, start, end, use_cache=True):
        if mode == "empty":
            return {}
        idx = seg_index.setdefault(segment.name, len(seg_index))
        frag_start = _to_utc(start) + timedelta(seconds=FRAG_OFFSET_S)
        out = {}
        for j, suffix in enumerate(("A", "B")):
            rng = np.random.default_rng(1000 + 10 * idx + j)
            data = rng.standard_normal(int(FRAG_SECONDS * FS))
            out[f"XX.S{idx}{suffix}..BHZ"] = _Stream([_Trace(data, FS, frag_start)])
        return out

    fetcher.fetch_segment_waveforms = fetch
    return fetcher


def _mk_ens(EN, FC, SD, mode="good", loader=None):
    """Fresh ensemble wiring per KAT: real monitor + real analysis chain, stubbed network
    edge, spy capsule loader, spy-wrapped analyze_region (records then DELEGATES)."""
    mon = FC.FaultCorrelationMonitor(data_fetcher=_mk_fetcher(SD, mode=mode),
                                     window_hours=24, decorrelation_threshold=0.3)
    seen = {"analyze": [], "loader": []}
    real_analyze = mon.analyze_region

    def spy_analyze(region, target_date, **kw):
        seen["analyze"].append((region, target_date, kw.get("calibration")))
        return real_analyze(region, target_date, **kw)

    mon.analyze_region = spy_analyze
    ens = EN.GeoSpecEnsemble.__new__(EN.GeoSpecEnsemble)
    ens.region = REGION
    ens.fault_corr_monitor = mon

    def spy_loader(region, day_str):
        seen["loader"].append((region, day_str))
        if loader is not None:
            return loader(region, day_str)
        return {"region": region, "threshold": 0.30}

    ens.capsule_loader = spy_loader
    return ens, seen


def _shape_ok(mr):
    """codex point 1 green condition: the mixed-datetime defect shape is ABSENT and the
    component either SCORED or returned only an honest data-quality/calibration result."""
    notes = str(getattr(mr, "notes", ""))
    absent = MIXED_SHAPE not in notes
    honest = (mr.available is True) or any(notes.startswith(p) for p in HONEST_PREFIXES)
    return absent and honest, notes


def _aware_utc(dt):
    return isinstance(dt, datetime) and dt.tzinfo is not None and dt.utcoffset() == timedelta(0)


def main():
    _install_obspy_stub()
    import seismic_data as SD
    import fault_correlation as FC
    import ensemble as EN

    need = (hasattr(EN.GeoSpecEnsemble, "compute_fault_correlation_risk")
            and hasattr(EN, "component_frozen")
            and hasattr(SD.SeismicDataFetcher, "get_segment_envelopes")
            and hasattr(SD, "compute_band_envelope_supported")
            and hasattr(FC, "CalibrationUnavailable"))
    check("DT-0 rev-3 seams present (ensemble entry + segmented-support chain)", need)
    if not need:
        return

    # =====================================================================
    # DT-1 (codex point 1) — THE red KAT: naive runner date, REAL placement path
    # =====================================================================
    ens1, seen1 = _mk_ens(EN, FC, SD)
    try:
        mr1 = ens1.compute_fault_correlation_risk(NAIVE_MIDNIGHT)[0]
        ok1, notes1 = _shape_ok(mr1)
        check("DT-1 NAIVE runner target_date traverses the REAL rev-3 timestamp-placement "
              "path (aware-UTC fragment starts) with NO mixed-datetime TypeError; the "
              "component scores or returns only an honest data-quality result",
              ok1, f"available={mr1.available} notes={notes1!r}")
    except Exception as exc:
        check("DT-1 NAIVE runner target_date traverses the REAL rev-3 timestamp-placement "
              "path (aware-UTC fragment starts) with NO mixed-datetime TypeError; the "
              "component scores or returns only an honest data-quality result",
              False, f"RAISED {type(exc).__name__}: {exc}")

    # =====================================================================
    # DT-2 (codex point 2) — ONE normalized binding: loader day == analyze instant/day
    # =====================================================================
    la = seen1["loader"]
    an = seen1["analyze"]
    got = an[0][1] if an else None
    ok2 = (len(la) == 1 and la[0][1] == DAY
           and len(an) == 1 and _aware_utc(got)
           and got == AWARE_MIDNIGHT
           and got.strftime("%Y-%m-%d") == la[0][1])
    check("DT-2 capsule loader and analyze_region observe the SAME normalized value: "
          "loader day '2026-08-12', analyze target AWARE UTC at the legacy-midnight "
          "instant, day binding unchanged",
          ok2, f"loader={la} analyze_target={got!r}")

    # =====================================================================
    # DT-3a (codex point 3) — aware-UTC input VALUE-PRESERVED (also DT-1's
    # greenability proof: the post-repair normalized path on the same fixture)
    # =====================================================================
    ens3a, seen3a = _mk_ens(EN, FC, SD)
    try:
        mr3a = ens3a.compute_fault_correlation_risk(AWARE_MIDNIGHT)[0]
        got3a = seen3a["analyze"][0][1] if seen3a["analyze"] else None
        shape3a, notes3a = _shape_ok(mr3a)
        ok3a = (shape3a and mr3a.available is True
                and _aware_utc(got3a) and got3a == AWARE_MIDNIGHT
                and seen3a["loader"] and seen3a["loader"][0][1] == DAY)
        check("DT-3a already-AWARE-UTC input is value-preserved end-to-end and the component "
              "SCORES on this fixture (available=True, same instant at analyze, day "
              f"'{DAY}' at the loader)",
              ok3a, f"available={mr3a.available} notes={notes3a!r} "
                    f"analyze_target={got3a!r} loader={seen3a['loader']}")
    except Exception as exc:
        check("DT-3a already-AWARE-UTC input is value-preserved end-to-end and the component "
              "SCORES on this fixture (available=True, same instant at analyze, day "
              f"'{DAY}' at the loader)", False, f"RAISED {type(exc).__name__}: {exc}")

    # =====================================================================
    # DT-3b (codex point 3) — non-UTC aware input CONVERTED (astimezone), never
    # relabelled (replace): same instant, UTC day binding
    # =====================================================================
    ens3b, seen3b = _mk_ens(EN, FC, SD)
    try:
        mr3b = ens3b.compute_fault_correlation_risk(PLUS3_INPUT)[0]
        got3b = seen3b["analyze"][0][1] if seen3b["analyze"] else None
        shape3b, notes3b = _shape_ok(mr3b)
        ok3b = (shape3b
                and _aware_utc(got3b) and got3b == PLUS3_UTC_INSTANT
                and seen3b["loader"] and seen3b["loader"][0][1] == DAY)
        check("DT-3b a +03:00 aware input is CONVERTED to the same UTC instant "
              "(2026-08-13T01:00+03:00 -> 2026-08-12T22:00Z): analyze sees utcoffset 0 at "
              f"the SAME instant and the capsule binds the UTC day '{DAY}' -- a replace() "
              "relabel (day '2026-08-13', shifted instant) FAILS this",
              ok3b, f"available={mr3b.available} notes={notes3b!r} "
                    f"analyze_target={got3b!r} loader={seen3b['loader']}")
    except Exception as exc:
        check("DT-3b a +03:00 aware input is CONVERTED to the same UTC instant "
              "(2026-08-13T01:00+03:00 -> 2026-08-12T22:00Z): analyze sees utcoffset 0 at "
              f"the SAME instant and the capsule binds the UTC day '{DAY}' -- a replace() "
              "relabel (day '2026-08-13', shifted instant) FAILS this",
              False, f"RAISED {type(exc).__name__}: {exc}")

    # =====================================================================
    # DT-4 (codex point 4) — change-scope FREEZE: the repair moves NONE of this
    # =====================================================================
    frozen_pairs = (("tokyo_kanto", True), ("japan_tohoku", True), ("ridgecrest", True),
                    ("istanbul_marmara", False), ("socal_saf_coachella", False),
                    ("socal_coachella", False), ("turkey_kahramanmaras", False))
    ok4a = all(EN.component_frozen(r, "fault_correlation") is want for r, want in frozen_pairs)
    check("DT-4a POST-lift freeze set exact and carrier-keyed both directions (Phase-5 "
          "scope, receipt da321397): blocked {tokyo_kanto, japan_tohoku, ridgecrest} stay "
          "frozen; the three lifted carriers stay unfrozen",
          ok4a, f"{[(r, EN.component_frozen(r, 'fault_correlation')) for r, _ in frozen_pairs]}")

    ok4b = bool(seen1["analyze"]) and seen1["analyze"][0][2] is not None \
        and seen1["analyze"][0][2].get("threshold") == 0.30
    check("DT-4b the capsule threshold (0.30) reaches analyze_region unchanged through the "
          "normalized call (no hardcoded verdict constant)",
          ok4b, f"calibration={seen1['analyze'][0][2] if seen1['analyze'] else None}")

    def _raise_cal(region, day_str):
        raise FC.CalibrationUnavailable(["no capsule registered"])

    ens4c, _ = _mk_ens(EN, FC, SD, loader=None)
    ens4c.capsule_loader = _raise_cal
    try:
        mr4c = ens4c.compute_fault_correlation_risk(NAIVE_MIDNIGHT)[0]
        ok4c = mr4c.available is False and str(mr4c.notes).startswith("calibration unavailable")
        check("DT-4c CalibrationUnavailable stays FAIL-CLOSED on the naive legacy input "
              "(available=False, honest note, zero effective weight)",
              ok4c, f"available={mr4c.available} notes={mr4c.notes!r}")
    except Exception as exc:
        check("DT-4c CalibrationUnavailable stays FAIL-CLOSED on the naive legacy input "
              "(available=False, honest note, zero effective weight)",
              False, f"RAISED {exc}")

    ens4d, _ = _mk_ens(EN, FC, SD, mode="empty")
    try:
        mr4d = ens4d.compute_fault_correlation_risk(NAIVE_MIDNIGHT)[0]
        ok4d = (mr4d.available is False
                and str(mr4d.notes).startswith("data quality gate failed")
                and MIXED_SHAPE not in str(mr4d.notes))
        check("DT-4d the data-quality gate stays FAIL-CLOSED on the naive legacy input when "
              "no waveforms exist (available=False, honest QC note, never the mixed-datetime "
              "error, never a fabricated score)",
              ok4d, f"available={mr4d.available} notes={mr4d.notes!r}")
    except Exception as exc:
        check("DT-4d the data-quality gate stays FAIL-CLOSED on the naive legacy input when "
              "no waveforms exist (available=False, honest QC note, never the mixed-datetime "
              "error, never a fabricated score)",
              False, f"RAISED {exc}")


main()
print()
if FAILS:
    print(f"D2 PHASE-5 DATETIME-SEAM RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 PHASE-5 DATETIME-SEAM RED-KATs PASS (one aware-UTC scored-instant binding "
      "across admission + analysis; convert-not-relabel; freeze/fail-closed scope intact)")
