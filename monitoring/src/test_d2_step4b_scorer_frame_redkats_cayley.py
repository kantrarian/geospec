#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B SCORER-FRAME red-KATs (cayley, 2026-08-10) — freezes the fix-A contract ruling
(cayley 1257 `35928ce` + codex 1303 concurrence) for the acceptance-HOLD finding: a
(carrier, day) is ADMITTED only when EVERY DECLARED SEGMENT survives the pre-outcome support
rule. Subset-frame scoring (dropping a dark segment and admitting on survivors) is the
defect this bar makes unreachable. HERMETIC: SD/FC are injected parameters of `_score_day`;
the FC stand-in reproduces codex's executable counterexamples (an aggregator that does NOT
check station eligibility — exactly the real module's behavior the repair must compensate
for) plus the real gate contract (AND-mask common floor). obspy/scipy stubbed for import.

CONTRACT (codex 1303 repair, grassmann implements in `d2_step4b_campaign_run._score_day`
UNEDITED; the 0123 acceptance is NOT relaxed):
  1. declared_segments = sorted(seg_station_es) — every key is part of the frame;
  2. each segment's station series filter through `FC.station_eligible` (individual
     supported coverage >= 0.50; 0.50 REMAINS eligible), distinct NET.STA contributors only;
  3. BEFORE matrix construction, fail closed if ANY declared segment has < 2 eligible
     distinct stations, no aggregate, or aggregate valid bins < 43,200;
  4. the correlation matrix is computed ONLY over the complete declared segment list; the
     existing all-segment AND-mask common-support >= 43,200 gate is retained;
  5. an ADMITTED row satisfies
     set(segment_support) == set(segment_names) == set(declared_segments).
  REJECTED rows carry deterministic segment-naming reasons in SORTED segment order:
     `INSUFFICIENT_ELIGIBLE_STATIONS:<segment>` / `SEGMENT_AGGREGATE_BELOW_FLOOR:<segment>`
  and mint NO scalar/matrix/eigenvalue fields (all null).

RED AS AUTHORED vs GeoSpec `dd5b5e0`: exactly ['A1', 'A2'] (the two codex-reproduced
counterexamples — current `_score_day` ADMITS both). A3 and the companions are green-side
semantics pins (boundary >= eligibility must SURVIVE the repair — a lazy `> 0.5` or
`> 43200` repair fails A3).
"""
import sys
import types

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []
GRID = 86400
FLOOR = 43200


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def _install_import_stubs():
    for name in ("obspy", "scipy"):
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            m.__version__ = "0.0-stub"
            sys.modules[name] = m
    for name in ("seismic_data", "fault_correlation"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


class _ES:
    """EnvelopeSeries stand-in: coverage + 86,400-bin validity mask."""

    def __init__(self, mask):
        self.valid_mask = np.asarray(mask, dtype=bool)
        self.coverage = float(self.valid_mask.sum()) / GRID


def _mask(n_valid, offset=0):
    m = np.zeros(GRID, dtype=bool)
    m[offset:offset + n_valid] = True
    return m


class _FC:
    """FC stand-in reproducing the REAL module's seams for this contract:
    - station_eligible: individual supported coverage >= 0.50 (0.50 eligible);
    - aggregate_segment_supported: median-stack that does NOT check eligibility (codex's
      counterexample #2 depends on this — the repair must filter BEFORE aggregation) and
      carries the >=1-station intersection mask;
    - compute_correlation_matrix_supported: full k x k unit-diagonal matrix over the given
      series; enforces the AND-mask common-support >= 43,200 gate (returns (None, names, qc)
      below it), mirroring the retained real gate."""

    @staticmethod
    def station_eligible(es):
        return es is not None and es.coverage >= 0.5

    @staticmethod
    def aggregate_segment_supported(series):
        series = [s for s in series if s is not None]
        if len(series) < 2:
            return None
        stack = np.vstack([s.valid_mask for s in series]).astype(int)
        agg_mask = stack.sum(axis=0) >= 2          # bins where >=2 stations contribute
        if not agg_mask.any():
            return None
        return _ES(agg_mask)

    @staticmethod
    def compute_correlation_matrix_supported(seg_series, seg_names):
        k = len(seg_series)
        if k < 2:
            return None, list(seg_names), ["INSUFFICIENT_SEGMENTS"]
        masks = np.vstack([s.valid_mask for s in seg_series])
        common = int(masks.all(axis=0).sum())
        if common < FLOOR:
            return None, list(seg_names), ["COMMON_SUPPORT_BELOW_FLOOR"]
        C = np.full((k, k), 0.3)
        np.fill_diagonal(C, 1.0)
        return C.tolist(), list(seg_names), []


class _SD:
    class DataUnavailable(Exception):
        pass


def _sorted_reject_reasons(row):
    return [r for r in row.get("qc_reasons", [])
            if r.startswith(("INSUFFICIENT_ELIGIBLE_STATIONS:",
                             "SEGMENT_AGGREGATE_BELOW_FLOOR:"))]


def main():
    _install_import_stubs()
    from datetime import datetime, timezone
    import d2_step4b_campaign_run as RUN
    session_start = datetime(2026, 5, 1, 7, 0, 13, 94647, tzinfo=timezone.utc)

    def score(seg_station_es):
        return RUN._score_day(_SD, _FC, seg_station_es, session_start)

    full = _mask(GRID)

    # ---- A1: SUBSET REFUSAL (codex counterexample #1, real-data dark segment) ----
    day = score({
        "alive_a": [("KO.A1..HHZ", _ES(full)), ("KO.A2..HHZ", _ES(full))],
        "alive_b": [("KO.B1..HHZ", _ES(full)), ("KO.B2..HHZ", _ES(full))],
        "dark_required": [("KO.D1..HHZ", None), ("KO.D2..HHZ", None)],
    })
    named = _sorted_reject_reasons(day)
    check("A1 SUBSET REFUSAL: two fully supported segments + one declared DARK segment -> "
          "REJECTED, the failed segment NAMED in a typed reason, and NO "
          "scalar/matrix/eigenvalue fields minted (the scorer may not drop the segment and "
          "admit on survivors)",
          day["status"] == "REJECTED"
          and any(r.endswith(":dark_required") for r in named)
          and day["ratio"] is None and day["correlation_matrix"] is None
          and day["ordered_eigenvalues"] is None
          and day["participation_ratio"] is None,
          f"status={day['status']} reasons={day.get('qc_reasons')} ratio={day.get('ratio')}")

    # ---- A2: STATION-FLOOR COMPOSITION (codex counterexample #2) -----------------
    # Three stations per segment at 0.49 coverage each (individually INELIGIBLE), cyclic
    # overlap so the eligibility-blind aggregator still yields an aggregate ABOVE the
    # 43,200 floor. The repair must filter through station_eligible BEFORE aggregation and
    # refuse (<2 eligible stations), regardless of how healthy the aggregate looks.
    # pairwise tiling (codex's 63,504 construction): region [0, 63504) in three equal
    # parts P1|P2|P3 of 21,168; station 0 covers P1+P2, station 1 covers P2+P3,
    # station 2 covers P1+P3 -> each station 42,336 bins (0.49, ineligible), every covered
    # bin has EXACTLY 2 contributors -> the >=2 aggregate is 63,504 bins (>= floor).
    part = 21168
    spans = {0: [(0, 2 * part)], 1: [(part, 3 * part)],
             2: [(0, part), (2 * part, 3 * part)]}
    cyc = {}
    for seg in ("seg_x", "seg_y"):
        stations = []
        for i in range(3):
            m = np.zeros(GRID, dtype=bool)
            for a, b in spans[i]:
                m[a:b] = True
            stations.append((f"KO.{seg[-1].upper()}{i}..HHZ", _ES(m)))
        cyc[seg] = stations
    agg_probe = _FC.aggregate_segment_supported([es for (_n, es) in cyc["seg_x"]])
    day2 = score(cyc)
    named2 = _sorted_reject_reasons(day2)
    check("A2 STATION-FLOOR COMPOSITION: 3x0.49-coverage stations (each individually "
          "ineligible) whose cyclic overlap aggregates ABOVE the floor must REJECT with "
          "INSUFFICIENT_ELIGIBLE_STATIONS (eligibility filters BEFORE aggregation; an "
          "agg-is-None-only repair fails here)",
          agg_probe is not None
          and int(agg_probe.valid_mask.sum()) >= FLOOR      # the trap is genuinely armed
          and day2["status"] == "REJECTED"
          and any(r.startswith("INSUFFICIENT_ELIGIBLE_STATIONS:") for r in named2)
          and day2["ratio"] is None,
          f"agg_bins={agg_probe and int(agg_probe.valid_mask.sum())} "
          f"status={day2['status']} reasons={day2.get('qc_reasons')}")
    check("A2b typed reasons are emitted in SORTED segment order and name every failed "
          "segment",
          named2 == sorted(named2)
          and any(r.endswith(":seg_x") for r in named2)
          and any(r.endswith(":seg_y") for r in named2),
          f"named={named2}")

    # ---- A3: EXACT BOUNDARY (>=, not >) ------------------------------------------
    # Two stations per declared segment at EXACTLY 0.50 coverage on the SAME bins ->
    # segment aggregates and the all-segment AND-mask common support are EXACTLY 43,200.
    # This day must remain ELIGIBLE: 0.50 stations eligible, 43,200-bin aggregates and
    # common support at the floor admit. A lazy '>' repair fails here.
    half = _mask(FLOOR)
    at_floor = {
        "seg_p": [("KO.P1..HHZ", _ES(half)), ("KO.P2..HHZ", _ES(half))],
        "seg_q": [("KO.Q1..HHZ", _ES(half)), ("KO.Q2..HHZ", _ES(half))],
    }
    day3 = score(at_floor)
    check("A3 EXACT BOUNDARY: >=2 stations per declared segment at exactly 0.50 coverage, "
          "segment aggregates and common support exactly 43,200 -> ADMITTED (locks >= "
          "semantics at every gate)",
          day3["status"] == "ADMITTED"
          and day3["common_support_count"] == FLOOR
          and day3["ratio"] is not None,
          f"status={day3['status']} common={day3.get('common_support_count')} "
          f"reasons={day3.get('qc_reasons')}")
    check("A3b an ADMITTED row's frame is COMPLETE: "
          "set(segment_support) == set(segment_names) == declared",
          day3["status"] == "ADMITTED"
          and set(day3["segment_support"]) == set(day3["segment_names"])
          == {"seg_p", "seg_q"},
          f"support={sorted(day3.get('segment_support', {}))} "
          f"names={day3.get('segment_names')}")

    # companion: the retained all-segment common gate still refuses BELOW the floor
    just_below = {
        "seg_p": [("KO.P1..HHZ", _ES(_mask(FLOOR))), ("KO.P2..HHZ", _ES(_mask(FLOOR)))],
        "seg_q": [("KO.Q1..HHZ", _ES(_mask(FLOOR, offset=1))),
                  ("KO.Q2..HHZ", _ES(_mask(FLOOR, offset=1)))],
    }                                             # AND-mask across segments = 43,199
    day4 = score(just_below)
    check("A3c the retained all-segment common-support gate refuses at 43,199 (one bin "
          "below the floor) even though every declared segment individually forms",
          day4["status"] == "REJECTED" and day4["ratio"] is None,
          f"status={day4['status']} reasons={day4.get('qc_reasons')}")


main()
print()
if FAILS:
    print(f"D2 STEP-4B SCORER-FRAME RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B SCORER-FRAME RED-KATs PASS (complete declared frame + "
      "eligibility-before-aggregation + exact >= boundary)")
