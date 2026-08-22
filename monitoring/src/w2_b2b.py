#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 B2B family engine (cayley) -- churn-tolerant partition runs
per the FROZEN annex docs/f2g_window2_freeze/annex_b2b.md (design
freeze CLOSED @ 12161f6/5fba544) and grassmann's bar seam pin
(`w2_b2b_family(panel, ...)`, calendar-frame, Phase-B result-shape
conventions). Seam name FIXED as `w2_b2b`.

Frozen statistic (annex, verbatim semantics): over the evaluation
calendar positions in order -- (1) a day is a CANDIDATE iff its capsule
passes the registered B2-class intrinsic gate battery (gap check +
`_b2_partition` gates, IMPORTED from the pinned Phase-B engine, never
reimplemented); (2) for each adjacent candidate pair,
I_d = MEASURED(d_prev) & MEASURED(d) with overlap floor
ceil(2/3 * |registry_carrier|) (below -> INTERSECTION_BELOW_FLOOR,
run terminates, never bridged); (3) the registered Fiedler partition is
recomputed on the subgraph induced by I_d for BOTH days (identical
eigengap/coordinate gates, typed INDUCED_*); (4) label-invariant
comparison: equal iff the two partitions induce the same UNORDERED
bipartition of I_d; (5) each side must hold >= 2 stations of I_d
(PARTITION_DEGENERATE_SIDE); (6) absences and typed refusals terminate
the current run atomically; (7) family statistic = total run count over
carriers, one-sided LOW.

Interpretation pins (disclosed, R1.2-able):
- PAIR-level refusal semantics: the refusal terminates the run that the
  reference day belonged to; the CURRENT day remains an intrinsically
  valid candidate, so it OPENS a new run (+1) and becomes the next
  reference. Day-level (intrinsic) refusals clear the reference
  entirely, exactly as `_b2a_runs` does (codex 8621baf2: excluded
  positions clear the frame-comparison reference).
- INDUCED_NODESET_INCOMPLETE: if an induced partition's node set is not
  exactly I_d (isolated stations fall outside the largest component),
  the "same bipartition of I_d" premise fails -> typed, run terminates.
  Refusal beats a silent partial comparison (the annex's never-mis-count
  requirement).
- MEASURED_EDGE_INCONSISTENT: a finite edge endpoint absent from the
  day's measured set is a producer defect -> typed intrinsic refusal.
- Pair comparisons are memoized on the unordered day-index pair inside
  one family call (canonical low-high computation order). This is a
  pure-function cache: the annex's raw-recompute-per-draw requirement
  is honored mathematically -- identical inputs yield identical
  results; no observed value leaks across draws.

Null: the registered B2A calendar-position permutation scheme (ONE
common permutation across carriers per draw), intersection ->
admission -> induced partition -> comparison -> runs recomputed per
draw; 9,999 draws, add-one p, valid-draw floor per the pinned Phase-B
conventions; substream seed = derive_substream_seed(doc_sha256,
family="B2B", fold, "null") with doc_sha256 = the window-2 freeze sha.

This module opens no window-2 value; it runs only where the barrier
instrument authorizes.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d2_f2g_phase_b_stats as _pb  # the PINNED Phase-B engine

OVERLAP_FLOOR_NUM = 2
OVERLAP_FLOOR_DEN = 3
SIDE_MINIMUM = 2
ALPHA_FAMILY = 0.05  # family-wise via the non-circular Holm selector


class PanelInvalid(ValueError):
    """Typed CALENDAR_* / REGISTRY_* panel-shape refusals."""


def overlap_floor(registry_size):
    return math.ceil(OVERLAP_FLOOR_NUM * registry_size
                     / OVERLAP_FLOOR_DEN)


def _day_edge_weights(carrier, d):
    ew = {}
    for e, series in carrier["r"].items():
        if d in series:
            v = float(series[d])
            if math.isfinite(v):
                ew[_pb._canonical_edge(e)] = max(v, 0.0)
    return ew


def _intrinsic_states(carrier, calendar):
    """Per calendar position: (capsule dict, None) for a CANDIDATE day,
    else (None, typed refusal code). Gate battery = the pinned Phase-B
    B2-class battery, imported."""
    regset = set(carrier["registered_days"])
    measured = carrier.get("measured", {})
    states = []
    for d in calendar:
        if d not in regset:
            states.append((None, "NO_REGISTERED_SNAPSHOT"))
            continue
        if d not in measured:
            states.append((None, "MEASURED_SET_ABSENT"))
            continue
        ew = _day_edge_weights(carrier, d)
        if not ew:
            states.append((None, "GAP_NO_MEASURED_EDGES"))
            continue
        mset = frozenset(measured[d])
        endpoints = {s for e in ew for s in e}
        if not endpoints <= mset:
            states.append((None, "MEASURED_EDGE_INCONSISTENT"))
            continue
        part, code = _pb._b2_partition(ew)
        if code:
            states.append((None, code))
            continue
        states.append(({"ew": ew, "measured": mset}, None))
    return states


def _compare_pair(cap_a, cap_b, floor):
    """(equal_bool, None) or (None, typed code). Symmetric in count
    semantics; callers pass canonical (low, high) order so reported
    codes are deterministic."""
    I = cap_a["measured"] & cap_b["measured"]
    if len(I) < floor:
        return None, "INTERSECTION_BELOW_FLOOR"
    bips = []
    for cap in (cap_a, cap_b):
        induced = {e: w for e, w in cap["ew"].items()
                   if e[0] in I and e[1] in I}
        part, code = _pb._b2_partition(induced)
        if code:
            return None, f"INDUCED_{code}"
        if set(part) != I:
            return None, "INDUCED_NODESET_INCOMPLETE"
        side_pos = frozenset(s for s in part if part[s] == 1)
        side_neg = frozenset(s for s in part if part[s] == -1)
        if len(side_pos) < SIDE_MINIMUM or len(side_neg) < SIDE_MINIMUM:
            return None, "PARTITION_DEGENERATE_SIDE"
        bips.append(frozenset({side_pos, side_neg}))
    return bips[0] == bips[1], None


def _runs_over_order(states, order, floor, cache, days=None,
                     refusals=None, carrier=None):
    """Run count over one ordered capsule sequence; the annex state
    machine with the disclosed pair-refusal semantics."""
    runs = 0
    candidates = 0
    prev = None  # index of the current reference candidate
    for i in order:
        cap, code = states[i]
        if code:
            if refusals is not None:
                refusals.append({"carrier": carrier, "day": days[i],
                                 "code": code})
            prev = None
            continue
        candidates += 1
        if prev is None:
            runs += 1
            prev = i
            continue
        key = (prev, i) if prev < i else (i, prev)
        if key not in cache:
            cache[key] = _compare_pair(states[key[0]][0],
                                       states[key[1]][0], floor)
        equal, pcode = cache[key]
        if pcode is not None:
            if refusals is not None:
                refusals.append({"carrier": carrier, "day": days[i],
                                 "vs": days[prev], "code": pcode})
            runs += 1        # current day opens the next run
            prev = i
            continue
        if not equal:
            runs += 1        # SWITCH event
        prev = i
    return runs, candidates


def w2_b2b_family(panel, *, doc_sha256, n_draws=_pb.N_DRAWS,
                  return_null=False, power_contract=None, fold="full"):
    calendar = panel.get("calendar")
    if not isinstance(calendar, list) or len(calendar) < 2:
        raise PanelInvalid("CALENDAR_EMPTY: need >= 2 positions")
    if calendar != sorted(calendar) or len(set(calendar)) != \
            len(calendar):
        raise PanelInvalid("CALENDAR_UNORDERED")
    keys = sorted(panel["carriers"])
    n_pos = len(calendar)

    caps = {}
    floors = {}
    caches = {}
    day_refusals = []
    runs_by_carrier = {}
    R_obs = 0
    identity = list(range(n_pos))
    for k in keys:
        c = panel["carriers"][k]
        registry = c.get("registry")
        if not registry:
            raise PanelInvalid(f"REGISTRY_ABSENT: {k}")
        floors[k] = overlap_floor(len(registry))
        caps[k] = _intrinsic_states(c, calendar)
        caches[k] = {}
        runs, candidates = _runs_over_order(
            caps[k], identity, floors[k], caches[k], days=calendar,
            refusals=day_refusals, carrier=k)
        runs_by_carrier[k] = runs
        if candidates < 2:
            out = {"family": "B2B", "frame": "calendar-w2",
                   "runs_total": None, "T_obs": None,
                   "runs_by_carrier": runs_by_carrier,
                   "day_refusals": day_refusals,
                   "overlap_floors": floors, "n_draws": int(n_draws),
                   "alpha": ALPHA_FAMILY,
                   "alpha_note": "family-wise via non-circular Holm at "
                                 "the selector (v0.3 sec 5)",
                   "p_value": None, "n_valid_draws": 0,
                   "verdict": "CANNOT_DETERMINE_FAMILY_SCORABILITY "
                              "(CARRIER_NO_COMPARABLE_SEQUENCE)"}
            if return_null:
                out["null_R"] = []
            return out
        R_obs += runs

    out = {"family": "B2B", "frame": "calendar-w2",
           "runs_total": int(R_obs), "T_obs": int(R_obs),
           "runs_by_carrier": runs_by_carrier,
           "day_refusals": day_refusals, "overlap_floors": floors,
           "n_draws": int(n_draws), "alpha": ALPHA_FAMILY,
           "alpha_note": "family-wise via non-circular Holm at the "
                         "selector (v0.3 sec 5)",
           "fold": str(fold)}
    rng = _pb._rng(doc_sha256, "B2B", fold, "null")
    null_R = []
    null_orders = []
    null_rbc = []
    n_valid = le = 0
    for _ in range(int(n_draws)):
        perm = [int(x) for x in rng.permutation(n_pos)]
        R_d = 0
        rbc = {}
        for k in keys:
            runs, _cand = _runs_over_order(caps[k], perm, floors[k],
                                           caches[k])
            rbc[k] = int(runs)
            R_d += runs
        n_valid += 1
        null_R.append(int(R_d))
        null_orders.append(perm)
        null_rbc.append(rbc)
        if R_d <= R_obs:   # one-sided LOW (fewer runs = persistence)
            le += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_R"] = null_R
        out["null_orders"] = null_orders
        out["null_runs_by_carrier"] = null_rbc
    if n_valid < _pb._valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + le) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _pb._typed_verdict(p, power_contract)
    return out


# ---------------------------------------------------------------- selftest
def _mk_day(cluster_a, cluster_b, strong=5.0, weak=0.1):
    """Edge dict: strong intra-cluster, weak inter-cluster -> Fiedler
    bipartition {A, B}."""
    ew = {}
    nodes = list(cluster_a) + list(cluster_b)
    for i, x in enumerate(nodes):
        for y in nodes[i + 1:]:
            same = (x in cluster_a) == (y in cluster_a)
            ew["|".join(sorted((x, y)))] = strong if same else weak
    return ew


def _carrier(days, day_edges, measured, registry):
    r = {}
    for d, ew in zip(days, day_edges):
        for e, w in ew.items():
            r.setdefault(e, {})[d] = w
    return {"registry": list(registry), "registered_days": list(days),
            "measured": {d: sorted(m) for d, m in
                         zip(days, measured)},
            "r": r}


def _selftest():
    days = [f"2026-09-{i:02d}" for i in range(1, 7)]
    A = [f"A{i}" for i in range(5)]
    B = [f"B{i}" for i in range(5)]
    reg = A + B                       # floor = ceil(2/3*10) = 7
    assert overlap_floor(10) == 7 and overlap_floor(9) == 6

    def fam(car, n_draws=199):
        return w2_b2b_family(
            {"calendar": days, "carriers": {"x": car}},
            doc_sha256="ab" * 32, n_draws=n_draws)

    # 1. stable: identical days -> 1 run, p == 1 under permutation
    car = _carrier(days, [_mk_day(A, B)] * 6, [reg] * 6, reg)
    r = fam(car)
    assert r["runs_by_carrier"]["x"] == 1 and r["p_value"] == 1.0, r

    # 2a. genuine switch: last 3 days repartition (A0,A1,B0..) -> 2 runs
    A2 = A[:2] + B[:3]
    B2 = A[2:] + B[3:]
    car = _carrier(days, [_mk_day(A, B)] * 3 + [_mk_day(A2, B2)] * 3,
                   [reg] * 6, reg)
    r = fam(car)
    assert r["runs_by_carrier"]["x"] == 2, r

    # 2b. label-permutation invariance: same bipartition, cluster
    # arguments swapped -> NOT a switch (1 run)
    car = _carrier(days, [_mk_day(A, B)] * 3 + [_mk_day(B, A)] * 3,
                   [reg] * 6, reg)
    r = fam(car)
    assert r["runs_by_carrier"]["x"] == 1, r

    # 3. overlap floor boundary (registry 9 -> floor 6): shared 6
    # passes, shared 5 refuses typed
    reg9 = A + B[:4]
    m_all = reg9
    m6 = A[:3] + B[:3]                      # |I|=6 vs m_all
    car = _carrier(days[:2], [_mk_day(A, B[:4]),
                              _mk_day(A[:3], B[:3])],
                   [m_all, m6], reg9)
    r = w2_b2b_family({"calendar": days[:2], "carriers": {"x": car}},
                      doc_sha256="ab" * 32, n_draws=99)
    assert r["runs_by_carrier"]["x"] in (1, 2)
    assert not any(x["code"] == "INTERSECTION_BELOW_FLOOR"
                   for x in r["day_refusals"]), r
    m5 = A[:3] + B[:2]                      # |I|=5 < 6 -> refusal
    car = _carrier(days[:2], [_mk_day(A, B[:4]),
                              _mk_day(A[:3], B[:2])],
                   [m_all, m5], reg9)
    r = w2_b2b_family({"calendar": days[:2], "carriers": {"x": car}},
                      doc_sha256="ab" * 32, n_draws=99)
    assert any(x["code"] == "INTERSECTION_BELOW_FLOOR"
               for x in r["day_refusals"]), r
    assert r["runs_by_carrier"]["x"] == 2, r   # each opens its own run

    # 4. degenerate side: induced I with a 1-station side -> typed
    lone = ["A0"]
    rest = [f"C{i}" for i in range(4)]
    regd = lone + rest + ["D0", "D1", "D2"]   # registry 8 -> floor 6
    m_d = lone + rest + ["D0"]                # I = 6 stations, 5 vs 1
    day_d = _mk_day(rest + ["D0"], lone)      # cluster sizes 5 / 1
    car = _carrier(days[:2], [day_d, day_d], [m_d, m_d], regd)
    r = w2_b2b_family({"calendar": days[:2], "carriers": {"x": car}},
                      doc_sha256="ab" * 32, n_draws=99)
    assert any(x["code"] == "PARTITION_DEGENERATE_SIDE"
               for x in r["day_refusals"]), r

    # 5. adversarial alternating dropout: every pair below floor ->
    # every candidate opens a run (runs == candidates), ALL typed,
    # never mis-counted
    mA = A + B[:2]                            # 7 measured
    mB = A[:2] + B                            # 7 measured, |I|=4 < 7
    alt_days = [_mk_day(A, B[:2]) if i % 2 == 0 else
                _mk_day(A[:2], B) for i in range(6)]
    alt_meas = [mA if i % 2 == 0 else mB for i in range(6)]
    car = _carrier(days, alt_days, alt_meas, reg)
    r = fam(car)
    assert r["runs_by_carrier"]["x"] == 6, r
    assert sum(1 for x in r["day_refusals"]
               if x["code"] == "INTERSECTION_BELOW_FLOOR") == 5, r

    # 6. absence mid-sequence terminates atomically: 1 missing
    # registered day in a stable sequence -> 2 runs
    car = _carrier(days, [_mk_day(A, B)] * 6, [reg] * 6, reg)
    car["registered_days"] = [d for d in days if d != days[3]]
    r = fam(car)
    assert r["runs_by_carrier"]["x"] == 2, r
    assert any(x["code"] == "NO_REGISTERED_SNAPSHOT"
               for x in r["day_refusals"]), r

    # 7. determinism: identical inputs + seed -> identical p
    car = _carrier(days, [_mk_day(A, B)] * 3 + [_mk_day(A2, B2)] * 3,
                   [reg] * 6, reg)
    p1 = fam(car)["p_value"]
    p2 = fam(car)["p_value"]
    assert p1 == p2

    print("w2_b2b selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
