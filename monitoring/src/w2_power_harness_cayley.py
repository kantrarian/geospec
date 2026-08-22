#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 POWER-CERTIFICATION HARNESS, core (cayley) -- prereg v0.3
sec 6 ("full window-1 instrument ... executed through the ENTIRE sec-5
rejection graph") + the frozen family effect grids (annex_b2b sec
"Effect grid", annex_b1b sec "Effect grid + specificity gate").

WHAT THIS IS: the machinery -- synthetic generator under the REGISTERED
law, per-family injection operators, the full-four-member-Holm
replicate rule, and the verbatim Tier stopping contract (R=20 evaluate;
CP-LB >= 0.80 -> CERTIFIED; one-sided 95% exact UPPER < 0.80 ->
FAILED; else extend to R=40; straddle -> typed
CANNOT_DETERMINE_POWER_ESTIMATE, never rounded).

WHAT THIS IS NOT: a certification run. Certified numbers require the
registered scale (Tier-C n_draws=9,999), the PRESTART anticipated-mask
geometry (and later the Stage-3 true masks), and the codex round. The
selftest here is FIXTURE-SCALE MECHANISM VERIFICATION ONLY and is
labeled so in every artifact -- fixture numbers can NEVER populate a
certified MDE or a passed power contract (the pinned Tier-S rule,
applied a fortiori).

Registered generator law (pinned Phase-B constants, carried verbatim):
u_ab(d) = MU0 + GAMMA*G(d) + s_a(d) + s_b(d) + eps_ab(d), G ~ N(0,1)
drawn once per calendar day common to all carriers; values = tanh(u);
MCAR cell dropout. MU0 = atanh(0.30), SIGMA_S = 0.15, SIGMA_E = 0.20,
MCAR = 0.08, GAMMA = 0.05.

Injection operators (frozen annex classes):
- B2A / B2B swap: m in {1,2,3} block swaps at the registered onset.
- B2B churn-robustness: (m=2 swap) x synthetic per-day station dropout
  at {10%, 25%} applied to the injected panel (dropped stations leave
  the measured set AND their incident edges that day -- producer-
  consistent).
- B1B detection: the registered (delta_lat, k, n_e) burst grid.
- B1B specificity (the KOZT class): ONE station's incident-edge raw
  values scaled x{3, 10} from a registered onset; the certified
  contract requires <= 0.05 FAMILYWISE positive rate on this class
  alongside CP-LB >= 0.80 on detection classes.

Replicate rule (sec 5): every replicate runs ALL FOUR families
(B2A/B2B/B1B/B3A) on the same panel; Holm at alpha 0.05 over the four
p-values; recovery for target h iff h rejects under that Holm. A family
returning no p (typed) is a non-rejection with m held at 4
(conservative; disclosed pin).

Geometry is INJECTED (anticipated-mask envelope at PRESTART; reduced
fixtures here) -- carriers, registries, masks, calendar, B1B block
geometry. B2A/B3A run via the PINNED engine functions; B2B/B1B via the
w2 engines. This module opens no window-2 value.
"""
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import d2_f2g_phase_b_stats as _pb
import w2_b2b as _b2b
import w2_b1b as _b1b

# registered generator law (pinned)
MU0 = math.atanh(0.30)
SIGMA_S = 0.15
SIGMA_E = 0.20
MCAR = 0.08
GAMMA = 0.05

# stopping contract (pinned, verbatim)
CP_FLOOR = 0.80
R_FIRST = 20
R_MAX = 40
ALPHA_HOLM = 0.05
ARTIFACT_MAX_RATE = 0.05
GRAPH = ("B1B", "B2A", "B2B", "B3A")

TIER_LABEL_FIXTURE = "FIXTURE_SMOKE (never certifiable)"


class PowerHarnessError(ValueError):
    pass


# ------------------------------------------------ exact binomial bounds
def cp_lb(k, n):
    """One-sided 95% exact (Clopper-Pearson) LOWER bound."""
    if k <= 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(0.05, k, n - k + 1))


def cp_ub(k, n):
    """One-sided 95% exact UPPER bound."""
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(0.95, k + 1, n - k))


def certify(success_fn, r_first=R_FIRST, r_max=R_MAX, floor=CP_FLOOR):
    """The verbatim stopping rule. success_fn(r) -> bool, called
    lazily replicate by replicate; the trace records every step."""
    succ = [bool(success_fn(r)) for r in range(r_first)]
    k, n = sum(succ), len(succ)
    trace = [{"R": n, "k": k, "lb": cp_lb(k, n), "ub": cp_ub(k, n)}]
    if trace[-1]["lb"] >= floor:
        status = "CERTIFIED"
    elif trace[-1]["ub"] < floor:
        status = "FAILED"
    else:
        succ += [bool(success_fn(r)) for r in range(r_first, r_max)]
        k, n = sum(succ), len(succ)
        trace.append({"R": n, "k": k, "lb": cp_lb(k, n),
                      "ub": cp_ub(k, n)})
        if trace[-1]["lb"] >= floor:
            status = "CERTIFIED"
        elif trace[-1]["ub"] < floor:
            status = "FAILED"
        else:
            status = "CANNOT_DETERMINE_POWER_ESTIMATE"
    return {"status": status, "R": n, "k": k, "lb": trace[-1]["lb"],
            "ub": trace[-1]["ub"], "floor": floor, "trace": trace}


# ------------------------------------------------ geometry + generator
def fixture_geometry(n_days=84, n_stations=10, seed_tag="fx"):
    """Reduced fixture geometry: ONE carrier, n_days registered days
    (B2A needs > 60 for its 60-day walk-forward baseline), two segments.
    The PRESTART envelope supplies the real anticipated geometry."""
    days = [f"D{i:03d}" for i in range(n_days)]
    sts = [f"S{i:02d}" for i in range(n_stations)]
    return {"carriers": ["cx"], "registry": {"cx": sts},
            "masks": {"cx": days}, "calendar": days,
            "segments": {"cx": {s: ("segA" if i < n_stations // 2
                                    else "segB")
                                for i, s in enumerate(sts)}},
            "eval_start_index": 60,           # walk-forward baseline
            "b1b": {"n_blocks": 14, "block_len": 6,
                    "baseline_positions": 60, "testable_min": 18},
            "seed_tag": seed_tag}


def _edges_of(stations):
    return ["|".join(sorted((a, b))) for i, a in enumerate(stations)
            for b in stations[i + 1:]]


def rep_seed(seed_root, family, point, r):
    material = json.dumps([seed_root, family, point, int(r)],
                          sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def make_panel(geom, family, point, r, seed_root, inject=True):
    """One synthetic replicate panel under the registered law, with the
    family's injection applied to the (single) fixture carrier.
    Returns {"u_panel", "measured", "dropped"} raw pieces plus the
    assembled per-engine shapes via panel_views()."""
    rng = np.random.Generator(np.random.PCG64(
        rep_seed(seed_root, family, point, r)))
    cal = geom["calendar"]
    G = rng.standard_normal(len(cal))
    ck = geom["carriers"][0]
    sts = geom["registry"][ck]
    eds = _edges_of(sts)
    days = geom["masks"][ck]
    s = rng.normal(0.0, SIGMA_S, size=(len(sts), len(days)))
    eps = rng.normal(0.0, SIGMA_E, size=(len(eds), len(days)))
    mcar = rng.random((len(eds), len(days))) < MCAR
    six = {st: i for i, st in enumerate(sts)}
    cpos = {d: i for i, d in enumerate(cal)}
    gvec = np.array([G[cpos[d]] for d in days])
    u = np.empty((len(eds), len(days)))
    for j, e in enumerate(eds):
        a, b = e.split("|")
        u[j] = MU0 + GAMMA * gvec + s[six[a]] + s[six[b]] + eps[j]

    ev0 = geom["eval_start_index"]
    half = len(sts) // 2
    dropped = {d: set() for d in days}
    gain = {}

    if family in ("B2A", "B2B") and inject:
        m = int(point.get("m", 0))
        if m > 0:
            onset = ev0 + max(1, (len(days) - ev0) // 3)
            block = {st: (0 if i < half else 1)
                     for i, st in enumerate(sts)}
            swapped = dict(block)
            for st in sts[half - m:half]:
                swapped[st] = 1
            for st in sts[half:half + m]:
                swapped[st] = 0
            for j, e in enumerate(eds):
                a, b = e.split("|")
                u[j, :onset] += 0.9 if block[a] == block[b] else -0.5
                u[j, onset:] += 0.9 if swapped[a] == swapped[b] \
                    else -0.5
    if family == "B2B":
        rate = float(point.get("dropout", 0.0))
        if rate > 0.0:
            for t in range(ev0, len(days)):
                for st in sts:
                    if rng.random() < rate:
                        dropped[days[t]].add(st)
    if family == "B1B" and inject:
        if "gain" in point:            # the KOZT specificity class
            g = float(point["gain"])
            onset = ev0 + max(1, (len(days) - ev0) // 3)
            target = sts[int(rng.integers(0, len(sts)))]
            gain = {"station": target, "g": g, "onset": onset}
        else:                          # registered burst grid
            k = int(point["k"])
            n_e = int(point["n_e"])
            d_ = float(point["delta_lat"])
            max_start = (len(days) - ev0) - k
            start = ev0 + (0 if max_start <= 0 else
                           int(rng.integers(0, max_start + 1)))
            for e in eds[:n_e]:
                j = eds.index(e)
                u[j, start:start + k] += d_

    vals = np.tanh(u)
    r_series = {}
    for j, e in enumerate(eds):
        a, b = e.split("|")
        row = {}
        for t, d in enumerate(days):
            if mcar[j, t]:
                continue
            if a in dropped[d] or b in dropped[d]:
                continue
            v = float(vals[j, t])
            if gain and gain["station"] in (a, b) \
                    and t >= gain["onset"]:
                v *= gain["g"]
            row[d] = v
        r_series[e] = row
    measured = {d: sorted(set(sts) - dropped[d]) for d in days}
    return {"carrier": ck, "days": days, "stations": sts,
            "edges": eds, "r": r_series, "measured": measured,
            "segments": geom["segments"][ck]}


def panel_views(geom, raw):
    """Per-engine panel shapes from one raw replicate."""
    ck = raw["carrier"]
    ev0 = geom["eval_start_index"]
    eval_days = raw["days"][ev0:]
    graph_carrier = {"registered_days": list(raw["days"]),
                     "stations": raw["stations"],
                     "segments": raw["segments"], "r": raw["r"]}
    b2b_carrier = {"registry": list(raw["stations"]),
                   "registered_days": eval_days,
                   "measured": {d: raw["measured"][d]
                                for d in eval_days},
                   "r": {e: {d: v for d, v in ser.items()
                             if d in set(eval_days)}
                         for e, ser in raw["r"].items()}}
    return {"pb": {"carriers": {ck: graph_carrier}},
            "b2b": {"calendar": eval_days,
                    "carriers": {ck: b2b_carrier}},
            "b1b": {"calendar": list(raw["days"]),
                    "carriers": {ck: {"registry": raw["stations"],
                                      "registered_days": raw["days"],
                                      "r": raw["r"]}}}}


# ------------------------------------------------ replicate -> Holm
def replicate_pvalues(geom, views, n_draws, doc_sha):
    """All four families on the same replicate panel. Typed/None p is
    carried as None (non-rejection, m stays 4 -- disclosed pin)."""
    b = geom["b1b"]
    out = {}
    r1 = _pb.b2a_family(views["pb"], doc_sha256=doc_sha,
                        n_draws=n_draws)
    out["B2A"] = r1.get("p_value")
    r2 = _pb.b3a_family(views["pb"], doc_sha256=doc_sha,
                        n_draws=n_draws)
    out["B3A"] = r2.get("p_value")
    r3 = _b2b.w2_b2b_family(views["b2b"], doc_sha256=doc_sha,
                            n_draws=n_draws)
    out["B2B"] = r3.get("p_value")
    r4 = _b1b.w2_b1b_family(views["b1b"], doc_sha256=doc_sha,
                            n_draws=n_draws, n_blocks=b["n_blocks"],
                            block_len=b["block_len"],
                            baseline_positions=b["baseline_positions"],
                            testable_min=b["testable_min"])
    out["B1B"] = r4.get("p_value")
    return out


def holm_rejects(pvals, alpha=ALPHA_HOLM):
    """Holm over the FULL four-member graph; None p = non-reject with
    m held at 4."""
    m = len(GRAPH)
    known = sorted((h for h in GRAPH if pvals.get(h) is not None),
                   key=lambda h: pvals[h])
    rejected = set()
    still = True
    for i, h in enumerate(known):
        if still and pvals[h] <= alpha / (m - i):
            rejected.add(h)
        else:
            still = False
    return rejected


def run_point(geom, family, point, *, seed_root, n_draws,
              r_first=R_FIRST, r_max=R_MAX, tier=TIER_LABEL_FIXTURE):
    """Detection-class certification record for one grid point."""
    def success(r):
        raw = make_panel(geom, family, point, r, seed_root)
        pv = replicate_pvalues(geom, panel_views(geom, raw), n_draws,
                               seed_root)
        return family in holm_rejects(pv)
    rec = certify(success, r_first=r_first, r_max=r_max)
    rec.update(family=family, point=point, tier=tier,
               n_draws=int(n_draws),
               certifiable=(tier != TIER_LABEL_FIXTURE))
    return rec


def run_artifact_class(geom, point, *, seed_root, n_draws, R,
                       tier=TIER_LABEL_FIXTURE):
    """The B1B specificity gate: FAMILYWISE positive rate on the
    gain-step class must be <= 0.05. Rate = observed proportion over R
    (disclosed pin); counts + per-replicate outcomes recorded."""
    positives = 0
    outcomes = []
    for r in range(int(R)):
        raw = make_panel(geom, "B1B", point, r, seed_root)
        pv = replicate_pvalues(geom, panel_views(geom, raw), n_draws,
                               seed_root)
        rej = holm_rejects(pv)
        outcomes.append(sorted(rej))
        if rej:
            positives += 1
    rate = positives / int(R)
    return {"class": "B1B_GAIN_STEP_SPECIFICITY", "point": point,
            "R": int(R), "positives": positives, "rate": rate,
            "passes": rate <= ARTIFACT_MAX_RATE,
            "max_rate": ARTIFACT_MAX_RATE, "tier": tier,
            "outcomes": outcomes,
            "certifiable": (tier != TIER_LABEL_FIXTURE)}


# ---------------------------------------------------------------- selftest
def _selftest():
    # stopping rule: the three clear terminals
    rec = certify(lambda r: True)
    assert rec["status"] == "CERTIFIED" and rec["R"] == R_FIRST
    assert rec["lb"] >= CP_FLOOR
    rec = certify(lambda r: False)
    assert rec["status"] == "FAILED" and rec["R"] == R_FIRST
    seq = [True] * 17 + [False] * 3 + [True] * 17 + [False] * 3
    rec = certify(lambda r: seq[r])
    assert rec["status"] in ("CERTIFIED", "FAILED",
                             "CANNOT_DETERMINE_POWER_ESTIMATE")
    assert len(rec["trace"]) == 2 and rec["R"] == R_MAX
    assert cp_lb(20, 20) > CP_FLOOR and cp_ub(0, 20) < CP_FLOOR
    assert cp_lb(10, 20) < cp_lb(19, 20)          # monotone

    # generator determinism + injection non-identity
    geom = fixture_geometry()
    raw1 = make_panel(geom, "B2B", {"m": 2}, 0, "seedroot")
    raw2 = make_panel(geom, "B2B", {"m": 2}, 0, "seedroot")
    d1 = hashlib.sha256(json.dumps(raw1["r"], sort_keys=True)
                        .encode()).hexdigest()
    assert d1 == hashlib.sha256(json.dumps(raw2["r"], sort_keys=True)
                                .encode()).hexdigest()
    raw0 = make_panel(geom, "B2B", {"m": 2}, 0, "seedroot",
                      inject=False)
    assert d1 != hashlib.sha256(json.dumps(raw0["r"], sort_keys=True)
                                .encode()).hexdigest()

    # dropout produces producer-consistent thinning (measured shrinks
    # AND incident edges vanish those days)
    rawd = make_panel(geom, "B2B", {"m": 0, "dropout": 0.25}, 1,
                      "seedroot")
    ev_days = rawd["days"][geom["eval_start_index"]:]
    thinned = [d for d in ev_days
               if len(rawd["measured"][d]) < len(rawd["stations"])]
    assert thinned, "25% dropout must thin some eval days"
    for d in thinned[:3]:
        absent = set(rawd["stations"]) - set(rawd["measured"][d])
        for e, ser in rawd["r"].items():
            a, b = e.split("|")
            if a in absent or b in absent:
                assert d not in ser

    # gain-step artifact scales exactly the target's incident eval
    # edges (mechanism, not power)
    rawg = make_panel(geom, "B1B", {"gain": 10.0}, 2, "seedroot")
    rawn = make_panel(geom, "B1B", {"gain": 10.0}, 2, "seedroot",
                      inject=False)
    changed = {e for e in rawg["r"]
               if rawg["r"][e] != rawn["r"][e]}
    stations_in_changed = set()
    for e in changed:
        stations_in_changed.update(e.split("|"))
    common = set.intersection(*[set(e.split("|")) for e in changed])
    assert len(common) == 1, "gain must localize to ONE station"

    # replicate -> four p-values -> Holm; null-ish panel rarely rejects
    raw = make_panel(geom, "B2A", {"m": 0}, 3, "seedroot")
    pv = replicate_pvalues(geom, panel_views(geom, raw), 199,
                           "ab" * 32)
    assert set(pv) == set(GRAPH)
    assert all(v is None or 0.0 < v <= 1.0 for v in pv.values())
    # Holm hand cases incl None handling (m stays 4)
    assert holm_rejects({"B1B": 0.001, "B2A": 0.02, "B2B": 0.3,
                         "B3A": None}) == {"B1B"}
    assert holm_rejects({"B1B": 0.012, "B2A": 0.013, "B2B": 0.016,
                         "B3A": 0.04}) == {"B1B", "B2A", "B2B",
                                           "B3A"}
    assert holm_rejects({h: None for h in GRAPH}) == set()

    # end-to-end fixture-tier records (tiny R + n_draws; MECHANISM
    # only -- the tier label forbids certification claims)
    rec = run_point(geom, "B2B", {"m": 3}, seed_root="sr",
                    n_draws=99, r_first=4, r_max=6)
    assert rec["tier"] == TIER_LABEL_FIXTURE \
        and rec["certifiable"] is False
    assert rec["status"] in ("CERTIFIED", "FAILED",
                             "CANNOT_DETERMINE_POWER_ESTIMATE")
    art = run_artifact_class(geom, {"gain": 3.0}, seed_root="sr",
                             n_draws=99, R=4)
    assert art["certifiable"] is False and 0.0 <= art["rate"] <= 1.0
    assert len(art["outcomes"]) == 4

    print("w2_power_harness selftest: ALL PASS "
          "(fixture-tier mechanism only)")


if __name__ == "__main__":
    _selftest()
