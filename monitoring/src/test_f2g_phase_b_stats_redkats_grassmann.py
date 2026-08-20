#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED-first KATs -- fault2graph PHASE B STATISTICS ENGINE bar (grassmann).

Cross-authored per prereg section 6 (V-D governance): grassmann authors, cayley
implements the engine BAR-UNEDITED, codex verifies once. Disagreements with any
pin below go through inbox R1.2, never through editing this file.

Contract sources (the bar encodes the CORRECTED contracts, not prereg v1
verbatim): cayley prereg v1 draft (geospec 2dd451c, sha 16d9e0b7) as repaired
by codex R1.2 306d1a5-successor note f24eb2e (2026-08-20 0237Z):
  R1 B-1 null draws rotate the ENTIRE registered carrier-day vector BEFORE the
     60/remainder split -- ONE common circular offset per carrier per draw --
     then rerun the complete pipeline (split, floor, median/MAD, exclusions,
     z, family max, persistence). NEVER shift already-computed evaluation z
     (a circular shift is a permutation: max|shift(z)| == max|z| identically,
     p degenerates to 1 -- proven in-bar as G7a). 9999 draws, add-one p,
     < 9900 valid -> CANNOT_DETERMINE_NULL_SUPPORT.
  R2 B-2 identifiability gates: unique largest positive-weight component with
     >= 3 nodes (else LCC_TIE), exact station/index identity across compared
     days (else NODESET_MISMATCH), relative eigengap >= 1e-6 (else
     FIEDLER_DEGENERATE), |v2_i| > 1e-10 for every classified coordinate
     (else FIEDLER_ZERO_COORDINATE); orientation: lexicographically first
     component station's coordinate positive.
  R3 max-|z| is the SOLE verdict-bearing B-1 statistic; persistence is
     corrected secondary evidence and can never promote a verdict; a family
     without a sealed power contract types nonpositives
     CANNOT_DETERMINE_NO_POWER, never "no signal".
  R4 B-3 selection is deterministic: sort by (-|z|, station_a, station_b) over
     canonical unordered pairs, K = ceil(0.10*m), exactly first K; m == 0 ->
     INSUFFICIENT_DAILY_EDGES; space null permutes station->segment labels
     within carrier preserving exact segment sizes; NEVER conditions on B-1
     verdict/BY significance.
  R5 substream seeds per prereg rev-2 section (a44819d): seed_material =
     UTF-8("<frozen_doc_sha256_hex_lowercase>||<family>||<fold>||<purpose>")
     with DOUBLE-pipe separators, family tokens {B1, B2, B3}, fold in
     {full} + {loco:<STATION_ID>}, purpose in {null, power}; seed = first 8
     bytes of SHA256(seed_material) as big-endian uint64 -> PCG64; LOCO is
     a conjunctive gate only (every scorable fold must independently pass; a
     missing fold withholds -> LOCO_FOLD_UNSCORABLE; never promotes).

PINNED SEAMS (module monitoring/src/d2_f2g_phase_b_stats.py):
  walk_forward_split(registered_days) -> (baseline_days, eval_days)
  b1_family(panel, *, doc_sha256, n_draws=N_DRAWS, power_contract=...,
            return_null=False) -> result dict
  b2_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False)
  b3_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False)
  loco_gate(full_result, fold_results, alpha) -> dict
  derive_substream_seed(doc_sha256_hex, family, fold, purpose) -> int
  Constants: N_DRAWS=9999, ALPHA_FAMILY=0.05/3, BASELINE_DAYS=60,
             TESTABLE_MIN_BASELINE=45, PERSISTENCE_K=3, Z_PERSIST=3.0,
             TOP_DECILE=0.10, BY_Q=0.05
Family results MUST carry at least: family, p_value (float or None), T_obs,
n_valid_draws, verdict, excluded (dict of typed-exclusion counts), and for
fixtures: return_null=True adds null_T (list); b2 adds day_refusals
[{day, code}]; b3 adds selection {day: [edge, ...]} with edges as "A|B"
canonical sorted pairs. Nonpositive typing rides verdict (a verdict string
containing CANNOT_DETERMINE_NO_POWER / CANNOT_DETERMINE_NULL_SUPPORT / a
typed no-testable-edges refusal is typed, "NEGATIVE" alone is not).

FIXTURE PANEL SCHEMA (fixture-panel-v1, this bar's authority):
  {"carriers": {carrier_key: {
      "registered_days": [ISO ascending],
      "stations": [ids], "segments": {station: segment},
      "r": {"A|B": {day: float}}}}}   # absent (edge, day) = not measured

FIXTURE-ONLY (prereg governing rule): nothing in this bar touches the Phase-A
artifact, any real graph, or any waveform. No claims; Lambda_geo INCONCLUSIVE.
EXPECTED RED until cayley's engine lands. grassmann 2026-08-20.
"""
import hashlib
import math
import sys

import numpy as np

HERE_FAILS = []
_N = [0]


def check(name, ok, detail=""):
    _N[0] += 1
    tag = "PASS" if ok else "FAIL"
    print(f"    [{tag}] PB-{name}" + (f" - {detail}" if (detail and not ok) else ""))
    if not ok:
        HERE_FAILS.append(name)


def day(i):
    # deterministic ascending ISO dates
    return f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"


def mk_panel(n_days, edges, seed, sigma=1.0, base=0.0):
    """fixture-panel-v1: one carrier, gaussian r series per edge."""
    rng = np.random.default_rng(seed)
    days = [day(i) for i in range(n_days)]
    stations = sorted({s for e in edges for s in e.split("|")})
    r = {}
    for e in edges:
        vals = base + sigma * rng.standard_normal(n_days)
        r[e] = {d: float(v) for d, v in zip(days, vals)}
    return {"carriers": {"c_fix": {
        "registered_days": days, "stations": stations,
        "segments": {s: ("seg_a" if i % 2 == 0 else "seg_b")
                     for i, s in enumerate(stations)},
        "r": r}}}


def plant(panel, edges, day_idx, k, delta):
    """add delta to `edges` on k consecutive registered days from day_idx."""
    c = panel["carriers"]["c_fix"]
    for e in edges:
        for i in range(day_idx, day_idx + k):
            d = c["registered_days"][i]
            c["r"][e][d] = c["r"][e][d] + delta
    return panel


DOC = hashlib.sha256(b"pb-fixture-doc-v1").hexdigest()

try:
    import d2_f2g_phase_b_stats as E
    HAVE = True
except ImportError as exc:
    HAVE = False
    print(f"    [RED ] engine module absent ({exc}) -- EXPECTED RED until "
          f"cayley's d2_f2g_phase_b_stats lands; every check below reds")


def g7a():
    """codex CRITICAL finding, proven in-bar and ENGINE-INDEPENDENT: shifting
    already-computed evaluation z can never calibrate max-|z| -- a circular
    shift is a permutation, so every draw's family max equals the observed max
    exactly and the p-value degenerates to 1."""
    rng7 = np.random.default_rng(7)
    z = rng7.standard_normal((10, 40))            # 10 edges x 40 eval days of z
    t_obs = float(np.max(np.abs(z)))
    degenerate = all(
        float(np.max(np.abs(np.stack([np.roll(z[i], int(off[i]))
                                      for i in range(z.shape[0])])))) == t_obs
        for off in rng7.integers(0, 40, size=(200, z.shape[0])))
    check("G7a wrong-null degeneracy (in-bar proof)", degenerate,
          "a circular shift of computed z changed max|z| -- impossible")


def main():
    g7a()                       # engine-independent: runs (and must pass) NOW
    if not HAVE:
        for nm in ("G1 seams+constants", "G2 walk-forward split",
                   "G3 testable floor 45", "G4 degenerate baseline",
                   "G5 planted-signal recovery", "G6 null uniformity",
                   "G7b engine null is NOT the degenerate z-shift",
                   "G8 dependence preservation (common rotation)",
                   "G9 no-testable-edges -> typed CANNOT_DETERMINE",
                   "G10 persistence cannot promote",
                   "G11 no-power -> CANNOT_DETERMINE_NO_POWER",
                   "G12 B-2 identifiability refusals",
                   "G13 B-3 deterministic selection + small-day",
                   "G14 substream seed exact vector",
                   "G15 LOCO conjunctive gate"):
            check(nm, False, "ENGINE_ABSENT")
        return

    # G1 seams + registered constants ------------------------------------------
    ok1 = all(callable(getattr(E, f, None)) for f in
              ("walk_forward_split", "b1_family", "b2_family", "b3_family",
               "loco_gate", "derive_substream_seed"))
    ok1 = ok1 and E.N_DRAWS == 9999 and abs(E.ALPHA_FAMILY - 0.05 / 3) < 1e-15 \
        and E.BASELINE_DAYS == 60 and E.TESTABLE_MIN_BASELINE == 45 \
        and E.PERSISTENCE_K == 3 and float(E.Z_PERSIST) == 3.0 \
        and abs(E.TOP_DECILE - 0.10) < 1e-15 and abs(E.BY_Q - 0.05) < 1e-15
    check("G1 seams+constants", ok1)

    # G2 split ------------------------------------------------------------------
    days100 = [day(i) for i in range(100)]
    b, ev = E.walk_forward_split(days100)
    ok2 = list(b) == days100[:60] and list(ev) == days100[60:]
    try:
        E.walk_forward_split(days100[:60])
        ok2b, det2 = False, "accepted 60-day sequence with empty evaluation"
    except Exception as exc:
        ok2b, det2 = "INSUFFICIENT" in str(exc).upper(), f"{exc}"
    check("G2 walk-forward split", ok2 and ok2b, det2 if not ok2b else "")

    # G3 testable floor ---------------------------------------------------------
    p3 = mk_panel(90, ["AA.S1|AA.S2", "AA.S1|AA.S3"], seed=3)
    c3 = p3["carriers"]["c_fix"]
    for i, d in enumerate(c3["registered_days"][:60]):
        if i >= 44:                       # leave exactly 44 finite baseline obs
            del c3["r"]["AA.S1|AA.S3"][d]
    r3 = E.b1_family(p3, doc_sha256=DOC, n_draws=199,
                     power_contract={"passed": True})
    ok3 = r3["excluded"].get("INSUFFICIENT_BASELINE", 0) == 1 \
        and r3["testable_edges"] == 1
    check("G3 testable floor 45", ok3,
          f"excluded={r3['excluded']} testable={r3.get('testable_edges')}")

    # G4 MAD=0 ------------------------------------------------------------------
    p4 = mk_panel(90, ["AA.S1|AA.S2", "AA.S1|AA.S3"], seed=4)
    c4 = p4["carriers"]["c_fix"]
    for d in c4["registered_days"]:
        c4["r"]["AA.S1|AA.S2"][d] = 0.5          # constant -> MAD 0
    r4 = E.b1_family(p4, doc_sha256=DOC, n_draws=199,
                     power_contract={"passed": True})
    ok4 = r4["excluded"].get("DEGENERATE_BASELINE", 0) == 1
    check("G4 degenerate baseline", ok4, f"excluded={r4['excluded']}")

    # G5 planted recovery -------------------------------------------------------
    edges5 = [f"AA.S{i}|AA.S{j}" for i in range(1, 7) for j in range(i + 1, 7)]
    p5 = plant(mk_panel(100, edges5, seed=5), edges5[:3], 75, 3, 12.0)
    r5 = E.b1_family(p5, doc_sha256=DOC, n_draws=999,
                     power_contract={"passed": True})
    ok5 = r5["p_value"] is not None and r5["p_value"] <= E.ALPHA_FAMILY
    check("G5 planted-signal recovery", ok5, f"p={r5.get('p_value')}")

    # G6 null uniformity --------------------------------------------------------
    ps = []
    for s in range(20):
        pn = mk_panel(100, edges5[:6], seed=100 + s)
        rn = E.b1_family(pn, doc_sha256=DOC, n_draws=499,
                         power_contract={"passed": True})
        ps.append(rn["p_value"])
    frac_sig = sum(1 for p in ps if p is not None and p <= 0.05) / len(ps)
    ok6 = all(p is not None for p in ps) and frac_sig <= 0.25 \
        and 0.25 <= float(np.mean([p for p in ps])) <= 0.80 \
        and min(ps) >= 1.0 / 500
    check("G6 null uniformity", ok6,
          f"frac_sig={frac_sig} mean={float(np.mean(ps)):.3f} min={min(ps)}")

    # G7b engine null must NOT be the degenerate z-shift (G7a ran above) --------
    r7 = E.b1_family(p5, doc_sha256=DOC, n_draws=499,
                     power_contract={"passed": True}, return_null=True)
    null_t = np.asarray(r7["null_T"], dtype=float)
    finite = null_t[np.isfinite(null_t)]
    add_one = (1 + int(np.sum(finite >= r7["T_obs"]))) / (len(finite) + 1)
    ok7b = finite.size > 0 and float(np.std(finite)) > 0 \
        and not np.allclose(finite, r7["T_obs"]) \
        and abs(r7["p_value"] - add_one) < 1e-12
    check("G7b engine null is NOT the degenerate z-shift", ok7b,
          f"std={float(np.std(finite)) if finite.size else 'NA'} "
          f"p={r7['p_value']} add_one={add_one}")

    # G8 dependence preservation: N identical edges == 1 edge, draw-for-draw ----
    p8a = mk_panel(100, ["AA.S1|AA.S2"], seed=8)
    series = dict(p8a["carriers"]["c_fix"]["r"]["AA.S1|AA.S2"])
    p8b = mk_panel(100, edges5[:5], seed=8)
    for e in edges5[:5]:
        p8b["carriers"]["c_fix"]["r"][e] = dict(series)
    ra = E.b1_family(p8a, doc_sha256=DOC, n_draws=299,
                     power_contract={"passed": True}, return_null=True)
    rb = E.b1_family(p8b, doc_sha256=DOC, n_draws=299,
                     power_contract={"passed": True}, return_null=True)
    ok8 = np.array_equal(np.asarray(ra["null_T"]), np.asarray(rb["null_T"])) \
        and ra["T_obs"] == rb["T_obs"]
    check("G8 dependence preservation (common rotation)", ok8,
          "5 perfectly dependent edges produced a different null than 1 edge "
          "-- offsets are not common per carrier-draw")

    # G9 all-excluded -> typed, never a fabricated p ----------------------------
    p9 = mk_panel(90, ["AA.S1|AA.S2"], seed=9)
    for d in p9["carriers"]["c_fix"]["registered_days"]:
        p9["carriers"]["c_fix"]["r"]["AA.S1|AA.S2"][d] = 0.5
    r9 = E.b1_family(p9, doc_sha256=DOC, n_draws=199,
                     power_contract={"passed": True})
    ok9 = r9["p_value"] is None and "CANNOT_DETERMINE" in str(r9["verdict"])
    check("G9 no-testable-edges -> typed CANNOT_DETERMINE", ok9,
          f"verdict={r9.get('verdict')} p={r9.get('p_value')}")

    # G10 persistence is secondary and cannot promote ---------------------------
    r10 = E.b1_family(mk_panel(100, edges5[:6], seed=10), doc_sha256=DOC,
                      n_draws=499, power_contract={"passed": True})
    ok10 = isinstance(r10.get("persistence"), dict) \
        and r10["persistence"].get("verdict_bearing") is False \
        and (r10["p_value"] > E.ALPHA_FAMILY) == ("POSITIVE" not in
                                                  str(r10["verdict"]).upper())
    check("G10 persistence cannot promote", ok10,
          f"persistence={r10.get('persistence')} verdict={r10.get('verdict')}")

    # G11 no power contract -> typed nonpositive --------------------------------
    r11 = E.b1_family(mk_panel(100, edges5[:6], seed=11), doc_sha256=DOC,
                      n_draws=199, power_contract=None)
    ok11 = "CANNOT_DETERMINE_NO_POWER" in str(r11["verdict"])
    check("G11 no-power -> CANNOT_DETERMINE_NO_POWER", ok11,
          f"verdict={r11.get('verdict')}")

    # G12 B-2 identifiability refusals ------------------------------------------
    def two_day_panel(edges_by_day):
        n = 62
        days_ = [day(i) for i in range(n)]
        stations = sorted({s for eds in edges_by_day for e in eds
                           for s in e.split("|")})
        r = {}
        for eds in edges_by_day:
            for e in eds:
                r.setdefault(e, {})
        for i, d in enumerate(days_):
            if i < 60:
                for e in r:
                    r[e][d] = 0.5      # trivially fills baseline
            else:
                for e in edges_by_day[i - 60]:
                    r[e][d] = 0.8
        return {"carriers": {"c_fix": {
            "registered_days": days_, "stations": stations,
            "segments": {s: "seg_a" for s in stations}, "r": r}}}

    tie = ["AA.S1|AA.S2", "AA.S2|AA.S3", "AA.S4|AA.S5", "AA.S5|AA.S6"]
    r12a = E.b2_family(two_day_panel([tie, tie]), doc_sha256=DOC, n_draws=99)
    ok12a = any(x["code"] == "LCC_TIE" for x in r12a["day_refusals"])
    k4 = ["BB.S1|BB.S2", "BB.S1|BB.S3", "BB.S1|BB.S4",
          "BB.S2|BB.S3", "BB.S2|BB.S4", "BB.S3|BB.S4"]
    r12b = E.b2_family(two_day_panel([k4, k4]), doc_sha256=DOC, n_draws=99)
    ok12b = any(x["code"] == "FIEDLER_DEGENERATE" for x in r12b["day_refusals"])
    p3g = ["CC.S1|CC.S2", "CC.S2|CC.S3"]
    r12c = E.b2_family(two_day_panel([p3g, p3g]), doc_sha256=DOC, n_draws=99)
    ok12c = any(x["code"] == "FIEDLER_ZERO_COORDINATE"
                for x in r12c["day_refusals"])
    p4g1 = ["DD.S1|DD.S2", "DD.S2|DD.S3", "DD.S3|DD.S4"]
    p4g2 = ["DD.S2|DD.S3", "DD.S3|DD.S4", "DD.S4|DD.S5"]
    r12d = E.b2_family(two_day_panel([p4g1, p4g2]), doc_sha256=DOC, n_draws=99)
    ok12d = any(x["code"] == "NODESET_MISMATCH" for x in r12d["day_refusals"])
    r12e = E.b2_family(two_day_panel([p4g1, p4g1]), doc_sha256=DOC, n_draws=99)
    ok12e = not r12e.get("day_refusals") and r12e.get("max_switches") == 0
    check("G12 B-2 identifiability refusals",
          ok12a and ok12b and ok12c and ok12d and ok12e,
          f"tie={ok12a} degen={ok12b} zero={ok12c} nodeset={ok12d} "
          f"stable={ok12e} ({r12e.get('max_switches')})")

    # G13 B-3 deterministic selection + small-day -------------------------------
    e13 = [f"EE.S1|EE.S{j}" for j in range(2, 13)]          # m = 11 -> K = 2
    p13 = mk_panel(100, e13, seed=13, sigma=0.0)
    c13 = p13["carriers"]["c_fix"]
    for i, d in enumerate(c13["registered_days"]):
        for e in e13:
            c13["r"][e][d] = (0.001 if i % 2 == 0 else -0.001) if i < 60 else 0.0
    d_ev = c13["registered_days"][60]
    for e in e13:
        c13["r"][e][d_ev] = 0.01                          # all |z| exactly tied
    r13 = E.b3_family(p13, doc_sha256=DOC, n_draws=99)
    sel = r13["selection"][d_ev]
    expect = sorted(e13, key=lambda e: (e.split("|")[0], e.split("|")[1]))[:2]
    ok13a = list(sel) == expect
    p13b = mk_panel(62, ["FF.S1|FF.S2"], seed=14, sigma=0.0)
    c13b = p13b["carriers"]["c_fix"]
    for i, d in enumerate(c13b["registered_days"]):
        c13b["r"]["FF.S1|FF.S2"][d] = (0.001 if i % 2 == 0 else -0.001) \
            if i < 60 else c13b["r"]["FF.S1|FF.S2"][d]
    for d in c13b["registered_days"][60:]:
        del c13b["r"]["FF.S1|FF.S2"][d]                    # m = 0 on eval days
    r13b = E.b3_family(p13b, doc_sha256=DOC, n_draws=99)
    ok13b = any("INSUFFICIENT_DAILY_EDGES" in str(x)
                for x in r13b.get("day_refusals", []))
    check("G13 B-3 deterministic selection + small-day", ok13a and ok13b,
          f"sel={sel} expect={expect} smallday={ok13b}")

    # G14 substream seed exact vector (prereg rev-2 formula verbatim) -----------
    ok14 = True
    det14 = []
    for fam, fold, purpose in (("B1", "full", "null"),
                               ("B2", "loco:KO.GEML", "null"),
                               ("B3", "full", "power")):
        exp = int.from_bytes(hashlib.sha256(
            f"{DOC}||{fam}||{fold}||{purpose}".encode("utf-8"))
            .digest()[:8], "big")
        got = E.derive_substream_seed(DOC, fam, fold, purpose)
        if got != exp:
            ok14 = False
            det14.append(f"{fam}/{fold}/{purpose}: got={got} expect={exp}")
    check("G14 substream seed exact vector", ok14, "; ".join(det14))

    # G15 LOCO conjunctive gate --------------------------------------------------
    a = E.ALPHA_FAMILY
    full_pos = {"p_value": a / 2, "verdict": "POSITIVE"}
    f_pass, f_fail = {"p_value": a / 2}, {"p_value": a * 2}
    g1 = E.loco_gate(full_pos, [f_pass, f_pass, f_pass], a)
    g2 = E.loco_gate(full_pos, [f_pass, f_fail, f_pass], a)
    g3 = E.loco_gate(full_pos, [f_pass, None, f_pass], a)
    g4 = E.loco_gate({"p_value": a * 2, "verdict": "NEGATIVE"},
                     [f_pass, f_pass], a)
    ok15 = g1.get("pass") is True and g2.get("pass") is False \
        and g3.get("pass") is False and g4.get("pass") is False
    check("G15 LOCO conjunctive gate", ok15,
          f"all={g1.get('pass')} onefail={g2.get('pass')} "
          f"missing={g3.get('pass')} promote={g4.get('pass')}")


main()
print()
if HERE_FAILS:
    print(f"PHASE-B STATS RED-KAT FAILURES ({len(HERE_FAILS)}): {HERE_FAILS}")
    sys.exit(1)
print("ALL PHASE-B STATS RED-KATs PASS")
