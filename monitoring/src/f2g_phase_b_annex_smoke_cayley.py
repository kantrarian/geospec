#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault2graph Phase B power-annex corner smokes (cayley) -- FIXTURE-ONLY.

Purpose: sec-5 power-annex authorship smokes at each family's MOST FAVORABLE
corner, real-carrier geometry (110 registered days, 60/50 split, 12 stations /
66 edges). 10 replicates, reduced draws (B-1 499, B-2/B-3 999) -- these are
smoke bounds informing the pre-seal decision, NOT certified power estimates.
All panels synthetic; no Phase-A artifact, real graph, or waveform touched.
Run: 2026-08-20 vs engine @ geospec 6034419, frozen prereg rev-2 a44819d.
RESULT: B-1 5/10, B-2 0/10 (p==1.0 all reps: max-switch DEGENERATE under
day-order permutation), B-3 2/10 -- no family reaches 80% at alpha=0.05/3.
"""
import hashlib

import numpy as np

import d2_f2g_phase_b_stats as E

DOC = hashlib.sha256(b"pb-annex-smoke-v1").hexdigest()
ALPHA = E.ALPHA_FAMILY


def day(i):
    return f"20{26 + i // 336:02d}-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}"


DAYS110 = [day(i) for i in range(110)]
assert DAYS110 == sorted(DAYS110)


def panel(rng, n_st=12):
    sts = [f"AA.S{i:02d}" for i in range(n_st)]
    r = {}
    for i, a in enumerate(sts):
        for b in sts[i + 1:]:
            r[f"{a}|{b}"] = {d: float(0.1 * rng.standard_normal())
                             for d in DAYS110}
    segs = {s: ("seg_a" if i < n_st // 2 else "seg_b")
            for i, s in enumerate(sts)}
    return {"carriers": {"cx": {"registered_days": DAYS110, "stations": sts,
                                "segments": segs, "r": r}}}, sts


def b1_corner():
    hits, ps = 0, []
    for rep in range(10):
        rng = np.random.default_rng(1000 + rep)
        p, _ = panel(rng)
        c = p["carriers"]["cx"]
        for e in sorted(c["r"])[:33]:               # half the edges
            for d in DAYS110[60:]:                  # full 50-day eval window
                c["r"][e][d] += 12 * 0.1            # ~12 robust-z units
        r = E.b1_family(p, doc_sha256=DOC, n_draws=499,
                        power_contract={"passed": True})
        ps.append(r["p_value"])
        hits += r["p_value"] is not None and r["p_value"] <= ALPHA
    print(f"B-1 corner power {hits}/10  p={[round(x, 3) for x in ps]}")


def b2_corner():
    hits, ps = 0, []
    for rep in range(10):
        rng = np.random.default_rng(2000 + rep)
        sts = [f"AA.S{i:02d}" for i in range(12)]
        grp = {s: (0 if i < 6 else 1) for i, s in enumerate(sts)}
        grp2 = dict(grp)
        for s in sts[3:6]:
            grp2[s] = 1                              # persistent 3+3 swap
        for s in sts[6:9]:
            grp2[s] = 0
        r = {}
        for i, a in enumerate(sts):
            for b in sts[i + 1:]:
                ser = {}
                for j, d in enumerate(DAYS110):
                    g = grp if j < 85 else grp2      # swap at eval pos 25
                    ser[d] = float((0.7 if g[a] == g[b] else 0.05)
                                   + 0.1 * rng.standard_normal())
                r[f"{a}|{b}"] = ser
        pnl = {"carriers": {"cx": {"registered_days": DAYS110,
                                   "stations": sts,
                                   "segments": {s: "seg_a" for s in sts},
                                   "r": r}}}
        res = E.b2_family(pnl, doc_sha256=DOC, n_draws=999)
        ps.append(res["p_value"])
        hits += res["p_value"] is not None and res["p_value"] <= ALPHA
    print(f"B-2 corner power {hits}/10  p={ps}")


def b3_corner():
    hits, ps = 0, []
    for rep in range(10):
        rng = np.random.default_rng(3000 + rep)
        p, _ = panel(rng)
        c = p["carriers"]["cx"]
        segs = c["segments"]
        cross = [e for e in sorted(c["r"])
                 if segs[e.split("|")[0]] != segs[e.split("|")[1]]][:8]
        for e in cross:
            for d in DAYS110[60:]:
                c["r"][e][d] += 12 * 0.1
        res = E.b3_family(p, doc_sha256=DOC, n_draws=999)
        ps.append(res["p_value"])
        hits += res["p_value"] is not None and res["p_value"] <= ALPHA
    print(f"B-3 corner power {hits}/10  p={[round(x, 3) for x in ps]}")


if __name__ == "__main__":
    b1_corner()
    b2_corner()
    b3_corner()
