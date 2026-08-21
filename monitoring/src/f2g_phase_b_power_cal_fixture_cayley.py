#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive-fixture generator for the CAL verifier close packet (cayley).

FIXTURE-ONLY. Emits a stopping-consistent synthetic evidence capsule at the
CANONICAL path: one REAL gate row (executed equivalence gate) + for every
family: all grid points x reps 0..49 S1 rows (pre=False), top-8 S2 rows
(post=False), top-3 C candidates with reps 0..19 (post=False -> FAILED@20)
carrying REAL regenerable panel digests. The capsule is then committed and
the PINNED assembler produces the fixture result artifacts under
docs/fixtures/ -- the verifier's positive case reconstructs them
independently. Recovery flags are synthetic (this is a verifier fixture,
never admissible power evidence); panel digests for all SAMPLED rows are
real so reopen checks bind.
Usage: fixture.py <out_evidence_path>
"""
import json
import sys
import time

import f2g_phase_b_power_estimation_cal_cayley as DC


def pk(p):
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def main(out_path):
    t0 = time.time()
    rows = []
    gate = DC.equivalence_gate()
    rows.append(dict({"key": "gateCal", "stage": "gate"}, **gate))
    for fam in ("B1A", "B2A", "B3A"):
        grid = DC.grid_of(fam)
        for p in grid:
            for rep in range(50):
                rows.append({"key": f"S1|{fam}|{pk(p)}|r={rep}",
                             "stage": "S1", "family": fam, "point": p,
                             "rep": rep, "p": 1.0, "pre": False,
                             "panel_sha256": "fixture-unsampled"})
        rank1 = sorted(grid, key=pk)[:min(8, len(grid))]
        for p in rank1:
            for rep in range(50):
                rows.append({"key": f"S2|{fam}|{pk(p)}|r={rep}",
                             "stage": "S2", "family": fam, "point": p,
                             "rep": rep, "p": 1.0, "pre": False,
                             "post": False,
                             "panel_sha256": "fixture-unsampled"})
        cpts = sorted(rank1, key=pk)[:min(3, len(rank1))]
        for p in cpts:
            for rep in range(20):
                dig = DC.panel_digest(DC.make_panel(fam, p, rep))
                rows.append({"key": f"C|{fam}|{pk(p)}|r={rep}",
                             "stage": "C", "family": fam, "point": p,
                             "rep": rep, "p": 1.0, "pre": False,
                             "post": False, "panel_sha256": dig})
        print(f"[{fam}] fixture rows done ({time.time()-t0:.0f}s)",
              flush=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {out_path} rows={len(rows)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
