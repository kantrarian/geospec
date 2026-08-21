#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive-fixture generator for the CAL verifier close packet (cayley).

FIXTURE-ONLY, PRODUCTION-SHAPED (codex 5-fix recheck, findings 1+5).
Emits a stopping-consistent synthetic evidence capsule at the FIXTURES-ONLY
path (never the production canonical path): a purpose=fixture header row,
one REAL gate row (executed equivalence gate), then for every family the
exact rows a production run would emit -- all grid points x reps 0..49 S1
rows (pre=False, real dt), the registered-selector top-8 S2 rows
(post=False), top-3 C candidates reps 0..19 (FAILED@20), and one derived-
consistent Cverdict row per candidate. EVERY row carries the REAL
regenerable panel digest (memoized: the panel depends only on
(family, point, rep), so S1/S2/C rows for the same replicate share one
digest) and the driver's canonical D0.key_of key. Selection uses the
REGISTERED coordinate order (driver ckey): with all recoveries tied at 0.0
the selector reduces to pure ckey order, which is exactly the tied-selector
behavior the recheck mandates. Recovery flags are synthetic (this is a
verifier fixture, never admissible power evidence); digests are real so
reopen and cross-stage bindings hold.
Usage: fixture.py <out_evidence_path>
"""
import json
import sys
import time

import f2g_phase_b_power_estimation_cal_cayley as DC

D0 = DC.D0


def main(out_path):
    t0 = time.time()
    rows = [{"key": "header", "stage": "header", "purpose": "fixture",
             "schema": DC.EVIDENCE_SCHEMA,
             "calendar_authority_sha256": DC.E.CAL_AUTHORITY_SHA256,
             "amendment2_sha256": DC.AMENDMENT2_SHA}]
    gate = DC.equivalence_gate()
    rows.append(dict({"key": "gateCal", "stage": "gate"}, **gate))
    print(f"[gate] all_equal={gate['all_equal']} ({time.time()-t0:.0f}s)",
          flush=True)
    for fam in ("B1A", "B2A", "B3A"):
        grid = DC.grid_of(fam)
        memo = {}

        def pdig(p, rep):
            k = (json.dumps(p, sort_keys=True), rep)
            if k not in memo:
                memo[k] = DC.panel_digest(DC.make_panel(fam, p, rep))
            return memo[k]

        for p in grid:
            for rep in range(50):
                ts = time.time()
                dig = pdig(p, rep)
                rows.append({"key": D0.key_of("S1", fam, p, rep),
                             "stage": "S1", "family": fam, "point": p,
                             "rep": rep, "p": 1.0, "pre": False,
                             "panel_sha256": dig,
                             "dt": round(time.time() - ts, 2)})
        sel = sorted(grid, key=lambda q: DC.ckey(fam, q))
        s2_pts = sel[:min(8, len(grid))]
        for p in s2_pts:
            for rep in range(50):
                rows.append({"key": D0.key_of("S2", fam, p, rep),
                             "stage": "S2", "family": fam, "point": p,
                             "rep": rep, "p": 1.0, "pre": False,
                             "post": False, "panel_sha256": pdig(p, rep)})
        c_pts = s2_pts[:min(3, len(s2_pts))]
        for p in c_pts:
            for rep in range(20):
                rows.append({"key": D0.key_of("C", fam, p, rep),
                             "stage": "C", "family": fam, "point": p,
                             "rep": rep, "p": 1.0, "pre": False,
                             "post": False, "panel_sha256": pdig(p, rep)})
            rows.append({"key": D0.key_of("Cv", fam, p, 0),
                         "stage": "Cverdict", "family": fam, "point": p,
                         "n": 20, "successes": 0,
                         "lb95": D0.cp_lower(0, 20),
                         "ub95": D0.cp_upper(0, 20), "verdict": "FAILED"})
        print(f"[{fam}] fixture rows done ({time.time()-t0:.0f}s)",
              flush=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {out_path} rows={len(rows)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
