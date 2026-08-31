#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 power-certification SIZING BENCHMARK (cayley; owner
directive 2026-08-22T19:24Z "run the tier-s benchmark and tier-c
sizing", quote sha 7caab14d).

TIMING ONLY. Geometry is fixture-authority at the FROZEN CAP SIZES
(istanbul 16 / socal 20 / turkey 14 / cascadia 16 stations) on the
WINDOW-2 v2 CALENDAR (192-position fixed authority grid, baseline 60,
codex 1400Z ruling 1) so per-call costs are production-shaped, but NO
power,
recovery, or certification number is produced or reported -- every
engine output beyond wall-clock is DISCARDED. The pinned Tier rule
applies a fortiori: nothing here can populate a certified MDE.

Method: time the bound generator and each family engine at n_draws in
{99, 999}; fit cost(n) = a + b*n per family; project cost(9,999);
report Tier-S full-grid and Tier-C scenarios under DOCUMENTED
multiplier assumptions (grid sizes from the registered annexes/pinned
module; R = 20 vs 40; target-only vs all-four-families-per-replicate;
LOCO fold multiplier as a scenario row only -- the window-2 LOCO
binding is design-settled but its per-replicate composition is sized
here as a scenario, not asserted).

Writes docs/f2g_window2_execution/power_cert_sizing_v2.json (raw) --
v1 (the 132-day Phase-B-geometry timing) is RETAINED unedited; this
v2 artifact re-times at the bound window-2 geometry and supersedes v1
for PRESTART planning. The human summary is authored separately.
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_power_harness_cayley as PH
import d2_f2g_phase_b_stats as _pb

CAPS = {"c_ist": 16, "c_soc": 20, "c_tur": 14, "c_cas": 16}
DRAW_POINTS = (99, 999)
CERT_DRAWS = 9999
BENCH_BUDGET_99_S = 240.0     # skip the 999 run if the 99 run blows this

# registered grid sizes (annexes + pinned estimation module)
GRID_POINTS = {"B1B_detection": 48, "B1B_artifact": 2, "B2A": 3,
               "B2B": 5, "B3A": 24}
TIER_S_R = 50
TIER_C_CANDIDATES_PER_FAMILY = 3
LOCO_FOLDS_SCENARIO = 16      # cascadia NEW-registry stations


def build_capsule():
    frame = PH.w2_calendar_frame()
    eng = list(frame["engine_days"])
    regs = {ck: [f"{ck}S{i:02d}" for i in range(n)]
            for ck, n in CAPS.items()}
    return {"schema": PH.BOUND_GEOMETRY_SCHEMA, "bound": True,
            "calendar_authority_sha256": "bench-cal-auth",
            "seed_authority_sha256": "bench-seed-auth",
            "calendar_authority_mode": "fixture",
            "calendar_frame": frame,
            "carrier_masks": {ck: {"registered_days": list(eng),
                                   "available_days": list(eng)}
                              for ck in CAPS},
            "registries": regs,
            "segments": {ck: {s: ("sA" if i < len(regs[ck]) // 2
                                  else "sB")
                              for i, s in enumerate(regs[ck])}
                         for ck in CAPS},
            # cycle-6: the advanced schema's closed input_refs set.
            # TIMING-ONLY, like every other field here: these resolve
            # to no admitted pin, so this capsule cannot certify.
            "input_refs": {n: {"commit": "0" * 40,
                               "path": f"docs/bench/{n}.json",
                               "blob_sha256": "0" * 64}
                           for n in PH.GEOMETRY_INPUT_REFS}}


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    cap = build_capsule()
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "label": "TIMING ONLY -- fixture-authority geometry at "
                    "frozen cap sizes; no power numbers exist in "
                    "this artifact",
           "geometry": {"carriers": CAPS, "calendar_days": 192,
                        "frame_id": "w2-calendar-v2-noncal",
                        "baseline_positions": 60},
           "gen_seconds": None, "families": {}, "fit_9999": {},
           "projections": {}}

    t0 = time.perf_counter()
    panel, views, _dbg = PH.make_bound_panels(
        cap, "B2B", {"m": 2}, 0)
    out["gen_seconds"] = round(time.perf_counter() - t0, 3)
    print(f"generator: {out['gen_seconds']}s", flush=True)

    b1bg = cap["calendar_frame"]["b1b"]
    runners = {
        "B2A": lambda n: _pb.b2a_family(
            panel, doc_sha256="ab" * 32, n_draws=n),
        "B3A": lambda n: _pb.b3a_family(
            panel, doc_sha256="ab" * 32, n_draws=n),
        "B2B": lambda n: PH._b2b.w2_b2b_family(
            views["b2b"], doc_sha256="ab" * 32, n_draws=n),
        "B1B": lambda n: PH._b1b.w2_b1b_family(
            views["b1b"], doc_sha256="ab" * 32, n_draws=n,
            n_blocks=b1bg["n_blocks"],
            block_len=b1bg["block_len"],
            baseline_positions=b1bg["baseline_positions"]),
    }
    for fam, run in runners.items():
        rec = {}
        for n in DRAW_POINTS:
            if n != DRAW_POINTS[0] and \
                    rec.get(str(DRAW_POINTS[0]), 0) > BENCH_BUDGET_99_S:
                rec["skipped_999"] = True
                break
            t = timed(lambda: run(n))
            rec[str(n)] = round(t, 3)
            print(f"{fam} n_draws={n}: {t:.1f}s", flush=True)
        out["families"][fam] = rec
        # linear fit cost(n) = a + b n
        if str(DRAW_POINTS[1]) in rec:
            b = (rec[str(DRAW_POINTS[1])] - rec[str(DRAW_POINTS[0])]) \
                / (DRAW_POINTS[1] - DRAW_POINTS[0])
            a = rec[str(DRAW_POINTS[0])] - b * DRAW_POINTS[0]
        else:  # extrapolate from the 99 point alone (draw-dominated)
            b = rec[str(DRAW_POINTS[0])] / DRAW_POINTS[0]
            a = 0.0
        out["fit_9999"][fam] = round(max(a, 0.0) + b * CERT_DRAWS, 1)

    c9999_all = sum(out["fit_9999"].values()) + out["gen_seconds"]
    c999_all = sum(out["families"][f].get("999",
                   out["fit_9999"][f] / 10) for f in runners) \
        + out["gen_seconds"]
    total_points = sum(GRID_POINTS.values())
    tier_c_points = TIER_C_CANDIDATES_PER_FAMILY * 4 \
        + GRID_POINTS["B1B_artifact"]

    def hours(s):
        return round(s / 3600.0, 1)

    proj = {"assumptions": {
                "grid_points": GRID_POINTS,
                "tier_s_R": TIER_S_R,
                "tier_c_candidates_per_family":
                    TIER_C_CANDIDATES_PER_FAMILY,
                "all_four_families_per_replicate":
                    "the sec-5 Holm rule runs ALL FOUR engines per "
                    "replicate; the target-only column is the "
                    "screening-style lower bound",
                "loco_scenario_folds": LOCO_FOLDS_SCENARIO},
            "per_replicate_seconds": {
                "all_four_at_999": round(c999_all, 1),
                "all_four_at_9999": round(c9999_all, 1)},
            "tier_s_full_grid_hours": {
                "all_four": hours(total_points * TIER_S_R * c999_all)},
            "tier_c_hours": {}}
    for R in (20, 40):
        proj["tier_c_hours"][f"R{R}_all_four"] = hours(
            tier_c_points * R * c9999_all)
        proj["tier_c_hours"][f"R{R}_all_four_with_loco_x"
                             f"{LOCO_FOLDS_SCENARIO + 1}"] = hours(
            tier_c_points * R * c9999_all * (LOCO_FOLDS_SCENARIO + 1))
    out["projections"] = proj

    rel = "docs/f2g_window2_execution/power_cert_sizing_v2.json"
    p = os.path.join(os.path.abspath(os.path.join(_HERE, "..", "..")),
                     rel.replace("/", os.sep))
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(proj, indent=1, sort_keys=True), flush=True)
    print(f"wrote {rel}", flush=True)


if __name__ == "__main__":
    main()
