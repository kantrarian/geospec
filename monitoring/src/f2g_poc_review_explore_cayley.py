#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PoC IMPROVEMENT REVIEW -- exploratory decomposition v1 (cayley).

*** EXPLORATORY LANE. Owner-authorized (A4 disposition 2026-08-21): the
Phase-B one-shot is CONSUMED; nothing here carries claim weight, ever.
Purpose = hypothesis generation for the NEXT window's prereg (3 carriers
+ cascadia). Every output is labeled EXPLORATORY. ***

Decomposes the sealed-run statistics on the consumed window:
  - B2A: per-carrier partition-state sequences, observed max
    identical-partition runs, switch timeline, exploratory 999-draw null
    context (the sealed p=0.0953 was one-sided LOW at 9999 draws).
  - B1A: per-carrier / per-edge / per-window top contributions to the
    family max window-mean |z| (sealed p=0.0928).
  - Coverage: per-station presence, edges/day -- the data-addition case.
Usage: explore.py <repo>
"""
import json
import os
import sys

import numpy as np

import f2g_sealed_run_instrument_cayley as I


def main(repo):
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import d2_f2g_phase_b_stats as E
    import f2g_phase_b_power_estimation_cal_cayley as PD
    panel = I.build_panel(repo, os.path.join(repo, I.ARTIFACT_ROOT,
                                             "snapshots"),
                          allow_real=True)
    cal, carriers = E._cal_load(panel)
    eval_days = cal[E.CAL_BASELINE_POSITIONS:]
    out = {"schema": "f2g-poc-review-exploratory-v1",
           "lane": "EXPLORATORY -- zero claim weight; hypothesis "
                   "generation for the next-window prereg only",
           "sealed_reference": {"B2A_p": 0.0953, "B1A_p": 0.0928,
                                "B3A_p": 0.3945},
           "carriers": {}}
    # ---- B2A decomposition (runs = engine COUNT; lengths walked with
    # the same registered semantics for exploration) ----
    for ck in sorted(carriers):
        regset = set(panel["carriers"][ck]["registered_days"])
        src = {"r": panel["carriers"][ck]["r"]}
        states = E._b2a_cal_states(src, cal, regset)
        runs_ct, refs, accepted = E._b2a_runs(states, eval_days)
        lengths = []
        cur = None
        ln = 0
        last_ns = None
        for (part, code), d in zip(states, eval_days):
            if code:
                if ln:
                    lengths.append(ln)
                cur, ln, last_ns = None, 0, None
                continue
            ns = frozenset(part)
            if last_ns is not None and ns != last_ns:
                if ln:
                    lengths.append(ln)
                cur, ln, last_ns = None, 0, None
                continue
            last_ns = ns
            if cur is not None and part == cur:
                ln += 1
            else:
                if ln:
                    lengths.append(ln)
                cur, ln = part, 1
        if ln:
            lengths.append(ln)
        code_hist = {}
        for r_ in refs:
            code_hist[r_["code"]] = code_hist.get(r_["code"], 0) + 1
        out["carriers"][ck] = {
            "b2a": {"accepted_days": accepted, "n_runs": runs_ct,
                    "run_lengths_top": sorted(lengths, reverse=True)[:8],
                    "max_run_len": max(lengths) if lengths else 0,
                    "refusal_codes": code_hist}}
    # ---- B1A decomposition (window-mean |z| top contributors) ----
    memo = PD.B1ACalMemo(panel)
    ident = list(range(E.B1A_CAL_BLOCKS))
    for ck in memo.keys:
        W = memo.edge_window_max(ck, ident)
        edges = memo.edges[ck]
        fin = [(edges[i], float(W[i])) for i in range(len(edges))
               if np.isfinite(W[i])]
        fin.sort(key=lambda x: -x[1])
        out["carriers"][ck]["b1a_top_edges"] = [
            {"edge": e, "window_max_mean_abs_z": round(v, 3)}
            for e, v in fin[:5]]
        out["carriers"][ck]["b1a_carrier_max"] = (round(fin[0][1], 3)
                                                 if fin else None)
    # ---- coverage / data-addition facts ----
    for ck in sorted(carriers):
        r = panel["carriers"][ck]["r"]
        days = panel["carriers"][ck]["registered_days"]
        sts = panel["carriers"][ck]["stations"]
        per_day = {d: 0 for d in days}
        st_presence = {s: 0 for s in sts}
        for e, row in r.items():
            a, b = e.split("|")
            for d in row:
                per_day[d] += 1
        for s in sts:
            sdays = set()
            for e, row in r.items():
                if s in e.split("|"):
                    sdays.update(row.keys())
            st_presence[s] = round(len(sdays) / len(days), 3)
        vals = list(per_day.values())
        out["carriers"][ck]["coverage"] = {
            "n_stations": len(sts), "n_edges_seen": len(r),
            "edges_per_day_median": float(np.median(vals)),
            "edges_per_day_min": int(min(vals)),
            "station_presence_min": min(st_presence.values()),
            "stations_below_half": [s for s, v in st_presence.items()
                                    if v < 0.5]}
    outdir = os.path.join(repo, "docs", "f2g_poc_review")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "exploratory_decomposition_v1.json"),
              "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "."))
