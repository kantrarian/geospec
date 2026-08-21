#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault2graph Phase B power estimation driver -- CALENDAR LANE (cayley).

FIXTURE-ONLY (synthetic values). Implements common protocol rev-1.5 sec 6:
true-geometry power panels emitted with calendar_authority_mode="bound" --
synthetic VALUES on the REAL frozen carrier names, REAL 132-day calendar,
and EXACT real masks, so the admitted engine's bound-mode byte-verification
applies to every power panel (codex 0131Z lock; fixture mode is unit-KATs
only and never an admissible power input). Registered draw order:
G[132] first (gamma=0.05), then per sorted carrier s/eps/MCAR over MASK
days, then effect placement on evaluation-window mask days. Replicate
streams at the FROZEN AMENDMENT-2 seed root. Each emitted replicate row
records the input-panel canonical digest; result artifacts must echo
calendar_authority_mode + calendar authority sha + panel digests, and the
verifier refuses non-bound panels typed POWER_GEOMETRY_UNBOUND.

B1A reduction (rev-1.4/1.5 sec 6 equivalence rule): baseline block-sets
C(11,6)=462 memoization on the 12-day calendar blocks + per-edge window-max
vectors; MANDATORY gate = exact full-data AND every-LOCO-fold p equality vs
E.b1a_family_cal before any table row; direct-engine fallback on failure.
B2A/B3A call the admitted cal engine directly. Estimation LAUNCH remains
gated on the loop's admissions; gate/smoke may run pre-admission (no
tables).
"""
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np

import d2_f2g_phase_b_stats as E
import f2g_phase_b_power_estimation_cayley as D0

AMENDMENT2_SHA = ("58b513b6c30b70c8014510788da9d7d819ce8971ca59b7dfdc11c57a"
                  "1664586f")
ALPHA = E.ALPHA_FAMILY
GAMMA = 0.05
B1A_GRID = [{"delta_lat": d, "k": k, "n_e": n}
            for d in (0.3, 0.6, 1.2, 2.4) for k in (3, 10, 25, 50)
            for n in (3, 10, 33)]
B2A_GRID = [{"m": m} for m in (1, 2, 3)]
B3A_GRID = [{"delta_lat": d, "n_cross": n, "k": k}
            for d in (0.3, 0.6, 1.2, 2.4) for n in (3, 8) for k in (10, 25, 50)]

_AUTH = E._cal_bound_authority()
assert ((E.CAL_POSITIONS, E.CAL_BASELINE_POSITIONS, E.CAL_EVAL_POSITIONS,
         E.B1A_CAL_BLOCK_LEN, E.B1A_CAL_BLOCKS) == (132, 72, 60, 12, 11)
        ), "CAL_CONSTANTS_DRIFT"
CAL = list(_AUTH["shared_calendar_days"])
CAL_POS = {d: i for i, d in enumerate(CAL)}
CARRIERS = sorted(_AUTH["carrier_masks"])
MASKS = {c: list(_AUTH["carrier_masks"][c]["registered_days"])
         for c in CARRIERS}
N_ST = {CARRIERS[0]: 12, CARRIERS[1]: 12, CARRIERS[2]: 11}
SEG_SIZES = {CARRIERS[0]: (4, 4, 4), CARRIERS[1]: (4, 4, 4),
             CARRIERS[2]: (4, 4, 3)}


def stations_of(ck):
    tag = ck.upper().replace("_", "")[:6]
    return [f"{tag}.S{i:02d}" for i in range(N_ST[ck])]


def segments_of(ck):
    sts = stations_of(ck)
    a, b, _c = SEG_SIZES[ck]
    return {s: ("seg_1" if i < a else ("seg_2" if i < a + b else "seg_3"))
            for i, s in enumerate(sts)}


def edges_of(ck):
    sts = stations_of(ck)
    return [f"{x}|{y}" for i, x in enumerate(sts) for y in sts[i + 1:]]


def rep_seed(family, r):
    master = np.random.Generator(np.random.PCG64(
        E.derive_substream_seed(AMENDMENT2_SHA, family, "full", "power")))
    return int(master.integers(0, 2 ** 63, size=r + 1, dtype=np.int64)[r])


def eval_mask_days(ck):
    return [d for d in MASKS[ck] if CAL_POS[d] >= 72]


def make_panel(family, point, r):
    rng = np.random.Generator(np.random.PCG64(rep_seed(family, r)))
    G = rng.standard_normal(len(CAL))
    lat = {}
    for ck in CARRIERS:
        sts = stations_of(ck)
        eds = edges_of(ck)
        days_ = MASKS[ck]
        s = rng.normal(0.0, D0.SIGMA_S, size=(len(sts), len(days_)))
        eps = rng.normal(0.0, D0.SIGMA_E, size=(len(eds), len(days_)))
        mcar = rng.random((len(eds), len(days_))) < D0.MCAR
        six = {st: i for i, st in enumerate(sts)}
        u = np.empty((len(eds), len(days_)))
        gvec = np.array([G[CAL_POS[d]] for d in days_])
        for j, e in enumerate(eds):
            a, b = e.split("|")
            u[j] = D0.MU0 + GAMMA * gvec + s[six[a]] + s[six[b]] + eps[j]
        if family == "B2A" and ck == CARRIERS[0]:
            m = point["m"]
            onset = next(d for d in days_ if CAL_POS[d] >= 102)  # calendar position 103, 1-based
            block = {st: (0 if i < 6 else 1) for i, st in enumerate(sts)}
            swapped = dict(block)
            for st in sts[6 - m:6]:
                swapped[st] = 1
            for st in sts[6:6 + m]:
                swapped[st] = 0
            cut = days_.index(onset)
            for j, e in enumerate(eds):
                a, b = e.split("|")
                u[j, :cut] += 0.9 if block[a] == block[b] else -0.5
                u[j, cut:] += 0.9 if swapped[a] == swapped[b] else -0.5
        lat[ck] = {"u": u, "mcar": mcar, "edges": eds, "days": days_}
    if family in ("B1A", "B3A"):
        ck0 = CARRIERS[0]
        if family == "B1A":
            k, n_e, d_ = point["k"], point["n_e"], point["delta_lat"]
            targets = lat[ck0]["edges"][:n_e]
        else:
            k, n_e, d_ = point["k"], point["n_cross"], point["delta_lat"]
            seg = segments_of(ck0)
            targets = [e for e in lat[ck0]["edges"]
                       if seg[e.split("|")[0]] != seg[e.split("|")[1]]][:n_e]
        # codex 0138Z fix 1: seeded length-k interval of consecutive TARGET
        # CALENDAR POSITIONS wholly within 73-132 (0-based 72..131); inject
        # ONLY at positions present in the carrier's frozen mask; absences
        # stay absent. k=50 = calendar duration, not present observations.
        max_start = 60 - k
        start = 72 + (0 if max_start <= 0
                      else int(rng.integers(0, max_start + 1)))
        interval = set(CAL[start:start + k])
        inj_days = interval & set(MASKS[ck0])
        idx = {e: j for j, e in enumerate(lat[ck0]["edges"])}
        dpos = {d: j for j, d in enumerate(lat[ck0]["days"])}
        for e in targets:
            for d in inj_days:
                lat[ck0]["u"][idx[e], dpos[d]] += d_
    carriers = {}
    for ck in CARRIERS:
        L = lat[ck]
        vals = np.tanh(L["u"])
        rr = {}
        for j, e in enumerate(L["edges"]):
            row = {}
            for t, d in enumerate(L["days"]):
                if not L["mcar"][j, t]:
                    row[d] = float(vals[j, t])
            rr[e] = row
        carriers[ck] = {"registered_days": list(MASKS[ck]),
                        "stations": stations_of(ck),
                        "segments": segments_of(ck), "r": rr}
    return {"schema": "fixture-panel-cal-v1",
            "calendar_authority_mode": "bound",
            "shared_calendar_days": list(CAL),
            "carriers": carriers}


def panel_digest(panel):
    return hashlib.sha256(json.dumps(panel, sort_keys=True,
                                     separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def all_stations():
    out = []
    for ck in CARRIERS:
        out.extend(stations_of(ck))
    return out


def drop_station(panel, st):
    out = {k: panel[k] for k in ("schema", "calendar_authority_mode",
                                 "shared_calendar_days")}
    cs = {}
    for ck, c in panel["carriers"].items():
        if st not in c["stations"]:
            cs[ck] = c
            continue
        cs[ck] = {"registered_days": c["registered_days"],
                  "stations": [x for x in c["stations"] if x != st],
                  "segments": {k: v for k, v in c["segments"].items()
                               if k != st},
                  "r": {e: s for e, s in c["r"].items()
                        if st not in e.split("|")}}
    out["carriers"] = cs
    return out


# ---------------- B1A calendar memoized reduction ----------------

class B1ACalMemo:
    def __init__(self, panel):
        _cal, carriers = E._cal_load(panel)
        self.keys = sorted(carriers)
        self.V = {k: carriers[k]["V"] for k in self.keys}
        self.edges = {k: carriers[k]["edges"] for k in self.keys}
        self.memo = {k: {} for k in self.keys}

    def _basefit(self, k, baseset):
        key = tuple(sorted(baseset))
        hit = self.memo[k].get(key)
        if hit is not None:
            return hit
        cols = [b * E.B1A_CAL_BLOCK_LEN + i for b in key
                for i in range(E.B1A_CAL_BLOCK_LEN)]
        base = self.V[k][:, cols]
        cnt = np.isfinite(base).sum(axis=1)
        rows = np.where(cnt >= E.TESTABLE_MIN_BASELINE)[0]
        if rows.size:
            med = np.nanmedian(base[rows], axis=1)
            mad = np.nanmedian(np.abs(base[rows] - med[:, None]), axis=1)
            keep = mad > 0
            rows, med, mad = rows[keep], med[keep], mad[keep]
        else:
            med = mad = np.empty(0)
        out = (rows, med, mad)
        self.memo[k][key] = out
        return out

    def edge_window_max(self, k, perm):
        rows, med, mad = self._basefit(k, perm[:6])
        n_edges = self.V[k].shape[0]
        W = np.full(n_edges, np.nan)
        if rows.size == 0:
            return W
        ev_cols = [b * E.B1A_CAL_BLOCK_LEN + i for b in perm[6:]
                   for i in range(E.B1A_CAL_BLOCK_LEN)]
        ev = self.V[k][:, ev_cols][rows]
        z = (ev - med[:, None]) / (E.MAD_SCALE * mad[:, None])
        az = np.abs(z)
        fin = np.isfinite(az)
        w, wmin = E.B1A_WINDOW, E.B1A_WINDOW_MIN
        best = np.full(rows.size, np.nan)
        for s in range(0, az.shape[1] - w + 1):
            wf = fin[:, s:s + w]
            nf = wf.sum(axis=1)
            ok = nf >= wmin
            if not ok.any():
                continue
            means = np.full(rows.size, np.nan)
            means[ok] = np.nansum(
                np.where(wf, az[:, s:s + w], 0.0), axis=1)[ok] / nf[ok]
            best = np.where(np.isnan(best), means,
                            np.fmax(best, np.where(np.isnan(means),
                                                   -np.inf, means)))
        W[rows] = best
        return W

    def family_p(self, n_draws, fold, excluded_station=None):
        masks = {}
        for k in self.keys:
            if excluded_station is None:
                masks[k] = np.ones(len(self.edges[k]), dtype=bool)
            else:
                masks[k] = np.array(
                    [excluded_station not in e.split("|")
                     for e in self.edges[k]])

        def fam_T(perm):
            total = 0.0
            for k in self.keys:
                W = self.edge_window_max(k, perm)
                sub = W[masks[k]]
                fin = sub[np.isfinite(sub)]
                if fin.size == 0:
                    return None
                total += float(fin.max())
            return total

        T_obs = fam_T(list(range(E.B1A_CAL_BLOCKS)))
        if T_obs is None:
            return None, 0, None
        rng = E._rng(AMENDMENT2_SHA, "B1A", fold, "null")
        n_valid = ge = 0
        for _ in range(int(n_draws)):
            perm = [int(x) for x in rng.permutation(E.B1A_CAL_BLOCKS)]
            T_d = fam_T(perm)
            if T_d is None:
                continue
            n_valid += 1
            if T_d >= T_obs:
                ge += 1
        if n_valid < E._valid_floor(n_draws):
            return None, n_valid, T_obs
        return (1 + ge) / (n_valid + 1), n_valid, T_obs


def equivalence_gate():
    """rev-1.4/1.5 sec 6 MANDATORY gate: memoized B1A reduction byte-equal
    to E.b1a_family_cal on full + EVERY one of the 35 LOCO folds."""
    panel = make_panel("B1A", {"delta_lat": 1.2, "k": 25, "n_e": 10}, 0)
    memo = B1ACalMemo(panel)
    checks = []
    d_p, _nv, d_T = memo.family_p(99, "full")
    eng = E.b1a_family_cal(panel, doc_sha256=AMENDMENT2_SHA, n_draws=99,
                           power_contract={"certified": True})
    checks.append(("full", d_p == eng["p_value"] and d_T == eng["T_obs"]))
    for st in all_stations():
        fp, _n, fT = memo.family_p(99, f"loco:{st}", excluded_station=st)
        ef = E.b1a_family_cal(drop_station(panel, st),
                              doc_sha256=AMENDMENT2_SHA, n_draws=99,
                              power_contract={"certified": True},
                              fold=f"loco:{st}")
        checks.append((f"loco:{st}",
                       fp == ef["p_value"] and fT == ef["T_obs"]))
    ok = all(c for _n2, c in checks)
    return {"full_equal": bool(checks[0][1]),
            "fold_equal_all": bool(all(c for _n2, c in checks[1:])),
            "folds_checked": len(checks) - 1, "all_equal": bool(ok),
            "engine_commit_bound": "24b0d8f",
            "calendar_authority_sha256": E.CAL_AUTHORITY_SHA256}


def recovery(family, point, r, n_draws, run_folds):
    panel = make_panel(family, point, r)
    pdig = panel_digest(panel)
    if family == "B1A":
        memo = B1ACalMemo(panel)
        p, _nv, _T = memo.family_p(n_draws, "full")
        pre = p is not None and p <= ALPHA
        if not pre or not run_folds:
            return {"p": p, "pre": pre,
                    "post": False if run_folds else None,
                    "panel_sha256": pdig}
        folds = []
        for st in all_stations():
            fp, _n, _t = memo.family_p(n_draws, f"loco:{st}",
                                       excluded_station=st)
            folds.append({"p_value": fp})
        gate = E.loco_gate({"p_value": p}, folds, ALPHA)
        return {"p": p, "pre": True, "post": bool(gate["pass"]),
                "panel_sha256": pdig}
    fn = E.b2a_family_cal if family == "B2A" else E.b3a_family_cal
    full = fn(panel, doc_sha256=AMENDMENT2_SHA, n_draws=n_draws,
              power_contract={"certified": True})
    p = full["p_value"]
    pre = p is not None and p <= ALPHA
    if not pre or not run_folds:
        return {"p": p, "pre": pre, "post": False if run_folds else None,
                "panel_sha256": pdig}
    folds = []
    for st in all_stations():
        fr = fn(drop_station(panel, st), doc_sha256=AMENDMENT2_SHA,
                n_draws=n_draws, power_contract={"certified": True},
                fold=f"loco:{st}")
        folds.append({"p_value": fr["p_value"]})
    gate = E.loco_gate({"p_value": p}, folds, ALPHA)
    return {"p": p, "pre": True, "post": bool(gate["pass"]),
            "panel_sha256": pdig}


def grid_of(family):
    return {"B1A": B1A_GRID, "B2A": B2A_GRID, "B3A": B3A_GRID}[family]


def stage_s1(ckpt, done):
    for family in ("B1A", "B2A", "B3A"):
        for point in grid_of(family):
            for r in range(D0.TIER_S_R):
                k = D0.key_of("S1", family, point, r)
                if k in done:
                    continue
                t0 = time.time()
                rec = recovery(family, point, r, D0.TIER_S_DRAWS,
                               run_folds=False)
                D0.emit(ckpt, {"key": k, "stage": "S1", "family": family,
                               "point": point, "rep": r, "p": rec["p"],
                               "pre": rec["pre"],
                               "panel_sha256": rec["panel_sha256"],
                               "dt": round(time.time() - t0, 2)})
        print(f"[S1] {family} complete", flush=True)


def s_rank(done, family, stage, field):
    from collections import defaultdict
    agg = defaultdict(list)
    for row in done.values():
        if row.get("stage") == stage and row.get("family") == family:
            agg[json.dumps(row["point"], sort_keys=True)].append(
                bool(row[field]))
    ranked = sorted(agg.items(),
                    key=lambda kv: (-sum(kv[1]) / len(kv[1]), kv[0]))
    return [(json.loads(pk), sum(v) / len(v)) for pk, v in ranked]


def stage_s2(ckpt, done):
    for family in ("B1A", "B2A", "B3A"):
        for point, _pre in s_rank(done, family, "S1",
                                  "pre")[:D0.PRESCREEN_TOP]:
            for r in range(D0.TIER_S_R):
                k = D0.key_of("S2", family, point, r)
                if k in done:
                    continue
                rec = recovery(family, point, r, D0.TIER_S_DRAWS,
                               run_folds=True)
                D0.emit(ckpt, {"key": k, "stage": "S2", "family": family,
                               "point": point, "rep": r, "p": rec["p"],
                               "pre": rec["pre"], "post": rec["post"],
                               "panel_sha256": rec["panel_sha256"]})
        print(f"[S2] {family} complete", flush=True)


def stage_c(ckpt, done):
    for family in ("B1A", "B2A", "B3A"):
        for point, _post in s_rank(done, family, "S2",
                                   "post")[:D0.TIER_C_CANDIDATES]:
            n_run = successes = 0
            verdict = None
            for r in range(40):
                k = D0.key_of("C", family, point, r)
                if k in done:
                    row = done[k]
                else:
                    rec = recovery(family, point, r, D0.TIER_C_DRAWS,
                                   run_folds=True)
                    row = {"key": k, "stage": "C", "family": family,
                           "point": point, "rep": r, "p": rec["p"],
                           "pre": rec["pre"], "post": rec["post"],
                           "panel_sha256": rec["panel_sha256"]}
                    D0.emit(ckpt, row)
                n_run += 1
                successes += bool(row["post"])
                if n_run == 20:
                    if D0.cp_lower(successes, 20) >= 0.80:
                        verdict = "CERTIFIED"
                        break
                    if D0.cp_upper(successes, 20) < 0.80:
                        verdict = "FAILED"
                        break
                if n_run == 40:
                    verdict = ("CERTIFIED"
                               if D0.cp_lower(successes, 40) >= 0.80 else
                               ("FAILED" if D0.cp_upper(successes, 40) < 0.80
                                else "CANNOT_DETERMINE_POWER_ESTIMATE"))
            D0.emit(ckpt, {"key": D0.key_of("Cv", family, point, 0),
                           "stage": "Cverdict", "family": family,
                           "point": point, "n": n_run,
                           "successes": successes,
                           "lb95": D0.cp_lower(successes, n_run),
                           "ub95": D0.cp_upper(successes, n_run),
                           "verdict": verdict})
            print(f"[C] {family} {point} -> {verdict} "
                  f"({successes}/{n_run})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stages", default="gate,S1,S2,C")
    args = ap.parse_args()
    stages = args.stages.split(",")
    if "gate" in stages:
        gate = equivalence_gate()
        print(f"[gate] {json.dumps(gate)}", flush=True)
        if not gate["all_equal"]:
            print("EQUIVALENCE GATE FAILED -- no tables admissible; direct "
                  "engine fallback required", flush=True)
            sys.exit(2)
        D0.emit(args.ckpt, {"key": "gateCal", "stage": "gate", **gate})
    done = D0.load_done(args.ckpt)
    if "S1" in stages:
        stage_s1(args.ckpt, done)
        done = D0.load_done(args.ckpt)
    if "S2" in stages:
        stage_s2(args.ckpt, done)
        done = D0.load_done(args.ckpt)
    if "C" in stages:
        stage_c(args.ckpt, done)
    print("DRIVER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
