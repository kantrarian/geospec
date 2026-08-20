#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault2graph Phase B power estimation driver -- AMENDED families (cayley).

FIXTURE-ONLY. Implements the amended-lane estimation of annexes B1A/B2A/B3A
v1 rev-1.1 (@ a0cc87c) under common protocol rev-1.3 (@ 60ea20e / b6352e91):
generator with the shared calendar factor gamma=0.05 (G[110] drawn FIRST per
the registered order), three-segment frozen geometry, B1A/B2A/B3A replicate
streams at the FROZEN AMENDMENT seed root (f3d0830b...), Tier-S/S2/C with the
post-LOCO endpoint and Clopper-Pearson stopping.

B1A exact reduction (common sec 4, PROVEN by the in-driver gate before any
table row): the 60/50 split is block-aligned under 11x10-day blocks, so per-
edge baseline median/MAD depend only on WHICH 6 blocks occupy the baseline
(median is order-invariant) -- memoized per replicate on the C(11,6)=462
baseline block-sets; per-draw per-edge window maxima then make every LOCO
fold an exact masked subset maximum (each edge's z depends only on its own
series). Gate: byte-equal add-one p vs E.b1a_family on registered fixtures
(full + fold); on gate failure the driver must fall back to direct engine
calls. B2A/B3A call the engine directly.

Estimation LAUNCH is sequenced behind bar admission + annex admission; this
module's gate/smoke may run pre-admission (no tables emitted).
"""
import argparse
import json
import math
import sys
import time

import numpy as np

import d2_f2g_phase_b_stats as E
import f2g_phase_b_power_estimation_cayley as D0

AMENDMENT_SHA = ("f3d0830b38869d8b6f0b03d113d45ae0f111e8645bd4a2934582b21e"
                 "48e909e8")
ALPHA = E.ALPHA_FAMILY
GAMMA = 0.05
N_DAYS = 110
N_BASE = 60
DAYS = D0.DAYS
CARRIERS = D0.CARRIERS
B1A_GRID = [{"delta_lat": d, "k": k, "n_e": n}
            for d in (0.3, 0.6, 1.2, 2.4) for k in (3, 10, 25, 50)
            for n in (3, 10, 33)]
B2A_GRID = [{"m": m} for m in (1, 2, 3)]
B3A_GRID = [{"delta_lat": d, "n_cross": n, "k": k}
            for d in (0.3, 0.6, 1.2, 2.4) for n in (3, 8) for k in (10, 25, 50)]


def rep_seed(family, r):
    master = np.random.Generator(np.random.PCG64(
        E.derive_substream_seed(AMENDMENT_SHA, family, "full", "power")))
    return int(master.integers(0, 2 ** 63, size=r + 1, dtype=np.int64)[r])


def gen_latent(rng, family, point):
    """Registered order (common rev-1.3 sec 1): G[110] FIRST, then per sorted
    carrier s/eps/mcar, then effect placement (in apply_effect)."""
    G = rng.standard_normal(N_DAYS)
    lat = {}
    for ck in sorted(CARRIERS):
        sts = D0.stations_of(ck)
        eds = D0.edges_of(ck)
        s = rng.normal(0.0, D0.SIGMA_S, size=(len(sts), N_DAYS))
        eps = rng.normal(0.0, D0.SIGMA_E, size=(len(eds), N_DAYS))
        mcar = rng.random((len(eds), N_DAYS)) < D0.MCAR
        six = {st: i for i, st in enumerate(sts)}
        u = np.empty((len(eds), N_DAYS))
        for j, e in enumerate(eds):
            a, b = e.split("|")
            u[j] = D0.MU0 + GAMMA * G + s[six[a]] + s[six[b]] + eps[j]
        if family == "B2A" and ck == "c1":
            m = point["m"]
            block = {st: (0 if i < 6 else 1) for i, st in enumerate(sts)}
            swapped = dict(block)
            for st in sts[6 - m:6]:
                swapped[st] = 1
            for st in sts[6:6 + m]:
                swapped[st] = 0
            for j, e in enumerate(eds):
                a, b = e.split("|")
                u[j, :85] += 0.9 if block[a] == block[b] else -0.5
                u[j, 85:] += 0.9 if swapped[a] == swapped[b] else -0.5
        lat[ck] = {"u": u, "mcar": mcar, "edges": eds, "stations": sts}
    return lat


def apply_effect(rng, family, point, lat):
    if family == "B1A":
        k, n_e, d = point["k"], point["n_e"], point["delta_lat"]
        targets = lat["c1"]["edges"][:n_e]
    elif family == "B3A":
        k, n_e, d = point["k"], point["n_cross"], point["delta_lat"]
        seg = D0.segments_of("c1")
        cross = [e for e in lat["c1"]["edges"]
                 if seg[e.split("|")[0]] != seg[e.split("|")[1]]]
        targets = cross[:n_e]
    else:
        return
    start = N_BASE if k == 50 else N_BASE + int(rng.integers(0, 50 - k + 1))
    idx = {e: j for j, e in enumerate(lat["c1"]["edges"])}
    for e in targets:
        lat["c1"]["u"][idx[e], start:start + k] += d


def make_panel(family, point, r):
    rng = np.random.Generator(np.random.PCG64(rep_seed(family, r)))
    lat = gen_latent(rng, family, point)
    apply_effect(rng, family, point, lat)
    return D0.to_panel(lat)


# ---------------- B1A memoized reduction ----------------

class B1AMemo:
    """Per-panel exact reduction. For each carrier: V (edges x 110); per
    baseline block-set (sorted 6-tuple): testable-row indices, med, mad.
    Per draw: per-edge window-max vector over the permuted eval order; family
    T = sum over carriers of max; LOCO folds = masked subset maxima."""

    def __init__(self, panel):
        self.keys = sorted(panel["carriers"])
        self.V = {}
        self.edges = {}
        self.stations = {}
        self.memo = {}
        for k in self.keys:
            self.V[k] = E._b1a_load(panel["carriers"][k], N_DAYS)
            loaded_edges = sorted({"|".join(E._canonical_edge(e))
                                   for e in panel["carriers"][k]["r"]})
            self.edges[k] = loaded_edges
            self.stations[k] = sorted(panel["carriers"][k]["stations"])
            self.memo[k] = {}

    def _basefit(self, k, baseset):
        key = tuple(sorted(baseset))
        hit = self.memo[k].get(key)
        if hit is not None:
            return hit
        cols = [b * 10 + i for b in key for i in range(10)]
        base = self.V[k][:, cols]
        cnt = np.isfinite(base).sum(axis=1)
        rows = np.where(cnt >= E.TESTABLE_MIN_BASELINE)[0]
        if rows.size:
            med = np.nanmedian(base[rows], axis=1)
            mad = np.nanmedian(np.abs(base[rows] - med[:, None]), axis=1)
            keep = mad > 0
            rows = rows[keep]
            med, mad = med[keep], mad[keep]
        else:
            med = mad = np.empty(0)
        out = (rows, med, mad)
        self.memo[k][key] = out
        return out

    def edge_window_max(self, k, perm):
        """Per-edge window-max vector (NaN = unscorable edge) for one draw."""
        baseset = perm[:6]
        rows, med, mad = self._basefit(k, baseset)
        n_edges = self.V[k].shape[0]
        W = np.full(n_edges, np.nan)
        if rows.size == 0:
            return W
        ev_cols = [b * 10 + i for b in perm[6:] for i in range(10)]
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

    def draw_offsets(self, n_draws, fold):
        rng = E._rng(AMENDMENT_SHA, "B1A", fold, "null")
        return [[int(x) for x in rng.permutation(E.B1A_BLOCKS)]
                for _ in range(int(n_draws))]

    def family_p(self, n_draws, fold, excluded_station=None):
        masks = {}
        for k in self.keys:
            if excluded_station is None:
                masks[k] = np.ones(len(self.edges[k]), dtype=bool)
            else:
                masks[k] = np.array(
                    [excluded_station not in e.split("|")
                     for e in self.edges[k]])
        identity = list(range(E.B1A_BLOCKS))

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

        T_obs = fam_T(identity)
        if T_obs is None:
            return None, 0, None
        n_valid = ge = 0
        for perm in self.draw_offsets(n_draws, fold):
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
    """Common rev-1.3 sec 4 gate: memoized B1A reduction must be byte-equal
    to E.b1a_family (add-one p, full + fold) on registered fixtures."""
    pan = make_panel("B1A", {"delta_lat": 1.2, "k": 25, "n_e": 10}, 0)
    memo = B1AMemo(pan)
    d_p, _nv, d_T = memo.family_p(199, "full")
    eng = E.b1a_family(pan, doc_sha256=AMENDMENT_SHA, n_draws=199,
                       power_contract={"certified": True})
    ok_full = (d_p == eng["p_value"]) and (d_T == eng["T_obs"])
    st = "C1.S03"
    d_fp, _nv2, d_fT = memo.family_p(199, f"loco:{st}", excluded_station=st)
    eng_f = E.b1a_family(D0.drop_station(pan, st), doc_sha256=AMENDMENT_SHA,
                         n_draws=199, power_contract={"certified": True},
                         fold=f"loco:{st}")
    ok_fold = (d_fp == eng_f["p_value"]) and (d_fT == eng_f["T_obs"])
    return {"full_equal": bool(ok_full), "fold_equal": bool(ok_fold),
            "engine_p": eng["p_value"], "driver_p": d_p,
            "engine_fold_p": eng_f["p_value"], "driver_fold_p": d_fp,
            "engine_commit_bound": "b4a7eee"}


# ---------------- recovery ----------------

def recovery(family, point, r, n_draws, run_folds):
    panel = make_panel(family, point, r)
    if family == "B1A":
        memo = B1AMemo(panel)
        p, _nv, _T = memo.family_p(n_draws, "full")
        pre = p is not None and p <= ALPHA
        if not pre or not run_folds:
            return {"p": p, "pre": pre, "post": False if run_folds else None}
        folds = []
        for st in D0.all_stations(panel):
            fp, _n, _t = memo.family_p(n_draws, f"loco:{st}",
                                       excluded_station=st)
            folds.append({"p_value": fp})
        gate = E.loco_gate({"p_value": p}, folds, ALPHA)
        return {"p": p, "pre": True, "post": bool(gate["pass"])}
    fn = E.b2a_family if family == "B2A" else E.b3a_family
    full = fn(panel, doc_sha256=AMENDMENT_SHA, n_draws=n_draws,
              power_contract={"certified": True})
    p = full["p_value"]
    pre = p is not None and p <= ALPHA
    if not pre or not run_folds:
        return {"p": p, "pre": pre, "post": False if run_folds else None}
    folds = []
    for st in D0.all_stations(panel):
        fr = fn(D0.drop_station(panel, st), doc_sha256=AMENDMENT_SHA,
                n_draws=n_draws, power_contract={"certified": True},
                fold=f"loco:{st}")
        folds.append({"p_value": fr["p_value"]})
    gate = E.loco_gate({"p_value": p}, folds, ALPHA)
    return {"p": p, "pre": True, "post": bool(gate["pass"])}


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
        for point, _pre in s_rank(done, family, "S1", "pre")[:D0.PRESCREEN_TOP]:
            for r in range(D0.TIER_S_R):
                k = D0.key_of("S2", family, point, r)
                if k in done:
                    continue
                rec = recovery(family, point, r, D0.TIER_S_DRAWS,
                               run_folds=True)
                D0.emit(ckpt, {"key": k, "stage": "S2", "family": family,
                               "point": point, "rep": r, "p": rec["p"],
                               "pre": rec["pre"], "post": rec["post"]})
        print(f"[S2] {family} complete", flush=True)


def stage_c(ckpt, done):
    for family in ("B1A", "B2A", "B3A"):
        cands = s_rank(done, family, "S2", "post")[:D0.TIER_C_CANDIDATES]
        for point, _post in cands:
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
                           "pre": rec["pre"], "post": rec["post"]}
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
                    verdict = "CERTIFIED" if D0.cp_lower(successes, 40) >= 0.80 \
                        else ("FAILED" if D0.cp_upper(successes, 40) < 0.80
                              else "CANNOT_DETERMINE_POWER_ESTIMATE")
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
        if not (gate["full_equal"] and gate["fold_equal"]):
            print("EQUIVALENCE GATE FAILED -- no tables admissible; fix or "
                  "fall back to direct engine calls", flush=True)
            sys.exit(2)
        D0.emit(args.ckpt, {"key": "gateA", "stage": "gate", **gate})
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
