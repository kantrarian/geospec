#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault2graph Phase B power-annex estimation driver (cayley) -- FIXTURE-ONLY.

Implements the registered estimation of the rev-1.1 power annexes:
  common  docs/f2g_phase_b_power_annex_common.md  @ d3aa25f  (baddf2aa...)
  B-1     docs/f2g_phase_b_power_annex_b1.md      @ b816f88  (fb2883f5...)
  B-2     docs/f2g_phase_b_power_annex_b2.md      @ b816f88  (dca9dede...)
  B-3     docs/f2g_phase_b_power_annex_b3.md      @ b816f88  (9414bd59...)
against the byte-stable engine d2_f2g_phase_b_stats.py @ 6034419 and the
FROZEN prereg rev-2 (a44819d / d1929c31...). No Phase-A artifact, real graph,
or waveform is touched.

Computational note (common sec 4): for B-1 the rotation null has at most
n_days distinct offsets per carrier, so per-(edge, offset) max-|z| profiles
are precomputed once per replicate and the 9,999 frozen-substream draws are
exact lookups; per-station LOCO folds are exact subset aggregations of the
same profiles (an edge's z depends only on its own series, so removing a
station's edges never changes another edge's z). The in-driver EQUIVALENCE
GATE must show byte-equal add-one p vs the engine before any table row is
written. Early exit: post-LOCO recovery requires the full-data pass first,
so folds run only for replicates whose full-data p <= alpha (exact).

Deterministic draw orders (registered here, disclosed in results):
  per replicate rng (from the family power substream, r-th uint64):
    for each carrier in sorted order: station shocks s (n_st x 110) row-major,
    then edge noise eps (n_edges x 110) row-major (edges sorted), then MCAR
    mask (n_edges x 110); after ALL carriers: family effect placement draws
    (B-1/B-3 onset start position when k < 50).
  B-2 swap membership: the lexicographically LAST m stations of block A and
    FIRST m of block B exchange blocks at eval position 25 (day index 85).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

import d2_f2g_phase_b_stats as E

FROZEN_DOC_SHA = "d1929c3127f6d76b87e95319e0f56c4d8ddf4b0cd6226aa9c2e1fe564e44e04e"
ALPHA = E.ALPHA_FAMILY
MU0 = math.atanh(0.30)
SIGMA_S = 0.15
SIGMA_E = 0.20
MCAR = 0.08
N_DAYS = 110
N_BASE = 60
N_EVAL = 50
CARRIERS = {"c1": 12, "c2": 12, "c3": 11}
SEG_SIZES = {"c1": (4, 4, 4), "c2": (4, 4, 4), "c3": (4, 4, 3)}

DAYS = []
for _i in range(N_DAYS):
    DAYS.append(f"20{26 + _i // 336:02d}-{1 + (_i // 28) % 12:02d}-{1 + _i % 28:02d}")
assert DAYS == sorted(DAYS) and len(set(DAYS)) == N_DAYS

B1_GRID = [{"delta_lat": d, "k": k, "n_e": n}
           for d in (0.3, 0.6, 1.2, 2.4) for k in (3, 10, 25, 50)
           for n in (3, 10, 33)]
B2_GRID = [{"m": m} for m in (1, 2, 3)]
B3_GRID = [{"delta_lat": d, "n_cross": n, "k": k}
           for d in (0.3, 0.6, 1.2, 2.4) for n in (3, 8) for k in (10, 25, 50)]

TIER_S_R = 50
TIER_S_DRAWS = 999
TIER_C_DRAWS = 9999
PRESCREEN_TOP = 8
TIER_C_CANDIDATES = 3


def stations_of(ck):
    return [f"{ck.upper()}.S{i:02d}" for i in range(CARRIERS[ck])]


def segments_of(ck):
    sts = stations_of(ck)
    a, b, c = SEG_SIZES[ck]
    out = {}
    for i, s in enumerate(sts):
        out[s] = "seg_1" if i < a else ("seg_2" if i < a + b else "seg_3")
    return out


def edges_of(ck):
    sts = stations_of(ck)
    return [f"{x}|{y}" for i, x in enumerate(sts) for y in sts[i + 1:]]


def rep_seed(family, r):
    master = np.random.Generator(np.random.PCG64(
        E.derive_substream_seed(FROZEN_DOC_SHA, family, "full", "power")))
    seeds = master.integers(0, 2 ** 63, size=r + 1, dtype=np.int64)
    return int(seeds[r])


def gen_latent(rng, family, point):
    """Latent u tensors + MCAR masks per carrier, in the registered order."""
    lat = {}
    for ck in sorted(CARRIERS):
        sts = stations_of(ck)
        eds = edges_of(ck)
        s = rng.normal(0.0, SIGMA_S, size=(len(sts), N_DAYS))
        eps = rng.normal(0.0, SIGMA_E, size=(len(eds), N_DAYS))
        mcar = rng.random((len(eds), N_DAYS)) < MCAR
        six = {st: i for i, st in enumerate(sts)}
        u = np.empty((len(eds), N_DAYS))
        for j, e in enumerate(eds):
            a, b = e.split("|")
            u[j] = MU0 + s[six[a]] + s[six[b]] + eps[j]
        if family == "B2" and ck == "c1":
            m = point["m"]
            block = {st: (0 if i < 6 else 1) for i, st in enumerate(sts)}
            swapped = dict(block)
            for st in sts[6 - m:6]:
                swapped[st] = 1
            for st in sts[6:6 + m]:
                swapped[st] = 0
            for j, e in enumerate(eds):
                a, b = e.split("|")
                pre = 0.9 if block[a] == block[b] else -0.5
                post = 0.9 if swapped[a] == swapped[b] else -0.5
                u[j, :85] += pre
                u[j, 85:] += post
        lat[ck] = {"u": u, "mcar": mcar, "edges": eds, "stations": sts}
    return lat


def apply_effect(rng, family, point, lat):
    if family == "B1":
        k, n_e, d = point["k"], point["n_e"], point["delta_lat"]
        targets = lat["c1"]["edges"][:n_e]
    elif family == "B3":
        k, n_e, d = point["k"], point["n_cross"], point["delta_lat"]
        seg = segments_of("c1")
        cross = [e for e in lat["c1"]["edges"]
                 if seg[e.split("|")[0]] != seg[e.split("|")[1]]]
        targets = cross[:n_e]
    else:
        return
    start = N_BASE if k == N_EVAL else \
        N_BASE + int(rng.integers(0, N_EVAL - k + 1))
    idx = {e: j for j, e in enumerate(lat["c1"]["edges"])}
    for e in targets:
        lat["c1"]["u"][idx[e], start:start + k] += d


def to_panel(lat):
    carriers = {}
    for ck in sorted(CARRIERS):
        L = lat[ck]
        r = {}
        vals = np.tanh(L["u"])
        for j, e in enumerate(L["edges"]):
            row = {}
            for t in range(N_DAYS):
                if not L["mcar"][j, t]:
                    row[DAYS[t]] = float(vals[j, t])
            r[e] = row
        carriers[ck] = {"registered_days": list(DAYS),
                        "stations": list(L["stations"]),
                        "segments": segments_of(ck), "r": r}
    return {"carriers": carriers}


def make_panel(family, point, r):
    rng = np.random.Generator(np.random.PCG64(rep_seed(family, r)))
    lat = gen_latent(rng, family, point)
    apply_effect(rng, family, point, lat)
    return to_panel(lat)


def drop_station(panel, st):
    out = {"carriers": {}}
    for ck, c in panel["carriers"].items():
        if st not in c["stations"]:
            out["carriers"][ck] = c
            continue
        out["carriers"][ck] = {
            "registered_days": c["registered_days"],
            "stations": [x for x in c["stations"] if x != st],
            "segments": {k: v for k, v in c["segments"].items() if k != st},
            "r": {e: s for e, s in c["r"].items() if st not in e.split("|")}}
    return out


def all_stations(panel):
    out = []
    for ck in sorted(panel["carriers"]):
        out.extend(panel["carriers"][ck]["stations"])
    return out


# ---------------- B-1 profile machinery (exact, memoized) ----------------

def b1_profiles(panel):
    """Per carrier: M[edge, offset] = that edge's max finite |z| under the
    carrier rotation by `offset` (NaN when not testable / no finite z)."""
    prof = {}
    for ck in sorted(panel["carriers"]):
        loaded = E._load_carrier(panel["carriers"][ck])
        V = loaded["V"]
        n = V.shape[1]
        M = np.full((V.shape[0], n), np.nan)
        for o in range(n):
            z, _exc, keep = E._b1_carrier(np.roll(V, o, axis=1))
            if z is None:
                continue
            az = np.abs(z)
            az[~np.isfinite(az)] = np.nan
            with np.errstate(all="ignore"):
                row_max = np.nanmax(az, axis=1)
            for r_i, ei in enumerate(keep):
                M[ei, o] = row_max[r_i]
        prof[ck] = {"M": M, "edges": loaded["edges"], "n_days": n,
                    "stations": loaded["stations"]}
    return prof


def b1_p_from_profiles(prof, n_draws, fold, excluded_station=None):
    """Byte-equal reproduction of E.b1_family's add-one p using profile
    lookups. Edge inclusion mask excludes edges incident to
    excluded_station (exact: per-edge z is independent of other edges)."""
    keys = sorted(prof)
    masks = {}
    for ck in keys:
        eds = prof[ck]["edges"]
        if excluded_station is None:
            masks[ck] = np.ones(len(eds), dtype=bool)
        else:
            masks[ck] = np.array(
                [excluded_station not in e.split("|") for e in eds])
    T_obs = None
    for ck in keys:
        col = prof[ck]["M"][masks[ck], 0]
        fin = col[np.isfinite(col)]
        if fin.size:
            m = float(fin.max())
            T_obs = m if T_obs is None else max(T_obs, m)
    if T_obs is None:
        return None, 0, None
    rng = np.random.Generator(np.random.PCG64(
        E.derive_substream_seed(FROZEN_DOC_SHA, "B1", fold, "null")))
    n_valid = ge = 0
    for _ in range(int(n_draws)):
        T_d = None
        for ck in keys:
            o = int(rng.integers(0, prof[ck]["n_days"]))
            col = prof[ck]["M"][masks[ck], o]
            fin = col[np.isfinite(col)]
            if fin.size:
                m = float(fin.max())
                T_d = m if T_d is None else max(T_d, m)
        if T_d is None:
            continue
        n_valid += 1
        if T_d >= T_obs:
            ge += 1
    if n_valid < E._valid_floor(n_draws):
        return None, n_valid, T_obs
    return (1 + ge) / (n_valid + 1), n_valid, T_obs


def equivalence_gate():
    """Common sec 4: driver must reproduce the engine BYTE-EQUAL before any
    table row is admissible. Small config: 2 carriers, 199 draws, full+fold."""
    rng = np.random.default_rng(424242)
    pan = {"carriers": {}}
    for ck, nst in (("c1", 5), ("c2", 4)):
        sts = [f"{ck.upper()}.S{i:02d}" for i in range(nst)]
        r = {}
        for i, a in enumerate(sts):
            for b in sts[i + 1:]:
                vals = np.tanh(MU0 + 0.2 * rng.standard_normal(N_DAYS))
                r[f"{a}|{b}"] = {d: float(v) for d, v in zip(DAYS, vals)}
        pan["carriers"][ck] = {"registered_days": list(DAYS),
                               "stations": sts,
                               "segments": {s: "seg_1" for s in sts}, "r": r}
    eng = E.b1_family(pan, doc_sha256=FROZEN_DOC_SHA, n_draws=199,
                      power_contract={"passed": True})
    prof = b1_profiles(pan)
    drv_p, _nv, drv_T = b1_p_from_profiles(prof, 199, "full")
    ok_full = (drv_p == eng["p_value"]) and (drv_T == eng["T_obs"])
    st = "C1.S01"
    eng_f = E.b1_family(drop_station(pan, st), doc_sha256=FROZEN_DOC_SHA,
                        n_draws=199, power_contract={"passed": True},
                        fold=f"loco:{st}")
    drv_fp, _nv2, drv_fT = b1_p_from_profiles(prof, 199, f"loco:{st}",
                                              excluded_station=st)
    ok_fold = (drv_fp == eng_f["p_value"]) and (drv_fT == eng_f["T_obs"])
    return {"full_equal": bool(ok_full), "fold_equal": bool(ok_fold),
            "engine_p": eng["p_value"], "driver_p": drv_p,
            "engine_fold_p": eng_f["p_value"], "driver_fold_p": drv_fp}


# ---------------- recovery evaluation ----------------

def b1_recovery(panel, n_draws, run_folds):
    prof = b1_profiles(panel)
    p, _nv, _T = b1_p_from_profiles(prof, n_draws, "full")
    pre = p is not None and p <= ALPHA
    if not pre or not run_folds:
        return {"p": p, "pre": pre, "post": False if run_folds else None}
    folds = []
    for st in all_stations(panel):
        fp, fnv, _ = b1_p_from_profiles(prof, n_draws, f"loco:{st}",
                                        excluded_station=st)
        folds.append({"p_value": fp})
    gate = E.loco_gate({"p_value": p}, folds, ALPHA)
    return {"p": p, "pre": True, "post": bool(gate["pass"])}


def bx_recovery(family, panel, n_draws, run_folds):
    fn = E.b2_family if family == "B2" else E.b3_family
    full = fn(panel, doc_sha256=FROZEN_DOC_SHA, n_draws=n_draws,
              power_contract={"passed": True})
    p = full["p_value"]
    pre = p is not None and p <= ALPHA
    if not pre or not run_folds:
        return {"p": p, "pre": pre, "post": False if run_folds else None}
    folds = []
    for st in all_stations(panel):
        fr = fn(drop_station(panel, st), doc_sha256=FROZEN_DOC_SHA,
                n_draws=n_draws, power_contract={"passed": True},
                fold=f"loco:{st}")
        folds.append({"p_value": fr["p_value"]})
    gate = E.loco_gate({"p_value": p}, folds, ALPHA)
    return {"p": p, "pre": True, "post": bool(gate["pass"])}


def recovery(family, point, r, n_draws, run_folds):
    panel = make_panel(family, point, r)
    if family == "B1":
        return b1_recovery(panel, n_draws, run_folds)
    return bx_recovery(family, panel, n_draws, run_folds)


# ---------------- binomial bounds & stopping ----------------

def binom_sf_geq(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def cp_lower(k, n, conf=0.95):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf_geq(k, n, mid) > 1 - conf:
            hi = mid
        else:
            lo = mid
    return lo


def cp_upper(k, n, conf=0.95):
    # One-sided Clopper-Pearson upper bound: UB solves P(X <= k | UB) = 1-conf,
    # i.e. sf_geq(k+1, UB) = conf. DEFECT FIXED 2026-08-20: v1 used `1 - conf`
    # here (wrong tail; e.g. 0/20 gave 0.003 instead of 0.139). No Tier-C
    # verdict was affected (all counts were 0; correct UB 0.139 < 0.80 still
    # FAILED), but reported bounds before this fix are invalid.
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf_geq(k + 1, n, mid) < conf:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------- checkpointed stages ----------------

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["key"]] = row
    return done


def emit(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def key_of(stage, family, point, r):
    pk = ",".join(f"{k}={point[k]}" for k in sorted(point))
    return f"{stage}|{family}|{pk}|r={r}"


def grid_of(family):
    return {"B1": B1_GRID, "B2": B2_GRID, "B3": B3_GRID}[family]


def stage_tier_s1(ckpt, done):
    for family in ("B1", "B2", "B3"):
        for point in grid_of(family):
            for r in range(TIER_S_R):
                k = key_of("S1", family, point, r)
                if k in done:
                    continue
                t0 = time.time()
                rec = recovery(family, point, r, TIER_S_DRAWS, run_folds=False)
                emit(ckpt, {"key": k, "stage": "S1", "family": family,
                            "point": point, "rep": r, "p": rec["p"],
                            "pre": rec["pre"], "dt": round(time.time() - t0, 2)})
        print(f"[S1] {family} complete", flush=True)


def s1_ranking(done, family):
    agg = {}
    for row in done.values():
        if row.get("stage") == "S1" and row["family"] == family:
            pk = json.dumps(row["point"], sort_keys=True)
            agg.setdefault(pk, []).append(bool(row["pre"]))
    ranked = sorted(agg.items(),
                    key=lambda kv: (-sum(kv[1]) / len(kv[1]), kv[0]))
    return [(json.loads(pk), sum(v) / len(v)) for pk, v in ranked]


def stage_tier_s2(ckpt, done):
    for family in ("B1", "B2", "B3"):
        top = s1_ranking(done, family)[:PRESCREEN_TOP]
        for point, _pre in top:
            for r in range(TIER_S_R):
                k = key_of("S2", family, point, r)
                if k in done:
                    continue
                rec = recovery(family, point, r, TIER_S_DRAWS, run_folds=True)
                emit(ckpt, {"key": k, "stage": "S2", "family": family,
                            "point": point, "rep": r, "p": rec["p"],
                            "pre": rec["pre"], "post": rec["post"]})
        print(f"[S2] {family} complete", flush=True)


def s2_ranking(done, family):
    agg = {}
    for row in done.values():
        if row.get("stage") == "S2" and row["family"] == family:
            pk = json.dumps(row["point"], sort_keys=True)
            agg.setdefault(pk, []).append(bool(row["post"]))
    ranked = sorted(agg.items(),
                    key=lambda kv: (-sum(kv[1]) / len(kv[1]), kv[0]))
    return [(json.loads(pk), sum(v) / len(v)) for pk, v in ranked]


def stage_tier_c(ckpt, done):
    for family in ("B1", "B2", "B3"):
        cands = s2_ranking(done, family)[:TIER_C_CANDIDATES]
        for point, _post in cands:
            n_run = 0
            successes = 0
            verdict = None
            for r in range(40):
                k = key_of("C", family, point, r)
                if k in done:
                    row = done[k]
                    n_run += 1
                    successes += bool(row["post"])
                else:
                    rec = recovery(family, point, r, TIER_C_DRAWS,
                                   run_folds=True)
                    emit(ckpt, {"key": k, "stage": "C", "family": family,
                                "point": point, "rep": r, "p": rec["p"],
                                "pre": rec["pre"], "post": rec["post"]})
                    n_run += 1
                    successes += bool(rec["post"])
                if n_run == 20:
                    if cp_lower(successes, 20) >= 0.80:
                        verdict = "CERTIFIED"
                        break
                    if cp_upper(successes, 20) < 0.80:
                        verdict = "FAILED"
                        break
                if n_run == 40:
                    verdict = "CERTIFIED" if cp_lower(successes, 40) >= 0.80 \
                        else ("FAILED" if cp_upper(successes, 40) < 0.80
                              else "CANNOT_DETERMINE_POWER_ESTIMATE")
            emit(ckpt, {"key": key_of("Cv", family, point, 0) ,
                        "stage": "Cverdict", "family": family, "point": point,
                        "n": n_run, "successes": successes,
                        "lb95": cp_lower(successes, n_run),
                        "ub95": cp_upper(successes, n_run),
                        "verdict": verdict})
            print(f"[C] {family} {point} -> {verdict} "
                  f"({successes}/{n_run})", flush=True)


def stage_b2_lemma(ckpt, done):
    """Annex B-2 sec L executable check. The lemma's hypothesis is
    PER-CARRIER (a two-value eligible-partition sequence), so the fixture is
    the injection carrier ALONE (m=3, no missingness): with a single carrier
    the family statistic reduces to the lemma's setting and the exact
    pipeline must return p == 1.0 identically.
    FIXTURE-SCOPE DEFECT FIXED 2026-08-20: v1 (stage key B2lemma) fed the
    full 3-carrier panel, so the noise carriers' partition churn entered the
    family max and p came out in (0.55, 0.99) -- a wrong-scope check, not a
    lemma failure. Corrected rows use stage key B2lemma_rev2; v1 rows are
    retained in the checkpoint as the disclosed defect record."""
    for r in range(10):
        k = f"L2|B2|m=3|r={r}"
        if k in done:
            continue
        rng = np.random.Generator(np.random.PCG64(rep_seed("B2", 1000 + r)))
        lat = gen_latent(rng, "B2", {"m": 3})
        for ck in lat:
            lat[ck]["mcar"][:] = False
        panel = to_panel(lat)
        panel = {"carriers": {"c1": panel["carriers"]["c1"]}}
        res = E.b2_family(panel, doc_sha256=FROZEN_DOC_SHA,
                          n_draws=TIER_C_DRAWS if r == 0 else TIER_S_DRAWS,
                          power_contract={"passed": True})
        emit(ckpt, {"key": k, "stage": "B2lemma_rev2", "rep": r,
                    "p": res["p_value"], "max_switches": res["max_switches"],
                    "n_draws": TIER_C_DRAWS if r == 0 else TIER_S_DRAWS})
    print("[L] B2 lemma checks complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stages", default="gate,S1,L,S2,C")
    args = ap.parse_args()
    stages = args.stages.split(",")
    if "gate" in stages:
        gate = equivalence_gate()
        print(f"[gate] {json.dumps(gate)}", flush=True)
        if not (gate["full_equal"] and gate["fold_equal"]):
            print("EQUIVALENCE GATE FAILED -- no tables are admissible",
                  flush=True)
            sys.exit(2)
        emit(args.ckpt, {"key": "gate", "stage": "gate", **gate})
    done = load_done(args.ckpt)
    if "S1" in stages:
        stage_tier_s1(args.ckpt, done)
        done = load_done(args.ckpt)
    if "L" in stages:
        stage_b2_lemma(args.ckpt, done)
        done = load_done(args.ckpt)
    if "S2" in stages:
        stage_tier_s2(args.ckpt, done)
        done = load_done(args.ckpt)
    if "C" in stages:
        stage_tier_c(args.ckpt, done)
    print("DRIVER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
