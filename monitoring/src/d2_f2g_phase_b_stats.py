#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault2graph Phase B statistics engine (cayley).

Contract: docs/f2g_phase_b_prereg_v1.md rev-2 (geospec a44819d, sha256
d1929c3127f6d76b87e95319e0f56c4d8ddf4b0cd6226aa9c2e1fe564e44e04e), which
encodes codex R1.2 repairs 1-5 and rulings 1-3 (inbox f24eb2e).
Red-first bar (BAR-UNEDITED, grassmann): test_f2g_phase_b_stats_redkats_grassmann.py.

FIXTURE-ONLY until codex freeze + asylum seal: this module never opens the
Phase-A artifact; a separate sealed run driver feeds it panels after the seal.
Panel schema: fixture-panel-v1 (the bar's in-file authority).
"""
import hashlib
import math

import numpy as np

N_DRAWS = 9999
ALPHA_FAMILY = 0.05 / 3
BASELINE_DAYS = 60
TESTABLE_MIN_BASELINE = 45
PERSISTENCE_K = 3
Z_PERSIST = 3.0
TOP_DECILE = 0.10
BY_Q = 0.05
MAD_SCALE = 1.4826
EIGENGAP_MIN = 1e-6
FIEDLER_COORD_MIN = 1e-10
# prereg sec 6 registers >= 9,900 valid of 9,999; fixtures run smaller n_draws,
# so the floor generalizes proportionally (exact 9,900 at the registered 9,999)
_FLOOR_NUM, _FLOOR_DEN = 9900, 9999


def _valid_floor(n_draws):
    return math.ceil(n_draws * _FLOOR_NUM / _FLOOR_DEN)


def derive_substream_seed(doc_sha256_hex, family, fold, purpose):
    material = f"{doc_sha256_hex}||{family}||{fold}||{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _rng(doc_sha256, family, fold, purpose):
    return np.random.Generator(np.random.PCG64(
        derive_substream_seed(doc_sha256, family, fold, purpose)))


def walk_forward_split(registered_days):
    days = [str(d) for d in registered_days]
    if len(set(days)) != len(days) or days != sorted(days):
        raise ValueError(
            "UNORDERED_REGISTERED_DAYS: the registered-day sequence must be "
            "strictly ascending ISO dates (prereg rev-2 sec 2)")
    if len(days) <= BASELINE_DAYS:
        raise ValueError(
            f"INSUFFICIENT_EVALUATION_DAYS: {len(days)} registered days leave "
            f"no evaluation window past the {BASELINE_DAYS}-day baseline")
    return days[:BASELINE_DAYS], days[BASELINE_DAYS:]


def _canonical_edge(e):
    a, b = str(e).split("|")
    return (a, b) if a < b else (b, a)


def _load_carrier(cdata):
    baseline_days, eval_days = walk_forward_split(cdata["registered_days"])
    days = list(baseline_days) + list(eval_days)
    pos = {d: j for j, d in enumerate(days)}
    edges = sorted({"|".join(_canonical_edge(e)) for e in cdata["r"]})
    idx = {e: i for i, e in enumerate(edges)}
    V = np.full((len(edges), len(days)), np.nan)
    for e, series in cdata["r"].items():
        i = idx["|".join(_canonical_edge(e))]
        for d, v in series.items():
            v = float(v)
            if d in pos and math.isfinite(v):
                V[i, pos[d]] = v
    return {"days": days, "eval_days": list(eval_days), "edges": edges,
            "V": V, "stations": sorted(cdata.get("stations", [])),
            "segments": dict(cdata.get("segments", {}))}


def _merge_excluded(total, part):
    for k, v in part.items():
        total[k] = total.get(k, 0) + v


def _b1_carrier(V):
    """One carrier's pipeline on an (already rotated, for null draws) matrix:
    positional split -> finite-support floor -> median/MAD fit -> degeneracy
    exclusions -> evaluation z. Returns (z, excluded, testable_row_indices)."""
    base = V[:, :BASELINE_DAYS]
    ev = V[:, BASELINE_DAYS:]
    cnt = np.isfinite(base).sum(axis=1)
    ins = cnt < TESTABLE_MIN_BASELINE
    excluded = {}
    if ins.any():
        excluded["INSUFFICIENT_BASELINE"] = int(ins.sum())
    rows = np.where(~ins)[0]
    if rows.size == 0:
        return None, excluded, []
    med = np.nanmedian(base[rows], axis=1)
    mad = np.nanmedian(np.abs(base[rows] - med[:, None]), axis=1)
    degen = mad == 0.0
    if degen.any():
        excluded["DEGENERATE_BASELINE"] = int(degen.sum())
    keep = rows[~degen]
    if keep.size == 0:
        return None, excluded, []
    scale = (MAD_SCALE * mad[~degen])[:, None]
    z = (ev[keep] - med[~degen][:, None]) / scale
    return z, excluded, keep.tolist()


def _b1_stat(loaded, offsets=None):
    """Family statistic across carriers. offsets: {carrier: int} rotates each
    carrier's ENTIRE registered day vector by ONE common offset (codex R1.2
    repair 1) before the positional split; None = observed data."""
    T = None
    pers = 0
    excluded = {}
    n_testable = 0
    zmap = {}
    for key in sorted(loaded):
        c = loaded[key]
        V = c["V"] if offsets is None else np.roll(c["V"], offsets[key], axis=1)
        z, exc, tr = _b1_carrier(V)
        _merge_excluded(excluded, exc)
        n_testable += len(tr)
        if z is None or z.size == 0:
            continue
        zmap[key] = (z, tr)
        fin = np.isfinite(z)
        if fin.any():
            m = float(np.max(np.abs(z[fin])))
            T = m if T is None else max(T, m)
        mask = fin & (np.abs(z) > Z_PERSIST)
        width = mask.shape[1] - PERSISTENCE_K + 1
        if width >= 1:
            run = mask[:, :width].copy()
            for j in range(1, PERSISTENCE_K):
                run &= mask[:, j:j + width]
            pers += int(run.any(axis=1).sum())
    return T, pers, excluded, n_testable, zmap


def _typed_verdict(p, power_contract):
    # bar AMENDMENT 1 G16b: power_contract carries {"certified": bool, ...};
    # not-certified => typed no-power, certified => plain NEGATIVE. The
    # original bar's {"passed": True} form stays accepted for its fixtures.
    if p <= ALPHA_FAMILY:
        return "POSITIVE_PRE_LOCO"
    if power_contract is not None and (power_contract.get("certified") is True
                                       or power_contract.get("passed") is True):
        return "NEGATIVE"
    return "CANNOT_DETERMINE_NO_POWER"


def b1_family(panel, *, doc_sha256, n_draws=N_DRAWS, power_contract=None,
              return_null=False, fold="full"):
    loaded = {k: _load_carrier(c) for k, c in panel["carriers"].items()}
    T_obs, pers_obs, excluded, n_testable, _ = _b1_stat(loaded)
    out = {"family": "B1", "T_obs": T_obs, "excluded": excluded,
           "testable_edges": int(n_testable), "n_draws": int(n_draws),
           "alpha": ALPHA_FAMILY, "fold": str(fold)}
    if n_testable == 0 or T_obs is None:
        out.update(p_value=None, n_valid_draws=0,
                   verdict="CANNOT_DETERMINE_NO_TESTABLE_EDGES",
                   persistence={"count_obs": None, "p_value": None,
                                "verdict_bearing": False})
        if return_null:
            out["null_T"] = []
        return out
    rng = _rng(doc_sha256, "B1", fold, "null")
    keys = sorted(loaded)
    null_T = []
    n_valid = ge = pers_ge = 0
    for _ in range(int(n_draws)):
        # ONE offset per carrier per draw; every edge of the carrier rotates
        # together (dependence preservation, codex repair 1 / bar G8)
        offsets = {k: int(rng.integers(0, loaded[k]["V"].shape[1]))
                   for k in keys}
        T_d, pers_d, _exc, _nt, _zm = _b1_stat(loaded, offsets)
        if T_d is None:
            null_T.append(float("nan"))
            continue
        n_valid += 1
        null_T.append(float(T_d))
        if T_d >= T_obs:
            ge += 1
        if pers_d >= pers_obs:
            pers_ge += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_T"] = null_T
    if n_valid < _valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT",
                   persistence={"count_obs": int(pers_obs), "p_value": None,
                                "verdict_bearing": False})
        return out
    p = (1 + ge) / (n_valid + 1)
    out["p_value"] = float(p)
    out["persistence"] = {"count_obs": int(pers_obs),
                          "p_value": float((1 + pers_ge) / (n_valid + 1)),
                          "verdict_bearing": False}
    out["verdict"] = _typed_verdict(p, power_contract)
    return out


def _b2_partition(edge_weights):
    """Fiedler-sign partition of the unique largest positive-weight component.
    Returns (partition dict station->+/-1, refusal code or None). Gate-1
    failures (tie, largest < 3 nodes, or no positive-weight edge at all) all
    type LCC_TIE per codex repair 2's condition bundling."""
    adj = {}
    for (a, b), w in edge_weights.items():
        if w > 0.0:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    if not adj:
        return None, "LCC_TIE"
    comps, seen = [], set()
    for s in sorted(adj):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    big = max(len(c) for c in comps)
    if big < 3 or sum(1 for c in comps if len(c) == big) > 1:
        return None, "LCC_TIE"
    nodes = next(c for c in comps if len(c) == big)
    n = len(nodes)
    ix = {s: i for i, s in enumerate(nodes)}
    W = np.zeros((n, n))
    for (a, b), w in edge_weights.items():
        if w > 0.0 and a in ix and b in ix:
            W[ix[a], ix[b]] = W[ix[b], ix[a]] = w
    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    lam2, lam3 = float(vals[1]), float(vals[2])
    if (lam3 - lam2) / max(lam3, 1e-12) < EIGENGAP_MIN:
        return None, "FIEDLER_DEGENERATE"
    v2 = vecs[:, 1]
    v2 = v2 / np.linalg.norm(v2)
    if np.any(np.abs(v2) <= FIEDLER_COORD_MIN):
        return None, "FIEDLER_ZERO_COORDINATE"
    if v2[0] < 0:  # nodes sorted -> index 0 is the lexicographically first
        v2 = -v2
    return {s: (1 if v2[ix[s]] > 0 else -1) for s in nodes}, None


def _b2_switches(seq):
    """Max switch count over adjacent comparable partitions in seq; pairwise
    NODESET_MISMATCH refusals for non-identical classified nodesets."""
    mx = None
    refs = []
    for (d1, p1), (d2, p2) in zip(seq, seq[1:]):
        if set(p1) != set(p2):
            refs.append({"day": d2, "vs": d1, "code": "NODESET_MISMATCH"})
            continue
        s = sum(1 for st in p1 if p1[st] != p2[st])
        mx = s if mx is None else max(mx, s)
    return mx, refs


def b2_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False,
              power_contract=None, fold="full"):
    day_refusals = []
    per_carrier = {}
    for key in sorted(panel["carriers"]):
        c = panel["carriers"][key]
        _b, ev_days = walk_forward_split(c["registered_days"])
        parts = []
        for d in ev_days:
            ew = {}
            for e, series in c["r"].items():
                if d in series and math.isfinite(float(series[d])):
                    ew[_canonical_edge(e)] = max(float(series[d]), 0.0)
            part, code = _b2_partition(ew)
            if code:
                day_refusals.append({"carrier": key, "day": d, "code": code})
            else:
                parts.append((d, part))
        per_carrier[key] = parts
    T_obs = None
    for key in sorted(per_carrier):
        mx, refs = _b2_switches(per_carrier[key])
        for r_ in refs:
            r_["carrier"] = key
        day_refusals.extend(refs)
        if mx is not None:
            T_obs = mx if T_obs is None else max(T_obs, mx)
    excluded = {}
    for r_ in day_refusals:
        excluded[r_["code"]] = excluded.get(r_["code"], 0) + 1
    out = {"family": "B2", "T_obs": T_obs, "max_switches": T_obs,
           "day_refusals": day_refusals, "excluded": excluded,
           "n_draws": int(n_draws), "alpha": ALPHA_FAMILY, "fold": str(fold)}
    if T_obs is None:
        out.update(p_value=None, n_valid_draws=0,
                   verdict="CANNOT_DETERMINE_NO_COMPARABLE_DAYS")
        if return_null:
            out["null_T"] = []
        return out
    rng = _rng(doc_sha256, "B2", fold, "null")
    keys = sorted(per_carrier)
    null_T = []
    n_valid = ge = 0
    for _ in range(int(n_draws)):
        T_d = None
        for key in keys:
            parts = per_carrier[key]
            if len(parts) < 2:
                continue
            perm = rng.permutation(len(parts))
            mx, _refs = _b2_switches([parts[i] for i in perm])
            if mx is not None:
                T_d = mx if T_d is None else max(T_d, mx)
        if T_d is None:
            null_T.append(float("nan"))
            continue
        n_valid += 1
        null_T.append(float(T_d))
        if T_d >= T_obs:
            ge += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_T"] = null_T
    if n_valid < _valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + ge) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _typed_verdict(p, power_contract)
    return out


def b3_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False,
              power_contract=None, fold="full"):
    day_refusals = []
    selection = {}
    sel_records = []
    seg_by_carrier = {}
    excluded = {}
    multi = len(panel["carriers"]) > 1
    for key in sorted(panel["carriers"]):
        loaded = _load_carrier(panel["carriers"][key])
        seg_by_carrier[key] = (loaded["stations"], dict(loaded["segments"]))
        z, exc, tr = _b1_carrier(loaded["V"])
        _merge_excluded(excluded, exc)
        edges = loaded["edges"]
        for j, d in enumerate(loaded["eval_days"]):
            cells = []
            if z is not None and z.size:
                for row_i, zv in zip(tr, z[:, j]):
                    if np.isfinite(zv):
                        a, b = edges[row_i].split("|")
                        cells.append((-abs(float(zv)), a, b))
            m = len(cells)
            if m == 0:
                day_refusals.append({"carrier": key, "day": d,
                                     "code": "INSUFFICIENT_DAILY_EDGES"})
                continue
            cells.sort()
            chosen = [(a, b) for (_nz, a, b) in cells[:math.ceil(TOP_DECILE * m)]]
            selection[f"{key}::{d}" if multi else d] = \
                [f"{a}|{b}" for a, b in chosen]
            sel_records.append((key, chosen))
    for r_ in day_refusals:
        excluded[r_["code"]] = excluded.get(r_["code"], 0) + 1
    out = {"family": "B3", "day_refusals": day_refusals, "excluded": excluded,
           "selection": selection, "n_draws": int(n_draws),
           "alpha": ALPHA_FAMILY, "fold": str(fold)}

    def frac(chosen, seg):
        return sum(1 for a, b in chosen if seg.get(a) != seg.get(b)) / len(chosen)

    if not sel_records:
        out.update(T_obs=None, p_value=None, n_valid_draws=0,
                   verdict="CANNOT_DETERMINE_NO_SELECTABLE_DAYS")
        if return_null:
            out["null_T"] = []
        return out
    T_obs = max(frac(chosen, seg_by_carrier[key][1])
                for key, chosen in sel_records)
    out["T_obs"] = float(T_obs)
    rng = _rng(doc_sha256, "B3", fold, "null")
    null_T = []
    n_valid = ge = 0
    for _ in range(int(n_draws)):
        permseg = {}
        for key in sorted(seg_by_carrier):
            stations, seg = seg_by_carrier[key]
            labels = [seg[s] for s in stations]
            perm = rng.permutation(len(stations))
            permseg[key] = {stations[i]: labels[perm[i]]
                            for i in range(len(stations))}
        T_d = max(frac(chosen, permseg[key]) for key, chosen in sel_records)
        n_valid += 1
        null_T.append(float(T_d))
        if T_d >= T_obs:
            ge += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_T"] = null_T
    if n_valid < _valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + ge) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _typed_verdict(p, power_contract)
    return out


def loco_gate(full_result, fold_results, alpha):
    """Conjunctive robustness gate ONLY (prereg rev-2 sec 4): full-data family
    must pass AND every fold must be scorable and independently pass. Can
    never create or promote a positive."""
    folds = list(fold_results)
    full_p = None if full_result is None else full_result.get("p_value")
    full_pass = full_p is not None and full_p <= alpha
    n_unscorable = sum(1 for f in folds
                       if f is None or f.get("p_value") is None)
    n_pass = sum(1 for f in folds
                 if f is not None and f.get("p_value") is not None
                 and f["p_value"] <= alpha)
    ok = bool(full_pass and folds and n_unscorable == 0
              and n_pass == len(folds))
    out = {"pass": ok, "full_pass": bool(full_pass), "n_folds": len(folds),
           "n_pass": n_pass, "n_unscorable": n_unscorable}
    if n_unscorable:
        out["code"] = "LOCO_FOLD_UNSCORABLE"
    return out


# ============================================================================
# AMENDMENT 1 (frozen F2G-PB-A1-R3-FREEZE-CODEX-20260820T1325Z, authority
# 7c3ca7b / f3d0830b...): amended families B1A/B2A/B3A supersede B1/B2/B3
# for verdict purposes; the frozen functions above are retained untouched as
# evidence surfaces. Bar authority: grassmann AMENDMENT 2 (BAR-UNEDITED).
# ============================================================================

B1A_BLOCKS = 11
B1A_BLOCK_LEN = 10
B1A_WINDOW = 7
B1A_WINDOW_MIN = 4


def _block_order(perm, block_len):
    return [b * block_len + i for b in perm for i in range(block_len)]


def _b1a_load(cdata, n_days):
    days = [str(d) for d in cdata["registered_days"]]
    if len(days) != n_days or days != sorted(days) or len(set(days)) != n_days:
        raise ValueError(
            f"B1A_REGISTERED_DAYS_MISMATCH: need {n_days} strictly ascending")
    pos = {d: j for j, d in enumerate(days)}
    edges = sorted({"|".join(_canonical_edge(e)) for e in cdata["r"]})
    idx = {e: i for i, e in enumerate(edges)}
    V = np.full((len(edges), n_days), np.nan)
    for e, series in cdata["r"].items():
        i = idx["|".join(_canonical_edge(e))]
        for d, v in series.items():
            v = float(v)
            if d in pos and math.isfinite(v):
                V[i, pos[d]] = v
    return V


def _b1a_carrier_T(V, order, baseline_len, testable_min, window, window_min):
    """Per-carrier B1A term: max over (testable edge, window) of the mean of
    finite |z| cells (scored iff >= window_min finite). None when the carrier
    has no scorable (edge, window)."""
    Vr = V[:, order]
    base = Vr[:, :baseline_len]
    ev = Vr[:, baseline_len:]
    cnt = np.isfinite(base).sum(axis=1)
    rows = np.where(cnt >= testable_min)[0]
    if rows.size == 0:
        return None
    med = np.nanmedian(base[rows], axis=1)
    mad = np.nanmedian(np.abs(base[rows] - med[:, None]), axis=1)
    keep = mad > 0
    rows = rows[keep]
    if rows.size == 0:
        return None
    z = (ev[rows] - med[keep][:, None]) / (MAD_SCALE * mad[keep][:, None])
    az = np.abs(z)
    fin = np.isfinite(az)
    best = None
    for s in range(0, az.shape[1] - window + 1):
        wf = fin[:, s:s + window]
        nf = wf.sum(axis=1)
        ok = nf >= window_min
        if not ok.any():
            continue
        sums = np.nansum(np.where(wf, az[:, s:s + window], 0.0), axis=1)
        m = float(np.max(sums[ok] / nf[ok]))
        if best is None or m > best:
            best = m
    return best


def b1a_family(panel, *, doc_sha256, n_draws=N_DRAWS, power_contract=None,
               return_null=False, exhaustive=False, n_blocks=B1A_BLOCKS,
               block_len=B1A_BLOCK_LEN, baseline_len=BASELINE_DAYS,
               testable_min=TESTABLE_MIN_BASELINE, window=B1A_WINDOW,
               window_min=B1A_WINDOW_MIN, fold="full"):
    n_days = int(n_blocks) * int(block_len)
    if baseline_len % block_len != 0 or not 0 < baseline_len < n_days:
        raise ValueError("B1A_SPLIT_NOT_BLOCK_ALIGNED")
    keys = sorted(panel["carriers"])
    mats = {k: _b1a_load(panel["carriers"][k], n_days) for k in keys}
    identity = list(range(n_days))

    def fam_T(order):
        total = 0.0
        for k in keys:
            t = _b1a_carrier_T(mats[k], order, baseline_len, testable_min,
                               window, window_min)
            if t is None:
                return None
            total += t
        return total

    T_obs = fam_T(identity)
    out = {"family": "B1A", "T_obs": T_obs, "n_draws": int(n_draws),
           "alpha": ALPHA_FAMILY, "fold": str(fold),
           "bound_carriers": keys, "excluded": {}}
    if T_obs is None:
        out.update(p_value=None, n_valid_draws=0,
                   verdict="CANNOT_DETERMINE_FAMILY_SCORABILITY")
        if return_null:
            out["null_T"] = []
        return out
    if exhaustive:
        import itertools
        ge = tot = 0
        for perm in itertools.permutations(range(n_blocks)):
            tot += 1
            T_p = fam_T(_block_order(perm, block_len))
            if T_p is not None and T_p >= T_obs:
                ge += 1
        out["p_exact"] = ge / tot
    rng = _rng(doc_sha256, "B1A", fold, "null")
    null_T = []
    n_valid = ge_n = 0
    for _ in range(int(n_draws)):
        perm = [int(x) for x in rng.permutation(int(n_blocks))]
        T_d = fam_T(_block_order(perm, block_len))
        if T_d is None:
            null_T.append(float("nan"))
            continue
        n_valid += 1
        null_T.append(float(T_d))
        if T_d >= T_obs:
            ge_n += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_T"] = null_T
    if n_valid < _valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + ge_n) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _typed_verdict(p, power_contract)
    return out


def _b2a_day_state(cdata, d):
    """One day's atomic capsule state: (partition dict or None, refusal code
    or None). A gap day (nothing measured) is a typed terminating refusal."""
    ew = {}
    for e, series in cdata["r"].items():
        if d in series and math.isfinite(float(series[d])):
            ew[_canonical_edge(e)] = max(float(series[d]), 0.0)
    if not ew:
        return None, "GAP_NO_MEASURED_EDGES"
    return _b2_partition(ew)


def _b2a_runs(states, day_names):
    """Maximal identical-partition runs over an ordered capsule sequence.
    Any refusal (gate failure, gap, nodeset mismatch vs the last ACCEPTED
    day) terminates the current run and is never bridged."""
    runs = 0
    refusals = []
    cur_part = None
    in_run = False
    last_nodeset = None
    accepted = 0
    for (part, code), d in zip(states, day_names):
        if code:
            # codex ruling 8621baf2: every excluded position clears the
            # frame-comparison reference (never bridge a gap/refusal)
            refusals.append({"day": d, "code": code})
            in_run = False
            cur_part = None
            last_nodeset = None
            continue
        nodeset = frozenset(part)
        if last_nodeset is not None and nodeset != last_nodeset:
            refusals.append({"day": d, "code": "NODESET_MISMATCH"})
            in_run = False
            cur_part = None
            last_nodeset = None
            continue
        last_nodeset = nodeset
        accepted += 1
        if in_run and part == cur_part:
            continue
        runs += 1
        cur_part = part
        in_run = True
    return runs, refusals, accepted


def _b2a_runs_dyn(states, order):
    """Runs over one ordered sequence of moved capsules, per the frozen sec-A2
    authority as adjudicated by codex (ccc339d5): each capsule moves its
    INTRINSIC state atomically (partition, intrinsic gate refusal, gap, exact
    nodeset/frame); the RELATIONAL NODESET_MISMATCH, adjacency, and runs are
    RECOMPUTED after the common permutation (mismatch judged against the last
    ACCEPTED day in the permuted order). Any refusal terminates a run and is
    never bridged."""
    runs = 0
    cur_part = None
    in_run = False
    last_nodeset = None
    for i in order:
        part, code = states[i]
        if code:
            # codex ruling 8621baf2: excluded positions clear the reference
            in_run = False
            cur_part = None
            last_nodeset = None
            continue
        nodeset = frozenset(part)
        if last_nodeset is not None and nodeset != last_nodeset:
            in_run = False
            cur_part = None
            last_nodeset = None
            continue
        last_nodeset = nodeset
        if in_run and part == cur_part:
            continue
        runs += 1
        cur_part = part
        in_run = True
    return runs


def b2a_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False,
               power_contract=None, fold="full", audit_orders=None):
    keys = sorted(panel["carriers"])
    caps = {}
    n_eval = None
    for k in keys:
        c = panel["carriers"][k]
        _b, ev_days = walk_forward_split(c["registered_days"])
        if n_eval is None:
            n_eval = len(ev_days)
        elif len(ev_days) != n_eval:
            raise ValueError("B2A_JOINT_CAPSULE_LENGTH_MISMATCH")
        caps[k] = {"days": list(ev_days),
                   "states": [_b2a_day_state(c, d) for d in ev_days]}
    R_obs = 0
    runs_by_carrier = {}
    day_refusals = []
    for k in keys:
        runs, refs, accepted = _b2a_runs(caps[k]["states"], caps[k]["days"])
        for r_ in refs:
            r_["carrier"] = k
        day_refusals.extend(refs)
        runs_by_carrier[k] = runs
        if accepted < 2:
            out = {"family": "B2A", "runs_total": None, "T_obs": None,
                   "runs_by_carrier": runs_by_carrier,
                   "day_refusals": day_refusals, "excluded": {},
                   "n_draws": int(n_draws), "alpha": ALPHA_FAMILY,
                   "p_value": None, "n_valid_draws": 0,
                   "verdict": "CANNOT_DETERMINE_FAMILY_SCORABILITY "
                              "(CARRIER_NO_COMPARABLE_SEQUENCE)"}
            if return_null:
                out["null_R"] = []
            return out
        R_obs += runs
    out = {"family": "B2A", "runs_total": int(R_obs), "T_obs": int(R_obs),
           "runs_by_carrier": runs_by_carrier, "day_refusals": day_refusals,
           "excluded": {}, "n_draws": int(n_draws), "alpha": ALPHA_FAMILY,
           "fold": str(fold)}
    if audit_orders:
        # codex 1616Z ruling sub-cases: deterministic recomputation for the
        # EXPLICIT requested orders on the same moved capsules
        out["audit_runs_by_carrier"] = [
            {k: int(_b2a_runs_dyn(caps[k]["states"], [int(i) for i in o]))
             for k in keys}
            for o in audit_orders]
    rng = _rng(doc_sha256, "B2A", fold, "null")
    null_R = []
    null_orders = []
    null_rbc = []
    n_valid = le = 0
    for _ in range(int(n_draws)):
        perm = [int(x) for x in rng.permutation(n_eval)]
        R_d = 0
        rbc = {}
        for k in keys:
            runs = _b2a_runs_dyn(caps[k]["states"], perm)
            rbc[k] = int(runs)
            R_d += runs
        n_valid += 1
        null_R.append(int(R_d))
        # A5k audit vector (codex 1427Z repair 2 + binding closure): the ONE
        # common capsule position order per draw + per-carrier runs; null_R[i]
        # is by construction sum(null_runs_by_carrier[i].values()).
        null_orders.append(perm)
        null_rbc.append(rbc)
        if R_d <= R_obs:
            le += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_R"] = null_R
        out["null_orders"] = null_orders
        out["null_runs_by_carrier"] = null_rbc
    if n_valid < _valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + le) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _typed_verdict(p, power_contract)
    return out


def _balanced_exact_day_p(stations, seg_sizes, chosen, thresh):
    """Exact probability over all balanced station->segment labelings that
    the FIXED selected edge set has cross-segment count >= thresh."""
    import itertools
    n = len(stations)
    six = {s: i for i, s in enumerate(stations)}
    edges_i = [(six[a], six[b]) for a, b in chosen]
    hits = total = 0
    pool = list(range(n))
    for seg1 in itertools.combinations(pool, seg_sizes[0]):
        rest1 = [s for s in pool if s not in seg1]
        for seg2 in itertools.combinations(rest1, seg_sizes[1]):
            lab = {}
            for s in seg1:
                lab[s] = 0
            for s in seg2:
                lab[s] = 1
            for s in rest1:
                if s not in lab:
                    lab[s] = 2
            cross = sum(1 for a, b in edges_i if lab[a] != lab[b])
            total += 1
            if cross >= thresh:
                hits += 1
    return hits / total


def b3a_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False,
               power_contract=None, fold="full", exhaustive_space=False):
    day_refusals = []
    selection = {}
    sel_records = []
    seg_by_carrier = {}
    excluded = {}
    multi = len(panel["carriers"]) > 1
    for key in sorted(panel["carriers"]):
        loaded = _load_carrier(panel["carriers"][key])
        seg_by_carrier[key] = (loaded["stations"], dict(loaded["segments"]))
        z, exc, tr = _b1_carrier(loaded["V"])
        _merge_excluded(excluded, exc)
        edges = loaded["edges"]
        for j, d in enumerate(loaded["eval_days"]):
            cells = []
            if z is not None and z.size:
                for row_i, zv in zip(tr, z[:, j]):
                    if np.isfinite(zv):
                        a, b = edges[row_i].split("|")
                        cells.append((-abs(float(zv)), a, b))
            m = len(cells)
            if m == 0:
                day_refusals.append({"carrier": key, "day": d,
                                     "code": "INSUFFICIENT_DAILY_EDGES"})
                continue
            K = math.ceil(TOP_DECILE * m)
            if K < 2:
                day_refusals.append({"carrier": key, "day": d,
                                     "code": "DAY_K_UNSCORABLE"})
                continue
            cells.sort()
            chosen = [(a, b) for (_nz, a, b) in cells[:K]]
            selection[f"{key}::{d}" if multi else d] = \
                [f"{a}|{b}" for a, b in chosen]
            sel_records.append((key, d, chosen, K))
    for r_ in day_refusals:
        excluded[r_["code"]] = excluded.get(r_["code"], 0) + 1
    out = {"family": "B3A", "day_refusals": day_refusals,
           "excluded": excluded, "selection": selection,
           "n_draws": int(n_draws), "alpha": ALPHA_FAMILY, "fold": str(fold)}

    def cross_count(chosen, seg):
        return sum(1 for a, b in chosen if seg.get(a) != seg.get(b))

    if not sel_records:
        out.update(T_obs=None, C_obs=None, p_value=None, n_valid_draws=0,
                   verdict="CANNOT_DETERMINE_FAMILY_SCORABILITY "
                           "(NO_SCORABLE_DAYS)")
        if return_null:
            out["null_C"] = []
        return out
    C_obs = sum(1 for key, _d, chosen, K in sel_records
                if cross_count(chosen, seg_by_carrier[key][1]) >= K - 1)
    out["C_obs"] = int(C_obs)
    out["T_obs"] = int(C_obs)
    if exhaustive_space:
        exact = {}
        for key, d, chosen, K in sel_records:
            stations, seg = seg_by_carrier[key]
            labels = sorted(set(seg.values()))
            sizes = tuple(sum(1 for s in stations if seg[s] == lb)
                          for lb in labels)
            exact[f"{key}::{d}" if multi else d] = _balanced_exact_day_p(
                stations, sizes, chosen, K - 1)
        out["exact_space_p"] = exact
    rng = _rng(doc_sha256, "B3A", fold, "null")
    null_C = []
    n_valid = ge = 0
    for _ in range(int(n_draws)):
        permseg = {}
        for key in sorted(seg_by_carrier):
            stations, seg = seg_by_carrier[key]
            labels = [seg[s] for s in stations]
            perm = rng.permutation(len(stations))
            permseg[key] = {stations[i]: labels[perm[i]]
                            for i in range(len(stations))}
        C_d = sum(1 for key, _d, chosen, K in sel_records
                  if cross_count(chosen, permseg[key]) >= K - 1)
        n_valid += 1
        null_C.append(int(C_d))
        if C_d >= C_obs:
            ge += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_C"] = null_C
    if n_valid < _valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + ge) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _typed_verdict(p, power_contract)
    return out
