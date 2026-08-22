#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 B1B family engine (cayley) -- robust burst statistic per the
FROZEN annex docs/f2g_window2_freeze/annex_b1b.md (design freeze CLOSED
@ 12161f6/5fba544) and grassmann's bar seam pin (`w2_b1b_family`,
calendar-frame, Phase-B result-shape conventions). Seam FIXED as
`w2_b1b`.

The registered B1A calendar pipeline (per-edge baseline median/MAD
robust z, length-7 calendar windows scored at >= 4/7 finite, mean over
finite, max over (edge, window); block-permutation null,
shift-raw-then-recompute) with TWO frozen transforms applied
IDENTICALLY to observed, null, LOCO, and injected panels:

1. Per-station robust renormalization (SYMMETRIC, single-valued per
   edge): S_req = endpoint set of the frozen tested edge set (the
   carrier's edge registry). EVERY s in S_req needs >= 20 finite
   baseline |z| values on incident edges AND finite positive
   m_s = MAD(those values); ANY failure -> the WHOLE CARRIER is typed
   ZERO_SCALE_REFUSAL -- no edge deletion, no graph shrinkage.
   m_car = median({m_s}), required finite positive (same refusal);
   q_s = max(1, m_s / m_car); z' = z / max(q_a, q_b) applied ONCE per
   edge, endpoint-order invariant.
2. Winsorization: |z'| capped at c = 8.0 BEFORE window means (a
   statistic transform, never a deletion).

Family T = MAX over carriers of the per-carrier max edge window-mean
winsorized |z'| (the annex's explicit aggregation; the "(B1A form)"
parenthetical binds the window-mean construction). One-sided HIGH.

Interpretation pins (disclosed, R1.2-able):
- m_s is computed from RAW baseline |z| (unwinsorized): the scale map
  is BUILT from baseline dispersion, then applied to evaluation z; the
  c=8 cap binds "BEFORE window means" per the annex and MAD is already
  robust. Winsorizing the m_s inputs would double-transform.
- A ZERO_SCALE_REFUSAL arising INSIDE a null draw invalidates that
  draw (NaN, outside the valid-draw count); the observed-path refusal
  types the whole family verdict. Never-shrink holds on every path.
- A carrier with no scorable (edge, window) refuses family scorability
  (all carriers required -- skipping one under a MAX aggregation would
  be graph shrinkage).
- Geometry (baseline positions, block length) comes from the window-2
  calendar fixed at PRESTART; scoring constants default to the PINNED
  Phase-B values (testable_min 45, window 7/4, MAD scale 1.4826) and
  are overridable ONLY for bar fixtures at reduced geometry. The annex
  constants (support 20, c = 8.0) are frozen module literals, never
  overridable.

Health admission (annex sec 1) is enforced by the accrual/barrier
instrument (the registry handed in here must be the pre-evaluation
w2_selection output); HEALTH_ADMISSION_VIOLATION is typed there. LOCO
and injected panels route through THIS same function (the four-path
identity the bar asserts). This module opens no window-2 value.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import d2_f2g_phase_b_stats as _pb  # the PINNED Phase-B engine

RENORM_MIN_SUPPORT = 20   # frozen (annex); never overridable
WINSOR_C = 8.0            # frozen (annex); never overridable
ALPHA_FAMILY = 0.05       # family-wise via the non-circular Holm selector


class PanelInvalid(ValueError):
    """Typed CALENDAR_* / REGISTRY_* / EDGE_* panel-shape refusals."""


def edge_scale(z, q_a, q_b):
    """The single-valued symmetric edge transform: z / max(q_a, q_b).
    (The bar's exact fixture: (8, 2, 3) -> 8/3.)"""
    return z / max(q_a, q_b)


def _load_edges_V(carrier, calendar):
    pos = {d: j for j, d in enumerate(calendar)}
    edges = sorted({"|".join(_pb._canonical_edge(e))
                    for e in carrier["r"]})
    idx = {e: i for i, e in enumerate(edges)}
    V = np.full((len(edges), len(calendar)), np.nan)
    regset = set(str(d) for d in carrier["registered_days"])
    for e, series in carrier["r"].items():
        i = idx["|".join(_pb._canonical_edge(e))]
        for d, v in series.items():
            v = float(v)
            if d in pos and d in regset and math.isfinite(v):
                V[i, pos[d]] = v
    return edges, V


def _carrier_T(edges, V, order, baseline_positions, testable_min,
               window, window_min):
    """One carrier under one position order. Returns (T, refusal):
    refusal = ("ZERO_SCALE_REFUSAL", station-or-None) types the whole
    carrier (never shrink); T None with no refusal = no scorable
    (edge, window)."""
    s_req = sorted({s for e in edges for s in e.split("|")})
    Vr = V[:, order]
    base = Vr[:, :baseline_positions]
    ev = Vr[:, baseline_positions:]
    cnt = np.isfinite(base).sum(axis=1)
    rows = np.where(cnt >= testable_min)[0]
    if rows.size == 0:
        return None, None
    med = np.nanmedian(base[rows], axis=1)
    mad = np.nanmedian(np.abs(base[rows] - med[:, None]), axis=1)
    keep = mad > 0
    rows = rows[keep]
    if rows.size == 0:
        return None, None
    med, mad = med[keep], mad[keep]
    zb = (base[rows] - med[:, None]) / (_pb.MAD_SCALE * mad[:, None])
    zev = (ev[rows] - med[:, None]) / (_pb.MAD_SCALE * mad[:, None])

    # transform 1: per-station scales over EXACTLY S_req, raw baseline
    # |z| support; ANY failure -> whole-carrier ZERO_SCALE_REFUSAL
    azb = np.abs(zb)
    incident = {s: [] for s in s_req}
    for ri, ei in enumerate(rows):
        a, b = edges[ei].split("|")
        incident[a].append(ri)
        incident[b].append(ri)
    m = {}
    for s in s_req:
        vals = (np.concatenate([azb[ri][np.isfinite(azb[ri])]
                                for ri in incident[s]])
                if incident[s] else np.empty(0))
        if vals.size < RENORM_MIN_SUPPORT:
            return None, ("ZERO_SCALE_REFUSAL", s)
        m_s = float(np.median(np.abs(vals - np.median(vals))))
        if not (math.isfinite(m_s) and m_s > 0):
            return None, ("ZERO_SCALE_REFUSAL", s)
        m[s] = m_s
    m_car = float(np.median(list(m.values())))
    if not (math.isfinite(m_car) and m_car > 0):
        return None, ("ZERO_SCALE_REFUSAL", None)
    q = {s: max(1.0, m[s] / m_car) for s in s_req}
    div = np.array([max(q[a], q[b]) for a, b in
                    (edges[ei].split("|") for ei in rows)])

    # transform 2: winsorize AFTER renormalization, BEFORE window means
    az = np.minimum(np.abs(zev / div[:, None]), WINSOR_C)

    fin = np.isfinite(az)
    best = None
    for s0 in range(0, az.shape[1] - window + 1):
        wf = fin[:, s0:s0 + window]
        nf = wf.sum(axis=1)
        ok = nf >= window_min
        if not ok.any():
            continue
        sums = np.nansum(np.where(wf, az[:, s0:s0 + window], 0.0),
                         axis=1)
        v = float(np.max(sums[ok] / nf[ok]))
        if best is None or v > best:
            best = v
    return best, None


def w2_b1b_family(panel, *, doc_sha256, n_draws=_pb.N_DRAWS,
                  return_null=False, power_contract=None, fold="full",
                  n_blocks=None, block_len=None, baseline_positions=None,
                  testable_min=_pb.TESTABLE_MIN_BASELINE,
                  window=_pb.B1A_WINDOW, window_min=_pb.B1A_WINDOW_MIN):
    calendar = panel.get("calendar")
    if not isinstance(calendar, list) or len(calendar) < 2:
        raise PanelInvalid("CALENDAR_EMPTY")
    if calendar != sorted(calendar) or len(set(calendar)) != \
            len(calendar):
        raise PanelInvalid("CALENDAR_UNORDERED")
    n_pos = len(calendar)
    if n_blocks is None or block_len is None or \
            baseline_positions is None:
        raise PanelInvalid("GEOMETRY_ABSENT: n_blocks/block_len/"
                           "baseline_positions come from the window-2 "
                           "calendar fixed at PRESTART")
    if n_blocks * block_len != n_pos or \
            baseline_positions % block_len != 0 or \
            not 0 < baseline_positions < n_pos:
        raise PanelInvalid("GEOMETRY_NOT_BLOCK_ALIGNED")

    keys = sorted(panel["carriers"])
    data = {}
    for k in keys:
        c = panel["carriers"][k]
        registry = set(c.get("registry") or ())
        if not registry:
            raise PanelInvalid(f"REGISTRY_ABSENT: {k}")
        edges, V = _load_edges_V(c, calendar)
        if not edges:
            raise PanelInvalid(f"EDGE_SET_EMPTY: {k}")
        endpoints = {s for e in edges for s in e.split("|")}
        if not endpoints <= registry:
            raise PanelInvalid(f"EDGE_ENDPOINT_NOT_IN_REGISTRY: {k}")
        data[k] = (edges, V)

    def fam_T(order):
        T = None
        for k in keys:
            t, refusal = _carrier_T(data[k][0], data[k][1], order,
                                    baseline_positions, testable_min,
                                    window, window_min)
            if refusal:
                return None, (k,) + refusal
            if t is None:
                return None, (k, "NO_SCORABLE", None)
            T = t if T is None else max(T, t)   # MAX over carriers
        return T, None

    identity = list(range(n_pos))
    T_obs, obs_refusal = fam_T(identity)
    out = {"family": "B1B", "frame": "calendar-w2", "T_obs": T_obs,
           "statistic": "max-carrier max-edge window-mean winsorized "
                        "renormalized |z| (one-sided HIGH)",
           "n_draws": int(n_draws), "alpha": ALPHA_FAMILY,
           "alpha_note": "family-wise via non-circular Holm at the "
                         "selector (v0.3 sec 5)",
           "fold": str(fold), "bound_carriers": keys,
           "winsor_c": WINSOR_C,
           "renorm_min_support": RENORM_MIN_SUPPORT}
    if obs_refusal is not None:
        k, code, station = obs_refusal
        if code == "ZERO_SCALE_REFUSAL":
            out.update(p_value=None, n_valid_draws=0,
                       verdict=f"ZERO_SCALE_REFUSAL (carrier={k}, "
                               f"station={station})")
        else:
            out.update(p_value=None, n_valid_draws=0,
                       verdict="CANNOT_DETERMINE_FAMILY_SCORABILITY "
                               f"(carrier={k})")
        if return_null:
            out["null_T"] = []
        return out

    def block_order(perm):
        return [b * block_len + i for b in perm
                for i in range(block_len)]

    rng = _pb._rng(doc_sha256, "B1B", fold, "null")
    null_T = []
    n_valid = ge_n = 0
    for _ in range(int(n_draws)):
        perm = [int(x) for x in rng.permutation(int(n_blocks))]
        T_d, refusal = fam_T(block_order(perm))
        if refusal is not None or T_d is None:
            null_T.append(float("nan"))   # refusal-in-draw: invalid
            continue
        n_valid += 1
        null_T.append(float(T_d))
        if T_d >= T_obs:
            ge_n += 1
    out["n_valid_draws"] = n_valid
    if return_null:
        out["null_T"] = null_T
    if n_valid < _pb._valid_floor(n_draws):
        out.update(p_value=None, verdict="CANNOT_DETERMINE_NULL_SUPPORT")
        return out
    p = (1 + ge_n) / (n_valid + 1)
    out["p_value"] = float(p)
    out["verdict"] = _pb._typed_verdict(p, power_contract)
    return out


# ---------------------------------------------------------------- selftest
def _mk_carrier(rng, stations, n_pos, days, spike_edge=None,
                spike_positions=(), spike_value=100.0,
                sparse_station=None, sparse_keep=5):
    """Deterministic synthetic carrier: full graph, unit-scale noise."""
    r = {}
    for i, a in enumerate(stations):
        for b in stations[i + 1:]:
            e = f"{a}|{b}"
            series = {}
            for j, d in enumerate(days):
                v = float(rng.normal(0.0, 1.0))
                if spike_edge == e and j in spike_positions:
                    v = spike_value
                if sparse_station in (a, b) and j >= sparse_keep:
                    continue      # withhold data -> under-support
                series[d] = v
            r[e] = series
    return {"registry": list(stations), "registered_days": list(days),
            "r": r}


def _selftest():
    # exact bar fixture: (z=8, q_A=2, q_B=3) -> 8/3, order-invariant
    assert edge_scale(8.0, 2.0, 3.0) == edge_scale(8.0, 3.0, 2.0) \
        == 8.0 / 3.0

    n_blocks, block_len, base_pos = 8, 6, 24   # 48 positions, eval 24
    n_pos = n_blocks * block_len
    days = [f"D{i:03d}" for i in range(n_pos)]   # ordered labels
    sts = [f"S{i}" for i in range(5)]
    rng = np.random.Generator(np.random.PCG64(7))

    def fam(car, n_draws=99):
        return w2_b1b_family(
            {"calendar": days, "carriers": {"x": car}},
            doc_sha256="cd" * 32, n_draws=n_draws, n_blocks=n_blocks,
            block_len=block_len, baseline_positions=base_pos,
            testable_min=18, window=7, window_min=4)

    # 1. nominal quiet panel: computes, p unexceptional, q ~ 1
    car = _mk_carrier(rng, sts, n_pos, days)
    r = fam(car)
    assert r["T_obs"] is not None and r["p_value"] is not None, r
    assert r["verdict"].startswith(("NEGATIVE", "POSITIVE",
                                    "CANNOT_DETERMINE")), r

    # 2. winsor cap: an eval spike of 100 sigma scores <= 8.0 exactly
    car = _mk_carrier(rng, sts, n_pos, days, spike_edge="S0|S1",
                      spike_positions=range(base_pos, base_pos + 7))
    r = fam(car)
    assert r["T_obs"] is not None and r["T_obs"] <= WINSOR_C + 1e-12, r
    assert r["T_obs"] > 7.5, r    # the capped window mean is ~8

    # 3. ZERO_SCALE_REFUSAL: one under-supported station in an
    # otherwise valid graph -> WHOLE carrier typed, no partial T
    car = _mk_carrier(rng, sts, n_pos, days, sparse_station="S4")
    r = fam(car)
    assert r["T_obs"] is None and r["p_value"] is None, r
    assert "ZERO_SCALE_REFUSAL" in r["verdict"] \
        and "S4" in r["verdict"], r

    # 4. never-shrink under MAX aggregation: a second healthy carrier
    # does NOT rescue the family from carrier x's refusal
    car_bad = _mk_carrier(rng, sts, n_pos, days, sparse_station="S4")
    car_ok = _mk_carrier(rng, sts, n_pos, days)
    r = w2_b1b_family(
        {"calendar": days, "carriers": {"x": car_bad, "y": car_ok}},
        doc_sha256="cd" * 32, n_draws=49, n_blocks=n_blocks,
        block_len=block_len, baseline_positions=base_pos,
        testable_min=18, window=7, window_min=4)
    assert "ZERO_SCALE_REFUSAL" in r["verdict"], r

    # 5. determinism: identical inputs + seed -> identical p
    car = _mk_carrier(np.random.Generator(np.random.PCG64(11)), sts,
                      n_pos, days)
    car2 = _mk_carrier(np.random.Generator(np.random.PCG64(11)), sts,
                       n_pos, days)
    assert fam(car)["p_value"] == fam(car2)["p_value"]

    print("w2_b1b selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
