#!/usr/bin/env python3
"""precip_residual.py - Amendment R5: precipitation-regressed residual anomaly measure.

Implements docs/AMENDMENT_2026-07-29_precip_residual.md (registered + owner-signed):

    log(ratio_t) = b0 + b1*API7_t + b2*R30_t + eps_t        (per region, log-OLS, one 2% trim)
    anomaly_t    = percentile of eps_t in the training-window residuals
    stat_t       = rank-remap of that percentile onto the training-window RATIO
                   distribution (monotone by construction), so the downstream risk
                   mapping is unchanged in distribution.

Discipline (all registered):
  - fit window [today-395d, today-30d]  (R3-aligned 30-day self-absorption lag)
  - weekly refit (model store age > 7d), per-region coefficients, never transferred
  - per-region auto-activation only at >= MIN_FIT_DAYS matched history days
  - FAIL-OPEN: any failure returns None and the caller stays on the R3 ratio path
  - dual publication: docs/r5_daily.json (raw ratio + residual percentile + coeffs)

Precipitation: Open-Meteo historical archive (public, keyless), cached per region in
monitoring/data/precip_cache/. Ratio history: monitoring/data/ensemble_results/*.json.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / "monitoring" / "data" / "precip_cache"
MODEL_FILE = REPO / "monitoring" / "data" / "baselines" / "precip_regression.json"
RESULTS_DIR = REPO / "monitoring" / "data" / "ensemble_results"
R5_DAILY = REPO / "docs" / "r5_daily.json"

R5_CONFIG = {
    "spec": "AMENDMENT_2026-07-29_precip_residual (R5)",
    "api7_lambda": 0.9,
    "fit_window_start_days": 395,
    "fit_window_lag_days": 30,
    "refit_age_days": 7,
    "trim_frac": 0.02,
    "min_fit_days": 90,
    "r5_daily_keep_days": 400,
    "max_condition": 1e6,        # R5-4: normal-equations condition-number ceiling
    "max_leverage": 0.5,         # R5-4: max hat-matrix diagonal in the fit
    "envelope_factor": 1.5,      # R5-4: reject transform if today's predictors exceed
                                 #        fit range by > this factor (extrapolation guard)
}


# ---------------------------------------------------------------- precipitation

def fetch_precip(region: str, lat: float, lon: float, end: str) -> Optional[Dict[str, float]]:
    """Daily precipitation date->mm through `end`, cached; fetches only the missing tail."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{region}.json"
    data: Dict[str, float] = {}
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
        except Exception:
            data = {}
    need_start = ( _d(end) - timedelta(days=R5_CONFIG["fit_window_start_days"] + 40)).isoformat()
    have_through = max(data) if data else ""
    fetch_from = need_start if have_through < need_start else (
        (_d(have_through) + timedelta(days=1)).isoformat())
    if fetch_from <= end:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}&start_date={fetch_from}&end_date={end}"
               "&daily=precipitation_sum&timezone=UTC")
        with urllib.request.urlopen(url, timeout=45) as r:
            got = json.load(r)
        for t, p in zip(got["daily"]["time"], got["daily"]["precipitation_sum"]):
            if p is not None:
                data[t] = float(p)
        cache.write_text(json.dumps(data))
    return data or None


def indices(precip: Dict[str, float], day: str) -> Optional[tuple]:
    """(API7, R30) for `day`, needing 30 prior days present."""
    d0 = _d(day)
    api = 0.0
    lam = R5_CONFIG["api7_lambda"]
    vals30 = []
    for k in range(60, -1, -1):          # 60d warmup for the exponential index
        key = (d0 - timedelta(days=k)).isoformat()
        p = precip.get(key)
        if p is None:
            if k <= 30:
                return None              # a gap inside the R30 window disqualifies the day
            p = 0.0
        api = lam * api + p
        if k <= 29:
            vals30.append(p)
    return api, sum(vals30)


def _d(s: str) -> date:
    return date(*map(int, s[:10].split("-")))


# ---------------------------------------------------------------- ratio history

def load_ratio_history(region: str) -> Dict[str, float]:
    """date -> lambda_geo raw ratio, from the accumulated daily ensemble results."""
    out: Dict[str, float] = {}
    if not RESULTS_DIR.exists():
        return out
    for f in sorted(RESULTS_DIR.glob("ensemble_*.json")):
        try:
            d = json.loads(f.read_text())
            day = str(d.get("date", ""))[:10]
            lg = d.get("regions", {}).get(region, {}).get("components", {}).get("lambda_geo", {})
            # R5-1 fix: the fitter must see the RAW R3 ratio, never the operational
            # value (which post-activation is the R5 stat -> recursion). Prefer the
            # immutable lineage field; fall back to raw_value only for pre-activation
            # dates that predate R5 (method_epoch absent or 'r3'/'r5_shadow').
            epoch = lg.get("method_epoch", "r3")
            v = lg.get("raw_r3_ratio")
            if v is None and epoch in ("r3", "r5_shadow"):
                v = lg.get("raw_value")
            if day and lg.get("available") and v is not None and v > 0:
                out[day] = float(v)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------- fitting

def fit_region(region: str, ratios: Dict[str, float], precip: Dict[str, float],
               today: str, sig_digits: int = 11) -> Optional[dict]:
    """Log-OLS with one trim pass on the lagged window. Returns the model dict or None.

    r5-canonical-numerics-v1 (codex 185751Z §1, owner-gated; cayley bar a6ce01e):
    the SAME registered estimator and operation order (lstsq -> residuals ->
    deterministic trim -> lstsq refit), with the canonical projection q(sig_digits)
    applied at EVERY persisted/hashed/ranked/compared value: training-row inputs at
    row build; first-pass residuals before trim ordering (ordered by
    (abs(q_residual), day) — day is the fixed tie-breaker); refit beta, final
    residuals, condition, leverage, and predictor ranges before storage or gates;
    ratio_sorted is formed from q(original_ratio) (no exp(log) round trip). Gates
    compare canonical Decimals with the ±1 ulp11 ambiguity band and fail closed to
    R3 with R5_NUMERIC_BOUNDARY_AMBIGUITY. sig_digits: 11 = the product; 12/17 =
    the acceptance bar's negative-control and raw-view precisions."""
    from canonical_numerics import gate_compare, qfloat
    q = lambda v: qfloat(v, sig_digits)
    end = (_d(today) - timedelta(days=R5_CONFIG["fit_window_lag_days"])).isoformat()
    start = (_d(today) - timedelta(days=R5_CONFIG["fit_window_start_days"])).isoformat()
    rows = []
    for day, ratio in ratios.items():
        if start <= day <= end:
            ix = indices(precip, day)
            if ix is not None:
                # §1.1 quantize API7, R30, log(ratio) at row build; §1.5 retain the
                # original training ratio in the row.
                rows.append((day, q(math.log(max(ratio, 1e-3))), q(ix[0]), q(ix[1]),
                             q(ratio)))
    if len(rows) < R5_CONFIG["min_fit_days"]:
        return None
    def ols(rs):
        import numpy as np
        X = np.array([[1.0, a, r] for _, _, a, r, _ in rs])
        y = np.array([v for _, v, _, _, _ in rs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return beta.tolist(), resid.tolist()
    beta, resid = ols(rows)
    # §1.2: quantize first-pass residuals before trim ordering; order by
    # (abs(q_residual), day) — day breaks ties deterministically; drop the final k.
    k = max(1, int(len(rows) * R5_CONFIG["trim_frac"]))
    keep = sorted(range(len(rows)),
                  key=lambda i: (abs(q(resid[i])), rows[i][0]))[:-k]
    kept = [rows[i] for i in keep]
    beta, resid = ols(kept)
    import numpy as np
    Xk = np.array([[1.0, a, r] for _, _, a, r, _ in kept])
    try:
        cond = float(np.linalg.cond(Xk.T @ Xk))
    except Exception:
        return None
    XtX_inv = np.linalg.pinv(Xk.T @ Xk)
    lev = np.array([float(x @ XtX_inv @ x) for x in Xk])   # hat-matrix diagonal
    max_lev = float(lev.max())
    # §1.3 quantize before gates; §1 gate rule: canonical-vs-exact-Decimal compare
    # with the ambiguity band failing closed to R3.
    cond_q, lev_q = q(cond), q(max_lev)
    g_cond = gate_compare(cond_q, R5_CONFIG["max_condition"])
    g_lev = gate_compare(lev_q, R5_CONFIG["max_leverage"])
    if g_cond == "ambiguous" or g_lev == "ambiguous":
        logger.warning(f"R5 fit for {region}: R5_NUMERIC_BOUNDARY_AMBIGUITY on "
                       f"{'condition' if g_cond == 'ambiguous' else 'leverage'} gate; "
                       f"failing closed to R3")
        return None
    if g_cond == "ge" or g_lev == "ge":
        logger.warning(f"R5 fit for {region} fails leverage/condition gate "
                       f"(cond={cond_q}, max_lev={lev_q}); staying on R3")
        return None
    return {
        "cond": cond_q, "max_leverage": lev_q,
        "api7_range": [min(a for _, _, a, _, _ in kept), max(a for _, _, a, _, _ in kept)],
        "r30_range": [min(r for _, _, _, r, _ in kept), max(r for _, _, _, r, _ in kept)],
        "region": region, "fitted_date": today, "n": len(keep),
        "beta": [q(b) for b in beta],
        "resid_sorted": sorted(q(v) for v in resid),
        "ratio_sorted": sorted(orig for _, _, _, _, orig in kept),
        "window": [start, end],
    }


def load_models() -> dict:
    if MODEL_FILE.exists():
        try:
            return json.loads(MODEL_FILE.read_text())
        except Exception:
            pass
    return {}


def get_model(region: str, lat: float, lon: float, today: str):
    """Weekly-refit model store. Returns (model|None, reason).

    R5-2 fix: NEVER serve a model whose age exceeds refit_age_days. A fresh model that
    is still within age is served directly. When a refit is due, ANY failure
    (precip/fit/eligibility) returns None with a reason code -> the caller falls back to
    the R3 ratio path for that run (the signed rule). No unbounded stale-model service."""
    models = load_models()
    m = models.get(region)
    if m:
        age = (_d(today) - _d(m["fitted_date"])).days
        win_ok = m.get("window", ["", ""])[1] <= (_d(today) - timedelta(
            days=R5_CONFIG["fit_window_lag_days"])).isoformat()
        # R5-R1: reject future-dated models (age<0, e.g. a replay date before the fit)
        # and models whose fit window is not fully lagged before today.
        if 0 <= age <= R5_CONFIG["refit_age_days"] and win_ok:
            return m, "fresh"
    # refit is due (or no model): a fresh fit must succeed or we fall back to R3
    precip = fetch_precip(region, lat, lon, today)
    if not precip:
        return None, "fallback_r3:precip_unavailable"
    ratios = load_ratio_history(region)
    fresh = fit_region(region, ratios, precip, today)
    if fresh is None:
        return None, "fallback_r3:fit_ineligible"
    models[region] = fresh
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(models, indent=1))
    return fresh, "refit"


# ---------------------------------------------------------------- the transform

def percentile_of(sorted_vals, v) -> float:
    """Rank of v: (# strictly-less) / n, in [0, 1). R5-5: paired with rank_index/
    quantile_at so that mapping each training residual back yields its own ratio
    exactly (sorted composed mapping == ratio_sorted; verified by KAT)."""
    import bisect
    if not sorted_vals:
        return 0.5
    return bisect.bisect_left(sorted_vals, v) / len(sorted_vals)


def quantile_at(sorted_vals, p) -> float:
    """Inverse of percentile_of on the training grid: index floor(p*n), clamped."""
    if not sorted_vals:
        return 1.0
    # +1e-9 absorbs float round-trip error (k/n * n can be k-epsilon), so
    # quantile_at(percentile_of(v)) == v exactly for unique training values (R5-5).
    i = min(len(sorted_vals) - 1, max(0, int(p * len(sorted_vals) + 1e-9)))
    return sorted_vals[i]


def r5_transform(region: str, lat: float, lon: float, ratio: float,
                 today: str, historical: bool = False) -> Optional[dict]:
    """The R5 statistic for today's ratio. None on ANY failure (caller stays on R3 path).

    historical=True (codex 1246 replay-purity contract; cayley bar ac05448): an explicit
    historical replay (`--date` caller) fits DETERMINISTICALLY as-of `today` from the
    target-date window and NEVER reads or writes the persistent live model store — so a
    date's R5 record is identical across replay orders and store states. The historical
    output additionally publishes the as-of provenance: fitted_date, window, beta, and
    model_sha256. The normal-cadence weekly-refit store policy (R5-2/R5-R1) is untouched."""
    try:
        if historical:
            precip = fetch_precip(region, lat, lon, today)
            if not precip:
                return None
            ratios = load_ratio_history(region)
            model = fit_region(region, ratios, precip, today)
            if model is None:
                logger.info(f"R5 {region}: historical as-of fit ineligible; R3")
                return None
        else:
            model, reason = get_model(region, lat, lon, today)
            if model is None:
                logger.info(f"R5 {region}: {reason}")
                return None
            precip = fetch_precip(region, lat, lon, today)
            if not precip:
                return None
        ix = indices(precip, today)
        if ix is None:
            return None
        # r5-canonical-numerics-v1 §1.4: quantize today's predictors and residual;
        # every published R5 float is canonical.
        from canonical_numerics import gate_compare, model_digest, qfloat
        api7, r30 = qfloat(ix[0]), qfloat(ix[1])
        # R5-4: extrapolation guard -- if today's predictors are far outside the fit
        # envelope, the linear residual is unreliable; stay on R3.
        ar = model.get("api7_range"); rr = model.get("r30_range")
        if not ar or not rr:
            logger.warning(f"R5 {region}: model lacks predictor ranges; R3 (R5-R2)")
            return None
        ef = R5_CONFIG["envelope_factor"]
        # R5-R3: guard BOTH sides via the canonical gate rule (ambiguity fails closed).
        def out(v, lo, hi):
            hi_c = gate_compare(v, hi * ef)
            lo_c = "lt" if lo == 0 and v >= 0 else gate_compare(v, lo / ef)
            if hi_c == "ambiguous" or lo_c == "ambiguous":
                return "ambiguous"
            return "out" if (hi_c == "ge" and v > hi * ef) or \
                            (lo_c == "lt" and v < lo / ef) else "in"
        oa, orr = out(api7, ar[0], ar[1]), out(r30, rr[0], rr[1])
        if oa == "ambiguous" or orr == "ambiguous":
            logger.warning(f"R5 {region}: R5_NUMERIC_BOUNDARY_AMBIGUITY on envelope "
                           f"guard; failing closed to R3")
            return None
        if oa == "out" or orr == "out":
            logger.warning(f"R5 {region}: predictors outside fit envelope "
                           f"(api7={api7:.0f} in [{ar[0]:.0f},{ar[1]:.0f}], "
                           f"r30={r30:.0f} in [{rr[0]:.0f},{rr[1]:.0f}]); R3")
            return None
        b = model["beta"]
        resid = qfloat(math.log(max(ratio, 1e-3)) - (b[0] + b[1] * api7 + b[2] * r30))
        # §1.6: rank/index is the claim-bearing provenance; the displayed percentile
        # is derived from the exact rational rank and then canonicalized.
        import bisect
        rank = bisect.bisect_left(model["resid_sorted"], resid)
        n_grid = len(model["resid_sorted"])
        p = qfloat(rank / n_grid) if n_grid else 0.5
        stat = qfloat(quantile_at(model["ratio_sorted"], rank / n_grid if n_grid else 0.5))
        rec = {"stat": stat, "residual_percentile": p,
               "residual_rank_index": rank, "raw_ratio": qfloat(ratio),
               "api7": api7, "r30": r30, "beta": b, "n_fit": model["n"],
               "fitted_date": model["fitted_date"], "r5_computed": True,
               "r5_active": False}   # R5-R5: shadow -- computed, NOT operational
        if historical:
            # As-of provenance (codex 1246 + 185751Z): window + the CANONICAL model
            # digest, recomputable from persisted content + the declared policy alone.
            rec["window"] = list(model["window"])
            rec["model_sha256"] = model_digest(model, n=11)
        return rec
    except Exception as e:                                     # FAIL-OPEN, always
        logger.warning(f"R5 transform failed for {region} ({e}); falling back to R3 ratio path")
        return None


# ---------------------------------------------------------------- dual publication

def publish_r5_daily(today: str, per_region: Dict[str, Optional[dict]]) -> None:
    """Append today's dual record (raw + residual) to docs/r5_daily.json (rolling)."""
    try:
        data = {}
        if R5_DAILY.exists():
            data = json.loads(R5_DAILY.read_text())
        days = data.get("days", {})
        days[today] = {reg: (r if r else {"r5_active": False}) for reg, r in per_region.items()}
        cutoff = (_d(today) - timedelta(days=R5_CONFIG["r5_daily_keep_days"])).isoformat()
        data = {"config": R5_CONFIG, "updated": datetime.now().isoformat(),
                "days": {d: v for d, v in days.items() if d >= cutoff}}
        R5_DAILY.write_text(json.dumps(data, indent=1))
    except Exception as e:
        logger.warning(f"R5 dual publication failed ({e}); daily record not updated")
