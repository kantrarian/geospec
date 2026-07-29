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
            v = lg.get("raw_value")
            if day and lg.get("available") and v is not None and v > 0:
                out[day] = float(v)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------- fitting

def fit_region(region: str, ratios: Dict[str, float], precip: Dict[str, float],
               today: str) -> Optional[dict]:
    """Log-OLS with one trim pass on the lagged window. Returns the model dict or None."""
    end = (_d(today) - timedelta(days=R5_CONFIG["fit_window_lag_days"])).isoformat()
    start = (_d(today) - timedelta(days=R5_CONFIG["fit_window_start_days"])).isoformat()
    rows = []
    for day, ratio in ratios.items():
        if start <= day <= end:
            ix = indices(precip, day)
            if ix is not None:
                rows.append((day, math.log(max(ratio, 1e-3)), ix[0], ix[1]))
    if len(rows) < R5_CONFIG["min_fit_days"]:
        return None
    def ols(rs):
        # normal equations for [1, api, r30] (stdlib; 3x3 solve)
        import numpy as np
        X = np.array([[1.0, a, r] for _, _, a, r in rs])
        y = np.array([v for _, v, _, _ in rs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return beta.tolist(), resid.tolist()
    beta, resid = ols(rows)
    # single trim pass: drop the largest trim_frac |residuals|, refit
    k = max(1, int(len(rows) * R5_CONFIG["trim_frac"]))
    keep = sorted(range(len(rows)), key=lambda i: abs(resid[i]))[:-k]
    beta, resid = ols([rows[i] for i in keep])
    return {
        "region": region, "fitted_date": today, "n": len(keep), "beta": beta,
        "resid_sorted": sorted(resid),
        "ratio_sorted": sorted(math.exp(v) for _, v, _, _ in (rows[i] for i in keep)),
        "window": [start, end],
    }


def load_models() -> dict:
    if MODEL_FILE.exists():
        try:
            return json.loads(MODEL_FILE.read_text())
        except Exception:
            pass
    return {}


def get_model(region: str, lat: float, lon: float, today: str) -> Optional[dict]:
    """Weekly-refit model store (R3 pattern)."""
    models = load_models()
    m = models.get(region)
    if m and (_d(today) - _d(m["fitted_date"])).days <= R5_CONFIG["refit_age_days"]:
        return m
    precip = fetch_precip(region, lat, lon, today)
    if not precip:
        return m                     # stale model beats no model (fail-open ladder)
    ratios = load_ratio_history(region)
    fresh = fit_region(region, ratios, precip, today)
    if fresh is None:
        return m
    models[region] = fresh
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(models, indent=1))
    return fresh


# ---------------------------------------------------------------- the transform

def percentile_of(sorted_vals, v) -> float:
    import bisect
    if not sorted_vals:
        return 0.5
    return bisect.bisect_right(sorted_vals, v) / len(sorted_vals)


def quantile_at(sorted_vals, p) -> float:
    if not sorted_vals:
        return 1.0
    i = min(len(sorted_vals) - 1, max(0, int(p * len(sorted_vals))))
    return sorted_vals[i]


def r5_transform(region: str, lat: float, lon: float, ratio: float,
                 today: str) -> Optional[dict]:
    """The R5 statistic for today's ratio. None on ANY failure (caller stays on R3 path)."""
    try:
        model = get_model(region, lat, lon, today)
        if model is None:
            return None
        precip = fetch_precip(region, lat, lon, today)
        if not precip:
            return None
        ix = indices(precip, today)
        if ix is None:
            return None
        api7, r30 = ix
        b = model["beta"]
        resid = math.log(max(ratio, 1e-3)) - (b[0] + b[1] * api7 + b[2] * r30)
        p = percentile_of(model["resid_sorted"], resid)
        stat = quantile_at(model["ratio_sorted"], p)
        return {"stat": stat, "residual_percentile": p, "raw_ratio": ratio,
                "api7": api7, "r30": r30, "beta": b, "n_fit": model["n"],
                "fitted_date": model["fitted_date"], "r5_active": True}
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
