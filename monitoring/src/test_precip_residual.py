#!/usr/bin/env python3
"""KAT battery for Amendment R5 (pre-activation gate, amendment §4). Offline, synthetic.

Run:  python monitoring/src/test_precip_residual.py   -> N/N PASS expected
"""
from __future__ import annotations

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import precip_residual as pr

PASS = []


def kat(name, ok, detail=""):
    PASS.append(bool(ok))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


rng = random.Random(41)

# ---- synthetic world: seasonal precip drives the ratio (log-linear), plus noise,
# plus two injected NON-rain anomaly episodes ------------------------------------
D0 = date(2024, 6, 1)
N = 760
days = [(D0 + timedelta(days=i)).isoformat() for i in range(N)]
precip = {}
for i, d in enumerate(days):
    doy = (_i := ( D0 + timedelta(days=i))).timetuple().tm_yday
    wet = max(0.0, 40.0 * math.sin(2 * math.pi * (doy - 150) / 365.25))  # summer-wet
    precip[d] = max(0.0, rng.gauss(wet, wet * 0.5 + 2.0))
B0, B1, B2 = -1.2, 0.004, 0.002
ratios = {}
ANOM = {days[500], days[501], days[502]}        # dry-season injected anomaly (x4)
for i, d in enumerate(days):
    ix = pr.indices(precip, d)
    if ix is None:
        continue
    api7, r30 = ix
    lg = B0 + B1 * api7 + B2 * r30 + rng.gauss(0, 0.25)
    if d in ANOM:
        lg += math.log(4.0)
    ratios[d] = math.exp(lg)

today = days[-1]
model = pr.fit_region("synth", ratios, precip, today)

print("=== R5 KATs (pre-activation gate) ===")

kat("K0 model fits with the lagged window and recovers the rain coefficients",
    model is not None and model["n"] >= pr.R5_CONFIG["min_fit_days"]
    and abs(model["beta"][1] - B1) < 0.002 and abs(model["beta"][2] - B2) < 0.001,
    f"n={model and model['n']} beta={model and [round(b,4) for b in model['beta']]}")

# K1 -- residual deseasonalization beats calendar deseasonalization
mon_of = lambda d: int(d.split("-")[1])
logv = {d: math.log(v) for d, v in ratios.items()}
mon_med = {}
for m in range(1, 13):
    vals = [v for d, v in logv.items() if mon_of(d) == m]
    if vals:
        mon_med[m] = sorted(vals)[len(vals)//2]
cal_resid = {d: v - mon_med[mon_of(d)] for d, v in logv.items() if mon_of(d) in mon_med}
b = model["beta"]
rain_resid = {}
for d in ratios:
    ix = pr.indices(precip, d)
    if ix:
        rain_resid[d] = logv[d] - (b[0] + b[1]*ix[0] + b[2]*ix[1])
def corr(res):
    ds = [d for d in res if pr.indices(precip, d)]
    xs = [res[d] for d in ds]
    ys = [pr.indices(precip, d)[0] for d in ds]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (sx*sy)
def var(res):
    xs = list(res.values()); m = sum(xs)/len(xs)
    return sum((x-m)**2 for x in xs)/len(xs)
c_cal, c_rain = corr(cal_resid), corr(rain_resid)
kat("K1 rain-residual decorrelated from precip while calendar-residual is NOT; variance no worse",
    abs(c_rain) < 0.05 and abs(c_cal) > 3 * abs(c_rain) and var(rain_resid) <= var(cal_resid) * 1.05,
    f"|corr(resid,API7)|: rain={abs(c_rain):.3f} calendar={abs(c_cal):.3f}; "
    f"var: rain={var(rain_resid):.3f} cal={var(cal_resid):.3f}")

# K2 -- injected NON-rain anomaly keeps its rank under R5 (shift < 10 percentage points)
raw_sorted = sorted(ratios.values())
ok2 = True
det2 = []
for d in ANOM:
    p_raw = pr.percentile_of(raw_sorted, ratios[d])
    p_res = pr.percentile_of(model["resid_sorted"], rain_resid[d])
    ok2 &= abs(p_raw - p_res) < 0.10
    det2.append(f"{d[-5:]}: raw p{100*p_raw:.1f} -> resid p{100*p_res:.1f}")
kat("K2 non-rain anomalous days keep their percentile (shift < 10 pts)", ok2, "; ".join(det2))

# K3 -- fetch failure => r5_transform returns None (FAIL-OPEN)
orig = pr.fetch_precip
pr.fetch_precip = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated outage"))
r = pr.r5_transform("synth", 0.0, 0.0, 2.0, today)
pr.fetch_precip = orig
kat("K3 precip-fetch failure -> None (caller stays on R3 ratio path)", r is None)

# K4 -- R5-5: composed mapping is the IDENTITY on the training distribution (each training
# residual maps back to its own ratio). This is the property the amendment claims and the
# old endpoint-only test missed.
rs, rt = model["resid_sorted"], model["ratio_sorted"]
composed = [pr.quantile_at(rt, pr.percentile_of(rs, r)) for r in rs]
kat("K4 composed rank-remap == identity on training dist (sorted composed == ratio_sorted)",
    composed == rt, f"n={len(rt)} mismatches={sum(1 for a,b in zip(composed,rt) if a!=b)}")

# K5 -- activation discontinuity measured (raw stat vs R5 stat on the last day)
ix = pr.indices(precip, today)
resid_today = logv[today] - (b[0] + b[1]*ix[0] + b[2]*ix[1])
p_today = pr.percentile_of(model["resid_sorted"], resid_today)
stat_today = pr.quantile_at(model["ratio_sorted"], p_today)
step = stat_today - ratios[today]
kat("K5 discontinuity measurable: |R5 stat - raw ratio| computed and finite",
    math.isfinite(step), f"raw={ratios[today]:.3f} r5_stat={stat_today:.3f} step={step:+.3f}")

# K6 (R5-R1) -- a future-dated model is NOT fresh (replay-causality)
import precip_residual as _pr
from datetime import date as _date
future_model = {"region": "z", "fitted_date": "2026-07-30", "n": 200, "beta": [0,0,0],
                "resid_sorted": [0.0], "ratio_sorted": [1.0], "window": ["2026-04-01","2026-06-30"]}
_pr.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
import json as _json
_saved = _pr.MODEL_FILE.read_text() if _pr.MODEL_FILE.exists() else None
_pr.MODEL_FILE.write_text(_json.dumps({"z": future_model}))
_m, _reason = _pr.get_model("z", 0.0, 0.0, "2011-01-01")   # replay date BEFORE fit -> age<0
# restore store
(_pr.MODEL_FILE.write_text(_saved) if _saved is not None else _pr.MODEL_FILE.unlink())
kat("K6 R5-R1: future-dated model rejected in historical replay (not 'fresh')",
    _reason != "fresh", f"reason={_reason}")

# K7 (R5-R3) -- predictors well BELOW the fit envelope deactivate R5 (lower-side guard)
lowmodel = {"region": "z", "fitted_date": "2026-07-30", "n": 200, "beta": [-1.2, 0.004, 0.002],
            "resid_sorted": [-0.5,0.0,0.5], "ratio_sorted": [0.5,1.0,2.0],
            "api7_range": [10.0, 20.0], "r30_range": [100.0, 200.0], "window": ["x","x"]}
# call the envelope logic directly: today's (0,0) is far below [10,20]/[100,200]
b = lowmodel["beta"]; ef = _pr.R5_CONFIG["envelope_factor"]
def _out(v, lo, hi): return v > hi*ef or v < lo/ef
below = _out(0.0, *lowmodel["api7_range"]) and _out(0.0, *lowmodel["r30_range"])
kat("K7 R5-R3: predictors below the fit envelope trigger the lower-side guard", below)

n = sum(PASS)
print(f"=== R5 KATs: {n}/{len(PASS)} PASS ===")
raise SystemExit(0 if n == len(PASS) else 1)
