#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R5 HISTORICAL-REPLAY PURITY red-KATs (cayley, 2026-08-15) — the acceptance bar for
codex `1246`'s R5 replay-order HIGH (WORKS-WITH-FIX) on the re-score package, grassmann
`1654` root-cause confirmation (live `precip_regression.json` rewritten mid-replay,
pre-replay bytes unrecoverable — disclosed shadow-layer record loss).

THE DEFECT (third instance of the dormant-seam class): `get_model` serves/refits from the
SHARED LIVE model store unconditionally. An owner-authorized historical `--date` replay
therefore (a) REWRITES the live store (R5-R1 rejects the future-dated model, refits as-of
the replay date, writes), and (b) makes each date's R5 shadow record depend on WHICH
replay ran first — codex's finding: 55 semantic changes under days.2026-08-13 driven by
replay order, `fitted_date 2026-08-13 -> 2026-08-12`, all betas moved.

THE RULED REPAIR (codex 1246 contract, accepted verbatim by grassmann 1654): an explicit
historical-replay model policy in `precip_residual.get_model` / the `--date` caller —
fit deterministically AS-OF the target date from the target-date window, do NOT read or
write the persistent live model store, and PUBLISH the resulting fitted_date, window,
coefficients, and model digest. Then regenerate only the affected R5 entries with dated
correction provenance; the FC artifacts that passed are not rerun or altered.

SEAM PINNED BY THIS BAR (the bar's naming decision implementing the contract): the
`--date` caller invokes `r5_transform(region, lat, lon, ratio, today, historical=True)`;
`historical=True` selects the replay policy. Provenance keys pinned in the historical
output: `fitted_date` (== the target date, per fit_region's as-of semantics), `window`
(the [start, end] fit window as-of the target date), `beta` (3 coefficients), and
`model_sha256` (64-hex digest of the as-of model; the bar asserts presence, shape, and
cross-run equality — not a pinned value). Shadow discipline unchanged: `r5_active` False.

CODEX 1246 RED-KAT (verbatim): "execute [08-12, 08-13] and [08-13, 08-12] from different
model-store seeds; each date's R5 JSON must be byte-identical across orders and the live
model-store SHA must remain unchanged."

KAT MAP (red on current tree, green after the repair):
  RP-1  order/seed invariance through the pinned historical seam: run A = seed S1
        (future-dated store with poisoned beta=[9,9,9]) order [08-12, 08-13]; run B =
        seed S2 (absent store) order [08-13, 08-12]; each date's R5 output byte-identical
        across runs + repeat-call deterministic. Poisoned-beta seed proves NO store READ.
        [RED now: the `historical` seam does not exist]
  RP-2  store purity: after all historical calls, S1's store bytes are UNCHANGED and S2's
        store file still does not exist (no write, no create).      [RED now]
  RP-3  provenance publication: each historical output carries fitted_date == target
        date, window == [D-395d, D-30d], beta (3 floats), model_sha256 (64-hex), equal
        across runs for the same date; r5_computed True, r5_active False. [RED now]
  RP-4  normal-cadence scope lock (green BOTH sides): the plain (non-historical) path
        keeps the signed R5-2/R5-R1 policy — a fresh in-age store is served with NO
        store write; a future-dated store is NOT served (age<0 -> refit path).

HERMETIC: MODEL_FILE redirected to per-run tempdirs; fetch_precip/load_ratio_history
stubbed with deterministic seeded fixtures (pure functions of (region, date range)) that
satisfy the fit gates (>= 90 eligible window days, 30-day gap-free R30 windows). No
network, no repo state touched. FREEZE DISCIPLINE: grassmann implements the codex-1246
contract WITHOUT editing this file; one verify; close. Nothing here authorizes the R5
entry regeneration timing, any FC rerun, or renewal-scope items.
"""
import json
import os
import random
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


REGION = "istanbul_marmara"
LAT, LON = 40.9, 28.9
RATIO = 1.37                      # today's raw R3 ratio fed to the transform
D1, D2 = "2026-08-12", "2026-08-13"

FIX_START = date(2025, 5, 1)      # fixture range comfortably covers window+warmup
FIX_END = date(2026, 8, 20)


def _mk_precip(region):
    """Deterministic daily precip mm over the fixture range (pure in region)."""
    rng = random.Random(f"precip:{region}")
    out = {}
    d = FIX_START
    while d <= FIX_END:
        out[d.isoformat()] = round(abs(rng.gauss(3.0, 4.0)), 2)
        d += timedelta(days=1)
    return out


def _mk_ratios(region):
    """Deterministic daily ratio history (pure in region)."""
    rng = random.Random(f"ratio:{region}")
    out = {}
    d = FIX_START
    while d <= FIX_END:
        out[d.isoformat()] = round(1.0 + abs(rng.gauss(0.3, 0.25)), 4)
        d += timedelta(days=1)
    return out


def _poisoned_store_body():
    """A syntactically valid, future-dated model with obviously wrong coefficients.
    If any historical output ever reflects beta=[9,9,9] or fitted_date 2026-08-14,
    the store was READ — a purity violation the seeds are designed to expose."""
    m = {"region": REGION, "fitted_date": "2026-08-14", "n": 99,
         "beta": [9.0, 9.0, 9.0],
         "resid_sorted": [-0.2, -0.1, 0.0, 0.1, 0.2],
         "ratio_sorted": [1.0, 1.1, 1.2, 1.3, 1.4],
         "api7_range": [0.0, 100.0], "r30_range": [0.0, 500.0],
         "cond": 10.0, "max_leverage": 0.1,
         "window": ["2025-07-15", "2026-07-15"]}
    return json.dumps({REGION: m}, indent=1)


def _install_stubs(PR, model_path):
    PR.MODEL_FILE = Path(model_path)
    precip = _mk_precip(REGION)
    ratios = _mk_ratios(REGION)
    PR.fetch_precip = lambda region, lat, lon, end: {
        k: v for k, v in precip.items() if k <= end}
    PR.load_ratio_history = lambda region: dict(ratios)


def _hist_call(PR, today):
    """The pinned historical seam. Raises on a pre-repair tree (no such kwarg)."""
    return PR.r5_transform(REGION, LAT, LON, RATIO, today, historical=True)


def _canon(d):
    return json.dumps(d, sort_keys=True) if d is not None else "None"


def main():
    import precip_residual as PR

    need = (hasattr(PR, "r5_transform") and hasattr(PR, "get_model")
            and hasattr(PR, "fit_region") and hasattr(PR, "MODEL_FILE"))
    check("RP-0 R5 seams present (r5_transform/get_model/fit_region/MODEL_FILE)", need)
    if not need:
        return

    real = (PR.MODEL_FILE, PR.fetch_precip, PR.load_ratio_history)
    try:
        # ---------------- run A: seed S1 (poisoned, future-dated), order [D1, D2]
        dirA = tempfile.mkdtemp()
        storeA = os.path.join(dirA, "precip_regression.json")
        open(storeA, "w").write(_poisoned_store_body())
        bytesA0 = open(storeA, "rb").read()
        _install_stubs(PR, storeA)
        outA, errA = {}, None
        try:
            outA[D1] = _hist_call(PR, D1)
        except TypeError as e:
            errA = f"seam absent: {e}"
        if errA is None:
            outA[D1 + "#2"] = _hist_call(PR, D1)          # repeat-call determinism
            outA[D2] = _hist_call(PR, D2)
        bytesA1 = open(storeA, "rb").read()

        # ---------------- run B: seed S2 (absent store), order [D2, D1]
        dirB = tempfile.mkdtemp()
        storeB = os.path.join(dirB, "precip_regression.json")   # never created
        _install_stubs(PR, storeB)
        outB, errB = {}, None
        try:
            outB[D2] = _hist_call(PR, D2)
            outB[D1] = _hist_call(PR, D1)
        except TypeError as e:
            errB = f"seam absent: {e}"
        storeB_exists = os.path.exists(storeB)

        # RP-1 — order/seed invariance + determinism through the pinned seam
        ok1 = (errA is None and errB is None
               and outA.get(D1) is not None and outA.get(D2) is not None
               and _canon(outA.get(D1)) == _canon(outB.get(D1))
               and _canon(outA.get(D2)) == _canon(outB.get(D2))
               and _canon(outA.get(D1)) == _canon(outA.get(D1 + "#2")))
        check("RP-1 historical seam (`historical=True`) exists and each date's R5 output "
              "is BYTE-IDENTICAL across replay orders [D1,D2]/[D2,D1] AND across "
              "different store seeds (poisoned vs absent), repeat-call deterministic",
              ok1, f"errA={errA} errB={errB} "
                   f"d1_eq={_canon(outA.get(D1)) == _canon(outB.get(D1)) if not (errA or errB) else 'n/a'}")

        # RP-2 — store purity: no write, no create, no read-influence
        beta_leak = False
        for o in (outA.get(D1), outA.get(D2), outB.get(D1), outB.get(D2)):
            if o and (o.get("beta") == [9.0, 9.0, 9.0] or o.get("fitted_date") == "2026-08-14"):
                beta_leak = True
        ok2 = (errA is None and errB is None
               and bytesA1 == bytesA0 and not storeB_exists and not beta_leak)
        check("RP-2 live model store is PURE under historical replays: seed-S1 bytes "
              "unchanged, seed-S2 file never created, and no poisoned-store content "
              "leaks into any output (no read)",
              ok2, f"errA={errA} store_changed={bytesA1 != bytesA0} "
                   f"storeB_created={storeB_exists} poisoned_leak={beta_leak}")

        # RP-3 — provenance publication in the historical output
        def prov_ok(o, D):
            if not o:
                return False
            w_start = (date.fromisoformat(D) - timedelta(days=395)).isoformat()
            w_end = (date.fromisoformat(D) - timedelta(days=30)).isoformat()
            sha = o.get("model_sha256")
            return (o.get("fitted_date") == D
                    and list(o.get("window", [])) == [w_start, w_end]
                    and isinstance(o.get("beta"), list) and len(o["beta"]) == 3
                    and isinstance(sha, str) and len(sha) == 64
                    and all(c in "0123456789abcdef" for c in sha)
                    and o.get("r5_computed") is True and o.get("r5_active") is False)
        sha_eq = (errA is None and errB is None and outA.get(D1) and outB.get(D1)
                  and outA[D1].get("model_sha256") == outB[D1].get("model_sha256")
                  and outA.get(D2) and outB.get(D2)
                  and outA[D2].get("model_sha256") == outB[D2].get("model_sha256"))
        ok3 = (errA is None and errB is None
               and prov_ok(outA.get(D1), D1) and prov_ok(outA.get(D2), D2)
               and bool(sha_eq))
        check("RP-3 historical outputs PUBLISH the as-of provenance: fitted_date == "
              "target date, window == [D-395d, D-30d], beta (3 coeffs), model_sha256 "
              "(64-hex, equal across runs per date); shadow flags unchanged "
              "(r5_computed True, r5_active False)",
              ok3, f"errA={errA} d1_prov={prov_ok(outA.get(D1), D1) if errA is None else 'n/a'} "
                   f"sha_eq={sha_eq}")

    finally:
        PR.MODEL_FILE, PR.fetch_precip, PR.load_ratio_history = real

    # RP-4 — normal-cadence scope lock (plain path, green both sides)
    try:
        dirC = tempfile.mkdtemp()
        storeC = os.path.join(dirC, "precip_regression.json")
        _install_stubs(PR, storeC)
        # (a) fresh in-age store is SERVED with no write
        T = "2026-08-15"
        fresh_model = {"region": REGION, "fitted_date": "2026-08-13", "n": 120,
                       "beta": [0.1, 0.001, 0.0002],
                       "resid_sorted": [-0.1, 0.0, 0.1], "ratio_sorted": [1.1, 1.2, 1.3],
                       "api7_range": [0.0, 100.0], "r30_range": [0.0, 500.0],
                       "window": ["2025-07-14", "2026-07-14"]}
        open(storeC, "w").write(json.dumps({REGION: fresh_model}, indent=1))
        c0 = open(storeC, "rb").read()
        m, reason = PR.get_model(REGION, LAT, LON, T)
        c1 = open(storeC, "rb").read()
        ok4a = (reason == "fresh" and m is not None
                and m.get("fitted_date") == "2026-08-13" and c0 == c1)
        # (b) R5-R1: a future-dated store is NOT served on the plain path
        open(storeC, "w").write(_poisoned_store_body())     # fitted 2026-08-14
        m2, reason2 = PR.get_model(REGION, LAT, LON, D1)    # today=08-12 -> age<0
        ok4b = reason2 != "fresh" and (m2 is None or m2.get("fitted_date") != "2026-08-14")
        check("RP-4 normal-cadence policy UNCHANGED (signed R5-2/R5-R1): a fresh in-age "
              "store is served with no store write; a future-dated store is never served "
              "(age<0 -> refit path)",
              ok4a and ok4b, f"fresh={reason}/{ok4a} future={reason2}/{ok4b}")
    finally:
        PR.MODEL_FILE, PR.fetch_precip, PR.load_ratio_history = real


main()
print()
if FAILS:
    print(f"R5 REPLAY-PURITY RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R5 REPLAY-PURITY RED-KATs PASS (historical replays deterministic as-of the "
      "target date, store-pure, provenance-published; normal-cadence policy intact)")
