#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R5 CANONICAL-NUMERICS red-KATs (cayley, 2026-08-15) — the acceptance bar for codex's
`r5-canonical-numerics-v1` contract (inbox 185751Z / `54886bd`), sequence step 1
("Cayley cuts and commits the red bar from this contract"). Implementation and the
public regeneration remain behind ASYLUM'S EXPLICIT GATE (step 2); this bar authorizes
neither.

THE PROBLEM (codex runtime HIGH `aa24c5d` + grassmann fingerprint `f0ac43b`): the 18
regenerated R5 model/record bytes depend on the producer's numerical runtime
(Windows py3.11.9 / NumPy 2.4.6 / scipy-openblas DYNAMIC_ARCH on Zen 3). Raw
lstsq/cond/pinv float bytes differ across Python/NumPy/BLAS/CPU; a container pin
cannot fix per-CPU kernel dispatch and cannot run the Windows wheel elsewhere.

THE RULED REPAIR: canonical numerics — an explicit information-losing quotient
(`q11` = 11 significant decimal digits, ROUND_HALF_EVEN, Decimal.from_float) applied at
every persisted/hashed/ranked/compared value, per the contract's §1 field list; digest
domain = SHA-256 over compact sorted-key JSON of the §3 envelope with every float
represented as {"$f11": "<canonical text>"}; gates compare canonical Decimals with the
±1 ulp11 ambiguity refusal (R5_NUMERIC_BOUNDARY_AMBIGUITY -> fail closed to R3).
Empirical calibration frozen by codex: q11 = 18/18 identical across its three verifier
lanes; q12 = the NEGATIVE CONTROL (still runtime-divergent on this corpus); worst
observed raw delta 0.077716 of one q11 quantum (bar threshold 0.1).

SEAMS PINNED BY THIS BAR (naming decisions implementing the contract):
  module `monitoring/src/canonical_numerics.py` exposing:
    POLICY == {"id": "r5-canonical-numerics-v1", "rounding": "ROUND_HALF_EVEN",
               "significant_decimal_digits": 11}
    qsig(x, n=11) -> decimal.Decimal          (§2 recipe, parametric; q11 = qsig(x, 11))
    canonical_text(x, n=11) -> str            (§2 text rules)
    parse_canonical_token(s, n=11) -> Decimal (decimal-preserving; REFUSES non-canonical)
    model_digest(model, n=11) -> str          (§3 envelope; excludes digest fields)
    canonical_record_bytes(record, n=11) -> bytes
    gate_compare(x, threshold) -> "lt"|"ge"|"ambiguous"   (§1 ulp11 rule)
  `precip_residual.fit_region(..., sig_digits=11)` — optional kwarg selecting the
  canonicalization precision (11 = product; 12 = the bar's negative control; 17 = the
  bar's raw-view for the delta measurement; the §1 pipeline applies the SAME precision
  at every quantization point including trim ordering).

REV 2 (codex `dfad58d` C-prime ruling, 2026-08-15): output-only q11 left BLAS-dependent
residual-grid cell selection (two straddles found on 2026-08-13/istanbul: grid[211] across
producer/geomen, grid[226] on py3.14 — three distinct digests for one entry). Repair =
CANONICAL RESIDUAL EVALUATION: project beta + each row operand to q11 FIRST, evaluate
`log_ratio - (b0 + b1*api7 + b2*r30)` exactly in Decimal in that declared operation order,
project ONCE to q11; same evaluator for first-pass trim ordering, the final grid, and
today's transform before bisect. New pinned seams: `canonical_numerics.canonical_residual`
+ `fit_region(..., diagnostics=dict)` out-param (kept_days + first_pass_beta, non-digested).
SCOPE (per ruling): the green claim is the NAMED FOUR-LANE capsule result — codex's three
lanes + the producer lane — not "any host".

MODES (the bar is cross-runtime by construction):
  default            in-lane checks CN-0..CN-5 (red today: seams absent + the landed
                     capsule verifier baseline fails off-producer)
  --emit PATH        additionally write this lane's summary JSON (fingerprint + per-entry
                     q11/q12/q17 model digests + canonical record shas + q17 raw views)
  --compare A B ...  cross-lane assertions over >= 2 emitted summaries:
                     CN-6 q11 digests + record bytes identical 18/18 in every lane;
                     CN-7 q12 negative control DIVERGES somewhere (proves the corpus
                     still probes runtime sensitivity — if q12 matches everywhere the
                     fixture lost its discrimination);
                     CN-8 max |raw q17 delta| / q11 quantum <= 0.1 per field.
GREEN requires: default-mode PASS in EVERY named lane (codex lanes py3.14.0/np2.5.1,
py3.11.9/np2.4.6, py3.11.9/np1.26.4 + grassmann's Windows/Zen3 producer lane) AND one
--compare PASS over all four summaries. FREEZE DISCIPLINE: implementer edits nothing
here; one verify; codex closes per its §Sequence.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = Path(HERE).resolve().parents[1]
CAPSULE = REPO / "monitoring" / "replay_capsules" / "r5_20260812_20260813"

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def ref_qsig(x, n=11):
    """Bar-local §2 reference implementation (independent of the pinned module)."""
    d = Decimal.from_float(float(x))
    if not d.is_finite():
        raise ValueError("non-finite")
    if d == 0:
        return Decimal(0)
    return d.quantize(Decimal(1).scaleb(d.adjusted() - (n - 1)), rounding=ROUND_HALF_EVEN)


def ref_text(x, n=11):
    q = ref_qsig(x, n)
    if q == 0:
        return "0"
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def lane_fingerprint():
    try:
        import numpy
        np_v = numpy.__version__
    except Exception:
        np_v = "absent"
    return {"python": sys.version.split()[0], "numpy": np_v,
            "platform": platform.platform(), "machine": platform.machine()}


def manifest_entries():
    m = json.loads((CAPSULE / "manifest.json").read_text())
    return m["entries"]


def rebuild_lane_summary(PR, CN):
    """Per capsule entry: q11/q12/q17 model digests + q11 canonical record sha + q17
    raw views of the delta-measured fields."""
    out = {"fingerprint": lane_fingerprint(), "q11_model": {}, "q11_record": {},
           "q12_model": {}, "q17_raw": {}}
    for key in sorted(manifest_entries().keys()):
        day, region = key.split("/", 1)
        d = CAPSULE / day / region
        ratios = json.loads((d / "ratios.json").read_text())
        precip = json.loads((d / "precip.json").read_text())
        query = json.loads((d / "query.json").read_text())
        diag = {}
        m11 = PR.fit_region(region, ratios, precip, day, sig_digits=11, diagnostics=diag)
        m12 = PR.fit_region(region, ratios, precip, day, sig_digits=12)
        m17 = PR.fit_region(region, ratios, precip, day, sig_digits=17)
        if m11 is None or m12 is None or m17 is None:
            out["q11_model"][key] = "FIT_NONE"
            continue
        out["q11_model"][key] = CN.model_digest(m11, n=11)
        out["q12_model"][key] = CN.model_digest(m12, n=12)
        # REV 2: explicit cross-lane surfaces per the C-prime ruling
        out.setdefault("trim_days", {})[key] = hashlib.sha256(json.dumps(
            sorted(diag.get("kept_days", [])), separators=(",", ":")).encode()).hexdigest()
        out.setdefault("resid_grid", {})[key] = hashlib.sha256(json.dumps(
            [CN.canonical_text(v, 11) for v in m11["resid_sorted"]],
            separators=(",", ":")).encode()).hexdigest()
        out["q17_raw"][key] = {"beta": [ref_text(b, 17) for b in m17["beta"]],
                               "cond": ref_text(m17["cond"], 17),
                               "max_leverage": ref_text(m17["max_leverage"], 17),
                               "resid_sorted": [ref_text(v, 17) for v in m17["resid_sorted"]]}
        old_fetch, old_hist, old_store = PR.fetch_precip, PR.load_ratio_history, PR.MODEL_FILE
        try:
            PR.fetch_precip = lambda *a, _p=precip, **k: dict(_p)
            PR.load_ratio_history = lambda *a, _r=ratios, **k: dict(_r)
            PR.MODEL_FILE = Path(os.devnull)
            rec = PR.r5_transform(region, query["lat"], query["lon"],
                                  query["raw_ratio"], day, historical=True)
        finally:
            PR.fetch_precip, PR.load_ratio_history, PR.MODEL_FILE = old_fetch, old_hist, old_store
        out["q11_record"][key] = (hashlib.sha256(CN.canonical_record_bytes(rec)).hexdigest()
                                  if rec is not None else "RECORD_NONE")
        out.setdefault("rank", {})[key] = (rec.get("residual_rank_index")
                                           if rec is not None else None)
    return out


def in_lane_checks():
    # CN-0 — pinned seams present
    try:
        import canonical_numerics as CN
    except ImportError as e:
        check("CN-0 pinned module canonical_numerics with POLICY/qsig/canonical_text/"
              "parse_canonical_token/model_digest/canonical_record_bytes/gate_compare",
              False, f"import failed: {e}")
        return None
    need = all(hasattr(CN, a) for a in
               ("POLICY", "qsig", "canonical_text", "parse_canonical_token",
                "model_digest", "canonical_record_bytes", "gate_compare",
                "canonical_residual"))
    policy_ok = getattr(CN, "POLICY", None) == {
        "id": "r5-canonical-numerics-v1", "rounding": "ROUND_HALF_EVEN",
        "significant_decimal_digits": 11}
    check("CN-0 pinned module canonical_numerics with POLICY/qsig/canonical_text/"
          "parse_canonical_token/model_digest/canonical_record_bytes/gate_compare",
          need and policy_ok, f"attrs={need} policy={policy_ok}")
    if not (need and policy_ok):
        return None

    import precip_residual as PR
    import inspect
    params = inspect.signature(PR.fit_region).parameters
    sig_ok = "sig_digits" in params and "diagnostics" in params
    check("CN-0b fit_region exposes sig_digits (11 product / 12 negative-control / "
          "17 raw-view) AND the REV-2 diagnostics out-param (kept_days + first_pass_beta, "
          "non-digested)", sig_ok)
    if not sig_ok:
        return None

    # CN-1 — §2 q11 semantics against the bar-local reference
    vectors = [0.1, -0.0, 0.0, 2.5e-11, 123456789012.345, 1.0000000000049,
               0.07771600000123, -3.3306690738754696e-16, 9.99999999995e-1]
    ok1 = True
    detail = ""
    for v in vectors:
        if CN.qsig(v, 11) != ref_qsig(v, 11) or CN.canonical_text(v, 11) != ref_text(v, 11):
            ok1, detail = False, f"mismatch at {v!r}: {CN.canonical_text(v, 11)} vs {ref_text(v, 11)}"
            break
    for bad in (float("nan"), float("inf")):
        try:
            CN.qsig(bad, 11)
            ok1, detail = False, f"accepted {bad}"
        except Exception:
            pass
    ok1 = ok1 and CN.canonical_text(-0.0, 11) == "0" and CN.canonical_text(0.0, 11) == "0"
    check("CN-1 q11 semantics == §2 (from_float, 11 sig digits ROUND_HALF_EVEN, signed "
          "zero -> 0, fixed-point text, NaN/inf refusal)", ok1, detail)

    # CN-1b -- REV 2: canonical residual evaluator (C-prime). Operands projected FIRST,
    # exact Decimal evaluation in the declared order, ONE final projection.
    ok1b = True
    d1b = ""
    try:
        if CN.canonical_residual(0.12345678901234, [0.0, 0.0, 0.0], 0.0, 0.0) != \
                ref_qsig(0.12345678901234, 11):
            ok1b, d1b = False, "identity vector"
        if ok1b and CN.canonical_residual(0.25, [0.1, 0.001, 0.0002], 10.0, 100.0) != \
                Decimal("0.12"):
            ok1b, d1b = False, "exact-arithmetic vector"
        if ok1b and CN.canonical_residual(0.12345678901, [5e-12, 0.0, 0.0], 0.0, 0.0) != \
                Decimal("0.123456789"):
            ok1b, d1b = False, "HALF_EVEN tie vector"
    except Exception as e:
        ok1b, d1b = False, f"raised {e}"
    check("CN-1b canonical_residual == C-prime (operands q11-projected first, exact "
          "Decimal in declared order b1*api7 + b2*r30 + b0 then subtract, ONE final "
          "projection; HALF_EVEN tie vector exact)", ok1b, d1b)

    # CN-2 — §3 digest domain on a fixture
    fixture = {"region": "x", "n": 42, "beta": [0.1, -0.0, 2.5e-11],
               "note": "str stays", "flag": True, "none": None}
    env = {"model": {"beta": [{"$f11": "0.1"}, {"$f11": "0"}, {"$f11": "0.000000000025"}],
                     "flag": True, "n": 42, "none": None, "note": "str stays",
                     "region": "x"},
           "numeric_policy": {"id": "r5-canonical-numerics-v1",
                              "rounding": "ROUND_HALF_EVEN",
                              "significant_decimal_digits": 11},
           "schema": "r5-model-canonical-v1"}
    expect = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    ok2 = CN.model_digest(fixture, n=11) == expect
    try:
        CN.model_digest({"bad": float("inf")}, n=11)
        ok2 = False
    except Exception:
        pass
    try:
        CN.parse_canonical_token("0.10000000000000001", n=11)   # not q11-canonical
        ok2 = False
    except Exception:
        pass
    ok2 = ok2 and CN.parse_canonical_token("0.1", n=11) == Decimal("0.1")
    check("CN-2 digest domain == §3 (tagged $f11 view, sorted keys, compact separators, "
          "ints/strings/bools/null preserved, non-finite refusal, decimal-preserving "
          "parser refuses non-canonical tokens)", ok2)

    # CN-3 — §1 gate ambiguity rule
    T = 0.2069784073219127
    ulp = Decimal(1).scaleb(Decimal.from_float(T).adjusted() - 10)
    ok3 = (CN.gate_compare(T - 1e-3, T) == "lt" and CN.gate_compare(T + 1e-3, T) == "ge"
           and CN.gate_compare(float(Decimal.from_float(T) + ulp / 2), T) == "ambiguous")
    check("CN-3 gate_compare implements the ±ulp11 ambiguity band "
          "(R5_NUMERIC_BOUNDARY_AMBIGUITY -> fail-closed-to-R3)", ok3)

    # CN-4 — scientific invariance vs the published 18 records at q11
    summary = rebuild_lane_summary(PR, CN)
    ok4 = True
    d4 = ""
    for key in sorted(manifest_entries().keys()):
        day, region = key.split("/", 1)
        rec_pub = json.loads((CAPSULE / day / region / "record.json").read_text())
        m11 = summary["q11_model"].get(key)
        if m11 in ("FIT_NONE", None):
            ok4, d4 = False, f"{key}: fit returned None"
            break
        # statistic + rank provenance + n_fit invariant at q11 precision
        d = CAPSULE / day / region
        ratios = json.loads((d / "ratios.json").read_text())
        precip = json.loads((d / "precip.json").read_text())
        query = json.loads((d / "query.json").read_text())
        old = (PR.fetch_precip, PR.load_ratio_history, PR.MODEL_FILE)
        try:
            PR.fetch_precip = lambda *a, _p=precip, **k: dict(_p)
            PR.load_ratio_history = lambda *a, _r=ratios, **k: dict(_r)
            PR.MODEL_FILE = Path(os.devnull)
            rec_new = PR.r5_transform(region, query["lat"], query["lon"],
                                      query["raw_ratio"], day, historical=True)
        finally:
            PR.fetch_precip, PR.load_ratio_history, PR.MODEL_FILE = old
        if rec_new is None:
            ok4, d4 = False, f"{key}: transform None"
            break
        if ref_qsig(rec_new["stat"], 11) != ref_qsig(rec_pub["stat"], 11):
            ok4, d4 = False, f"{key}: stat moved beyond q11"
            break
        if rec_new.get("n_fit") != rec_pub.get("n_fit"):
            ok4, d4 = False, f"{key}: n_fit {rec_pub.get('n_fit')} -> {rec_new.get('n_fit')}"
            break
        if "residual_rank_index" not in rec_new:
            ok4, d4 = False, f"{key}: residual_rank_index not persisted (§1.6)"
            break
        if rec_new.get("r5_active") is not False:
            ok4, d4 = False, f"{key}: shadow flag changed"
            break
        if rec_new.get("gate_state") == "ambiguous" or rec_new.get("fallback_reason") == \
                "R5_NUMERIC_BOUNDARY_AMBIGUITY":
            ok4, d4 = False, f"{key}: boundary ambiguity in historical set"
            break
    check("CN-4 scientific invariance on all 18: statistic equal at q11, n_fit equal, "
          "residual_rank_index persisted (§1.6), shadow flags unchanged, zero "
          "R5_NUMERIC_BOUNDARY_AMBIGUITY rows", ok4, d4)

    # CN-5 — the landed capsule verifier passes IN THIS LANE (cross-host closure)
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / "tools" / "r5_replay_capsule.py"),
                        "verify", "--mutation-selftest"],
                       capture_output=True, text=True, cwd=str(REPO))
    check("CN-5 tools/r5_replay_capsule.py verify --mutation-selftest exits 0 IN THIS "
          "LANE (capsule integrity + purity + mutation refusal + recompute, now "
          "host-independent)", r.returncode == 0, (r.stdout + r.stderr)[-300:])
    return summary


def compare(paths):
    lanes = [json.loads(Path(p).read_text()) for p in paths]
    names = [f"{ln['fingerprint']['python']}/np{ln['fingerprint']['numpy']}" for ln in lanes]
    keys = sorted(lanes[0]["q11_model"].keys())
    ok6 = len(lanes) >= 2 and all(
        ln["q11_model"] == lanes[0]["q11_model"] and ln["q11_record"] == lanes[0]["q11_record"]
        and ln.get("trim_days") == lanes[0].get("trim_days")
        and ln.get("resid_grid") == lanes[0].get("resid_grid")
        and ln.get("rank") == lanes[0].get("rank")
        for ln in lanes[1:]) and "FIT_NONE" not in lanes[0]["q11_model"].values()
    check(f"CN-6 q11 model digests + record shas + REV-2 trim membership + resid-grid "
          f"shas + today's rank identical across ALL lanes ({', '.join(names)}) for all "
          f"{len(keys)} entries", ok6)
    diverges = any(ln["q12_model"].get(k) != lanes[0]["q12_model"].get(k)
                   for ln in lanes[1:] for k in keys)
    check("CN-7 q12 NEGATIVE CONTROL diverges somewhere across lanes (the corpus still "
          "probes runtime sensitivity; all-match would mean the fixture lost its "
          "discrimination)", diverges)
    worst = 0.0
    for k in keys:
        for ln in lanes[1:]:
            a, b = lanes[0]["q17_raw"].get(k), ln["q17_raw"].get(k)
            if not a or not b:
                continue
            for fa, fb in [(x, y) for x, y in
                           zip(a["beta"] + [a["cond"], a["max_leverage"]],
                               b["beta"] + [b["cond"], b["max_leverage"]])]:
                da, db = Decimal(fa), Decimal(fb)
                ref = da if da != 0 else db
                if ref == 0:
                    continue
                quantum = Decimal(1).scaleb(ref.adjusted() - 10)
                worst = max(worst, float(abs(da - db) / quantum))
    check(f"CN-8 max |raw q17 delta| / q11 quantum <= 0.1 across lanes "
          f"(codex-measured corpus max 0.077716)", worst <= 0.1, f"worst={worst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", metavar="PATH", default=None)
    ap.add_argument("--compare", nargs="+", metavar="SUMMARY", default=None)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        summary = in_lane_checks()
        if args.emit and summary is not None:
            Path(args.emit).write_text(json.dumps(summary, indent=1, sort_keys=True))
            print(f"    lane summary -> {args.emit}")


main()
print()
if FAILS:
    print(f"R5 CANONICAL-NUMERICS RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("R5 CANONICAL-NUMERICS RED-KATs PASS (this mode; GREEN overall requires default-"
      "mode PASS in every named lane + one --compare PASS over all lane summaries)")
