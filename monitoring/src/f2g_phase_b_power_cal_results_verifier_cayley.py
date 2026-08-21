#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CALENDAR-LANE power-results verifier v4 (cayley) -- FIXTURE-ONLY.

Codex 0307Z consolidated architecture: the verifier RECONSTRUCTS every
claim-bearing surface from the COMMITTED evidence capsule and compares --
result echoes are never trusted.

verify(res, repo, expected_family, check_files=True) -> (ok, reasons):
  1. schema must be exactly f2g-phase-b-power-cal-results-v1; family must be
     a registered family AND equal expected_family (CLI derives it from the
     filename).
  2. geometry lock (typed POWER_GEOMETRY_UNBOUND) always runs.
  3. digests must echo every pinned authority (recomputed from the repo's
     git blobs: frozen Amendment 2, cal engine, admitted bar, cal driver,
     rev-1.6 annex set, PINNED ASSEMBLER).
  4. transitive source attestation before any regeneration/import: cal
     driver, engine, base driver, calendar authority, and assembler checkout
     bytes (LF) must equal their pinned blobs; the module cache is PURGED
     first so a preloaded fake can never satisfy attestation; Clopper-
     Pearson bounds use a VERIFIER-LOCAL implementation.
  5. evidence = the bound (geospec_commit, canonical path, blob sha) triple;
     rows schema-validated with globally unique (stage, family, point, rep)
     keys; then S1/S2 tables, Tier-C outcomes, verdict/stopping under the
     registered R=20/40 semantics, certified set, terminal, Pareto frontier
     + lex representative, and the gate receipt are RECONSTRUCTED from those
     rows and compared field-by-field to the artifact.
  6. panel_reopen_sample must equal the derived expectation exactly (rep 0
     of every registered grid point + every Stage-C row) and every sampled
     panel must regenerate to its recorded digest.
self_test(repo, fixture_dir) exercises the permanent negatives; each must
refuse (matrix in the close routing).
"""
import hashlib
import importlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict

SCHEMA = "f2g-phase-b-power-cal-results-v1"
FAMILIES = ("B1A", "B2A", "B3A")
CANONICAL_EVIDENCE_PATH = "docs/f2g_phase_b_power_cal_evidence.jsonl"
CAL_AUTH_REF = ("8111805", "docs/f2g_phase_b_shared_calendar_v1.json")
PINS = {
    "frozen_amendment2_sha256": ("337571c",
                                 "docs/f2g_phase_b_prereg_amendment2_DRAFT.md"),
    "engine_lf_sha256": ("24b0d8f", "monitoring/src/d2_f2g_phase_b_stats.py"),
    "admitted_bar_lf_sha256": (
        "89673dc", "monitoring/src/test_f2g_phase_b_stats_redkats_grassmann.py"),
    "driver_lf_sha256": (
        "975638d", "monitoring/src/f2g_phase_b_power_estimation_cal_cayley.py"),
    "annex_common_rev16_sha256": ("feb20bb",
                                  "docs/f2g_phase_b_power_annex_common.md"),
    "annex_b1a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b1a.md"),
    "annex_b2a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b2a.md"),
    "annex_b3a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b3a.md"),
    "assembler_lf_sha256": (
        "5377138",
        "monitoring/src/f2g_phase_b_power_cal_results_assembly_cayley.py"),
}
ENGINE_COMMIT = "24b0d8f"
# transitive attestation set: (module_name or None, pinned key)
ATTEST = [
    ("f2g_phase_b_power_estimation_cal_cayley", "driver_lf_sha256"),
    ("d2_f2g_phase_b_stats", "engine_lf_sha256"),
    (None, "assembler_lf_sha256"),
]
REQUIRED_KEYS = ("schema", "family", "calendar_authority_mode",
                 "calendar_authority_sha256", "digests",
                 "equivalence_receipt", "evidence_capsule",
                 "panel_reopen_sample", "tier_s1", "tier_s2", "tier_c",
                 "certified_points", "pareto_minimal_certified",
                 "pareto_lex_representative", "terminal_type",
                 "induced_effect_report", "env")


def _blob(repo, commit, path):
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _binom_sf_geq(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def cp_lower(k, n, conf=0.95):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_sf_geq(k, n, mid) > 1 - conf:
            hi = mid
        else:
            lo = mid
    return lo


def cp_upper(k, n, conf=0.95):
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_sf_geq(k + 1, n, mid) < conf:
            lo = mid
        else:
            hi = mid
    return hi


def pk(point):
    return json.dumps(point, sort_keys=True, separators=(",", ":"))


def _attest_sources(repo, pins=None):
    """Checkout LF bytes of every transitive source must equal the pinned
    blob; the module cache is purged so preloads cannot satisfy this."""
    import os
    pins = pins or PINS
    failures = []
    for mod, key in ATTEST:
        commit, path = pins[key]
        try:
            pinned = _blob(repo, commit, path)
        except Exception as exc:
            failures.append(f"pin {key} unreadable: {exc}")
            continue
        disk = open(os.path.join(repo, path), "rb").read().replace(
            b"\r\n", b"\n")
        if _sha(disk) != _sha(pinned):
            failures.append(f"DEPENDENCY_UNATTESTED: {path} checkout bytes "
                            "differ from the pinned blob")
        if mod and mod in sys.modules:
            del sys.modules[mod]
    # calendar authority checkout (used indirectly by the driver)
    try:
        pinned = _blob(repo, *CAL_AUTH_REF)
        disk = open(os.path.join(repo, CAL_AUTH_REF[1]), "rb").read().replace(
            b"\r\n", b"\n")
        if _sha(disk) != _sha(pinned):
            failures.append("DEPENDENCY_UNATTESTED: calendar authority "
                            "checkout differs from the pinned blob")
    except Exception as exc:
        failures.append(f"calendar authority unreadable: {exc}")
    return failures


def _load_driver(repo):
    import os
    src = os.path.join(repo, "monitoring", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        PD = importlib.import_module(
            "f2g_phase_b_power_estimation_cal_cayley")
    finally:
        os.chdir(cwd)
    return PD


def _validate_rows(rows, fam, grid_keys, reasons):
    seen = set()
    ok_rows = []
    for i, r in enumerate(rows):
        st = r.get("stage")
        if st in ("gate",):
            ok_rows.append(r)
            continue
        if st not in ("S1", "S2", "C", "Cverdict"):
            reasons.append(f"evidence row {i}: unknown stage {st!r}")
            continue
        rf = r.get("family")
        if rf not in FAMILIES:
            reasons.append(f"evidence row {i}: unknown family {rf!r}")
            continue
        key = (st, rf, pk(r.get("point")), r.get("rep", 0))
        if key in seen:
            reasons.append(f"duplicate evidence key: {key}")
            continue
        seen.add(key)
        if rf == fam and st in ("S1", "S2", "C") and \
                pk(r.get("point")) not in grid_keys:
            reasons.append(f"evidence row {i}: point outside the "
                           f"registered {fam} grid")
            continue
        ok_rows.append(r)
    return ok_rows


def _reconstruct(fam, rows, PD, reasons):
    grid = PD.grid_of(fam)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("family") == fam and r.get("stage") in ("S1", "S2", "C"):
            by[r["stage"]][pk(r["point"])].append(r)
    s1 = []
    for p in grid:
        v = by["S1"].get(pk(p), [])
        if sorted(x.get("rep") for x in v) != list(range(50)):
            reasons.append(f"evidence S1 {pk(p)}: not exactly reps 0..49")
            return None
        s1.append({"point": p, "pre_loco_recovery":
                   sum(bool(x["pre"]) for x in v) / 50, "n": 50})
    rank1 = sorted(s1, key=lambda x: (-x["pre_loco_recovery"],
                                      pk(x["point"])))
    s2_pts = [x["point"] for x in rank1[:min(8, len(grid))]]
    s2 = []
    for p in s2_pts:
        v = by["S2"].get(pk(p), [])
        if sorted(x.get("rep") for x in v) != list(range(50)):
            reasons.append(f"evidence S2 {pk(p)}: not exactly reps 0..49")
            return None
        s2.append({"point": p,
                   "pre_loco_recovery": sum(bool(x["pre"])
                                            for x in v) / 50,
                   "post_loco_recovery": sum(bool(x["post"])
                                             for x in v) / 50, "n": 50})
    rank2 = sorted(s2, key=lambda x: (-x["post_loco_recovery"],
                                      pk(x["point"])))
    c_pts = [x["point"] for x in rank2[:min(3, len(s2))]]
    cands = []
    for p in c_pts:
        v = sorted(by["C"].get(pk(p), []), key=lambda x: x["rep"])
        if [x["rep"] for x in v] != list(range(len(v))):
            reasons.append(f"evidence C {pk(p)}: reps not sequential")
            return None
        succ = 0
        verdict = None
        n_stop = None
        for i, r in enumerate(v):
            succ += bool(r["post"])
            n = i + 1
            if n == 20:
                if cp_lower(succ, 20) >= 0.80:
                    verdict, n_stop = "CERTIFIED", 20
                    break
                if cp_upper(succ, 20) < 0.80:
                    verdict, n_stop = "FAILED", 20
                    break
            if n == 40:
                verdict = ("CERTIFIED" if cp_lower(succ, 40) >= 0.80 else
                           ("FAILED" if cp_upper(succ, 40) < 0.80 else
                            "CANNOT_DETERMINE_POWER_ESTIMATE"))
                n_stop = 40
        if verdict is None or n_stop != len(v):
            reasons.append(f"evidence C {pk(p)}: rows inconsistent with "
                           "the registered stopping semantics")
            return None
        cands.append({"point": p,
                      "post_loco": {"successes": succ, "replicates": n_stop},
                      "lb95": cp_lower(succ, n_stop),
                      "ub95": cp_upper(succ, n_stop),
                      "verdict": verdict,
                      "stopping": f"{verdict}@R={n_stop}"})
    certified = [c for c in cands if c["verdict"] == "CERTIFIED"]

    def dominates(a, b):
        ka, kb = a["point"], b["point"]
        ks = sorted(ka)
        return all(ka[k] <= kb[k] for k in ks) and \
            any(ka[k] < kb[k] for k in ks)

    pareto = sorted([c for c in certified
                     if not any(dominates(o, c) for o in certified
                                if o is not c)],
                    key=lambda c: pk(c["point"]))
    reopen = [{"family": fam, "point": p, "rep": 0} for p in grid]
    for p in c_pts:
        for r in sorted(by["C"][pk(p)], key=lambda x: x["rep"]):
            reopen.append({"family": fam, "point": p, "rep": r["rep"],
                           "panel_sha256": r.get("panel_sha256")})
    return {"s1": s1, "s2": s2, "cands": cands, "certified": certified,
            "pareto": pareto, "reopen": reopen,
            "terminal": ("CERTIFIED" if certified else
                         "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH")}


def _cmp(name, got, want, reasons):
    if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
        reasons.append(f"RESULT_NOT_DERIVED_FROM_EVIDENCE: {name}")


def verify(res, repo, expected_family=None, check_files=True, pins=None):
    reasons = []
    pins = pins or PINS
    # ---- schema + family identity (always) ----
    if res.get("schema") != SCHEMA:
        reasons.append(f"RESULT_SCHEMA_MISSING: schema != {SCHEMA}")
    fam = res.get("family")
    if fam not in FAMILIES:
        reasons.append(f"unknown family {fam!r}")
    if expected_family is not None and fam != expected_family:
        reasons.append(f"family {fam!r} != expected {expected_family!r}")
    # ---- geometry lock (always) ----
    if res.get("calendar_authority_mode") != "bound":
        reasons.append("POWER_GEOMETRY_UNBOUND: calendar_authority_mode "
                       "must be 'bound'")
    try:
        if res.get("calendar_authority_sha256") != \
                _sha(_blob(repo, *CAL_AUTH_REF)):
            reasons.append("POWER_GEOMETRY_UNBOUND: calendar authority "
                           "sha mismatch")
    except Exception as exc:
        reasons.append(f"POWER_GEOMETRY_UNBOUND: authority unreadable "
                       f"({exc})")
    if not check_files:
        return (not reasons), reasons
    # ---- schema-closed key presence ----
    for key in REQUIRED_KEYS:
        if key not in res or res[key] is None and key != \
                "pareto_lex_representative":
            reasons.append(f"RESULT_SCHEMA_MISSING: {key}")
    d = res.get("digests") or {}
    r = res.get("equivalence_receipt") or {}
    ev = res.get("evidence_capsule") or {}
    # ---- authorities ----
    for key, (commit, path) in pins.items():
        try:
            want = _sha(_blob(repo, commit, path))
        except Exception as exc:
            reasons.append(f"authority {key} unreadable: {exc}")
            continue
        if d.get(key) != want:
            reasons.append(f"digest {key} mismatch/absent")
    if r.get("full_equal") is not True or r.get("fold_equal_all") is not True:
        reasons.append("equivalence booleans not strictly True")
    if r.get("folds_checked") != 35:
        reasons.append("folds_checked != 35")
    if r.get("engine_commit_bound") != ENGINE_COMMIT:
        reasons.append(f"engine_commit_bound != {ENGINE_COMMIT}")
    # ---- transitive attestation, then driver load ----
    att = _attest_sources(repo, pins)
    reasons.extend(att)
    if att:
        return False, reasons
    try:
        PD = _load_driver(repo)
        grid_keys = {pk(p) for p in PD.grid_of(fam)} if fam in FAMILIES \
            else set()
    except Exception as exc:
        reasons.append(f"driver load failed: {exc}")
        return False, reasons
    # ---- evidence triple + rows ----
    if ev.get("path") != CANONICAL_EVIDENCE_PATH:
        reasons.append("evidence path is not the canonical registered path")
    rows = []
    try:
        raw = _blob(repo, ev.get("geospec_commit", "HEAD"), ev.get("path",
                                                                   ""))
        if _sha(raw) != ev.get("git_blob_sha256"):
            reasons.append("evidence blob sha mismatch")
        lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
        if len(lines) != ev.get("rows"):
            reasons.append("evidence row count mismatch")
        rows = _validate_rows([json.loads(l) for l in lines], fam,
                              grid_keys, reasons)
    except Exception as exc:
        reasons.append(f"evidence unreadable: {exc}")
    # ---- gate receipt from evidence ----
    gates = [x for x in rows if x.get("stage") == "gate"]
    if len(gates) != 1:
        reasons.append("evidence must contain exactly one gate row")
    else:
        g = gates[0]
        for k in ("full_equal", "fold_equal_all", "folds_checked",
                  "engine_commit_bound"):
            if r.get(k) != g.get(k):
                reasons.append(f"receipt field {k} not derived from the "
                               "evidence gate row")
    # ---- reconstruction + field comparison ----
    recon = _reconstruct(fam, rows, PD, reasons) if fam in FAMILIES else None
    if recon is not None:
        _cmp("tier_s1.table", (res.get("tier_s1") or {}).get("table"),
             recon["s1"], reasons)
        _cmp("tier_s2.table", (res.get("tier_s2") or {}).get("table"),
             recon["s2"], reasons)
        _cmp("tier_c.candidates",
             (res.get("tier_c") or {}).get("candidates"), recon["cands"],
             reasons)
        _cmp("certified_points", res.get("certified_points"),
             recon["certified"], reasons)
        _cmp("pareto_minimal_certified",
             res.get("pareto_minimal_certified"), recon["pareto"], reasons)
        _cmp("pareto_lex_representative",
             res.get("pareto_lex_representative"),
             recon["pareto"][0]["point"] if recon["pareto"] else None,
             reasons)
        if res.get("terminal_type") != recon["terminal"]:
            reasons.append("RESULT_NOT_DERIVED_FROM_EVIDENCE: terminal_type")
        for tier, lbl, nd in (("tier_s1", "PRELIMINARY_SMOKE", 999),
                              ("tier_s2", "PRELIMINARY_SMOKE", 999),
                              ("tier_c", "CERTIFICATION", 9999)):
            t = res.get(tier) or {}
            if t.get("label") != lbl or t.get("n_draws") != nd:
                reasons.append(f"{tier} label/draws wrong")
        # ---- panel reopen: exact sample + regeneration ----
        sample = res.get("panel_reopen_sample") or []
        want_keys = Counter((x["family"], pk(x["point"]), x["rep"])
                            for x in recon["reopen"])
        got_keys = Counter((x.get("family"), pk(x.get("point")),
                            x.get("rep")) for x in sample)
        if want_keys != got_keys:
            reasons.append("panel_reopen_sample not exactly the derived "
                           "coverage")
        ev_dig = {(x["family"], pk(x["point"]), x["rep"]):
                  x.get("panel_sha256") for x in recon["reopen"]
                  if x.get("panel_sha256")}
        for x in sample:
            key = (x.get("family"), pk(x.get("point")), x.get("rep"))
            want_ev = ev_dig.get(key)
            if want_ev is not None and x.get("panel_sha256") != want_ev:
                reasons.append(f"reopen digest differs from evidence: {key}")
            try:
                pan = PD.make_panel(x["family"], x["point"], x["rep"])
                if PD.panel_digest(pan) != x.get("panel_sha256"):
                    reasons.append("POWER_GEOMETRY_UNBOUND: panel "
                                   f"{key} does not regenerate to its "
                                   "recorded digest")
            except Exception as exc:
                reasons.append(f"panel regeneration failed {key}: {exc}")
    return (not reasons), reasons


def self_test(repo, fixture_docs=None):
    """Permanent negatives; each must refuse. A committed POSITIVE fixture
    (assembled by the pinned assembler) is doctored per class; classes that
    need no fixture run standalone."""
    results = {}
    ok_all = True

    def rec(name, refused):
        results[name] = "REFUSED" if refused else "ACCEPTED -- DEFECT"
        return refused

    cal_sha = _sha(_blob(repo, *CAL_AUTH_REF))
    geo = {"schema": SCHEMA, "family": "B2A",
           "calendar_authority_mode": "bound",
           "calendar_authority_sha256": cal_sha}
    ok, _ = verify(geo, repo, expected_family="B2A", check_files=True)
    ok_all &= rec("geometry-only-schema-shaped", not ok)
    ok, _ = verify(dict(geo, schema="NOT-THE-REGISTERED-SCHEMA"), repo,
                   expected_family="B2A", check_files=False)
    ok_all &= rec("wrong-schema", not ok)
    ok, _ = verify(geo, repo, expected_family="B1A", check_files=False)
    ok_all &= rec("wrong-family-in-file", not ok)
    bad_pins = dict(PINS)
    # pin the driver to its STALE v1 commit (7028324): that blob genuinely
    # differs from the current checkout, so attestation must refuse -- the
    # executable form of a mutated/cached dependency
    bad_pins["driver_lf_sha256"] = (
        "7028324", "monitoring/src/f2g_phase_b_power_estimation_cal_cayley.py")
    att = _attest_sources(repo, bad_pins)
    ok_all &= rec("mutated-dependency", bool(att))
    if fixture_docs:
        base = json.loads(open(
            f"{fixture_docs}/f2g_phase_b_power_annex_b2a_cal_results.json",
            encoding="utf-8").read())
        okp, rp = verify(base, repo, expected_family="B2A",
                         check_files=True)
        results["positive-fixture"] = ("PASS" if okp else
                                       "REFUSED -- DEFECT: " + "; ".join(
                                           map(str, rp[:3])))
        ok_all &= okp
        import copy
        t1 = copy.deepcopy(base)
        t1["evidence_capsule"]["path"] = \
            "docs/f2g_phase_b_power_a_evidence.jsonl"
        ok, _ = verify(t1, repo, expected_family="B2A", check_files=True)
        ok_all &= rec("unrelated-committed-evidence", not ok)
        t2 = copy.deepcopy(base)
        if t2["tier_c"]["candidates"]:
            t2["tier_c"]["candidates"][0]["verdict"] = "CERTIFIED"
            t2["tier_c"]["candidates"][0]["post_loco"]["successes"] = 0
            t2["certified_points"] = [t2["tier_c"]["candidates"][0]]
            t2["terminal_type"] = "CERTIFIED"
        ok, _ = verify(t2, repo, expected_family="B2A", check_files=True)
        ok_all &= rec("zero-labeled-certified", not ok)
        t3 = copy.deepcopy(base)
        if t3["tier_s1"]["table"]:
            t3["tier_s1"]["table"][0]["pre_loco_recovery"] = 0.98
        ok, _ = verify(t3, repo, expected_family="B2A", check_files=True)
        ok_all &= rec("aggregate-mismatch", not ok)
        t4 = copy.deepcopy(base)
        if len(t4["tier_s1"]["table"]) >= 2:
            t4["tier_s1"]["table"][1] = dict(
                t4["tier_s1"]["table"][0])
        ok, _ = verify(t4, repo, expected_family="B2A", check_files=True)
        ok_all &= rec("duplicate-registered-point", not ok)
        t5 = copy.deepcopy(base)
        if t5["panel_reopen_sample"]:
            t5["panel_reopen_sample"] = t5["panel_reopen_sample"][:-1]
        ok, _ = verify(t5, repo, expected_family="B2A", check_files=True)
        ok_all &= rec("missing-reopen-rep", not ok)
    return ok_all, results


if __name__ == "__main__":
    repo = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--self-test":
        fdocs = sys.argv[3] if len(sys.argv) > 3 else None
        ok, results = self_test(repo, fdocs)
        print(json.dumps(results, indent=1))
        sys.exit(0 if ok else 2)
    ok_all = True
    for f, fam in (("b1a", "B1A"), ("b2a", "B2A"), ("b3a", "B3A")):
        path = f"{repo}/docs/f2g_phase_b_power_annex_{f}_cal_results.json"
        try:
            res = json.loads(open(path, encoding="utf-8").read())
        except Exception as exc:
            print(f"[{f}] UNREADABLE: {exc}")
            ok_all = False
            continue
        ok, reasons = verify(res, repo=repo, expected_family=fam,
                             check_files=True)
        print(f"[{f}] {'PASS' if ok else 'REFUSE: ' + '; '.join(reasons[:6])}")
        ok_all = ok_all and ok
    sys.exit(0 if ok_all else 2)
