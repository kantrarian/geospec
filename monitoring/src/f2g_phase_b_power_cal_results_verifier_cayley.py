#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CALENDAR-LANE power-results verifier v5 (cayley) -- FIXTURE-ONLY.

Codex 0307Z consolidated architecture + the codex 5-fix bounded-recheck
repairs (findings 1-5): the verifier RECONSTRUCTS every claim-bearing
surface from the COMMITTED evidence capsule and compares -- result echoes
are never trusted -- and every substantive verification runs in a CLEAN
`python -I` SUBPROCESS that imports the engine, BASE driver, and CAL
driver from their attested absolute paths (module-cache purge + explicit
sys.modules registration), so neither a preloaded fake module nor
sys.path shadowing can substitute code (finding 2).

verify(res, repo, expected_family, check_files=True, expected_purpose)
-> (ok, reasons). In the subprocess:
  1. schema/family identity; geometry lock (POWER_GEOMETRY_UNBOUND).
  2. EXACT top-level key set (RESULT_SCHEMA_MISSING / RESULT_SCHEMA_EXTRA)
     and EXACT nested schemas with NO defaults (finding 4): evidence
     triple = full 40-hex commit + purpose-registered path + blob sha +
     row count + purpose; tier labels/draws/replicates; receipt fields
     strictly True and derived from the gate row; typed env; quantitative
     induced-effect report recomputed from the attested driver.
  3. digests must equal every pinned authority recomputed from git blobs
     (frozen Amendment 2, engine, admitted bar, CAL driver, BASE driver,
     rev-1.6 annexes, pinned assembler).
  4. evidence rows pass an EXACT stage-discriminated schema (finding 1):
     strict Python booleans (a JSON string "false" refuses typed
     EVIDENCE_ROW_TYPE), p/alpha/pre consistency, post=>pre, canonical
     driver keys, exact grid membership, per-stage field sets, global key
     uniqueness, purpose header + single gate row; S2/C/Cverdict rows are
     refused OUTSIDE the derived selected sets for EVERY family
     (EVIDENCE_SELECTION_CLOSURE); Cverdict rows must equal the derived
     verdicts; cross-stage panel digests must agree per (point, rep).
  5. tables/candidates/verdicts/stopping/certified/Pareto (REGISTERED
     coordinate order, finding 3)/terminal reconstructed and compared
     field-by-field (RESULT_NOT_DERIVED_FROM_EVIDENCE).
  6. panel_reopen_sample == derived expectation EXACTLY including digests
     (grid rep-0 digests BOUND to the S1 evidence rows -- finding 1), and
     every sampled panel regenerates to its recorded digest.
verify_package(...) additionally requires the three family artifacts to
share one evidence triple, authority digests, and campaign env
(PACKAGE_INCONSISTENT). self_test(repo, fixture_docs) runs the permanent
negative matrix (all prior classes + the recheck's new classes).
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

SCHEMA = "f2g-phase-b-power-cal-results-v1"
EVIDENCE_SCHEMA = "f2g-phase-b-power-cal-evidence-v1"
FAMILIES = ("B1A", "B2A", "B3A")
CANONICAL_EVIDENCE_PATH = "docs/f2g_phase_b_power_cal_evidence.jsonl"
FIXTURE_EVIDENCE_PATH = \
    "docs/fixtures/f2g_phase_b_power_cal_evidence_fixture.jsonl"
PURPOSE_PATHS = {"production": CANONICAL_EVIDENCE_PATH,
                 "fixture": FIXTURE_EVIDENCE_PATH}
CAL_AUTH_REF = ("8111805", "docs/f2g_phase_b_shared_calendar_v1.json")
PINS = {
    "frozen_amendment2_sha256": ("337571c",
                                 "docs/f2g_phase_b_prereg_amendment2_DRAFT.md"),
    "engine_lf_sha256": ("24b0d8f", "monitoring/src/d2_f2g_phase_b_stats.py"),
    "admitted_bar_lf_sha256": (
        "89673dc", "monitoring/src/test_f2g_phase_b_stats_redkats_grassmann.py"),
    "driver_lf_sha256": (
        "60509f7", "monitoring/src/f2g_phase_b_power_estimation_cal_cayley.py"),
    "base_driver_lf_sha256": (
        "60509f7", "monitoring/src/f2g_phase_b_power_estimation_cayley.py"),
    "annex_common_rev16_sha256": ("feb20bb",
                                  "docs/f2g_phase_b_power_annex_common.md"),
    "annex_b1a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b1a.md"),
    "annex_b2a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b2a.md"),
    "annex_b3a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b3a.md"),
    "assembler_lf_sha256": (
        "2bf545a",
        "monitoring/src/f2g_phase_b_power_cal_results_assembly_cayley.py"),
}
ENGINE_COMMIT = "24b0d8f"
# attested import set, dependency order (finding 2: BASE driver included)
MODULES = [
    ("d2_f2g_phase_b_stats", "engine_lf_sha256"),
    ("f2g_phase_b_power_estimation_cayley", "base_driver_lf_sha256"),
    ("f2g_phase_b_power_estimation_cal_cayley", "driver_lf_sha256"),
]
ATTEST = [(m, k) for m, k in MODULES] + [(None, "assembler_lf_sha256")]
REQUIRED_KEYS = ("schema", "family", "calendar_authority_mode",
                 "calendar_authority_sha256", "digests",
                 "equivalence_receipt", "evidence_capsule",
                 "panel_reopen_sample", "tier_s1", "tier_s2", "tier_c",
                 "certified_points", "pareto_minimal_certified",
                 "pareto_lex_representative", "terminal_type",
                 "induced_effect_report", "env")
STAGE_FIELDS = {
    "header": {"key", "stage", "purpose", "schema",
               "calendar_authority_sha256", "amendment2_sha256"},
    "gate": {"key", "stage", "full_equal", "fold_equal_all",
             "folds_checked", "all_equal", "engine_commit_bound",
             "calendar_authority_sha256"},
    "S1": {"key", "stage", "family", "point", "rep", "p", "pre",
           "panel_sha256", "dt"},
    "S2": {"key", "stage", "family", "point", "rep", "p", "pre", "post",
           "panel_sha256"},
    "C": {"key", "stage", "family", "point", "rep", "p", "pre", "post",
          "panel_sha256"},
    "Cverdict": {"key", "stage", "family", "point", "n", "successes",
                 "lb95", "ub95", "verdict"},
}
KEYSTAGE = {"S1": "S1", "S2": "S2", "C": "C", "Cverdict": "Cv"}
VERDICTS = ("CERTIFIED", "FAILED", "CANNOT_DETERMINE_POWER_ESTIMATE")


def _blob(repo, commit, path):
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _hex64(s):
    return isinstance(s, str) and len(s) == 64 and \
        all(c in "0123456789abcdef" for c in s)


def _num(v):
    return type(v) in (int, float)


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
    """Checkout LF bytes of every transitive source (engine, BASE driver,
    CAL driver, assembler, calendar authority) must equal the pinned blob;
    the module cache is purged so preloads cannot satisfy this."""
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


def _load_modules(repo, reasons):
    """Attested absolute-path import of engine -> base driver -> CAL
    driver, each registered in sys.modules BEFORE dependents execute, so
    the CAL driver's own import statements bind the attested objects."""
    for name, _k in MODULES:
        sys.modules.pop(name, None)
    loaded = {}
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        for name, key in MODULES:
            commit, path = PINS[key]
            ap = os.path.abspath(os.path.join(repo, path))
            try:
                pinned = _blob(repo, commit, path)
            except Exception as exc:
                reasons.append(f"DEPENDENCY_UNATTESTED: pin {key} "
                               f"unreadable ({exc})")
                return None
            disk = open(ap, "rb").read().replace(b"\r\n", b"\n")
            if _sha(disk) != _sha(pinned):
                reasons.append(f"DEPENDENCY_UNATTESTED: {path} checkout "
                               "differs from the pinned blob")
                return None
            spec = importlib.util.spec_from_file_location(name, ap)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:
                reasons.append(f"DEPENDENCY_UNATTESTED: {name} failed to "
                               f"load from attested path ({exc})")
                return None
            got = getattr(mod, "__file__", None)
            if got is None or os.path.normcase(os.path.abspath(got)) != \
                    os.path.normcase(ap):
                reasons.append(f"DEPENDENCY_UNATTESTED: {name} __file__ is "
                               "not the attested path")
                return None
            loaded[name] = mod
    finally:
        os.chdir(cwd)
    PD = loaded["f2g_phase_b_power_estimation_cal_cayley"]
    if PD.D0 is not loaded["f2g_phase_b_power_estimation_cayley"] or \
            PD.E is not loaded["d2_f2g_phase_b_stats"]:
        reasons.append("DEPENDENCY_UNATTESTED: the CAL driver's D0/E are "
                       "not the attested modules")
        return None
    return PD


def _validate_and_reconstruct(rows, repo, PD, reasons):
    """EXACT stage-discriminated evidence schema + per-family selection
    closure + full reconstruction (finding 1). Returns
    {"header":, "gate":, "recon": {fam: {...}}} or None."""
    E = PD.E
    alpha = E.ALPHA_FAMILY
    try:
        cal_sha = _sha(_blob(repo, *CAL_AUTH_REF))
        a2_sha = _sha(_blob(repo, *PINS["frozen_amendment2_sha256"]))
    except Exception as exc:
        reasons.append(f"authority unreadable during row validation: {exc}")
        return None
    grid_keys = {f: {pk(p) for p in PD.grid_of(f)} for f in FAMILIES}
    grid_of = {f: {pk(p): p for p in PD.grid_of(f)} for f in FAMILIES}
    by = {f: {s: defaultdict(list) for s in ("S1", "S2", "C", "Cverdict")}
          for f in FAMILIES}
    headers, gates = [], []
    seen_keys = set()
    fatal = False
    for i, r in enumerate(rows):
        st = r.get("stage")
        if st not in STAGE_FIELDS:
            reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} unknown stage "
                           f"{st!r}")
            fatal = True
            continue
        if set(r) != STAGE_FIELDS[st]:
            reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} ({st}) field set "
                           f"{sorted(set(r) ^ STAGE_FIELDS[st])} deviates")
            fatal = True
            continue
        if r["key"] in seen_keys:
            reasons.append(f"EVIDENCE_ROW_SCHEMA: duplicate key {r['key']}")
            fatal = True
            continue
        seen_keys.add(r["key"])
        if st == "header":
            if i != 0:
                reasons.append("CAPSULE_PURPOSE_MISMATCH: header row must "
                               "be the first evidence row")
                fatal = True
            headers.append(r)
            continue
        if st == "gate":
            gates.append(r)
            continue
        fam = r.get("family")
        if fam not in FAMILIES:
            reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} unknown family "
                           f"{fam!r}")
            fatal = True
            continue
        pt = r.get("point")
        if not isinstance(pt, dict) or pk(pt) not in grid_keys[fam]:
            reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} ({st}) point "
                           f"outside the registered {fam} grid")
            fatal = True
            continue
        if st == "Cverdict":
            want_key = PD.D0.key_of("Cv", fam, pt, 0)
            if r["key"] != want_key:
                reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} key != "
                               "canonical driver key")
                fatal = True
                continue
            if type(r["n"]) is not int or type(r["successes"]) is not int \
                    or not (0 <= r["successes"] <= r["n"]) \
                    or not _num(r["lb95"]) or not _num(r["ub95"]) \
                    or r["verdict"] not in VERDICTS:
                reasons.append(f"EVIDENCE_ROW_TYPE: row {i} Cverdict "
                               "fields malformed")
                fatal = True
                continue
            by[fam]["Cverdict"][pk(pt)].append(r)
            continue
        rep = r.get("rep")
        rep_cap = 40 if st == "C" else 50
        if type(rep) is not int or not (0 <= rep < rep_cap):
            reasons.append(f"EVIDENCE_ROW_TYPE: row {i} rep {rep!r} out of "
                           f"range for {st}")
            fatal = True
            continue
        if r["key"] != PD.D0.key_of(KEYSTAGE[st], fam, pt, rep):
            reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} key != canonical "
                           "driver key")
            fatal = True
            continue
        p = r["p"]
        if not (p is None or (_num(p) and 0.0 <= p <= 1.0)):
            reasons.append(f"EVIDENCE_ROW_TYPE: row {i} p {p!r} not "
                           "numeric-in-[0,1] or null")
            fatal = True
            continue
        if type(r["pre"]) is not bool:
            reasons.append(f"EVIDENCE_ROW_TYPE: row {i} pre={r['pre']!r} "
                           "is not a strict boolean")
            fatal = True
            continue
        if r["pre"] is not (p is not None and p <= alpha):
            reasons.append(f"EVIDENCE_ROW_TYPE: row {i} pre inconsistent "
                           "with p vs the registered alpha")
            fatal = True
            continue
        if st in ("S2", "C"):
            if type(r["post"]) is not bool:
                reasons.append(f"EVIDENCE_ROW_TYPE: row {i} "
                               f"post={r['post']!r} is not a strict boolean")
                fatal = True
                continue
            if r["post"] is True and r["pre"] is not True:
                reasons.append(f"EVIDENCE_ROW_TYPE: row {i} post=>pre "
                               "violated")
                fatal = True
                continue
        if st == "S1" and not (_num(r["dt"]) and r["dt"] >= 0):
            reasons.append(f"EVIDENCE_ROW_TYPE: row {i} dt malformed")
            fatal = True
            continue
        if not _hex64(r["panel_sha256"]):
            reasons.append(f"EVIDENCE_ROW_SCHEMA: row {i} panel_sha256 is "
                           "not 64-hex")
            fatal = True
            continue
        by[fam][st][pk(pt)].append(r)
    # header + gate closure
    if len(headers) != 1:
        reasons.append("CAPSULE_PURPOSE_MISMATCH: exactly one header row "
                       "required")
        fatal = True
    else:
        h = headers[0]
        if h["key"] != "header" or h["schema"] != EVIDENCE_SCHEMA or \
                h["purpose"] not in PURPOSE_PATHS or \
                h["calendar_authority_sha256"] != cal_sha or \
                h["amendment2_sha256"] != a2_sha:
            reasons.append("CAPSULE_PURPOSE_MISMATCH: header row fields "
                           "not the registered authorities/purpose")
            fatal = True
    if len(gates) != 1:
        reasons.append("evidence must contain exactly one gate row")
        fatal = True
    else:
        g = gates[0]
        if g["key"] != "gateCal" or g["full_equal"] is not True or \
                g["fold_equal_all"] is not True or \
                g["all_equal"] is not True or g["folds_checked"] != 35 or \
                g["engine_commit_bound"] != ENGINE_COMMIT or \
                g["calendar_authority_sha256"] != cal_sha:
            reasons.append("EVIDENCE_ROW_TYPE: gate row fields not the "
                           "strict registered receipt")
            fatal = True
    if fatal:
        return None
    # per-family closure + reconstruction
    recon = {}
    for fam in FAMILIES:
        grid = PD.grid_of(fam)
        # cross-stage digest binding: one panel per (point, rep)
        dig_of = {}
        for st in ("S1", "S2", "C"):
            for pj, v in by[fam][st].items():
                for x in v:
                    kk = (pj, x["rep"])
                    if kk in dig_of and dig_of[kk] != x["panel_sha256"]:
                        reasons.append("EVIDENCE_DIGEST_INCONSISTENT: "
                                       f"{fam} {kk} digests differ across "
                                       "stages")
                        return None
                    dig_of[kk] = x["panel_sha256"]
        n_s1 = sum(len(v) for v in by[fam]["S1"].values())
        if n_s1 != len(grid) * 50:
            reasons.append(f"evidence S1 {fam}: {n_s1} rows != "
                           f"{len(grid) * 50}")
            return None
        s1 = []
        for p in grid:
            v = by[fam]["S1"].get(pk(p), [])
            if sorted(x["rep"] for x in v) != list(range(50)):
                reasons.append(f"evidence S1 {fam} {pk(p)}: not exactly "
                               "reps 0..49")
                return None
            s1.append({"point": p, "pre_loco_recovery":
                       sum(x["pre"] is True for x in v) / 50, "n": 50})
        rank1 = sorted(s1, key=lambda x: (-x["pre_loco_recovery"],
                                          PD.ckey(fam, x["point"])))
        s2_pts = [x["point"] for x in rank1[:min(8, len(grid))]]
        s2_set = {pk(p) for p in s2_pts}
        extra_s2 = set(by[fam]["S2"]) - s2_set
        if extra_s2:
            reasons.append(f"EVIDENCE_SELECTION_CLOSURE: {fam} S2 rows at "
                           f"unselected points {sorted(extra_s2)[:2]}")
            return None
        s2 = []
        for p in s2_pts:
            v = by[fam]["S2"].get(pk(p), [])
            if sorted(x["rep"] for x in v) != list(range(50)):
                reasons.append(f"evidence S2 {fam} {pk(p)}: not exactly "
                               "reps 0..49")
                return None
            s2.append({"point": p,
                       "pre_loco_recovery": sum(x["pre"] is True
                                                for x in v) / 50,
                       "post_loco_recovery": sum(x["post"] is True
                                                 for x in v) / 50, "n": 50})
        rank2 = sorted(s2, key=lambda x: (-x["post_loco_recovery"],
                                          PD.ckey(fam, x["point"])))
        c_pts = [x["point"] for x in rank2[:min(3, len(s2))]]
        c_set = {pk(p) for p in c_pts}
        for st in ("C", "Cverdict"):
            extra = set(by[fam][st]) - c_set
            if extra:
                reasons.append(f"EVIDENCE_SELECTION_CLOSURE: {fam} {st} "
                               f"rows at non-candidate points "
                               f"{sorted(extra)[:2]}")
                return None
        cands = []
        for p in c_pts:
            v = sorted(by[fam]["C"].get(pk(p), []), key=lambda x: x["rep"])
            if [x["rep"] for x in v] != list(range(len(v))):
                reasons.append(f"evidence C {fam} {pk(p)}: reps not "
                               "sequential")
                return None
            succ = 0
            verdict = None
            n_stop = None
            for i, r in enumerate(v):
                succ += r["post"] is True
                n = i + 1
                if n == 20:
                    if cp_lower(succ, 20) >= 0.80:
                        verdict, n_stop = "CERTIFIED", 20
                        break
                    if cp_upper(succ, 20) < 0.80:
                        verdict, n_stop = "FAILED", 20
                        break
                if n == 40:
                    verdict = ("CERTIFIED" if cp_lower(succ, 40) >= 0.80
                               else ("FAILED" if cp_upper(succ, 40) < 0.80
                                     else "CANNOT_DETERMINE_POWER_ESTIMATE"))
                    n_stop = 40
            if verdict is None or n_stop != len(v):
                reasons.append(f"EVIDENCE_STOPPING: {fam} C {pk(p)} rows "
                               "inconsistent with the registered stopping "
                               "semantics")
                return None
            cvs = by[fam]["Cverdict"].get(pk(p), [])
            if len(cvs) != 1:
                reasons.append(f"EVIDENCE_SELECTION_CLOSURE: {fam} "
                               f"{pk(p)}: exactly one Cverdict row required")
                return None
            cv = cvs[0]
            if cv["n"] != n_stop or cv["successes"] != succ or \
                    cv["lb95"] != cp_lower(succ, n_stop) or \
                    cv["ub95"] != cp_upper(succ, n_stop) or \
                    cv["verdict"] != verdict:
                reasons.append(f"EVIDENCE_STOPPING: {fam} {pk(p)} Cverdict "
                               "row not derived-consistent")
                return None
            cands.append({"point": p,
                          "post_loco": {"successes": succ,
                                        "replicates": n_stop},
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
                        key=lambda c: PD.ckey(fam, c["point"]))
        reopen = []
        for p in grid:
            r0 = [x for x in by[fam]["S1"][pk(p)] if x["rep"] == 0]
            reopen.append({"family": fam, "point": p, "rep": 0,
                           "panel_sha256": r0[0]["panel_sha256"]})
        for p in c_pts:
            for r in sorted(by[fam]["C"][pk(p)], key=lambda x: x["rep"]):
                reopen.append({"family": fam, "point": p, "rep": r["rep"],
                               "panel_sha256": r["panel_sha256"]})
        recon[fam] = {"s1": s1, "s2": s2, "cands": cands,
                      "certified": certified, "pareto": pareto,
                      "reopen": reopen,
                      "terminal": ("CERTIFIED" if certified else
                                   "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH")}
    return {"header": headers[0], "gate": gates[0], "recon": recon}


def _cmp(name, got, want, reasons):
    if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
        reasons.append(f"RESULT_NOT_DERIVED_FROM_EVIDENCE: {name}")


def _verify_inner(res, repo, expected_family, check_files,
                  expected_purpose):
    reasons = []
    if res.get("schema") != SCHEMA:
        reasons.append(f"RESULT_SCHEMA_MISSING: schema != {SCHEMA}")
    fam = res.get("family")
    if fam not in FAMILIES:
        reasons.append(f"unknown family {fam!r}")
    if expected_family is not None and fam != expected_family:
        reasons.append(f"family {fam!r} != expected {expected_family!r}")
    if res.get("calendar_authority_mode") != "bound":
        reasons.append("POWER_GEOMETRY_UNBOUND: calendar_authority_mode "
                       "must be 'bound'")
    try:
        cal_sha = _sha(_blob(repo, *CAL_AUTH_REF))
        if res.get("calendar_authority_sha256") != cal_sha:
            reasons.append("POWER_GEOMETRY_UNBOUND: calendar authority "
                           "sha mismatch")
    except Exception as exc:
        reasons.append(f"POWER_GEOMETRY_UNBOUND: authority unreadable "
                       f"({exc})")
    if not check_files:
        return (not reasons), reasons
    # ---- exact top-level schema, no defaults ----
    missing = [k for k in REQUIRED_KEYS if k not in res]
    extra = [k for k in res if k not in REQUIRED_KEYS]
    for k in missing:
        reasons.append(f"RESULT_SCHEMA_MISSING: {k}")
    for k in extra:
        reasons.append(f"RESULT_SCHEMA_EXTRA: {k}")
    for k in REQUIRED_KEYS:
        if k in res and res[k] is None and k != "pareto_lex_representative":
            reasons.append(f"RESULT_SCHEMA_MISSING: {k} is null")
    if missing or extra:
        return False, reasons
    d = res["digests"]
    rcpt = res["equivalence_receipt"]
    ev = res["evidence_capsule"]
    # ---- digests: exact key set, every authority recomputed ----
    if not isinstance(d, dict) or set(d) != set(PINS):
        reasons.append("RESULT_SCHEMA_MISSING: digests key set != the "
                       "pinned authority set")
    else:
        for key, (commit, path) in PINS.items():
            try:
                want = _sha(_blob(repo, commit, path))
            except Exception as exc:
                reasons.append(f"authority {key} unreadable: {exc}")
                continue
            if d.get(key) != want:
                reasons.append(f"digest {key} mismatch")
    # ---- receipt: exact key set, strictly True ----
    RCPT_KEYS = {"full_equal", "fold_equal_all", "folds_checked",
                 "all_equal", "engine_commit_bound"}
    if not isinstance(rcpt, dict) or set(rcpt) != RCPT_KEYS:
        reasons.append("RESULT_SCHEMA_MISSING: equivalence_receipt key set")
    else:
        if rcpt["full_equal"] is not True or \
                rcpt["fold_equal_all"] is not True or \
                rcpt["all_equal"] is not True:
            reasons.append("equivalence booleans not strictly True")
        if rcpt["folds_checked"] != 35:
            reasons.append("folds_checked != 35")
        if rcpt["engine_commit_bound"] != ENGINE_COMMIT:
            reasons.append(f"engine_commit_bound != {ENGINE_COMMIT}")
    # ---- evidence triple: exact, no defaults ----
    EV_KEYS = {"geospec_commit", "path", "git_blob_sha256", "rows",
               "purpose"}
    ev_ok = isinstance(ev, dict) and set(ev) == EV_KEYS
    if not ev_ok:
        reasons.append("RESULT_SCHEMA_MISSING: evidence_capsule key set "
                       "(commit/path/blob/rows/purpose all required, no "
                       "defaults)")
    else:
        if not (isinstance(ev["geospec_commit"], str)
                and len(ev["geospec_commit"]) == 40
                and all(c in "0123456789abcdef"
                        for c in ev["geospec_commit"])):
            reasons.append("evidence geospec_commit is not a full 40-hex "
                           "commit id")
            ev_ok = False
        if ev["purpose"] not in PURPOSE_PATHS:
            reasons.append(f"CAPSULE_PURPOSE_MISMATCH: unknown purpose "
                           f"{ev['purpose']!r}")
            ev_ok = False
        elif ev["path"] != PURPOSE_PATHS[ev["purpose"]]:
            reasons.append("CAPSULE_PURPOSE_MISMATCH: evidence path is not "
                           "the registered path for its purpose")
            ev_ok = False
        if expected_purpose is not None and \
                ev.get("purpose") != expected_purpose:
            reasons.append(f"CAPSULE_PURPOSE_MISMATCH: purpose "
                           f"{ev.get('purpose')!r} != expected "
                           f"{expected_purpose!r}")
        if not _hex64(ev.get("git_blob_sha256")):
            reasons.append("evidence git_blob_sha256 is not 64-hex")
            ev_ok = False
        if type(ev.get("rows")) is not int:
            reasons.append("evidence rows is not an integer")
            ev_ok = False
    # ---- env: typed ----
    envd = res["env"]
    if not isinstance(envd, dict) or set(envd) != {"python", "numpy"} or \
            not all(isinstance(envd[k], str) and envd[k]
                    for k in ("python", "numpy")):
        reasons.append("RESULT_SCHEMA_MISSING: env must carry non-empty "
                       "python + numpy versions")
    # ---- tier shells ----
    for tier, lbl, nd, inner in (("tier_s1", "PRELIMINARY_SMOKE", 999,
                                  "table"),
                                 ("tier_s2", "PRELIMINARY_SMOKE", 999,
                                  "table"),
                                 ("tier_c", "CERTIFICATION", 9999,
                                  "candidates")):
        t = res[tier]
        want_keys = {"label", "n_draws", inner} | \
            ({"replicates"} if inner == "table" else set())
        if not isinstance(t, dict) or set(t) != want_keys:
            reasons.append(f"RESULT_SCHEMA_MISSING: {tier} key set")
            continue
        if t["label"] != lbl or t["n_draws"] != nd:
            reasons.append(f"{tier} label/draws wrong")
        if inner == "table" and t["replicates"] != 50:
            reasons.append(f"{tier} replicates != 50")
    # ---- attestation + attested module load ----
    att = _attest_sources(repo)
    reasons.extend(att)
    if att:
        return False, reasons
    PD = _load_modules(repo, reasons)
    if PD is None:
        return False, reasons
    # ---- evidence rows from the bound blob (no defaults) ----
    rows = None
    if ev_ok:
        try:
            raw = _blob(repo, ev["geospec_commit"], ev["path"])
            if _sha(raw) != ev["git_blob_sha256"]:
                reasons.append("evidence blob sha mismatch")
            lines = [l for l in raw.decode("utf-8").splitlines()
                     if l.strip()]
            if len(lines) != ev["rows"]:
                reasons.append("evidence row count mismatch")
            rows = [json.loads(l) for l in lines]
        except Exception as exc:
            reasons.append(f"evidence unreadable: {exc}")
    if rows is None:
        return False, reasons
    vr = _validate_and_reconstruct(rows, repo, PD, reasons)
    if vr is None:
        return False, reasons
    # header purpose must match the artifact triple
    if ev_ok and vr["header"]["purpose"] != ev["purpose"]:
        reasons.append("CAPSULE_PURPOSE_MISMATCH: capsule header purpose "
                       "differs from the artifact evidence triple")
    # receipt fields derived from the gate row (incl all_equal)
    if isinstance(rcpt, dict) and set(rcpt) == RCPT_KEYS:
        for k in RCPT_KEYS:
            if rcpt[k] != vr["gate"].get(k):
                reasons.append(f"receipt field {k} not derived from the "
                               "evidence gate row")
    # ---- reconstruction comparison for this artifact's family ----
    recon = vr["recon"].get(fam) if fam in FAMILIES else None
    if recon is not None:
        _cmp("tier_s1.table", res["tier_s1"].get("table"), recon["s1"],
             reasons)
        _cmp("tier_s2.table", res["tier_s2"].get("table"), recon["s2"],
             reasons)
        _cmp("tier_c.candidates", res["tier_c"].get("candidates"),
             recon["cands"], reasons)
        _cmp("certified_points", res["certified_points"],
             recon["certified"], reasons)
        _cmp("pareto_minimal_certified", res["pareto_minimal_certified"],
             recon["pareto"], reasons)
        _cmp("pareto_lex_representative",
             res["pareto_lex_representative"],
             recon["pareto"][0]["point"] if recon["pareto"] else None,
             reasons)
        if res["terminal_type"] != recon["terminal"]:
            reasons.append("RESULT_NOT_DERIVED_FROM_EVIDENCE: "
                           "terminal_type")
        # induced-effect report recomputed from the attested driver
        try:
            _cmp("induced_effect_report", res["induced_effect_report"],
                 PD.induced_effect_report(fam), reasons)
        except Exception as exc:
            reasons.append(f"induced_effect_report recomputation failed: "
                           f"{exc}")
        # ---- reopen: exact multiset INCLUDING digests + regeneration ----
        sample = res["panel_reopen_sample"]
        entry_keys = {"family", "point", "rep", "panel_sha256"}
        if not isinstance(sample, list) or \
                any(not isinstance(x, dict) or set(x) != entry_keys
                    for x in sample):
            reasons.append("RESULT_SCHEMA_MISSING: panel_reopen_sample "
                           "entry shape")
        else:
            want = Counter(json.dumps(x, sort_keys=True)
                           for x in recon["reopen"])
            got = Counter(json.dumps(x, sort_keys=True) for x in sample)
            if want != got:
                reasons.append("panel_reopen_sample not exactly the "
                               "derived evidence-bound coverage")
            for x in recon["reopen"]:
                try:
                    pan = PD.make_panel(x["family"], x["point"], x["rep"])
                    if PD.panel_digest(pan) != x["panel_sha256"]:
                        reasons.append(
                            "POWER_GEOMETRY_UNBOUND: panel "
                            f"({x['family']},{pk(x['point'])},{x['rep']}) "
                            "does not regenerate to its recorded digest")
                except Exception as exc:
                    reasons.append(f"panel regeneration failed: {exc}")
    return (not reasons), reasons


def verify(res, repo, expected_family=None, check_files=True,
           expected_purpose=None):
    """Public entry: runs the full verification in a clean `python -I`
    subprocess so preloaded modules and sys.path shadowing in the calling
    process can never reach the derivation (finding 2)."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(res, f)
        out = subprocess.run(
            [sys.executable, "-I", os.path.abspath(__file__),
             "--verify-inner", tmp, repo, expected_family or "-",
             "1" if check_files else "0", expected_purpose or "-"],
            capture_output=True, text=True)
        if out.returncode != 0:
            return False, ["VERIFIER_SUBPROCESS_FAILURE: "
                           + (out.stderr or out.stdout or "")[-400:]]
        payload = json.loads(out.stdout.strip().splitlines()[-1])
        return payload["ok"], payload["reasons"]
    finally:
        os.unlink(tmp)


def verify_package_objs(arts, repo, expected_purpose):
    """arts: {family: artifact dict}. Individual verification of each,
    then package-level consistency: one evidence triple, one authority
    digest set, one campaign env (finding 4)."""
    reasons = []
    for fam in FAMILIES:
        if fam not in arts:
            reasons.append(f"PACKAGE_INCONSISTENT: {fam} artifact missing")
            continue
        ok, rs = verify(arts[fam], repo, expected_family=fam,
                        check_files=True, expected_purpose=expected_purpose)
        if not ok:
            reasons.append(f"{fam}: " + "; ".join(map(str, rs[:4])))
    if all(f in arts for f in FAMILIES):
        ref = arts["B1A"]
        for field in ("evidence_capsule", "digests", "env"):
            for fam in ("B2A", "B3A"):
                if json.dumps(arts[fam].get(field), sort_keys=True) != \
                        json.dumps(ref.get(field), sort_keys=True):
                    reasons.append(f"PACKAGE_INCONSISTENT: {field} differs "
                                   f"across family artifacts ({fam})")
    return (not reasons), reasons


def verify_package(repo, docs_dir, expected_purpose):
    arts = {}
    reasons = []
    for f, fam in (("b1a", "B1A"), ("b2a", "B2A"), ("b3a", "B3A")):
        path = f"{docs_dir}/f2g_phase_b_power_annex_{f}_cal_results.json"
        try:
            arts[fam] = json.loads(open(path, encoding="utf-8").read())
        except Exception as exc:
            reasons.append(f"PACKAGE_INCONSISTENT: {fam} unreadable "
                           f"({exc})")
    ok, rs = verify_package_objs(arts, repo, expected_purpose)
    return (ok and not reasons), reasons + rs


def _rowneg(repo, case):
    """Run one evidence-row negative in a clean subprocess; returns its
    reasons list (or extra payload for the KAT/probe cases)."""
    out = subprocess.run(
        [sys.executable, "-I", os.path.abspath(__file__), "--rowneg",
         case, repo], capture_output=True, text=True)
    if out.returncode != 0:
        return {"reasons": ["ROWNEG_SUBPROCESS_FAILURE: "
                            + (out.stderr or "")[-400:]]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def _rowneg_inner(repo, case):
    reasons = []
    PD = _load_modules(repo, reasons)
    if PD is None:
        print(json.dumps({"reasons": reasons}))
        return
    if case == "probe-d0":
        print(json.dumps({"reasons": [], "sigma_s": PD.D0.SIGMA_S,
                          "d0_file": PD.D0.__file__}))
        return
    raw = _blob(repo, "HEAD", FIXTURE_EVIDENCE_PATH)
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines()
            if l.strip()]
    if case == "string-false":
        for r in rows:
            if r.get("stage") == "C" and r.get("family") == "B2A":
                r["post"] = "false"
    elif case == "sample-vs-evidence-digest":
        for r in rows:
            if r.get("stage") == "S1" and r.get("family") == "B2A" and \
                    r.get("rep") == 0:
                r["panel_sha256"] = "0" * 64
                break
    elif case == "extra-unselected-stage-row":
        uns = sorted(PD.grid_of("B1A"),
                     key=lambda p: PD.ckey("B1A", p))[-1]
        dig = next(r["panel_sha256"] for r in rows
                   if r.get("stage") == "S1" and r.get("family") == "B1A"
                   and pk(r["point"]) == pk(uns) and r["rep"] == 0)
        rows.append({"key": PD.D0.key_of("S2", "B1A", uns, 0),
                     "stage": "S2", "family": "B1A", "point": uns,
                     "rep": 0, "p": 1.0, "pre": False, "post": False,
                     "panel_sha256": dig})
    elif case == "offgrid-cverdict-row":
        rows.append({"key": PD.D0.key_of("Cv", "B2A", {"m": 99}, 0),
                     "stage": "Cverdict", "family": "B2A",
                     "point": {"m": 99}, "n": 20, "successes": 0,
                     "lb95": 0.0, "ub95": 0.139, "verdict": "FAILED"})
    elif case == "duplicate-gate-row":
        rows.append(dict(next(r for r in rows
                              if r.get("stage") == "gate")))
    elif case == "duplicate-header-row":
        h2 = dict(rows[0])
        h2["key"] = "header2"
        rows.append(h2)
    elif case == "tied-selector-kat":
        vr = _validate_and_reconstruct(rows, repo, PD, reasons)
        payload = {"reasons": reasons}
        if vr is not None:
            payload["s2_b3a"] = [list(PD.ckey("B3A", x["point"]))
                                 for x in vr["recon"]["B3A"]["s2"]]
            payload["c_b3a"] = [list(PD.ckey("B3A", x["point"]))
                                for x in vr["recon"]["B3A"]["cands"]]
        print(json.dumps(payload))
        return
    else:
        print(json.dumps({"reasons": [f"unknown rowneg case {case!r}"]}))
        return
    _validate_and_reconstruct(rows, repo, PD, reasons)
    print(json.dumps({"reasons": reasons}))


def self_test(repo, fixture_docs=None):
    """Permanent negative matrix. Every negative must refuse WITH ITS
    TYPED REASON (a refusal for an unrelated reason -- e.g. a broken
    positive path -- is recorded as a DEFECT, never as a pass), the
    positive fixture must PASS, and the injected-base-driver reproduction
    must be provably inert (IMMUNE_PASS)."""
    results = {}
    ok_all = True

    def rec(name, ok, reasons, expect):
        refused = (not ok) and any(expect in str(r) for r in reasons)
        results[name] = (f"REFUSED ({expect})" if refused else
                         ("DEFECT -- ACCEPTED" if ok else
                          "DEFECT -- refused without the typed reason: "
                          + "; ".join(map(str, reasons[:3]))))
        return refused

    cal_sha = _sha(_blob(repo, *CAL_AUTH_REF))
    geo = {"schema": SCHEMA, "family": "B2A",
           "calendar_authority_mode": "bound",
           "calendar_authority_sha256": cal_sha}
    ok, r = verify(geo, repo, expected_family="B2A", check_files=True)
    ok_all &= rec("geometry-only-schema-shaped", ok, r,
                  "RESULT_SCHEMA_MISSING")
    ok, r = verify(dict(geo, schema="NOT-THE-REGISTERED-SCHEMA"), repo,
                   expected_family="B2A", check_files=False)
    ok_all &= rec("wrong-schema", ok, r, "RESULT_SCHEMA_MISSING")
    ok, r = verify(geo, repo, expected_family="B1A", check_files=False)
    ok_all &= rec("wrong-family-in-file", ok, r, "expected")
    # stale-pin attestation refusals (executable mutated-dependency forms)
    bad = dict(PINS)
    bad["driver_lf_sha256"] = (
        "7028324", "monitoring/src/f2g_phase_b_power_estimation_cal_cayley.py")
    att = _attest_sources(repo, bad)
    ok_all &= rec("mutated-dependency", not att, att,
                  "DEPENDENCY_UNATTESTED")
    bad = dict(PINS)
    bad["base_driver_lf_sha256"] = (
        "d4edfb2", "monitoring/src/f2g_phase_b_power_estimation_cayley.py")
    att = _attest_sources(repo, bad)
    ok_all &= rec("mutated-base-driver", not att, att,
                  "DEPENDENCY_UNATTESTED")
    # evidence-row negatives (clean-subprocess, doctored committed capsule)
    ROWNEG_EXPECT = {
        "string-false": "EVIDENCE_ROW_TYPE",
        "sample-vs-evidence-digest": "EVIDENCE_DIGEST_INCONSISTENT",
        "extra-unselected-stage-row": "EVIDENCE_SELECTION_CLOSURE",
        "offgrid-cverdict-row": "outside the registered",
        "duplicate-gate-row": "duplicate key",
        "duplicate-header-row": "CAPSULE_PURPOSE_MISMATCH",
    }
    for case, expect in ROWNEG_EXPECT.items():
        rr = _rowneg(repo, case)
        ok_all &= rec(case, not rr["reasons"], rr["reasons"], expect)
    # tied-selector KAT: derived B3A selection == the registered exact order
    kat = _rowneg(repo, "tied-selector-kat")
    exp8 = [[0.3, 3, 10], [0.3, 3, 25], [0.3, 3, 50], [0.3, 8, 10],
            [0.3, 8, 25], [0.3, 8, 50], [0.6, 3, 10], [0.6, 3, 25]]
    kat_ok = (not kat["reasons"] and kat.get("s2_b3a") == exp8
              and kat.get("c_b3a") == exp8[:3])
    results["tied-selector-kat"] = ("PASS (registered top-8/top-3 exact)"
                                    if kat_ok else "FAIL -- DEFECT")
    ok_all &= kat_ok
    if fixture_docs:
        import copy
        base = json.loads(open(
            f"{fixture_docs}/f2g_phase_b_power_annex_b2a_cal_results.json",
            encoding="utf-8").read())
        base3 = json.loads(open(
            f"{fixture_docs}/f2g_phase_b_power_annex_b3a_cal_results.json",
            encoding="utf-8").read())
        okp, rp = verify(base, repo, expected_family="B2A",
                         check_files=True, expected_purpose="fixture")
        results["positive-fixture"] = ("PASS" if okp else
                                       "REFUSED -- DEFECT: " + "; ".join(
                                           map(str, rp[:3])))
        ok_all &= okp
        # finding-2 reproduction, exact: preload a fake base driver in THIS
        # process; the clean-subprocess verifier must be unaffected and its
        # loaded D0 must be the attested one (SIGMA_S == 0.15)
        import types
        fake = types.ModuleType("f2g_phase_b_power_estimation_cayley")
        fake.SIGMA_S = 999
        sys.modules["f2g_phase_b_power_estimation_cayley"] = fake
        try:
            oki, ri = verify(base, repo, expected_family="B2A",
                             check_files=True, expected_purpose="fixture")
            probe = _rowneg(repo, "probe-d0")
            immune = (oki == okp and ri == rp
                      and probe.get("sigma_s") == 0.15)
        finally:
            sys.modules.pop("f2g_phase_b_power_estimation_cayley", None)
        results["injected-base-driver"] = (
            "IMMUNE_PASS (verdict unchanged; attested D0 SIGMA_S=0.15)"
            if immune else "FAIL -- DEFECT")
        ok_all &= immune
        t1 = copy.deepcopy(base)
        t1["evidence_capsule"]["path"] = \
            "docs/f2g_phase_b_power_a_evidence.jsonl"
        ok, r = verify(t1, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("unrelated-committed-evidence", ok, r,
                      "CAPSULE_PURPOSE_MISMATCH")
        t2 = copy.deepcopy(base)
        if t2["tier_c"]["candidates"]:
            t2["tier_c"]["candidates"][0]["verdict"] = "CERTIFIED"
            t2["tier_c"]["candidates"][0]["post_loco"]["successes"] = 0
            t2["certified_points"] = [t2["tier_c"]["candidates"][0]]
            t2["terminal_type"] = "CERTIFIED"
        ok, r = verify(t2, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("zero-labeled-certified", ok, r,
                      "RESULT_NOT_DERIVED_FROM_EVIDENCE")
        t3 = copy.deepcopy(base)
        t3["tier_s1"]["table"][0]["pre_loco_recovery"] = 0.98
        ok, r = verify(t3, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("aggregate-mismatch", ok, r,
                      "RESULT_NOT_DERIVED_FROM_EVIDENCE: tier_s1.table")
        t4 = copy.deepcopy(base)
        if len(t4["tier_s1"]["table"]) >= 2:
            t4["tier_s1"]["table"][1] = dict(t4["tier_s1"]["table"][0])
        ok, r = verify(t4, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("duplicate-registered-point", ok, r,
                      "RESULT_NOT_DERIVED_FROM_EVIDENCE: tier_s1.table")
        t5 = copy.deepcopy(base)
        t5["panel_reopen_sample"] = t5["panel_reopen_sample"][:-1]
        ok, r = verify(t5, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("missing-reopen-rep", ok, r, "panel_reopen_sample")
        # finding-4 reproductions
        t6 = copy.deepcopy(base)
        t6["tier_s1"]["replicates"] = 1
        t6["tier_s2"]["replicates"] = "fifty"
        t6["equivalence_receipt"]["all_equal"] = False
        t6["env"] = {}
        t6["induced_effect_report"] = {}
        ok, rs6 = verify(t6, repo, expected_family="B2A", check_files=True,
                         expected_purpose="fixture")
        hit = all(any(s in r for r in rs6) for s in
                  ("replicates", "strictly True", "env",
                   "induced_effect_report"))
        results["combined-nested-mutation"] = (
            "REFUSED (all four surfaces typed)" if (not ok and hit)
            else "DEFECT")
        ok_all &= (not ok and hit)
        t7 = copy.deepcopy(base)
        del t7["evidence_capsule"]["geospec_commit"]
        ok, r = verify(t7, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("missing-commit-no-default", ok, r,
                      "RESULT_SCHEMA_MISSING: evidence_capsule")
        # finding-5 reproductions at the artifact surface
        t8 = copy.deepcopy(base)
        t8["evidence_capsule"]["purpose"] = "production"
        ok, r = verify(t8, repo, expected_family="B2A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("fixture-relabeled-production", ok, r,
                      "CAPSULE_PURPOSE_MISMATCH")
        ok, r = verify(base, repo, expected_family="B2A",
                       check_files=True, expected_purpose="production")
        ok_all &= rec("fixture-under-production-expectation", ok, r,
                      "CAPSULE_PURPOSE_MISMATCH")
        # finding-3 reproduction: an artifact assembled under the OLD
        # JSON-key-order selector must refuse against reconstruction
        t9 = copy.deepcopy(base3)
        old_pt = {"delta_lat": 0.6, "n_cross": 8, "k": 10}
        for row in t9["tier_s2"]["table"]:
            if row["point"] == {"delta_lat": 0.6, "n_cross": 3, "k": 25}:
                row["point"] = old_pt
        ok, r = verify(t9, repo, expected_family="B3A", check_files=True,
                       expected_purpose="fixture")
        ok_all &= rec("wrong-selector-order", ok, r,
                      "RESULT_NOT_DERIVED_FROM_EVIDENCE")
        # package-level checks
        arts = {"B1A": json.loads(open(
            f"{fixture_docs}/f2g_phase_b_power_annex_b1a_cal_results.json",
            encoding="utf-8").read()), "B2A": base, "B3A": base3}
        ok, r = verify_package_objs(arts, repo, expected_purpose="fixture")
        results["package-consistent-positive"] = (
            "PASS" if ok else "DEFECT: " + "; ".join(map(str, r[:3])))
        ok_all &= ok
        artsx = dict(arts, B1A=copy.deepcopy(arts["B1A"]))
        artsx["B1A"]["env"] = {"python": "0.0.0", "numpy": "0.0.0"}
        ok, r = verify_package_objs(artsx, repo,
                                    expected_purpose="fixture")
        ok_all &= rec("package-inconsistent-env", ok, r,
                      "PACKAGE_INCONSISTENT")
    return ok_all, results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-inner":
        _tmp, _repo, _fam, _cf, _purp = sys.argv[2:7]
        _res = json.loads(open(_tmp, encoding="utf-8").read())
        _ok, _reasons = _verify_inner(
            _res, _repo, None if _fam == "-" else _fam, _cf == "1",
            None if _purp == "-" else _purp)
        print(json.dumps({"ok": _ok, "reasons": _reasons}))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--rowneg":
        _rowneg_inner(sys.argv[3], sys.argv[2])
        sys.exit(0)
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
                             check_files=True,
                             expected_purpose="production")
        print(f"[{f}] {'PASS' if ok else 'REFUSE: ' + '; '.join(reasons[:6])}")
        ok_all = ok_all and ok
    okp, rp = verify_package(repo, f"{repo}/docs", "production")
    print(f"[package] {'PASS' if okp else 'REFUSE: ' + '; '.join(rp[:6])}")
    sys.exit(0 if (ok_all and okp) else 2)
