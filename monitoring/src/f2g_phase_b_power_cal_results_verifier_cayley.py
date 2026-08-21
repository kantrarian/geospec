#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict CALENDAR-LANE power-results verifier v2 (cayley) -- FIXTURE-ONLY.

v2 = codex 0222Z fail-open repair: check_files=True is SCHEMA-CLOSED and
SOURCE-ATTESTED. v1's key-presence-conditional checks let an empty artifact
pass; v2 requires every result key, runs every check unconditionally from {}
defaults (absence -> typed reasons), pins the ADMITTED bar and driver blobs
in addition to the annex/engine/amendment authorities, requires the digest
block to echo them, attests the imported driver's bytes against the pinned
blob before any panel regeneration, and enforces exact panel_reopen_sample
coverage (rep 0 of every registered grid point + every Tier-C replicate row
derived from the committed evidence capsule; no dupes/extras/mismatches).

verify(res, repo, check_files=True) -> (ok, reasons). Geometry lock (typed
POWER_GEOMETRY_UNBOUND) always runs. self_test(repo) exercises the
permanent negatives; each must refuse.
"""
import hashlib
import json
import subprocess

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
}
ENGINE_COMMIT = "24b0d8f"
REQUIRED_KEYS = ("schema", "family", "digests", "equivalence_receipt",
                 "evidence_capsule", "panel_reopen_sample", "tier_s1",
                 "tier_s2", "tier_c", "certified_points",
                 "pareto_minimal_certified", "terminal_type")
GRID_SIZES = {"B1A": 48, "B2A": 3, "B3A": 24}


def _blob(repo, commit, path):
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo)


def _blob_sha(repo, commit, path):
    return hashlib.sha256(_blob(repo, commit, path)).hexdigest()


def _load_pinned_driver(repo):
    """Import the driver ONLY after attesting the checkout bytes equal the
    pinned blob (LF-normalized); refuse otherwise."""
    import os
    import sys
    commit, path = PINS["driver_lf_sha256"]
    pinned = _blob(repo, commit, path)
    disk = open(os.path.join(repo, path), "rb").read().replace(b"\r\n",
                                                               b"\n")
    if hashlib.sha256(disk).hexdigest() != hashlib.sha256(pinned).hexdigest():
        raise ValueError("DRIVER_SOURCE_UNATTESTED: checkout driver bytes "
                         "differ from the pinned blob")
    src = os.path.join(repo, "monitoring", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        import f2g_phase_b_power_estimation_cal_cayley as PD
    finally:
        os.chdir(cwd)
    return PD


def _scan_stage_c(ev_rows, fam, expected, reasons):
    """Collect Stage-C evidence keys; a DUPLICATE (family, canonical point,
    rep) key refuses typed and never silently overwrites (codex 0244Z)."""
    ev_c_seen = set()
    for row in ev_rows:
        if row.get("stage") == "C" and row.get("family") == fam:
            ck = (fam, json.dumps(row["point"], sort_keys=True),
                  row["rep"])
            if ck in ev_c_seen:
                reasons.append(f"duplicate Stage-C evidence key: {ck}")
                continue
            ev_c_seen.add(ck)
            expected[ck] = row.get("panel_sha256")


def verify(res, repo, check_files=True):
    reasons = []
    # ---- geometry lock (always) ----
    mode = res.get("calendar_authority_mode")
    if mode != "bound":
        reasons.append(f"POWER_GEOMETRY_UNBOUND: calendar_authority_mode="
                       f"{mode!r} (must be 'bound')")
    try:
        expect_cal_sha = _blob_sha(repo, *CAL_AUTH_REF)
        if res.get("calendar_authority_sha256") != expect_cal_sha:
            reasons.append("POWER_GEOMETRY_UNBOUND: calendar_authority_"
                           "sha256 does not match the pinned authority blob")
    except Exception as exc:
        reasons.append(f"POWER_GEOMETRY_UNBOUND: calendar authority blob "
                       f"unreadable from repo ({exc})")
    if not check_files:
        return (not reasons), reasons
    # ---- schema-closed: every key REQUIRED ----
    for key in REQUIRED_KEYS:
        if key not in res or res[key] is None:
            reasons.append(f"RESULT_SCHEMA_MISSING: {key}")
    d = res.get("digests") or {}
    r = res.get("equivalence_receipt") or {}
    ev = res.get("evidence_capsule") or {}
    sample = res.get("panel_reopen_sample") or []
    fam = res.get("family")
    # ---- authorities: every pin recomputed and echoed ----
    for key, (commit, path) in PINS.items():
        try:
            want = _blob_sha(repo, commit, path)
        except Exception as exc:
            reasons.append(f"authority {key} unreadable: {exc}")
            continue
        if d.get(key) != want:
            reasons.append(f"digest {key} mismatch/absent")
    # ---- receipt: strict, exact binding ----
    if r.get("full_equal") is not True or r.get("fold_equal_all") is not True:
        reasons.append("equivalence booleans not strictly True")
    if r.get("folds_checked") != 35:
        reasons.append("folds_checked != 35")
    if r.get("engine_commit_bound") != ENGINE_COMMIT:
        reasons.append(f"engine_commit_bound != {ENGINE_COMMIT}")
    # ---- evidence capsule: committed blob only ----
    ev_rows = []
    try:
        raw = _blob(repo, "HEAD", ev.get("path", ""))
        if hashlib.sha256(raw).hexdigest() != ev.get("git_blob_sha256"):
            reasons.append("evidence capsule git-blob sha mismatch")
        lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
        if len(lines) != ev.get("rows"):
            reasons.append("evidence row count mismatch")
        ev_rows = [json.loads(l) for l in lines]
    except Exception as exc:
        reasons.append(f"evidence capsule unreadable: {exc}")
    # ---- panel reopen: exact coverage, attested driver, digest match ----
    try:
        PD = _load_pinned_driver(repo)
        expected = {}
        if fam in GRID_SIZES:
            for pt in PD.grid_of(fam):
                expected[(fam, json.dumps(pt, sort_keys=True), 0)] = None
        _scan_stage_c(ev_rows, fam, expected, reasons)
        seen = set()
        for row in sample:
            key = (row.get("family"), json.dumps(row.get("point"),
                                                 sort_keys=True),
                   row.get("rep"))
            if key in seen:
                reasons.append(f"panel_reopen_sample duplicate: {key}")
                continue
            seen.add(key)
            if key not in expected:
                reasons.append(f"panel_reopen_sample extra/foreign: {key}")
                continue
            pan = PD.make_panel(row["family"], row["point"], row["rep"])
            if PD.panel_digest(pan) != row.get("panel_sha256"):
                reasons.append("POWER_GEOMETRY_UNBOUND: panel "
                               f"{key} cannot be reopened to its recorded "
                               "digest")
            want_ev = expected[key]
            if want_ev is not None and row.get("panel_sha256") != want_ev:
                reasons.append(f"panel digest differs from evidence row: "
                               f"{key}")
        missing = set(expected) - seen
        if missing:
            reasons.append(f"panel_reopen_sample missing coverage: "
                           f"{len(missing)} entries (e.g. "
                           f"{sorted(missing)[0]})")
        if not sample:
            reasons.append("panel_reopen_sample empty")
    except Exception as exc:
        reasons.append(f"panel reopen failed: {exc}")
    # ---- tiers / cardinality / determinism ----
    for tier, lbl, nd in (("tier_s1", "PRELIMINARY_SMOKE", 999),
                          ("tier_s2", "PRELIMINARY_SMOKE", 999),
                          ("tier_c", "CERTIFICATION", 9999)):
        t = res.get(tier) or {}
        if t.get("label") != lbl or t.get("n_draws") != nd:
            reasons.append(f"{tier} label/draws wrong")
    # codex 0244Z repair: exact canonical-key MULTISETS at every tier -- a
    # duplicate point can never displace a registered grid member
    from collections import Counter

    def pkeys(rows):
        return [json.dumps(x.get("point"), sort_keys=True,
                           separators=(",", ":")) for x in rows]

    s1t = (res.get("tier_s1") or {}).get("table") or []
    s2t = (res.get("tier_s2") or {}).get("table") or []
    cands = (res.get("tier_c") or {}).get("candidates") or []
    grid_keys = []
    if fam in GRID_SIZES:
        try:
            PD2 = _load_pinned_driver(repo)
            grid_keys = [json.dumps(p, sort_keys=True,
                                    separators=(",", ":"))
                         for p in PD2.grid_of(fam)]
        except Exception as exc:
            reasons.append(f"registered grid unreadable: {exc}")
    if grid_keys and Counter(pkeys(s1t)) != Counter(grid_keys):
        reasons.append("tier_s1 is not exactly the registered grid")
    if s1t:
        rank1 = sorted(s1t, key=lambda x: (-x.get("pre_loco_recovery", 0),
                                           json.dumps(x.get("point"),
                                                      sort_keys=True)))
        want_s2_list = pkeys(rank1[:min(8, len(grid_keys) or len(rank1))])
        if Counter(pkeys(s2t)) != Counter(want_s2_list):
            reasons.append("tier_s2 is not exactly the registered top-8")
    if s2t:
        rank2 = sorted(s2t, key=lambda x: (-x.get("post_loco_recovery", 0),
                                           json.dumps(x.get("point"),
                                                      sort_keys=True)))
        want_c_list = pkeys(rank2[:min(3, len(s2t))])
        if Counter(pkeys(cands)) != Counter(want_c_list):
            reasons.append("tier_c is not exactly the registered top-3")
    if not cands:
        reasons.append("tier_c candidates empty")
    import f2g_phase_b_power_estimation_cayley as D0
    for c in cands:
        k, n = c["post_loco"]["successes"], c["post_loco"]["replicates"]
        if abs(D0.cp_lower(k, n) - c["lb95"]) > 1e-12 or \
                abs(D0.cp_upper(k, n) - c["ub95"]) > 1e-12:
            reasons.append(f"CP bounds mismatch at {c['point']}")
    cert = res.get("certified_points")
    term = res.get("terminal_type")
    if (cert is not None) and (term is not None):
        if bool(cert) != (term == "CERTIFIED"):
            reasons.append("terminal/certified inconsistency")
        if term not in ("CERTIFIED",
                        "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH"):
            reasons.append("unknown terminal type")
    pareto = res.get("pareto_minimal_certified") or []
    if pareto and cert:
        if not any(p["point"] == c["point"] for p in pareto for c in cert):
            reasons.append("pareto member not in certified_points")
    if cert and not pareto:
        reasons.append("certified without a Pareto-minimal set")
    return (not reasons), reasons


def self_test(repo):
    """Permanent negatives (codex 0222Z item 5): each must refuse."""
    try:
        cal_sha = _blob_sha(repo, *CAL_AUTH_REF)
    except Exception:
        return False, ["repo unreadable"]
    geometry_only = {"calendar_authority_mode": "bound",
                     "calendar_authority_sha256": cal_sha}
    negatives = {"geometry-only": geometry_only}
    full = dict(geometry_only)
    full.update({k: {} for k in REQUIRED_KEYS if k not in full})
    full["schema"] = "x"
    full["family"] = "B2A"
    full["digests"] = {k: _blob_sha(repo, c, p)
                       for k, (c, p) in PINS.items()}
    full["equivalence_receipt"] = {"full_equal": True,
                                   "fold_equal_all": True,
                                   "folds_checked": 35,
                                   "engine_commit_bound": ENGINE_COMMIT}
    negatives["missing-evidence"] = dict(full, evidence_capsule={})
    negatives["empty-reopen-sample"] = dict(full, panel_reopen_sample=[])
    bogus = dict(full)
    bogus["equivalence_receipt"] = dict(full["equivalence_receipt"],
                                        engine_commit_bound="NOT_THE_ENGINE")
    negatives["bogus-engine-binding"] = bogus
    nopin = dict(full)
    nopin["digests"] = {k: v for k, v in full["digests"].items()
                        if k not in ("admitted_bar_lf_sha256",
                                     "driver_lf_sha256")}
    negatives["missing-bar-driver-pin"] = nopin
    # sixth permanent negative (codex 0244Z): B2A tier-membership collapse --
    # duplicate {m:1} displacing {m:3} must refuse via exact multisets
    member = dict(full)
    member["family"] = "B2A"
    member["tier_s1"] = {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                         "table": [
                             {"point": {"m": 1}, "pre_loco_recovery": 1.0},
                             {"point": {"m": 1}, "pre_loco_recovery": 1.0},
                             {"point": {"m": 2}, "pre_loco_recovery": 1.0}]}
    member["tier_s2"] = {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                         "table": [
                             {"point": {"m": 1}, "post_loco_recovery": 1.0},
                             {"point": {"m": 2}, "post_loco_recovery": 1.0}]}
    member["tier_c"] = {"label": "CERTIFICATION", "n_draws": 9999,
                        "candidates": [
                            {"point": {"m": 1},
                             "post_loco": {"successes": 20,
                                           "replicates": 20},
                             "lb95": 0.0, "ub95": 1.0}]}
    negatives["tier-membership-collapse"] = member
    results = {}
    ok_all = True
    for name, res in negatives.items():
        ok, reasons = verify(res, repo=repo, check_files=True)
        refused = not ok
        if name == "tier-membership-collapse":
            refused = refused and any("registered grid" in str(x)
                                      or "top-8" in str(x)
                                      or "top-3" in str(x) for x in reasons)
        results[name] = "REFUSED" if refused else "ACCEPTED -- DEFECT"
        ok_all = ok_all and refused
    # seventh permanent negative: duplicate Stage-C evidence key (unit seam)
    dup_reasons = []
    _scan_stage_c([{"stage": "C", "family": "B2A", "point": {"m": 1},
                    "rep": 0, "panel_sha256": "aa"},
                   {"stage": "C", "family": "B2A", "point": {"m": 1},
                    "rep": 0, "panel_sha256": "bb"}],
                  "B2A", {}, dup_reasons)
    dup_ok = any("duplicate Stage-C" in str(x) for x in dup_reasons)
    results["duplicate-stage-c-evidence"] = ("REFUSED" if dup_ok
                                             else "ACCEPTED -- DEFECT")
    ok_all = ok_all and dup_ok
    return ok_all, results


if __name__ == "__main__":
    import sys
    repo = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--self-test":
        ok, results = self_test(repo)
        print(json.dumps(results, indent=1))
        sys.exit(0 if ok else 2)
    ok_all = True
    for f in ("b1a", "b2a", "b3a"):
        path = f"{repo}/docs/f2g_phase_b_power_annex_{f}_cal_results.json"
        try:
            res = json.loads(open(path, encoding="utf-8").read())
        except Exception as exc:
            print(f"[{f}] UNREADABLE: {exc}")
            ok_all = False
            continue
        ok, reasons = verify(res, repo=repo, check_files=True)
        print(f"[{f}] {'PASS' if ok else 'REFUSE: ' + '; '.join(reasons)}")
        ok_all = ok_all and ok
    sys.exit(0 if ok_all else 2)
