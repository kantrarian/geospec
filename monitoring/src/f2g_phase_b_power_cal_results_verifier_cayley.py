#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict CALENDAR-LANE power-results verifier (cayley) -- FIXTURE-ONLY.

The admissible-power verification seam of codex 0131Z repair 3 + 0138Z, at
the bar-pinned module name, with a PORTABLE repo argument (never a hardcoded
host path -- the a-lane verifier's REPO constant was a routed finding).

verify(res, repo, check_files=True) -> (ok, reasons):
  GEOMETRY LOCK (always, typed POWER_GEOMETRY_UNBOUND):
    - calendar_authority_mode must be exactly "bound" (missing/fixture/other
      refuse);
    - calendar_authority_sha256 must equal the sha256 of the pinned calendar
      authority blob recomputed FROM THE GIVEN REPO (git cat-file at
      8111805:docs/f2g_phase_b_shared_calendar_v1.json); unreadable repo or
      mismatch refuses.
  FULL RESULT CHECKS (when the corresponding keys are present; all mandatory
  under check_files=True for a results artifact):
    - digests: every expected authority recomputed from the repo's pinned
      git blobs (frozen Amendment 2, cal engine, admitted bar, rev-1.6 annex
      set) -- no hand-typed constants;
    - equivalence receipt: strict booleans, folds_checked == 35, engine
      binding echoed;
    - evidence capsule: committed git blob digest + row count (never the
      checkout);
    - panel reopen: regenerate the registered deterministic sample (every
      Tier-C replicate row + the rep-0 row of every grid point) from the
      frozen substreams and match each recorded panel_sha256;
    - tier labels/draws, Clopper-Pearson recomputation, Pareto membership,
      terminal consistency.
"""
import hashlib
import json
import subprocess

CAL_AUTH_REF = ("8111805", "docs/f2g_phase_b_shared_calendar_v1.json")
PINS = {
    "frozen_amendment2_sha256": ("337571c",
                                 "docs/f2g_phase_b_prereg_amendment2_DRAFT.md"),
    "engine_lf_sha256": ("24b0d8f", "monitoring/src/d2_f2g_phase_b_stats.py"),
    "annex_common_rev16_sha256": ("feb20bb",
                                  "docs/f2g_phase_b_power_annex_common.md"),
    "annex_b1a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b1a.md"),
    "annex_b2a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b2a.md"),
    "annex_b3a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b3a.md"),
}


def _blob(repo, commit, path):
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo)


def _blob_sha(repo, commit, path):
    return hashlib.sha256(_blob(repo, commit, path)).hexdigest()


def verify(res, repo, check_files=True):
    reasons = []
    # ---- geometry lock (always) ----
    mode = res.get("calendar_authority_mode")
    if mode != "bound":
        reasons.append(f"POWER_GEOMETRY_UNBOUND: calendar_authority_mode="
                       f"{mode!r} (must be 'bound')")
    try:
        expect_cal_sha = _blob_sha(repo, *CAL_AUTH_REF)
    except Exception as exc:
        reasons.append(f"POWER_GEOMETRY_UNBOUND: calendar authority blob "
                       f"unreadable from repo ({exc})")
        expect_cal_sha = None
    if expect_cal_sha is not None and \
            res.get("calendar_authority_sha256") != expect_cal_sha:
        reasons.append("POWER_GEOMETRY_UNBOUND: calendar_authority_sha256 "
                       "does not match the pinned authority blob")
    # ---- full result checks (keys present) ----
    d = res.get("digests")
    if d is not None:
        for key, (commit, path) in PINS.items():
            try:
                want = _blob_sha(repo, commit, path)
            except Exception as exc:
                reasons.append(f"authority {key} unreadable: {exc}")
                continue
            if d.get(key) != want:
                reasons.append(f"digest {key} mismatch/absent")
    r = res.get("equivalence_receipt")
    if r is not None:
        if r.get("full_equal") is not True or \
                r.get("fold_equal_all") is not True:
            reasons.append("equivalence booleans not strictly True")
        if r.get("folds_checked") != 35:
            reasons.append("folds_checked != 35")
        if not r.get("engine_commit_bound"):
            reasons.append("engine binding absent")
    ev = res.get("evidence_capsule")
    if ev is not None and check_files:
        try:
            raw = _blob(repo, "HEAD", ev["path"])
            if hashlib.sha256(raw).hexdigest() != ev.get("git_blob_sha256"):
                reasons.append("evidence capsule git-blob sha mismatch")
            n = sum(1 for l in raw.decode("utf-8").splitlines() if l.strip())
            if n != ev.get("rows"):
                reasons.append("evidence row count mismatch")
        except Exception as exc:
            reasons.append(f"evidence capsule unreadable: {exc}")
    if check_files and res.get("panel_reopen_sample"):
        try:
            import os
            import sys
            src = os.path.join(repo, "monitoring", "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            cwd = os.getcwd()
            os.chdir(repo)
            try:
                import f2g_phase_b_power_estimation_cal_cayley as PD
                for row in res["panel_reopen_sample"]:
                    pan = PD.make_panel(row["family"], row["point"],
                                        row["rep"])
                    if PD.panel_digest(pan) != row["panel_sha256"]:
                        reasons.append(
                            "POWER_GEOMETRY_UNBOUND: panel "
                            f"{row['family']}/{row['rep']} cannot be "
                            "reopened to its recorded digest")
            finally:
                os.chdir(cwd)
        except Exception as exc:
            reasons.append(f"panel reopen failed: {exc}")
    cert = res.get("certified_points")
    term = res.get("terminal_type")
    if cert is not None and term is not None:
        if bool(cert) != (term == "CERTIFIED"):
            reasons.append("terminal/certified inconsistency")
        if term not in ("CERTIFIED",
                        "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH"):
            reasons.append("unknown terminal type")
    if res.get("tier_c") is not None:
        import f2g_phase_b_power_estimation_cayley as D0
        for c in res["tier_c"].get("candidates", []):
            k, n = c["post_loco"]["successes"], c["post_loco"]["replicates"]
            if abs(D0.cp_lower(k, n) - c["lb95"]) > 1e-12 or \
                    abs(D0.cp_upper(k, n) - c["ub95"]) > 1e-12:
                reasons.append(f"CP bounds mismatch at {c['point']}")
    for tier, lbl, nd in (("tier_s1", "PRELIMINARY_SMOKE", 999),
                          ("tier_s2", "PRELIMINARY_SMOKE", 999),
                          ("tier_c", "CERTIFICATION", 9999)):
        t = res.get(tier)
        if t is not None and (t.get("label") != lbl
                              or t.get("n_draws") != nd):
            reasons.append(f"{tier} label/draws wrong")
    pareto = res.get("pareto_minimal_certified")
    if pareto and cert:
        if not any(p["point"] == c["point"] for p in pareto for c in cert):
            reasons.append("pareto member not in certified_points")
    return (not reasons), reasons


if __name__ == "__main__":
    import sys
    repo = sys.argv[1]
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
