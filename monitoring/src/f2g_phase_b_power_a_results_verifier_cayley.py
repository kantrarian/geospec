#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict amended-results verifier (codex repair 3) -- FIXTURE-ONLY.

Independent, predeclared-schema verifier for the amended-family results
artifacts. Deliberately SEPARATE from the admitted bar's _results_bound
(which binds the superseded-family authorities by design and must return
False on these artifacts). Checks per artifact:
  1. exact amended authorities in digests (frozen amendment sha, engine
     493c2b9 LF sha, admitted bar LF sha, all four annex shas);
  2. strict booleans in equivalence_receipt (full_equal / fold_equal_all is
     True, folds_checked == 35, engine_commit_bound == '493c2b9', receipt
     file exists and matches its bound sha);
  3. evidence capsule exists, sha matches, row count matches;
  4. family/terminal consistency (certified_points nonempty <=> terminal
     CERTIFIED) and pareto membership in certified_points;
  5. tier labels + draw counts (S tiers PRELIMINARY_SMOKE @ 999, C tier
     CERTIFICATION @ 9999);
  6. Clopper-Pearson recomputation of every Tier-C lb95/ub95 from
     successes/replicates (exact);
  7. Pareto recomputation for 1-D B2A (min m among certified);
  8. NEGATIVE fixtures: a tampered copy (fake authority / stringly boolean /
     wrong LB) MUST refuse.
Exit 0 only if all artifacts pass and all negative fixtures refuse.
"""
import copy
import hashlib
import json
import subprocess
import sys

import f2g_phase_b_power_estimation_cayley as D0

REPO = "C:/geospec"


def _blob_sha(commit, path):
    """Recompute an authority digest from the pinned git blob -- NEVER a
    hand-typed constant (the transcription defect class), and independent of
    the producer's own DIGESTS map (no self-consistency)."""
    raw = subprocess.check_output(["git", "cat-file", "blob",
                                   f"{commit}:{path}"], cwd=REPO)
    return hashlib.sha256(raw).hexdigest()


EXPECT_DIGESTS = {
    "frozen_amendment_sha256": _blob_sha(
        "7c3ca7b", "docs/f2g_phase_b_prereg_amendment1_DRAFT.md"),
    "engine_lf_sha256": _blob_sha(
        "493c2b9", "monitoring/src/d2_f2g_phase_b_stats.py"),
    "admitted_bar_lf_sha256": _blob_sha(
        "ff211ca", "monitoring/src/test_f2g_phase_b_stats_redkats_grassmann.py"),
    "annex_common_rev13_sha256": _blob_sha(
        "60ea20e", "docs/f2g_phase_b_power_annex_common.md"),
    "annex_b1a_sha256": _blob_sha(
        "a0cc87c", "docs/f2g_phase_b_power_annex_b1a.md"),
    "annex_b2a_sha256": _blob_sha(
        "a0cc87c", "docs/f2g_phase_b_power_annex_b2a.md"),
    "annex_b3a_sha256": _blob_sha(
        "a0cc87c", "docs/f2g_phase_b_power_annex_b3a.md"),
}


def verify(res, repo=REPO, check_files=True):
    errs = []
    d = res.get("digests") or {}
    for k, v in EXPECT_DIGESTS.items():
        if d.get(k) != v:
            errs.append(f"digest {k} mismatch/absent")
    r = res.get("equivalence_receipt") or {}
    if r.get("full_equal") is not True or r.get("fold_equal_all") is not True:
        errs.append("equivalence booleans not strictly True")
    if r.get("folds_checked") != 35:
        errs.append("folds_checked != 35")
    if r.get("engine_commit_bound") != "493c2b9":
        errs.append("engine_commit_bound != 493c2b9")
    ev = res.get("evidence_capsule") or {}
    if check_files:
        try:
            raw = open(f"{repo}/{r['receipt_path']}", "rb").read()
            if hashlib.sha256(raw).hexdigest() != r.get("receipt_sha256"):
                errs.append("fullgate receipt sha mismatch")
        except Exception:
            errs.append("fullgate receipt unreadable")
        try:
            # codex 22:32Z repair 1: verify against the COMMITTED GIT BLOB,
            # never the checkout (CRLF conversion must not be able to pass)
            raw = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{ev['path']}"], cwd=repo)
            if hashlib.sha256(raw).hexdigest() != ev.get("git_blob_sha256"):
                errs.append("evidence capsule git-blob sha mismatch")
            n = sum(1 for l in raw.decode("utf-8").splitlines() if l.strip())
            if n != ev.get("rows"):
                errs.append("evidence row count mismatch")
        except Exception:
            errs.append("evidence capsule unreadable")
    cert = res.get("certified_points") or []
    term = res.get("terminal_type")
    if bool(cert) != (term == "CERTIFIED"):
        errs.append("terminal/certified inconsistency")
    if term not in ("CERTIFIED", "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH"):
        errs.append("unknown terminal type")
    for tier, lbl, nd in (("tier_s1", "PRELIMINARY_SMOKE", 999),
                          ("tier_s2", "PRELIMINARY_SMOKE", 999),
                          ("tier_c", "CERTIFICATION", 9999)):
        t = res.get(tier) or {}
        if t.get("label") != lbl or t.get("n_draws") != nd:
            errs.append(f"{tier} label/draws wrong")
    for c in (res.get("tier_c") or {}).get("candidates", []):
        k, n = c["post_loco"]["successes"], c["post_loco"]["replicates"]
        if abs(D0.cp_lower(k, n) - c["lb95"]) > 1e-12 or \
           abs(D0.cp_upper(k, n) - c["ub95"]) > 1e-12:
            errs.append(f"CP bounds mismatch at {c['point']}")
        want = ("CERTIFIED" if D0.cp_lower(k, n) >= 0.80 else
                ("FAILED" if D0.cp_upper(k, n) < 0.80 else
                 "CANNOT_DETERMINE_POWER_ESTIMATE"))
        if n in (20, 40) and c["verdict"] != want and not (
                n == 20 and want == "CANNOT_DETERMINE_POWER_ESTIMATE"):
            errs.append(f"verdict/bounds inconsistency at {c['point']}")
    pareto = res.get("pareto_minimal_certified") or []
    if res.get("family") == "B2A" and cert:
        want_min = min(c["point"]["m"] for c in cert)
        if not pareto or pareto[0]["point"]["m"] != want_min:
            errs.append("Pareto recomputation mismatch")
    if pareto and not any(p["point"] == c["point"] for p in pareto
                          for c in cert):
        errs.append("pareto member not in certified_points")
    return errs


def main():
    docs = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/docs"
    ok = True
    arts = {}
    for f in ("b1a", "b2a", "b3a"):
        path = f"{docs}/f2g_phase_b_power_annex_{f}_results.json"
        res = json.loads(open(path, encoding="utf-8").read())
        arts[f] = res
        errs = verify(res)
        print(f"[{f}] {'PASS' if not errs else 'REFUSE: ' + '; '.join(errs)}")
        ok = ok and not errs
    # negative fixtures (must refuse)
    neg = []
    t1 = copy.deepcopy(arts["b2a"])
    t1["digests"]["engine_lf_sha256"] = "0" * 64
    neg.append(("fake-authority", t1))
    t2 = copy.deepcopy(arts["b2a"])
    t2["equivalence_receipt"]["fold_equal_all"] = "true"
    neg.append(("stringly-boolean", t2))
    t3 = copy.deepcopy(arts["b2a"])
    t3["tier_c"]["candidates"][0]["lb95"] = 0.999
    neg.append(("wrong-LB", t3))
    t4 = copy.deepcopy(arts["b3a"])
    t4["terminal_type"] = "CERTIFIED"
    neg.append(("terminal-flip", t4))
    t5 = copy.deepcopy(arts["b2a"])
    blob = subprocess.check_output(
        ["git", "cat-file", "blob",
         f"HEAD:{t5['evidence_capsule']['path']}"], cwd=REPO)
    crlf = blob.replace(b"\n", b"\r\n")
    t5["evidence_capsule"]["git_blob_sha256"] = hashlib.sha256(
        crlf).hexdigest()
    neg.append(("crlf-checkout-hash", t5))
    for name, t in neg:
        # the CRLF fixture must exercise the file/blob path; the others are
        # structural and skip file IO
        errs = verify(t, check_files=(name == "crlf-checkout-hash"))
        refused = bool(errs)
        print(f"[neg:{name}] {'REFUSED (correct)' if refused else 'ACCEPTED -- VERIFIER DEFECT'}")
        ok = ok and refused
    print("VERIFIER:", "ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
