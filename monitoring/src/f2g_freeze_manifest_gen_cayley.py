#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 BYTE-PIN MANIFEST generator (prereg v0.3 sec 7 / codex R2
fix 4). Every inherited dependency pinned as (repo-relative path, the
LAST commit touching it, the blob sha256 at that commit). Nothing
hand-typed: commits and hashes are resolved from git at generation time.
The freeze verifier reopens these bytes and refuses on absence,
mismatch, dirty substitution, or unlisted dependency.
Usage: gen.py <repo>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

DEPS = {
    # the design authorities
    "prereg_v03": "docs/f2g_window2_prereg_DRAFT.md",
    "mag1_design_v02": "docs/mag1_channel_design_DRAFT.md",
    "mag1_coverage_admission": "docs/mag1_coverage_admission_v1.md",
    # the freeze package part 1
    "selection_constants": "docs/f2g_window2_freeze/selection_constants.md",
    "cascadia_capsule": "docs/f2g_window2_freeze/cascadia_carrier_capsule.md",
    "cascadia_receipt": "docs/f2g_window2_freeze/receipts/cascadia_UW_CC_CN_HHZ.txt",
    "annex_b2b": "docs/f2g_window2_freeze/annex_b2b.md",
    "annex_b1b": "docs/f2g_window2_freeze/annex_b1b.md",
    "annex_mf4": "docs/f2g_window2_freeze/annex_mf4.md",
    "mag1_instantiation": "docs/f2g_window2_freeze/mag1_instantiation.md",
    # inherited executable semantics (B2A/B3A engine, run instrument)
    "engine_b2a_b3a": "monitoring/src/d2_f2g_phase_b_stats.py",
    "sealed_driver": "monitoring/src/f2g_sealed_run_driver_cayley.py",
    "sealed_instrument": "monitoring/src/f2g_sealed_run_instrument_cayley.py",
    "sealed_verifier": "monitoring/src/f2g_sealed_run_result_verifier_cayley.py",
    # phase-b conventions + region/pool authorities
    "phaseb_annex_common_rev16": "docs/f2g_phase_b_power_annex_common.md",
    "region_polygons": "monitoring/src/fault_segments.py",
    "candidate_pool": "monitoring/src/d2_campaign_v2_candidate_pool.json",
    # provenance (non-quantitative pilot record)
    "poc_findings_nonquant": "docs/f2g_poc_review/findings_v1.md",
}


def main(repo):
    os.chdir(repo)
    pins = {}
    for key, path in sorted(DEPS.items()):
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", path]).decode().strip()
        blob = subprocess.check_output(
            ["git", "cat-file", "blob", f"{commit}:{path}"])
        pins[key] = {"path": path, "commit": commit,
                     "blob_sha256": hashlib.sha256(blob).hexdigest()}
    out = {"schema": "f2g-window2-byte-pin-manifest-v1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "rule": "the freeze verifier reopens every pinned blob and "
                   "refuses on absence, mismatch, dirty substitution, or "
                   "any unlisted dependency; later edits to any pinned "
                   "path do not alter the registered method",
           "pins": pins}
    p = "docs/f2g_window2_freeze/byte_pin_manifest.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {p} ({len(pins)} pins)")
    for k, v in sorted(pins.items()):
        print(f"  {k}: {v['commit'][:9]} {v['blob_sha256'][:12]}...")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "."))
