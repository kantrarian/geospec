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
    "cascadia_receipt_envelope": "docs/f2g_window2_freeze/receipts/cascadia_UW_CC_CN_HHZ.envelope.json",
    "phase_a_graph_builder": "monitoring/src/d2_f2g_graph_builder.py",
    "mag1_band_b_sos": "docs/f2g_window2_freeze/mag1_band_b_sos.json",
    "mag_vic_probe_body": "docs/f2g_window2_freeze/receipts/mag_vic_probe.json",
    "mag_vic_probe_envelope": "docs/f2g_window2_freeze/receipts/mag_vic_probe.envelope.json",
    "mag_vic_capsule": "docs/f2g_window2_freeze/mag_capsule_vic.json",
    "mag_new_probe_body": "docs/f2g_window2_freeze/receipts/mag_new_probe.json",
    "mag_new_probe_envelope": "docs/f2g_window2_freeze/receipts/mag_new_probe.envelope.json",
    "mag_new_capsule": "docs/f2g_window2_freeze/mag_capsule_new.json",
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


OBJECT_ID = {".md": "markdown-contract", ".json": "json-object",
             ".py": "python-source", ".txt": "fdsn-station-text"}


def main(repo, target):
    os.chdir(repo)
    target_full = subprocess.check_output(
        ["git", "rev-parse", f"{target}^{{commit}}"]).decode().strip()
    pins = {}
    for key, path in sorted(DEPS.items()):
        # codex revision-1 fix 3: resolve last-touch FROM THE DECLARED
        # TARGET, never from whatever HEAD runs the generator
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", target_full, "--",
             path]).decode().strip()
        subprocess.check_call(["git", "merge-base", "--is-ancestor",
                               commit, target_full])
        blob = subprocess.check_output(
            ["git", "cat-file", "blob", f"{commit}:{path}"])
        ext = os.path.splitext(path)[1]
        pins[key] = {"path": path, "commit": commit,
                     "blob_sha256": hashlib.sha256(blob).hexdigest(),
                     "object_id": OBJECT_ID.get(ext, "bytes"),
                     "imported_section": "whole-file",
                     "imported_section_sha256":
                         hashlib.sha256(blob).hexdigest()}
    out = {"schema": "f2g-window2-design-manifest-v2.1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "repository_url": "https://github.com/kantrarian/geospec",
           "design_target_commit": target_full,
           "target_ref": "origin/master",
           "pin_count": len(pins),
           "manifest_class": "DESIGN manifest (codex freeze-review fix 2): "
                             "pins the registered design/contract bytes. A "
                             "closed execution-manifest-v1 (generator/"
                             "verifier, selection + adapter + accrual "
                             "implementations, typed MAG capsules, "
                             "calibration ledgers, bars, producer code) plus "
                             "an executable verifier that reopens every pin, "
                             "checks repository identity + the execution "
                             "allowlist, and refuses missing/mismatched/"
                             "dirty/unlisted inputs are REQUIRED PRESTART "
                             "deliverables. Until that verifier exists, a "
                             "listed-pin integrity walk is not a "
                             "fire-verifier PASS.",
           "pins": pins}
    p = "docs/f2g_window2_freeze/byte_pin_manifest.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {p} ({len(pins)} pins)")
    for k, v in sorted(pins.items()):
        print(f"  {k}: {v['commit'][:9]} {v['blob_sha256'][:12]}...")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), sys.argv[2])
