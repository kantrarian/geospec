#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 EXECUTION-MANIFEST v1 generator (cayley). Schema contract:
docs/f2g_window2_execution/execution_manifest_schema_v1.md.

Binds the executable surface: BOUND slots pin (path, last-touch commit
resolved FROM the declared execution target, blob sha256); OPEN slots
carry owner + note and bind nothing. The manifest links to the CLOSED
design manifest (commit + blob sha + design target echo). manifest_state
is computed, never hand-typed.

Usage: gen.py <repo> <execution-target-commit> <design-manifest-commit>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

MANIFEST_PATH = "docs/f2g_window2_execution/execution_manifest.json"
DESIGN_MANIFEST_PATH = "docs/f2g_window2_freeze/byte_pin_manifest.json"
SCHEMA = "f2g-window2-execution-manifest-v1"

BOUND_SLOTS = {
    "execution_generator": {
        "owner": "cayley",
        "note": "this generator",
        "paths": ["monitoring/src/f2g_execution_manifest_gen_cayley.py"]},
    "execution_verifier": {
        "owner": "cayley",
        "note": "executable verifier (design linkage + slot/pin walk + "
                "prestart gate)",
        "paths": ["monitoring/src/f2g_execution_manifest_verifier_cayley.py"]},
    "design_pin_verifier": {
        "owner": "cayley",
        "note": "design-pin walk executable (landed b755ce1)",
        "paths": ["monitoring/src/f2g_design_pin_verifier_cayley.py"]},
    # bound at bar-sweep completion (grassmann 0916Z: 11/11 green,
    # "execution-manifest-v1 can pin the complete bar") per the
    # bind-on-bar-green policy
    "selection_impl": {
        "owner": "cayley",
        "note": "cutoff-stable selection REV 2 (codex 3xP1 repaired; "
                "W-SEL-b green incl production locks)",
        "paths": ["monitoring/src/w2_selection.py"]},
    "adapter_impl": {
        "owner": "cayley",
        "note": "family engines + registry resolution (W-CAS/W-B2B/"
                "W-B1B green; panel assembly pinned under accrual)",
        "paths": ["monitoring/src/w2_cascadia.py",
                  "monitoring/src/w2_b2b.py",
                  "monitoring/src/w2_b1b.py"]},
    "accrual_impl": {
        "owner": "cayley",
        "note": "barrier state machine + production instrument (core + "
                "3 seam layers) + accrual-lane engines (W-BARRIER/"
                "W-MF4/W-MAG green)",
        "paths": ["monitoring/src/w2_barrier.py",
                  "monitoring/src/w2_accrual_instrument_cayley.py",
                  "monitoring/src/w2_mf4.py",
                  "monitoring/src/w2_mag1.py"]},
    "mag_capsules": {
        "owner": "cayley",
        "note": "IZN/FRN/TUC typed capsules + probe bodies + envelopes "
                "(VIC/NEW already design-pinned; W-MAG frame paths "
                "green)",
        "paths": ["docs/f2g_window2_freeze/mag_capsule_izn.json",
                  "docs/f2g_window2_freeze/mag_capsule_frn.json",
                  "docs/f2g_window2_freeze/mag_capsule_tuc.json",
                  "docs/f2g_window2_freeze/receipts/mag_izn_probe.json",
                  "docs/f2g_window2_freeze/receipts/"
                  "mag_izn_probe.envelope.json",
                  "docs/f2g_window2_freeze/receipts/mag_frn_probe.json",
                  "docs/f2g_window2_freeze/receipts/"
                  "mag_frn_probe.envelope.json",
                  "docs/f2g_window2_freeze/receipts/mag_tuc_probe.json",
                  "docs/f2g_window2_freeze/receipts/"
                  "mag_tuc_probe.envelope.json"]},
    "bars": {
        "owner": "grassmann",
        "note": "the complete window-2 bar, 11/11 green @ 8aecf96 "
                "(grassmann's 0916Z completion declaration)",
        "paths": ["monitoring/src/"
                  "test_f2g_window2_redkats_grassmann.py"],
        "families": ["W-SEL", "W-CAS", "W-B2B", "W-B1B", "W-MF4",
                     "W-MAG", "W-BARRIER", "W-PIN"]},
}
# NOTE: the schema doc is design-adjacent prose, deliberately NOT a slot

OPEN_SLOTS = {
    "calibration_ledgers": ("cayley", "MF4/MAG subtraction "
                                      "coefficients + diagnostics -- "
                                      "PRODUCED at the availability "
                                      "cutoff, pre-evaluation only"),
    "producer_code": ("grassmann", "accrual producers (seismic + MAG "
                                   "raw byte acquisition; s4t-lane "
                                   "build next per 0916Z)"),
}


def _git(repo, args, binary=False):
    out = subprocess.check_output(["git", "-C", repo] + args)
    return out if binary else out.decode().strip()


def pin(repo, target_full, path):
    commit = _git(repo, ["log", "-1", "--format=%H", target_full, "--",
                         path])
    if not commit:
        raise SystemExit(f"PATH_NOT_AT_TARGET: {path}")
    subprocess.check_call(["git", "-C", repo, "merge-base",
                           "--is-ancestor", commit, target_full])
    blob = _git(repo, ["cat-file", "blob", f"{commit}:{path}"],
                binary=True)
    return {"path": path, "commit": commit,
            "blob_sha256": hashlib.sha256(blob).hexdigest()}


def main(repo, target, design_manifest_commit):
    target_full = _git(repo, ["rev-parse", f"{target}^{{commit}}"])
    dm_full = _git(repo, ["rev-parse",
                          f"{design_manifest_commit}^{{commit}}"])
    dm_blob = _git(repo, ["cat-file", "blob",
                          f"{dm_full}:{DESIGN_MANIFEST_PATH}"],
                   binary=True)
    dm_obj = json.loads(dm_blob.decode("utf-8"))

    slots = {}
    for name, spec in BOUND_SLOTS.items():
        slots[name] = {"status": "BOUND", "owner": spec["owner"],
                       "note": spec["note"],
                       "pins": [pin(repo, target_full, p)
                                for p in spec["paths"]]}
        if "families" in spec:
            slots[name]["families"] = list(spec["families"])
    for name, (owner, note) in OPEN_SLOTS.items():
        slots[name] = {"status": "OPEN", "owner": owner, "note": note,
                       "pins": []}
    state = "CLOSED" if all(s["status"] == "BOUND"
                            for s in slots.values()) else "OPEN"

    out = {"schema": SCHEMA,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "repository_url": "https://github.com/kantrarian/geospec",
           "execution_target_commit": target_full,
           "target_ref": "origin/master",
           "design_manifest_commit": dm_full,
           "design_manifest_blob_sha256":
               hashlib.sha256(dm_blob).hexdigest(),
           "design_target_commit": dm_obj["design_target_commit"],
           "manifest_state": state,
           "slots": slots}
    p = os.path.join(repo, MANIFEST_PATH.replace("/", os.sep))
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    bound = sum(1 for s in slots.values() if s["status"] == "BOUND")
    print(f"wrote {MANIFEST_PATH}: state={state}, "
          f"{bound}/{len(slots)} slots bound @ {target_full[:12]}")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3])
