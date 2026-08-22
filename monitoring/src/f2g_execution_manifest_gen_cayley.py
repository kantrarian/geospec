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
}
# NOTE: the schema doc is design-adjacent prose, deliberately NOT a slot

OPEN_SLOTS = {
    "selection_impl": ("cayley", "cutoff-stable selection per prereg "
                                 "v0.3 + selection_constants.md"),
    "adapter_impl": ("cayley", "window-2 family adapter (B2A/B2B/B1B/"
                               "B3A) over the frozen graph"),
    "accrual_impl": ("cayley", "sealed prediction accrual + two-stage "
                               "barrier instruments"),
    "mag_capsules": ("cayley", "typed at-freeze station capsules IZN/"
                               "FRN/TUC (VIC/NEW already design-pinned)"),
    "calibration_ledgers": ("cayley", "MAG-1 subtraction coefficients + "
                                      "diagnostics, pre-evaluation"),
    "bars": ("grassmann", "executable bar file(s); families must equal "
                          "the required 8 when BOUND"),
    "producer_code": ("grassmann", "accrual producers (seismic + MAG "
                                   "raw byte acquisition)"),
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
