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
SCHEMA = "f2g-window2-execution-manifest-v1.2"

BOUND_SLOTS = {
    "execution_generator": {
        "owner": "cayley",
        "note": "this generator",
        "paths": ["monitoring/src/f2g_execution_manifest_gen_cayley.py"]},
    "execution_verifier": {
        "owner": "cayley",
        "note": "executable verifier (design linkage + slot/pin walk + "
                "prestart gate) and the post-regeneration gate. codex "
                "0445Z item 5: verification CODE is its own authority, "
                "distinct from production operation bytes and from the "
                "execution records -- the gate must be pinned by the "
                "manifest it audits, or the check that certifies the "
                "pin set is itself unbound",
        "paths": ["monitoring/src/f2g_execution_manifest_verifier_cayley.py",
                  "monitoring/src/w2_regeneration_gate_cayley.py"]},
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
                "W-MF4/W-MAG green) + the FROZEN v3 static-contract "
                "authority and its generator (codex 2205Z item 1: "
                "capture admission requires the authority as a pin of "
                "THIS slot; producer_boundary additionally binds it "
                "at final boundary bind)",
        "paths": ["monitoring/src/w2_barrier.py",
                  "monitoring/src/w2_accrual_instrument_cayley.py",
                  "monitoring/src/w2_mf4.py",
                  "monitoring/src/w2_mag1.py",
                  "monitoring/src/w2_expected_contracts_gen_cayley.py",
                  "docs/f2g_window2_execution/"
                  "staged_expected_contracts_v3.json",
                  # the 2056-key three-way disposition capsule +
                  # its generator/verifier: capture_authorized
                  # reopens THIS pin and requires HTTP_CAPTURE
                  # membership before the opener, so an unpinned
                  # capsule fails the entrypoint closed
                  "monitoring/src/w2_disposition_capsule_grassmann.py",
                  "docs/f2g_window2_execution/"
                  "key_disposition_capsule_v4.json",
                  # the two-leg RESTAGED_LINEAGE verifier: my
                  # boundary calls it, so it must be reopenable
                  # from a pin rather than merely imported
                  "monitoring/src/w2_restage_lineage_grassmann.py",
                  # codex 0404Z item 1: the offline restager is part
                  # of the production OPERATION record, so it binds
                  # here and NOT in bars -- one path must not answer
                  # to two slot authorities
                  "monitoring/src/w2_restage_v4_grassmann.py",
                  # codex 0445Z item 2: the tools that PRODUCE the two
                  # execution records must themselves be bound, or a
                  # receipt could be emitted by code the manifest
                  # never pinned -- the record would authenticate its
                  # contents while its producer stayed unauthenticated
                  "monitoring/src/w2_restage_verify_batch_grassmann.py",
                  "monitoring/src/"
                  "w2_verification_run_summary_grassmann.py",
                  # codex 1716Z P0-2: two RUNTIME DEPENDENCIES of the
                  # admission path that were pinned nowhere at all.
                  # The sentinel produces a MEASURED fact in the
                  # pre-manifest record and is called by the batch and
                  # summary producers already bound here -- under
                  # execution_verifier the manifest-certifying
                  # authority would own a measurement belonging to a
                  # different producer authority. w2_producer is a
                  # runtime dependency of my own instrument and of
                  # restage lineage; producer_boundary is the LATER
                  # staged-envelope trust boundary, and pre-stage
                  # acquisition source there would contradict its rule
                  # that acquisition before staged bytes is
                  # receipt-attested, not source-code-attested.
                  "monitoring/src/w2_no_network_grassmann.py",
                  "monitoring/src/w2_producer_grassmann.py"]},
    # v1.1 (codex 1358Z item 4/5): the two repaired execution tools
    # join the runtime allowlist as explicit slots
    "power_harness": {
        "owner": "cayley",
        "note": "sec-6 power machinery (certification path constructs "
                "its own config; bound-geometry gate; codex items 1+2 "
                "repaired) + the campaign runner REV 2 and Tier-C "
                "selector with their registered amendments (codex "
                "1909Z items 1-4)",
        "paths": ["monitoring/src/w2_power_harness_cayley.py",
                  "docs/f2g_window2_execution/"
                  "loco_composition_amendment_v1.md",
                  "monitoring/src/w2_cert_runner_cayley.py",
                  "monitoring/src/w2_tier_selector_cayley.py",
                  "docs/f2g_window2_execution/"
                  "tier_selector_amendment_w2_v1.md",
                  "docs/f2g_window2_execution/"
                  "effect_grids_w2_v1.json",
                  "monitoring/src/w2_effect_grids_gen_cayley.py"]},
    "calibration_runner": {
        "owner": "cayley",
        "note": "calibration-ledger production runner (temporal "
                "boundary + M3 index equality + provenance-verifying "
                "receipts; codex items 3+4 repaired + canonical-UTC "
                "frame)",
        "paths": ["monitoring/src/w2_calibration_runner_cayley.py"]},
    # RE-BOUND ATOMICALLY at bar REV 9 green (codex 1335Z/1358Z
    # disposition; grassmann 1745Z, 14/14, dual-verified by cayley on
    # py3.14+py3.11)
    "mag_capsules": {
        "owner": "cayley",
        "note": "IZN/FRN/TUC capsules + bodies + envelopes at the "
                "RELOCATED execution tree (codex 0451Z); loaded via "
                "load_execution_capsule under the dynamic manifest "
                "authority; W-MAG-EXEC green",
        "paths": ["docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_izn.json",
                  "docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_frn.json",
                  "docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_tuc.json",
                  "docs/f2g_window2_execution/mag_capsules/receipts/"
                  "mag_izn_probe.json",
                  "docs/f2g_window2_execution/mag_capsules/receipts/"
                  "mag_izn_probe.envelope.json",
                  "docs/f2g_window2_execution/mag_capsules/receipts/"
                  "mag_frn_probe.json",
                  "docs/f2g_window2_execution/mag_capsules/receipts/"
                  "mag_frn_probe.envelope.json",
                  "docs/f2g_window2_execution/mag_capsules/receipts/"
                  "mag_tuc_probe.json",
                  "docs/f2g_window2_execution/mag_capsules/receipts/"
                  "mag_tuc_probe.envelope.json"]},
    "bars": {
        "owner": "grassmann",
        # codex 0404Z item 2: SCOPE ONLY. The previous note asserted
        # "REV 19, 18/18 green @ 2309ab7" -- wrong in three ways at
        # HEAD, and the dangerous one was that it described a
        # boundary-admission result the pinned bytes no longer
        # establish. No REV number, group count, commit nickname or
        # "green" assertion belongs here: pinning bytes does not, by
        # that act, prove those bytes were executed successfully, and
        # static prose making execution claims goes stale silently and
        # survives a zero-stale-pin regeneration. Time-varying
        # execution facts live ONLY in the pinned run summary.
        "note": "Window-2 verification surfaces. The shared bar's "
                "W2-ADMIT group is STRUCTURAL_KAT_ONLY, not closure-4 "
                "PASS; it carries admission_eligible=false and no "
                "production boundary digest or proof kinds. Closure-4 "
                "635/1420/1 and capsule pin binding are established "
                "by separately pinned locks. Other separately pinned "
                "locks cover the admitted partition, "
                "ADMITTED_ABSENCE, whole-authority serving, the "
                "frozen carrier set, and fixture/production schema "
                "separation. Interpreter identities and PRE-MANIFEST "
                "execution outcomes live only in the pinned compact "
                "run summary. Manifest-owned admission verification "
                "cannot appear in that summary at all: it can only "
                "run against the finished manifest commit, so a "
                "summary pinned inside that same manifest could never "
                "honestly contain its own post-manifest result. Those "
                "outcomes exist solely in the separate downstream "
                "post-manifest verification receipt, whose authority "
                "is the manifest it names rather than a pin the "
                "manifest recursively holds.",
        # codex 0404Z item 1: the COMPLETE verification record. The
        # shared bar alone stopped being sufficient the moment
        # W2-ADMIT was relabelled STRUCTURAL_KAT_ONLY -- after that,
        # closure 4 and the capsule pin-bind rest solely on the locks
        # below, and unbound evidence cannot support a packet claim.
        "paths": ["monitoring/src/"
                  "test_f2g_window2_redkats_grassmann.py",
                  # closure 4 (635/1420/1) and the P0 pin-bind: the
                  # two claims the shared bar no longer makes
                  "monitoring/src/"
                  "test_w2_capsule_pin_bind_redkats_cayley.py",
                  "monitoring/src/"
                  "test_w2_report_proof_kinds_redkats_cayley.py",
                  "monitoring/src/"
                  "test_w2_boundary_admitted_partition_redkats_cayley"
                  ".py",
                  "monitoring/src/"
                  "test_w2_admitted_absence_redkats_cayley.py",
                  "monitoring/src/"
                  "test_w2_authority_serves_every_key_redkats_cayley"
                  ".py",
                  "monitoring/src/"
                  "test_w2_frozen_carrier_set_redkats_cayley.py",
                  # grassmann's item-3 lock; lands before the single
                  # regeneration, which is when this list is read
                  "monitoring/src/"
                  "test_w2_fixture_schema_redkats_grassmann.py",
                  # the execution record itself: argv, interpreter,
                  # exit code, typed verdict per invocation, with
                  # PASS / COVERED_ELSEWHERE / REFUSE / NOT_RUN kept
                  # distinct so a missing input is a typed failure and
                  # never a green skip
                  "docs/f2g_window2_execution/"
                  "w2_verification_run_summary_v1.json"],
        "families": ["W-SEL", "W-CAS", "W-B2B", "W-B1B", "W-MF4",
                     "W-MAG", "W-BARRIER", "W-PIN"]},
}
# NOTE: the schema doc is design-adjacent prose, deliberately NOT a slot

OPEN_SLOTS = {
    "calibration_ledgers": ("cayley", "MF4/MAG subtraction "
                                      "coefficients + diagnostics -- "
                                      "PRODUCED at the availability "
                                      "cutoff, pre-evaluation only"),
    # v1.2 (codex 1400Z ruling 2, option (ii)): producer_code is
    # RENAMED producer_boundary -- the STAGED-ENVELOPE trust boundary
    # per producer_boundary_amendment_v1.md; grassmann's ratification
    # + real staging BIND it (amendment + envelope-verifier code +
    # per-lane/per-day envelope records + claim ceiling)
    "producer_boundary": ("grassmann",
                          "staged-envelope boundary (amendment v1); "
                          "OPEN until grassmann ratifies the mode "
                          "and stages: BOUND requires the amendment "
                          "doc + envelope verifier + transform code "
                          "+ envelope records; acquisition before "
                          "the staged bytes is receipt-attested, "
                          "never source-code-attested"),
}
PRODUCER_BOUNDARY_MODE = "staged_envelope"


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
    # the registered boundary mode rides the slot object itself
    slots["producer_boundary"]["boundary_mode"] = \
        PRODUCER_BOUNDARY_MODE
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
