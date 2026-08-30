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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# codex 1547Z repair 1: the ADMISSION CONSUMER owns the staged-prefix
# and operation-evidence path constants; this generator IMPORTS them,
# so a carrier rename cannot close in the generator while the
# consumer still refuses it (the exact defect codex reproduced).
import w2_accrual_instrument_cayley as _ACCM

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
                  "monitoring/src/w2_regeneration_gate_cayley.py",
                  # codex 0532Z: the pre-fire frame-readiness doctor.
                  # VERIFICATION authority, not operation bytes: the
                  # capture runner and RG-10 import the SAME one pinned
                  # verifier, so the readiness claim and its
                  # enforcement cannot drift apart. Binding it under
                  # accrual_impl would put a verification surface under
                  # an operation authority.
                  "monitoring/src/w2_frame_readiness_doctor_cayley.py"]},
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
                  "monitoring/src/w2_producer_grassmann.py",
                  # codex 0151Z P0-1: the executable that will
                  # FIRE the 635 was pinned nowhere. A clean
                  # plan produced by unbound working-tree code
                  # is not a reviewed capture executable.
                  "monitoring/src/w2_capture_run_v4_grassmann.py",
                  # codex 0413Z circular-pin repair: the registered
                  # lane transform is PRODUCTION-OPERATION code used
                  # by the predecessor bridge during plan, before the
                  # later producer_boundary can truthfully bind. Bind
                  # its operation role here; final-bind additionally
                  # binds the same file's boundary-code role, matching
                  # the existing authority-file dual-role pattern.
                  "monitoring/src/"
                  "w2_acquisition_capture_grassmann.py",
                  # P0-4 amended lane (the retry precedent): the
                  # ComCat acquisition is a NETWORK-SPENDING
                  # executable and may not exist unpinned; its own
                  # authority chain gates any fire
                  "monitoring/src/"
                  "w2_mf4_catalog_acquire_grassmann.py",
                  # codex 0532Z: the zero-HTTP VIC repair driver is
                  # PRODUCTION-OPERATION code -- it reaches the frozen
                  # store, the registered transform and the producer
                  # gate, and emits an operation record. It is
                  # deliberately NOT producer_boundary, which is the
                  # staged-envelope trust boundary rather than the
                  # code that performs an operation.
                  "monitoring/src/w2_capture_repair_v4_vic_cayley.py",
                  # codex 1345Z: the exact-key 404 retry one-shot. A
                  # NETWORK-SPENDING executable may not run unpinned --
                  # its own precheck refuses RETRY_MODULE_UNBOUND
                  # unless it is among the BOUND pins checked against
                  # executed disk bytes. Operation bytes, same
                  # authority as the capture runner it succeeds.
                  "monitoring/src/w2_capture_retry_404_v4_cayley.py",
                  # codex 2240Z P0-2: the ONE reviewed zero-network
                  # production finalizer (plan/apply) for the two
                  # remaining zero-HTTP class families -- operation
                  # orchestration, so accrual_impl authority
                  "monitoring/src/w2_zero_http_finalizer_cayley.py",
                  # codex 0532Z: already-committed production-operation
                  # code that was pinned NOWHERE. It stages the single
                  # non-capture key (MAG_WEATHER_FEED/omni/2026-01-01),
                  # so leaving it unbound would let the 2,056th key
                  # arrive through code the manifest never bound.
                  "monitoring/src/w2_predecessor_bridge_cayley.py"]},
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
                  # codex 1758Z P0-1: the two power-estimation engines
                  # behind this machinery, found by the dependency
                  # closure doctor and previously in NO registry.
                  # Placed TOGETHER: `_cal_` is the CALENDAR lane (not
                  # the calibration-ledger runner) and it IMPORTS the
                  # non-calendar engine, so calibration_runner would
                  # be the wrong authority and a split would separate
                  # a dependency from its dependent.
                  "monitoring/src/"
                  "f2g_phase_b_power_estimation_cayley.py",
                  "monitoring/src/"
                  "f2g_phase_b_power_estimation_cal_cayley.py",
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
        "paths": ["monitoring/src/w2_calibration_runner_cayley.py",
                  # P0-4 (codex 1733Z sequence): the feed producer
                  # derives the runner's inputs from committed
                  # staged/capsule bytes -- production machinery on
                  # the calibration lane, plus its lock suite
                  "monitoring/src/"
                  "w2_calibration_feed_producer_cayley.py",
                  "monitoring/src/"
                  "w2_calibration_feed_producer_kats_cayley.py",
                  # the amended MF4 catalog lane (codex 1758Z opt 1
                  # + 0317Z bytes-only boundary): adapter, archive
                  # capsule machinery, and their lock suite
                  "monitoring/src/"
                  "w2_mf4_catalog_adapter_grassmann.py",
                  "monitoring/src/"
                  "w2_mf4_archive_capsule_gen_grassmann.py",
                  "monitoring/src/"
                  "w2_mf4_archive_kats_grassmann.py",
                  # the REGISTERED calibration input surfaces --
                  # data pins, reopened by the producer/adapter
                  "docs/f2g_window2_execution/"
                  "mf4_archive_capsule_v1.json",
                  "docs/f2g_window2_execution/"
                  "mf4_archive_receipt_v1.json",
                  "docs/f2g_window2_execution/mf4_archive/"
                  "daily_risk_rows_v1.jsonl",
                  "docs/f2g_window2_execution/mf4_catalog_snapshot/"
                  "catalog_snapshot_v1.json",
                  "docs/f2g_window2_execution/mf4_catalog_snapshot/"
                  "acquisition_receipt_v1.json"]},
    # RE-BOUND ATOMICALLY at bar REV 9 green (codex 1335Z/1358Z
    # disposition; grassmann 1745Z, 14/14, dual-verified by cayley on
    # py3.14+py3.11)
    "mag_capsules": {
        "owner": "cayley",
        "note": "FOUR capsules at the execution tree -- IZN/FRN/TUC "
                "relocated under codex 0451Z with their probe "
                "receipts, plus VIC registered under codex 0532Z as an "
                "exact-byte copy of the preserved freeze capsule. VIC "
                "carries NO probe receipts here: its three historical "
                "receipts stay at the freeze path, and this slot must "
                "not imply otherwise. Loaded via "
                "load_execution_capsule under the dynamic manifest "
                "authority; W-MAG-EXEC green",
        "paths": ["docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_izn.json",
                  "docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_frn.json",
                  "docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_tuc.json",
                  # codex 0532Z: the carrier whose absence here fired
                  # 212 keys that could never be admitted. VIC reports
                  # sensor_orientation XYZS, which is not in
                  # REPORTED_CONVENTIONS, so the transform resolves the
                  # frame from THIS path or refuses.
                  "docs/f2g_window2_execution/mag_capsules/"
                  "mag_capsule_vic.json",
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

# ---- codex 0532Z P0-1: the explicit, fail-closed FINAL-BIND contract -
# DEFAULT generation is unchanged and must stay exactly the honest
# 10/12 OPEN shape. Final bind is a DISTINCT mode that takes explicit,
# sorted, unique, COMMITTED path specs for both OPEN slots.
#
# No glob, no directory walk, no `os.listdir` decides membership
# anywhere in this contract. Readiness inferred from what happens to be
# on disk is how a boundary closes over whatever was lying around; the
# spec has to say it, and every path in it has to resolve at the named
# target through the ordinary pin() -- which already refuses
# PATH_NOT_AT_TARGET for anything uncommitted.
FINAL_BIND_SCHEMA = "f2g-window2-final-bind-spec-v1"
FINAL_BIND_SLOTS = ("producer_boundary", "calibration_ledgers")
FINAL_BIND_SPEC_PATH = ("docs/f2g_window2_execution/"
                        "final_bind_spec_v1.json")

# the REGISTERED v4 staged prefix. The reviewed v4 producers
# (w2_capture_run_v4_grassmann.py, w2_restage_v4_grassmann.py) both
# write staged_envelopes_v4/. The legacy v3 prefix is RETIRED: it is
# never a valid final-bind path, and copying v4 output into it would
# hide an object-identity error rather than repair it. The two are
# cleanly distinguishable -- ".../staged_envelopes_v4/" does NOT
# startswith ".../staged_envelopes/" -- so no path satisfies both.
STAGED_PREFIX_V4 = _ACCM.STAGED_PREFIX
STAGED_PREFIX_LEGACY_RETIRED = _ACCM.STAGED_PREFIX_RETIRED
# codex 0655Z item 3: recognition comes from the ONE authority in
# the admission consumer, so the `.restage.json` provenance carrier
# the boundary parses is a carrier this generator pins.
STAGED_CLASS_SUFFIXES = tuple(_ACCM.STAGED_ALL_SUFFIXES)
# 2,056 authorized keys x 4 staged classes. Recomputed from the
# authority/class bijection by the boundary verifier -- this constant
# is the DECLARED expectation, never the proof.
FINAL_BIND_EXPECTED_CLASSES = 2056 * 4

# codex 0551Z repair 1: required classes are EXACT registered
# repo-relative paths, never basenames. A basename class collapsed
# path identity, authority identity and one-authority-per-path into
# cardinality -- six files under docs/attacker/ satisfied the whole
# producer contract because their basenames matched. Each class below
# is exactly ONE registered path; a same-basename file anywhere else
# satisfies nothing.
# The last three are the operation-evidence classes codex 0532Z/0551Z
# rules into the boundary: (a) the compact terminal receipt binding
# ledger + inventory, (b) the exact 212-key VIC repair operation-
# record set, (c) the predecessor-bridge operation record. They are
# AUDIT EVIDENCE: pinned and reopened by the boundary verifier, never
# a substitute for any staged class. terminal_receipt and
# predecessor_bridge_record REGISTER path names for artifacts
# grassmann produces at freeze/bridge time -- newly registered here,
# named for review, not yet committed. vic_repair_records must equal
# w2_capture_repair_v4_vic_cayley.REPAIR_LEDGER (selftest-asserted).
PRODUCER_BOUNDARY_REQUIRED = {
    "amendment": ("docs/f2g_window2_execution/"
                  "producer_boundary_amendment_v1.md"),
    "boundary_code": ("monitoring/src/"
                      "w2_acquisition_capture_grassmann.py"),
    "authority": ("docs/f2g_window2_execution/"
                  "staged_expected_contracts_v3.json"),
    "inventory": STAGED_PREFIX_V4 + "staged_body_inventory.json",
    "store_descriptor": STAGED_PREFIX_V4 + "store_descriptor.json",
    # the three operation-evidence classes + the ledger come FROM the
    # admission consumer, which reopens and recomputes them
    # (codex 1547Z repair 2) -- one authority, three consumers
    "ledger": _ACCM.CAPTURE_LEDGER_PATH,
    "terminal_receipt": _ACCM.TERMINAL_RECEIPT_PATH,
    "vic_repair_records": _ACCM.VIC_REPAIR_LEDGER_PATH,
    "predecessor_bridge_record": _ACCM.PREDECESSOR_RECORD_PATH,
    # codex 2240Z P0-1: the six-member CLOSED retry-operation chain,
    # including the mandatory transport receipt -- registered from
    # the admission consumer's table so the three layers cannot drift
    "retry_dispatch": _ACCM.RETRY_CHAIN_PATHS["dispatch"],
    "retry_transport_receipt":
        _ACCM.RETRY_CHAIN_PATHS["transport_receipt"],
    "retry_prepared": _ACCM.RETRY_CHAIN_PATHS["prepared"],
    "retry_result": _ACCM.RETRY_CHAIN_PATHS["result"],
    "retry_classes_complete":
        _ACCM.RETRY_CHAIN_PATHS["classes_complete"],
    "retry_index": _ACCM.RETRY_CHAIN_PATHS["index"],
}


# THE registered calibration production path set (P0-4 closure).
# MF4 amended lane: landed run d8324818 + final-bind ec9245ee under
# codex MF4_POST_RUN_LEDGER_PASS. MAG lane: landed fit c8089a13
# under codex MAG_POST_RUN_LEDGER_PASS (carrier bytes live in the
# content-addressed S4T store; the committed record binds them).
# Exact committed paths, literal by design -- the registration IS
# this list, cross-checked against the runner's constants in the
# selftest (a runtime import here would be circular: runner ->
# producer -> generator).
_CAL_DIR = "docs/f2g_window2_execution/calibration/"
REGISTERED_CALIBRATION_PATHS = tuple(sorted(
    [_CAL_DIR + "mf4_input_feed_amended.json",
     _CAL_DIR + "mf4_ledger_amended.json",
     _CAL_DIR + "mf4_ledger_amended.receipt.json",
     _CAL_DIR + "mf4_final_bind_record_v1.json",
     _CAL_DIR + "mag_carrier_record_v1.json",
     _CAL_DIR + "mag_ledgers.receipt.json",
     _CAL_DIR + "mag_final_bind_record_v1.json"]
    + [_CAL_DIR + f"mag_{o}_{c}_ledger.json"
       for o in ("frn", "izn", "new", "tuc", "vic")
       for c in ("X", "Y")]
    + [_CAL_DIR + f"mag_m3_{pair}_{c}_ledger.json"
       for pair in ("frn_on_tuc", "vic_on_new")
       for c in ("X", "Y")]))


class FinalBindRefusal(SystemExit):
    """Typed and fail-closed. The code leads the message."""


def _fb_refuse(code, detail):
    raise FinalBindRefusal(f"{code}: {detail}")


def _decode_final_bind_spec(raw, source):
    """Strict JSON decode: duplicate object keys are ambiguous
    authorities and therefore refuse before shape validation."""
    def no_dupes(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                _fb_refuse("FINAL_BIND_SPEC_DUPLICATE_KEY",
                           f"{source}: {key!r}")
            obj[key] = value
        return obj
    try:
        spec = json.loads(raw.decode("utf-8"),
                          object_pairs_hook=no_dupes)
    except Exception as exc:
        _fb_refuse("FINAL_BIND_SPEC_UNPARSEABLE",
                   f"{source}: {exc}")
    if not isinstance(spec, dict):
        _fb_refuse("FINAL_BIND_SPEC_NOT_CLOSED", "spec is not an object")
    want = {"schema"} | set(FINAL_BIND_SLOTS)
    if set(spec) != want:
        _fb_refuse("FINAL_BIND_SPEC_NOT_CLOSED",
                   f"keys {sorted(set(spec) ^ want)} unexpected/missing")
    if spec["schema"] != FINAL_BIND_SCHEMA:
        _fb_refuse("FINAL_BIND_SCHEMA_MISMATCH",
                   f"{spec['schema']!r} != {FINAL_BIND_SCHEMA!r}")
    for name in FINAL_BIND_SLOTS:
        slot = spec[name]
        if not isinstance(slot, dict) or set(slot) != {"paths"}:
            _fb_refuse("FINAL_BIND_SLOT_NOT_CLOSED",
                       f"{name} must carry exactly one key 'paths'")
        paths = slot["paths"]
        if not isinstance(paths, list) or not paths or \
                any(not isinstance(p, str) or not p for p in paths):
            _fb_refuse("FINAL_BIND_PATHS_UNTYPED",
                       f"{name}: paths must be a non-empty list of str")
        if len(set(paths)) != len(paths):
            dupes = sorted({p for p in paths if paths.count(p) > 1})
            _fb_refuse("FINAL_BIND_PATHS_DUPLICATE",
                       f"{name}: {dupes[:3]}")
        if paths != sorted(paths):
            _fb_refuse("FINAL_BIND_PATHS_UNSORTED",
                       f"{name}: paths must be sorted for review "
                       "determinism")
    # codex 0551Z repair 1: ONE authority per path. The same path
    # pinned in both slots would answer to two slot authorities.
    both = set(spec["producer_boundary"]["paths"])         & set(spec["calibration_ledgers"]["paths"])
    if both:
        _fb_refuse("FINAL_BIND_PATH_IN_BOTH_SLOTS",
                   f"{sorted(both)[:3]}")
    return spec


def load_final_bind_spec(path):
    """Parse a CLOSED spec object. Shape errors refuse before any path
    is resolved, so a malformed spec can never partially bind."""
    with open(path, "rb") as f:
        return _decode_final_bind_spec(f.read(), path)


def _check_prefixes(name, paths):
    """Legacy prefix is retired; mixed prefixes refuse. Applies to the
    STAGED CLASS paths only -- the amendment, code, authority,
    inventory, descriptor and ledger legitimately live elsewhere."""
    legacy = [p for p in paths
              if p.startswith(STAGED_PREFIX_LEGACY_RETIRED)]
    if legacy:
        _fb_refuse("FINAL_BIND_LEGACY_PREFIX",
                   f"{name}: {len(legacy)} path(s) under the RETIRED v3 "
                   f"prefix, e.g. {legacy[0]}")
    staged = [p for p in paths if p.endswith(STAGED_CLASS_SUFFIXES)]
    stray = [p for p in staged if not p.startswith(STAGED_PREFIX_V4)]
    if stray:
        _fb_refuse("FINAL_BIND_MIXED_PREFIX",
                   f"{name}: {len(stray)} staged-class path(s) outside "
                   f"the one registered prefix, e.g. {stray[0]}")
    return staged


def _check_producer_boundary(paths):
    staged = _check_prefixes("producer_boundary", paths)
    pathset = set(paths)
    for cls, exact in sorted(PRODUCER_BOUNDARY_REQUIRED.items()):
        if exact not in pathset:
            same_base = [p for p in paths
                         if os.path.basename(p)
                         == os.path.basename(exact)]
            hint = (f"; a same-basename file at {same_base[0]} "
                    "satisfies NOTHING -- the class is the exact "
                    "registered path" if same_base else "")
        # exact-path identity: uniqueness within the slot is already
        # guaranteed by load_final_bind_spec, so presence is the whole
        # check and no ambiguity case exists
            _fb_refuse("FINAL_BIND_CLASS_MISSING",
                       f"producer_boundary: required class {cls} "
                       f"missing its registered path {exact}{hint}")
    if len(staged) != FINAL_BIND_EXPECTED_CLASSES:
        _fb_refuse("FINAL_BIND_CLASS_COUNT",
                   f"producer_boundary: {len(staged)} staged classes, "
                   f"expected {FINAL_BIND_EXPECTED_CLASSES} "
                   f"(2056 keys x 4). The count is a DECLARED "
                   "expectation; the authority/class bijection is "
                   "recomputed by the boundary verifier and this never "
                   "substitutes for it")
    return staged


def final_bind_slots(spec, repo, target_full):
    """Both slots, or a typed refusal. Every path resolves through the
    ordinary pin(), so uncommitted paths refuse PATH_NOT_AT_TARGET."""
    _check_producer_boundary(spec["producer_boundary"]["paths"])
    _check_prefixes("calibration_ledgers",
                    spec["calibration_ledgers"]["paths"])

    # P0-4 CLOSED (codex 0532Z hold retired): the production
    # calibration set is now REGISTERED above -- both lanes fit
    # once, receipted, final-bound, and codex-passed. The spec must
    # equal that set EXACTLY; anything else refuses typed. The
    # original refusal shape survives as the divergence gate: a
    # final bind can still never invent a calibration list.
    got = sorted(spec["calibration_ledgers"]["paths"])
    want = sorted(REGISTERED_CALIBRATION_PATHS)
    if got != want:
        missing = [p for p in want if p not in set(got)]
        extra = [p for p in got if p not in set(want)]
        _fb_refuse("FINAL_BIND_CALIBRATION_SET_DIVERGENT",
                   "calibration_ledgers must equal the REGISTERED "
                   f"production set exactly (missing={missing[:3]}, "
                   f"extra={extra[:3]})")
    return {name: list(spec[name]["paths"])
            for name in FINAL_BIND_SLOTS}


def _git(repo, args, binary=False):
    out = subprocess.check_output(["git", "-C", repo] + args)
    return out if binary else out.decode().strip()


def load_final_bind_spec_at_target(repo, target_full, supplied_path,
                                   blob_reader=None):
    """The owner-gated spec is an object at the named target, never an
    unattested worktree/temporary-file authority. The supplied path is
    accepted only as the exact registered repo path; membership bytes
    are read ONLY from the target's Git object, so checkout newline
    conversion or a dirty disk copy can never steer the operation."""
    registered = os.path.realpath(os.path.join(
        repo, FINAL_BIND_SPEC_PATH.replace("/", os.sep)))
    supplied = os.path.realpath(
        supplied_path if os.path.isabs(supplied_path)
        else os.path.join(repo, supplied_path))
    if os.path.normcase(supplied) != os.path.normcase(registered):
        _fb_refuse("FINAL_BIND_SPEC_PATH_UNREGISTERED",
                   f"{supplied_path!r} != {FINAL_BIND_SPEC_PATH!r}")
    try:
        target_raw = (blob_reader(target_full, FINAL_BIND_SPEC_PATH)
                      if blob_reader is not None else
                      _git(repo, ["cat-file", "blob",
                                  f"{target_full}:{FINAL_BIND_SPEC_PATH}"],
                           binary=True))
    except Exception as exc:
        _fb_refuse("FINAL_BIND_SPEC_NOT_AT_TARGET", str(exc))
    return _decode_final_bind_spec(
        target_raw, f"{target_full}:{FINAL_BIND_SPEC_PATH}")


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


def main(repo, target, design_manifest_commit, final_bind_spec=None):
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
    # codex 0532Z P0-1: FINAL BIND is opt-in and fail-closed. Without a
    # spec the shape above is untouched and both slots stay honestly
    # OPEN -- the default generator can no more emit BOUND than it
    # could before. With a spec, every path is validated and then
    # resolved through the ordinary pin(), so an uncommitted path
    # refuses PATH_NOT_AT_TARGET rather than binding a promise.
    if final_bind_spec is not None:
        spec = load_final_bind_spec_at_target(
            repo, target_full, final_bind_spec)
        for name, paths in final_bind_slots(spec, repo,
                                            target_full).items():
            owner, note = OPEN_SLOTS[name]
            slots[name] = {"status": "BOUND", "owner": owner,
                           "note": note,
                           "pins": [pin(repo, target_full, p)
                                    for p in paths]}
        slots["producer_boundary"]["boundary_mode"] = \
            PRODUCER_BOUNDARY_MODE
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


def _selftest():
    """Final-bind contract controls. Pure shape/prefix/class logic --
    no repo, no network, no writes, so it runs anywhere."""
    import tempfile
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    def refuses(fn, code):
        try:
            fn()
        except FinalBindRefusal as exc:
            return str(exc).startswith(code)
        except Exception:
            return False
        return False

    # codex 0413Z: plan must not depend on a transform whose only pin
    # can appear after apply. Exactly one DEFAULT slot owns the
    # operation role; producer_boundary separately owns the same
    # file's later boundary-code role.
    _dispatcher = ("monitoring/src/"
                   "w2_acquisition_capture_grassmann.py")
    _default_owners = sorted(
        name for name, slot in BOUND_SLOTS.items()
        if _dispatcher in slot.get("paths", ()))
    check("C0 dispatcher operation role is default-bound exactly "
          "under accrual_impl and remains the registered later "
          "producer-boundary code role",
          _default_owners == ["accrual_impl"] and
          PRODUCER_BOUNDARY_REQUIRED["boundary_code"] == _dispatcher,
          f"default_owners={_default_owners} boundary_code="
          f"{PRODUCER_BOUNDARY_REQUIRED.get('boundary_code')}")

    def spec_file(obj):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return p

    staged = sorted(
        f"{STAGED_PREFIX_V4}mag_feed_vic_2026-01-{d:02d}{s}"
        for d in range(1, 3) for s in STAGED_CLASS_SUFFIXES)
    # every required class at its EXACT registered path
    others = sorted(PRODUCER_BOUNDARY_REQUIRED.values())
    good_pb = sorted(staged + others)

    def mk(pb=None, cal=None, schema=FINAL_BIND_SCHEMA):
        return {"schema": schema,
                "producer_boundary": {"paths": pb if pb is not None
                                      else good_pb},
                "calibration_ledgers": {"paths": cal if cal is not None
                                        else ["docs/x/cal.json"]}}

    # ---- DEFAULT SHAPE is the thing that must not regress
    check("C1 default mode still emits both slots OPEN with no pins",
          all(OPEN_SLOTS[n] for n in FINAL_BIND_SLOTS)
          and set(FINAL_BIND_SLOTS) == set(OPEN_SLOTS))

    # ---- spec shape
    check("C2 non-closed spec refuses",
          refuses(lambda: load_final_bind_spec(
              spec_file({"schema": FINAL_BIND_SCHEMA, "extra": 1,
                         "producer_boundary": {"paths": ["a"]},
                         "calibration_ledgers": {"paths": ["b"]}})),
              "FINAL_BIND_SPEC_NOT_CLOSED"))
    check("C2a duplicate JSON object keys refuse before shape use",
          refuses(lambda: _decode_final_bind_spec(
              (b'{"schema":"f2g-window2-final-bind-spec-v1",'
               b'"schema":"f2g-window2-final-bind-spec-v1"}'),
              "duplicate-fixture"),
              "FINAL_BIND_SPEC_DUPLICATE_KEY"))
    check("C3 schema mismatch refuses",
          refuses(lambda: load_final_bind_spec(
              spec_file(mk(schema="wrong"))),
              "FINAL_BIND_SCHEMA_MISMATCH"))
    check("C4 duplicate paths refuse",
          refuses(lambda: load_final_bind_spec(
              spec_file(mk(pb=["a", "a"]))),
              "FINAL_BIND_PATHS_DUPLICATE"))
    check("C5 unsorted paths refuse (review determinism)",
          refuses(lambda: load_final_bind_spec(
              spec_file(mk(pb=["b", "a"]))),
              "FINAL_BIND_PATHS_UNSORTED"))
    check("C6 empty/untyped paths refuse",
          refuses(lambda: load_final_bind_spec(spec_file(mk(pb=[]))),
                  "FINAL_BIND_PATHS_UNTYPED"))
    check("C7 a well-formed spec parses",
          load_final_bind_spec(spec_file(mk()))["schema"]
          == FINAL_BIND_SCHEMA)
    # The executing authority is the exact registered blob at the
    # named target; an arbitrary spec path or dirty worktree copy can
    # never steer membership after review.
    _target_repo = tempfile.mkdtemp(prefix="f2g-final-bind-target-")
    _target_path = os.path.join(
        _target_repo, FINAL_BIND_SPEC_PATH.replace("/", os.sep))
    os.makedirs(os.path.dirname(_target_path), exist_ok=True)
    _target_raw = json.dumps(mk()).encode("utf-8")
    with open(_target_path, "wb") as _tf:
        _tf.write(_target_raw)
    _reader = lambda _commit, _path: _target_raw
    check("C7a registered-path target blob parses",
          load_final_bind_spec_at_target(
              _target_repo, "target", FINAL_BIND_SPEC_PATH,
              blob_reader=_reader)["schema"] == FINAL_BIND_SCHEMA)
    check("C7b arbitrary spec path refuses",
          refuses(lambda: load_final_bind_spec_at_target(
              _target_repo, "target", spec_file(mk()),
              blob_reader=_reader),
              "FINAL_BIND_SPEC_PATH_UNREGISTERED"))
    with open(_target_path, "ab") as _tf:
        _tf.write(b" ")
    check("C7c dirty worktree spec cannot steer target blob parsing",
          load_final_bind_spec_at_target(
              _target_repo, "target", FINAL_BIND_SPEC_PATH,
              blob_reader=_reader)["schema"] == FINAL_BIND_SCHEMA)
    def _missing_reader(_commit, _path):
        raise FileNotFoundError("target blob absent")
    check("C7d missing registered spec at target refuses",
          refuses(lambda: load_final_bind_spec_at_target(
              _target_repo, "target", FINAL_BIND_SPEC_PATH,
              blob_reader=_missing_reader),
              "FINAL_BIND_SPEC_NOT_AT_TARGET"))

    # ---- prefix contract
    check("C8 a RETIRED legacy-prefix path refuses",
          refuses(lambda: _check_prefixes(
              "producer_boundary",
              [STAGED_PREFIX_LEGACY_RETIRED + "x.record.json"]),
              "FINAL_BIND_LEGACY_PREFIX"))
    check("C9 a staged class outside the one registered prefix refuses",
          refuses(lambda: _check_prefixes(
              "producer_boundary", ["docs/elsewhere/x.record.json"]),
              "FINAL_BIND_MIXED_PREFIX"))
    # NOT `... or True`: that reads as a control and cannot fail. The
    # real assertion is that these paths do not raise AND that only the
    # staged-suffixed ones are classified as staged.
    _c10_raised = None
    try:
        _c10 = _check_prefixes("producer_boundary", others)
    except FinalBindRefusal as exc:
        _c10, _c10_raised = None, str(exc)
    check("C10 non-staged paths outside the prefix are ALLOWED and are "
          "NOT counted as staged classes (amendment/code/ledger "
          "legitimately live elsewhere)",
          _c10_raised is None and _c10 == [],
          f"raised={_c10_raised} staged={_c10}")

    # ---- producer_boundary required classes (all nine, exact paths)
    for cls in sorted(PRODUCER_BOUNDARY_REQUIRED):
        drop = [p for p in good_pb
                if p != PRODUCER_BOUNDARY_REQUIRED[cls]]
        check(f"C11 dropping required class {cls} refuses",
              refuses(lambda d=drop: _check_producer_boundary(d),
                      "FINAL_BIND_CLASS_MISSING"))
    # codex 0551Z repair 1 KATs -- the attacks the basename contract
    # let through, each now a typed refusal:
    # (a) same-basename substitution: the amendment's basename under
    # docs/attacker/ instead of its registered path
    _amend = PRODUCER_BOUNDARY_REQUIRED["amendment"]
    _attack = sorted([p for p in good_pb if p != _amend]
                     + ["docs/attacker/"
                        + os.path.basename(_amend)])
    check("C11a same-basename file at an unregistered path does NOT "
          "satisfy the class",
          refuses(lambda: _check_producer_boundary(_attack),
                  "FINAL_BIND_CLASS_MISSING"))
    # (b) the same path in BOTH slots refuses at spec load
    check("C11b a path pinned in both slots refuses (one authority "
          "per path)",
          refuses(lambda: load_final_bind_spec(spec_file(
              mk(cal=[good_pb[0]]))),
              "FINAL_BIND_PATH_IN_BOTH_SLOTS"))
    # (c) count-preserving substitution: drop the inventory, add one
    # more staged path so the 8,224 count survives -- the CLASS check
    # must still refuse; cardinality is not identity
    _inv = PRODUCER_BOUNDARY_REQUIRED["inventory"]
    _sub = sorted([p for p in good_pb if p != _inv]
                  + [f"{STAGED_PREFIX_V4}k_extra_0001.record.json"])
    check("C11c preserving the staged count by substituting a "
          "duplicate-class path still refuses CLASS_MISSING",
          refuses(lambda: _check_producer_boundary(_sub),
                  "FINAL_BIND_CLASS_MISSING"))
    # (d) the registered VIC repair-record path equals the replay
    # module's own REPAIR_LEDGER constant (drift here would pass
    # authoring and refuse at bind time on the wrong path)
    import w2_capture_repair_v4_vic_cayley as _VICR
    _vic_reg = PRODUCER_BOUNDARY_REQUIRED["vic_repair_records"]
    _vic_mod = os.path.relpath(
        _VICR.REPAIR_LEDGER, _VICR.REPO).replace(os.sep, "/")
    check("C11d vic_repair_records path equals the replay module's "
          "REPAIR_LEDGER", _vic_reg == _vic_mod,
          f"registered={_vic_reg} module={_vic_mod}")
    check("C12 the declared staged-class count is enforced",
          refuses(lambda: _check_producer_boundary(good_pb),
                  "FINAL_BIND_CLASS_COUNT"))

    # ---- the P0-4 hold, which must NOT be quietly satisfiable.
    # An `or` between the class-count refusal and the calibration
    # refusal would pass on the FIRST and never exercise the second --
    # a live branch carrying a dead one. So build a producer_boundary
    # spec that FULLY satisfies its own contract (exactly 8,224 staged
    # classes + every required class), and require the calibration
    # refusal to be the one that fires.
    # codex 0655Z item 3: the satisfying spec carries the REAL
    # mixed provenance -- 636 native records + 1,420 restage lineage
    # carriers, each stem with exactly ONE provenance form plus the
    # three scientific classes (8,224 total; a 4-uniform fixture no
    # longer describes the staged space)
    _full_staged = []
    for n in range(FINAL_BIND_EXPECTED_CLASSES // 4):
        _stem = f"{STAGED_PREFIX_V4}k{n:05d}"
        _full_staged.append(
            _stem + (".record.json" if n < 636
                     else ".restage.json"))
        for _sfx in (".transcript.json", ".contract.json",
                     ".artifact.json"):
            _full_staged.append(_stem + _sfx)
    _full_staged = sorted(_full_staged)
    _full_pb = sorted(_full_staged + others)
    _pb_ok = True
    try:
        _check_producer_boundary(_full_pb)
    except FinalBindRefusal as exc:
        _pb_ok = False
        _pb_why = str(exc)
    check("C13a a FULLY satisfying producer_boundary spec passes its "
          f"own contract ({FINAL_BIND_EXPECTED_CLASSES} classes + all "
          "required classes) -- so the mode is not merely always-refuse",
          _pb_ok, "" if _pb_ok else _pb_why)
    # P0-4 hold RETIRED: with producer_boundary fully satisfied and
    # the REGISTERED calibration set supplied, final_bind_slots
    # RETURNS both slots' exact path lists (pin-free validation;
    # main() performs the pinning)
    _cal_ok = list(REGISTERED_CALIBRATION_PATHS)
    _fb_out = final_bind_slots(
        mk(pb=_full_pb, cal=_cal_ok), ".", "HEAD")
    check("C13b with both contracts satisfied final_bind_slots "
          "returns the exact two slot path lists",
          sorted(_fb_out) == ["calibration_ledgers",
                              "producer_boundary"]
          and _fb_out["calibration_ledgers"] == _cal_ok
          and _fb_out["producer_boundary"] == _full_pb)
    check("C14a a MISSING registered calibration path refuses "
          "typed",
          refuses(lambda: final_bind_slots(
              mk(pb=_full_pb, cal=_cal_ok[:-1]), ".", "HEAD"),
              "FINAL_BIND_CALIBRATION_SET_DIVERGENT"))
    check("C14b an EXTRA calibration path refuses typed",
          refuses(lambda: final_bind_slots(
              mk(pb=_full_pb,
                 cal=sorted(_cal_ok + ["docs/x/evil.json"])),
              ".", "HEAD"),
              "FINAL_BIND_CALIBRATION_SET_DIVERGENT"))
    check("C14c a RENAMED calibration path refuses typed",
          refuses(lambda: final_bind_slots(
              mk(pb=_full_pb,
                 cal=sorted(_cal_ok[:-1]
                            + [_cal_ok[-1] + ".evil"])),
              ".", "HEAD"),
              "FINAL_BIND_CALIBRATION_SET_DIVERGENT"))
    # cross-check the literal registration against the runner's own
    # constants (function-scoped import; module-level would be
    # circular through the feed producer)
    import w2_calibration_runner_cayley as _RUNX
    check("C14d the registered set carries the runner's own carrier "
          "record and receipt paths",
          _RUNX.MAG_CARRIER_RECORD_REL
          in REGISTERED_CALIBRATION_PATHS
          and (_CAL_DIR + "mag_ledgers.receipt.json")
          in REGISTERED_CALIBRATION_PATHS
          and (_CAL_DIR + "mf4_ledger_amended.receipt.json")
          in REGISTERED_CALIBRATION_PATHS)
    print()
    if fails:
        print(f"FINAL-BIND CONTRACT FAILURES ({len(fails)}): {fails}")
        return 1
    print("FINAL-BIND CONTRACT: ALL CONTROLS PASS")
    return 0


if __name__ == "__main__":
    _args = sys.argv[1:]
    if "--selftest" in _args:
        raise SystemExit(_selftest())
    _fb = None
    if "--final-bind" in _args:
        _i = _args.index("--final-bind")
        if _i + 1 >= len(_args):
            raise FinalBindRefusal(
                "FINAL_BIND_SPEC_MISSING: --final-bind needs a spec path")
        _fb = _args[_i + 1]
        _args = _args[:_i] + _args[_i + 2:]
    main(os.path.abspath(_args[0]), _args[1], _args[2],
         final_bind_spec=_fb)
