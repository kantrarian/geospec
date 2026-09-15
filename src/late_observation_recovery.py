#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Late-observation recovery record -- NOT a publication receipt (option B, codex 2026-09-14T16:00Z).

Proposed runner module (grassmann, 2026-09-14; r2 after codex 2026-09-14T17:05Z), scratch-only until the owner gives
separate write/apply authority.

When a scored day's canonical receipt was never written, this records what can still be observed honestly: the day,
its full commit, the reopened carrier hashes, and the retained server build record -- with the server's own times
kept apart from the time WE observed it and the time this record was created. It deliberately has no
`availability_utc`, no `built_utc`, no `deployment` and no `artifact_hashes`, and a different schema, so the canonical
admission path (`publication_receipt.admit_receipt`) refuses it: it never mints a VerifiedReceipt and restores no
hit, availability or deadline credit. The missing original receipt stays missing and is reported beside this record.

r2 (codex item 4): the record is installed create-once through `create_once.install_no_replace`, which never replaces
an existing destination. A concurrent first writer keeps its bytes and observation times.
"""
import datetime
import hashlib
import json
import os

import create_once as CO
import receipt_target_selector as SEL

SCHEMA = "geospec-late-observation-recovery-v1"
STATE = "LATE_OBSERVATION_RECOVERY_NOT_CONTEMPORANEOUS"
STANDING = ("NONE: not a publication receipt; never admitted by publication_receipt.admit_receipt; mints no "
            "VerifiedReceipt; restores no hit, availability or deadline credit. The original receipt remains missing "
            "and is reported beside this record, never replaced by it.")
REPORTED = "BESIDE_THIS_RECORD_NEVER_REPLACED"
MANDATORY_CARRIERS = ("docs/ensemble_latest.json", "docs/data.csv")
RECORD_KEYS = frozenset({"schema", "state", "scored_day", "commit_sha", "commit_subject", "carrier_date", "carrier_hashes",
                         "build", "raw_server_record_sha256", "observed_utc", "recovery_record_created_utc",
                         "original_receipt", "cause", "standing", "producer"})
BUILD_KEYS = frozenset({"id", "api_url", "status", "server_created_utc", "server_completed_utc"})
CANONICAL_RECEIPT_KEYS_NEVER_PRESENT = frozenset({"availability_utc", "built_utc", "deployment", "artifact_hashes"})
VERIFIED = "PUBLISHED_RECEIPT_VERIFIED_AGAINST_COMMIT"
_EVIDENCE_TIME_KEYS = ("observed_utc", "recovery_record_created_utc")


class RecoveryRefused(ValueError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def recovery_path(recoveries_dir, day):
    return os.path.join(recoveries_dir, f"{day}.late-observation-recovery.json")


def build_recovery_record(selection, *, commit_subject_loader, artifact_loader, observed_utc, created_utc,
                          original_receipt_state, cause, producer):
    """Build the record from a COMPLETE target selection only; every other state is refused, and nothing is written."""
    if selection.get("state") != SEL.COMPLETE:
        raise RecoveryRefused("RECOVERY_TARGET_NOT_COMPLETE", str(selection.get("state")))
    day, commit = selection["scored_day"], selection["commit_sha"]
    if original_receipt_state == VERIFIED:
        raise RecoveryRefused("RECOVERY_NOT_NEEDED_ORIGINAL_RECEIPT_VERIFIED", day)
    if not original_receipt_state:
        raise RecoveryRefused("RECOVERY_ORIGINAL_RECEIPT_STATE_REQUIRED", day)
    try:
        observed, created = SEL.parse_utc(observed_utc), SEL.parse_utc(created_utc)
        completed = SEL.parse_utc(selection["server_completed_utc"])
    except ValueError as exc:
        raise RecoveryRefused("RECOVERY_CLOCK_INVALID", str(exc)) from None
    if observed < completed:
        raise RecoveryRefused("RECOVERY_OBSERVED_BEFORE_SERVER_COMPLETION", f"{observed_utc} < {selection['server_completed_utc']}")
    if created < observed:
        raise RecoveryRefused("RECOVERY_CREATED_BEFORE_OBSERVATION", f"{created_utc} < {observed_utc}")
    hashes = {}
    try:
        for rel in MANDATORY_CARRIERS:
            hashes[rel] = hashlib.sha256(bytes(artifact_loader(commit, rel))).hexdigest()
        carrier_date = json.loads(bytes(artifact_loader(commit, MANDATORY_CARRIERS[0])).decode("utf-8")).get("date")
        subject = str(commit_subject_loader(commit)).strip()
    except Exception as exc:  # noqa: BLE001
        raise RecoveryRefused("RECOVERY_COMMIT_UNREADABLE", f"{type(exc).__name__}: {exc}") from None
    if carrier_date != day or subject != f"Daily monitoring {day}":
        raise RecoveryRefused("RECOVERY_DAY_MISMATCH", f"subject {subject!r}, carrier {carrier_date!r}, day {day}")
    build = selection["build"]
    record = dict(
        schema=SCHEMA, state=STATE, scored_day=day, commit_sha=commit, commit_subject=subject, carrier_date=carrier_date,
        carrier_hashes=hashes,
        build=dict(id=selection["build_id"], api_url=build["url"], status=build["status"],
                   server_created_utc=selection["server_created_utc"], server_completed_utc=selection["server_completed_utc"]),
        raw_server_record_sha256=canonical_sha256(selection["server_record"]),
        observed_utc=observed_utc, recovery_record_created_utc=created_utc,
        original_receipt=dict(state=original_receipt_state, reported=REPORTED),
        cause=cause, standing=STANDING, producer=producer)
    validate_record_shape(record)
    return record


def validate_record_shape(record):
    """Exact keyset and fixed schema/state/standing. A record carrying any canonical-receipt key is refused."""
    if not isinstance(record, dict):
        raise RecoveryRefused("RECOVERY_NOT_A_RECORD", type(record).__name__)
    if CANONICAL_RECEIPT_KEYS_NEVER_PRESENT & set(record):
        raise RecoveryRefused("RECOVERY_CARRIES_CANONICAL_RECEIPT_KEYS", sorted(CANONICAL_RECEIPT_KEYS_NEVER_PRESENT & set(record)))
    if frozenset(record) != RECORD_KEYS:
        raise RecoveryRefused("RECOVERY_KEYSET", sorted(set(record) ^ RECORD_KEYS))
    if record["schema"] != SCHEMA or record["state"] != STATE or record["standing"] != STANDING:
        raise RecoveryRefused("RECOVERY_SCHEMA_STATE_OR_STANDING", f"{record['schema']} / {record['state']}")
    if not isinstance(record["build"], dict) or frozenset(record["build"]) != BUILD_KEYS:
        raise RecoveryRefused("RECOVERY_BUILD_KEYSET", "build")
    if (record.get("original_receipt") or {}).get("reported") != REPORTED:
        raise RecoveryRefused("RECOVERY_ORIGINAL_NOT_REPORTED_BESIDE", "original_receipt")
    return True


def _evidence(record):
    return {k: v for k, v in record.items() if k not in _EVIDENCE_TIME_KEYS}


def write_recovery_record(record, recoveries_dir):
    """Create once, atomically, never replacing an existing record. An existing record with the same evidence is a
    no-op that keeps the FIRST complete bytes and their observation times; different evidence is a typed conflict."""
    validate_record_shape(record)
    dst = recovery_path(recoveries_dir, record["scored_day"])
    data = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if CO.install_no_replace(dst, data):
        return "written"
    with open(dst, "rb") as fh:
        existing_bytes = fh.read()
    try:
        existing = json.loads(existing_bytes.decode("utf-8"))
    except ValueError:
        raise RecoveryRefused("RECOVERY_RECORD_CONFLICT", f"{dst} holds unreadable bytes") from None
    if isinstance(existing, dict) and _evidence(existing) == _evidence(record):
        return "existing_recovery_noop"
    raise RecoveryRefused("RECOVERY_RECORD_CONFLICT", dst)


def report_day(original_receipt_state, recoveries_dir, day):
    """The day's original receipt state, UNCHANGED, with any late-observation recovery reported beside it."""
    path = recovery_path(recoveries_dir, day)
    recovery = None
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                rec = json.loads(fh.read().decode("utf-8"))
            validate_record_shape(rec)
            recovery = dict(state=rec["state"], observed_utc=rec["observed_utc"],
                            server_completed_utc=rec["build"]["server_completed_utc"], standing="NONE",
                            commit_sha=rec["commit_sha"])
        except (OSError, ValueError) as exc:
            recovery = dict(state="RECOVERY_RECORD_INVALID", detail=f"{type(exc).__name__}: {exc}"[:200])
    return dict(scored_day=day, original_receipt_state=original_receipt_state, late_observation_recovery=recovery)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
