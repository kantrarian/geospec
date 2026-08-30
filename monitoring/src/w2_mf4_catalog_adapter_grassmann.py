#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF4 calibration catalog adapter (grassmann) -- codex 2026-08-29
2359Z blocker 2: the frozen engine `w2_mf4.py` is byte-untouched; this
adapter is the ONLY sanctioned way to feed the late-repair catalog
snapshot into `w2_mf4.calibrate()`, and it consumes RECEIPT-BOUND
BYTES, never bare dicts:

- `load_verified_snapshot(snapshot_bytes, receipt_bytes)` is the sole
  entry for calibration material. It strict-parses both byte streams,
  requires the exact snapshot and acquisition-receipt schemas,
  requires sha256(snapshot_bytes) == receipt.snapshot_sha256, requires
  the canonical-event-table digest recomputed from the table to equal
  the digest BOUND IN BOTH the snapshot and the receipt (an absent
  digest refuses), requires the exact registered temporal role AND the
  exact registered temporal-role-policy literal, and requires the
  authorization identity + pinned query-contract digest to be bound
  identically in snapshot and receipt (the contract digest must equal
  the module's recomputed pinned contract). Only the verified object
  proceeds to conversion or calibration.
- `calibrate_with_snapshot(...)` takes the two byte streams, verifies
  them through the loader, guards BOTH temporal roles, runs the frozen
  `w2_mf4.calibrate` unchanged, then binds the AMENDED training
  digest:
      amended = sha256(engine_training_digest
                       || temporal_role_policy bytes
                       || canonical table sha256)
  and records the full binding including the verified snapshot bytes'
  sha256, so a policy, snapshot, or table change moves the digest.
- `live_prediction_events(view)` REFUSES EVERY view typed: the
  late-repair calibration snapshot refuses MF4_CATALOG_ROLE_VIOLATION,
  and ALL `ISSUE_TIME_VIEW` inputs refuse MF4_CATALOG_LIVE_UNVERIFIED
  because no registered issue-time-view receipt verifier exists yet.
  A truthy receipt string is not a receipt; until a verifier that
  binds exact event bytes, issue day, source, and view digest is
  registered and reviewed, there is no live path through this adapter.
"""
import datetime as dt
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import w2_mf4 as MF4
from w2_mf4_catalog_acquire_grassmann import (
    Refusal, TEMPORAL_ROLE_POLICY, _strict_loads,
    calibration_snapshot_role_guard, query_contract)

SNAPSHOT_SCHEMA = "geospec-mf4-calibration-catalog-snapshot-v1"
RECEIPT_SCHEMA = "geospec-mf4-catalog-acquisition-receipt-v1"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def snapshot_table_sha(snapshot):
    """Recompute the canonical-table digest and require it to equal
    the digest BOUND in the snapshot. An absent bound digest refuses
    (codex 2359Z blocker 2: unbound tables must never reach the
    frozen engine)."""
    table = snapshot.get("canonical_event_table")
    if not isinstance(table, list):
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      "snapshot lacks canonical_event_table")
    raw = json.dumps(table, sort_keys=True).encode("utf-8")
    dig = _sha(raw)
    bound = snapshot.get("canonical_event_table_sha256")
    if not isinstance(bound, str) or not bound:
        raise Refusal("MF4_CATALOG_TABLE_DIGEST",
                      "snapshot does not bind "
                      "canonical_event_table_sha256")
    if bound != dig:
        raise Refusal("MF4_CATALOG_TABLE_DIGEST",
                      f"{dig[:16]} != bound {bound[:16]}")
    return dig


def load_verified_snapshot(snapshot_bytes, receipt_bytes):
    """The ONLY sanctioned loader for calibration catalog material:
    receipt-bound bytes in, verified snapshot object out. Every check
    refuses typed BEFORE any conversion or calibration."""
    if not isinstance(snapshot_bytes, (bytes, bytearray)) \
            or not isinstance(receipt_bytes, (bytes, bytearray)):
        raise Refusal("MF4_CATALOG_RECEIPT_UNBOUND",
                      "loader requires snapshot BYTES and receipt "
                      "BYTES, not objects")
    try:
        rec = _strict_loads(bytes(receipt_bytes))
    except Exception:                                   # noqa: BLE001
        raise Refusal("MF4_CATALOG_RECEIPT_UNPARSEABLE",
                      "acquisition receipt is not strict JSON")
    try:
        snap = _strict_loads(bytes(snapshot_bytes))
    except Exception:                                   # noqa: BLE001
        raise Refusal("MF4_CATALOG_SNAPSHOT_UNPARSEABLE",
                      "snapshot is not strict JSON")
    if not isinstance(rec, dict) or rec.get("schema") != RECEIPT_SCHEMA:
        raise Refusal("MF4_CATALOG_RECEIPT_SCHEMA",
                      repr((rec or {}).get("schema")
                           if isinstance(rec, dict) else type(rec))[:60])
    if not isinstance(snap, dict) \
            or snap.get("schema") != SNAPSHOT_SCHEMA:
        raise Refusal("MF4_CATALOG_SNAPSHOT_SCHEMA",
                      repr((snap or {}).get("schema")
                           if isinstance(snap, dict)
                           else type(snap))[:60])
    snap_sha = _sha(bytes(snapshot_bytes))
    if rec.get("snapshot_sha256") != snap_sha:
        raise Refusal("MF4_CATALOG_RECEIPT_BINDING",
                      f"snapshot bytes {snap_sha[:16]} != receipt "
                      f"{str(rec.get('snapshot_sha256'))[:16]}")
    dig = snapshot_table_sha(snap)
    if rec.get("canonical_event_table_sha256") != dig:
        raise Refusal("MF4_CATALOG_TABLE_DIGEST",
                      "receipt does not bind the recomputed canonical "
                      "table digest")
    if snap.get("temporal_role") != "CALIBRATION_LATE_REPAIR":
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      repr(snap.get("temporal_role"))[:60])
    if snap.get("temporal_role_policy") != TEMPORAL_ROLE_POLICY:
        raise Refusal("MF4_CATALOG_POLICY_UNBOUND",
                      "temporal_role_policy is not the registered "
                      "policy literal")
    _, contract_sha = query_contract()
    for obj, name in ((snap, "snapshot"), (rec, "receipt")):
        if obj.get("query_contract_sha256") != contract_sha:
            raise Refusal("MF4_CATALOG_CONTRACT_UNBOUND",
                          f"{name} does not bind the pinned query "
                          "contract")
    sa, ra = snap.get("authorization"), rec.get("authorization")
    if not isinstance(sa, dict) or not isinstance(ra, dict) \
            or sa != ra \
            or not isinstance(sa.get("sha256"), str) \
            or not sa.get("sha256"):
        raise Refusal("MF4_CATALOG_AUTH_UNBOUND",
                      "snapshot/receipt authorization identities are "
                      "absent or diverge")
    return snap, {"snapshot_sha256": snap_sha,
                  "canonical_event_table_sha256": dig,
                  "receipt_schema": rec["schema"],
                  "authorization": sa}


def events_from_snapshot(snapshot, use):
    """Role-guarded conversion: canonical table -> the frozen engine's
    event shape. `day` derives from the EXACT time_ms (UTC date).
    Callers must supply a snapshot that came through
    `load_verified_snapshot` -- calibrate_with_snapshot enforces
    that; direct callers still hit the role guard + bound digest."""
    calibration_snapshot_role_guard(snapshot, use)
    dig = snapshot_table_sha(snapshot)
    events = []
    for ev in snapshot["canonical_event_table"]:
        t = dt.datetime.fromtimestamp(ev["time_ms"] / 1000.0,
                                      dt.timezone.utc)
        events.append({"day": t.date().isoformat(),
                       "lat": ev["lat"], "lon": ev["lon"],
                       "mag": ev["mag"]})
    return events, dig


def calibrate_with_snapshot(risk_by_region, snapshot_bytes,
                            receipt_bytes, bboxes, regions,
                            freeze_day, snapshot_end,
                            requested_issue_end=None):
    """The real calibration entrypoint for the amended lane: verified
    receipt-bound bytes in, frozen engine unchanged, amended training
    digest out (policy + snapshot identity bound)."""
    snapshot, ident = load_verified_snapshot(snapshot_bytes,
                                             receipt_bytes)
    feat_events, dig = events_from_snapshot(snapshot,
                                            "calibration_features")
    label_events, dig2 = events_from_snapshot(snapshot,
                                              "calibration_labels")
    if dig != dig2 or feat_events != label_events:
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      "feature/label event views diverged")
    policy = snapshot["temporal_role_policy"]
    ledger = MF4.calibrate(risk_by_region, feat_events, bboxes, regions,
                           freeze_day, snapshot_end,
                           requested_issue_end=requested_issue_end)
    engine_digest = ledger["training_digest"]
    amended = _sha(engine_digest.encode("utf-8")
                   + policy.encode("utf-8")
                   + dig.encode("utf-8"))
    ledger["amended_training_digest"] = amended
    ledger["amended_training_binding"] = {
        "engine_training_digest": engine_digest,
        "temporal_role_policy_sha256": _sha(policy.encode("utf-8")),
        "canonical_event_table_sha256": dig,
        "snapshot_sha256": ident["snapshot_sha256"],
        "receipt_schema": ident["receipt_schema"],
        "authorization_sha256": ident["authorization"]["sha256"],
        "policy_source": ("docs/f2g_window2_execution/amendment_mf4_"
                          "late_catalog_repair_20260829_correction3.md")}
    return ledger


def live_prediction_events(view):
    """Live-side entry: EVERY input refuses typed. The late-repair
    calibration snapshot refuses as a role violation; every
    ISSUE_TIME_VIEW refuses because no registered issue-time-view
    receipt verifier exists yet (codex 2359Z blocker 2: a truthy path
    string is not a receipt). This function gains a positive path only
    when a registered verifier binding exact event bytes, issue day,
    source, and view digest lands and passes review."""
    role = (view or {}).get("temporal_role")
    if role == "CALIBRATION_LATE_REPAIR":
        raise Refusal("MF4_CATALOG_ROLE_VIOLATION",
                      "late-repair calibration snapshot offered to a "
                      "live prediction")
    if role != "ISSUE_TIME_VIEW":
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND", repr(role))
    raise Refusal("MF4_CATALOG_LIVE_UNVERIFIED",
                  "no registered issue-time-view receipt verifier "
                  "exists; every ISSUE_TIME_VIEW refuses until one is "
                  "registered, reviewed, and receipted")
