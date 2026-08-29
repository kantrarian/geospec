#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF4 calibration catalog adapter (grassmann) -- codex 2026-08-29
2014Z item 5: the temporal-role guard wired onto the REAL consumer
path. The frozen engine `w2_mf4.py` is byte-untouched; this adapter is
the ONLY sanctioned way to feed the late-repair catalog snapshot into
`w2_mf4.calibrate()`:

- `events_from_snapshot(snapshot, use)` invokes
  `calibration_snapshot_role_guard` for the requested role, verifies
  the canonical-table digest against its own bytes, and converts the
  canonical event table into the frozen engine's event shape
  ({day, lat, lon, mag}; day = UTC date of the exact time_ms).
- `calibrate_with_snapshot(...)` guards BOTH roles (features + labels),
  runs the frozen `w2_mf4.calibrate` unchanged, then binds the
  AMENDED training digest:
      amended = sha256(engine_training_digest
                       || temporal_role_policy bytes
                       || canonical snapshot sha256)
  so the correction-3 policy and the exact snapshot identity enter the
  training identity (a policy or snapshot change moves the digest).
- `live_prediction_events(view)` is the live-side entry: it requires a
  separately receipted issue-time view (`temporal_role:
  ISSUE_TIME_VIEW` + a receipt reference) and REFUSES the late-repair
  calibration snapshot (`CALIBRATION_LATE_REPAIR`) typed.
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
    Refusal, calibration_snapshot_role_guard)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def snapshot_table_sha(snapshot):
    table = snapshot.get("canonical_event_table")
    if not isinstance(table, list):
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      "snapshot lacks canonical_event_table")
    raw = json.dumps(table, sort_keys=True).encode("utf-8")
    dig = _sha(raw)
    bound = snapshot.get("canonical_event_table_sha256")
    if bound is not None and bound != dig:
        raise Refusal("MF4_CATALOG_TABLE_DIGEST",
                      f"{dig[:16]} != bound {str(bound)[:16]}")
    return dig


def events_from_snapshot(snapshot, use):
    """Role-guarded conversion: canonical table -> the frozen engine's
    event shape. `day` derives from the EXACT time_ms (UTC date)."""
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


def calibrate_with_snapshot(risk_by_region, snapshot, bboxes, regions,
                            freeze_day, snapshot_end,
                            requested_issue_end=None):
    """The real calibration entrypoint for the amended lane. Both
    temporal roles are guarded; the frozen engine runs unchanged; the
    amended training digest binds policy + snapshot identity."""
    feat_events, dig = events_from_snapshot(snapshot,
                                            "calibration_features")
    label_events, dig2 = events_from_snapshot(snapshot,
                                              "calibration_labels")
    if dig != dig2 or feat_events != label_events:
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      "feature/label event views diverged")
    policy = snapshot.get("temporal_role_policy")
    if not isinstance(policy, str) or "AMENDED_AFTER_FREEZE" not in policy:
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      "temporal_role_policy absent or unamended")
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
        "policy_source": ("docs/f2g_window2_execution/amendment_mf4_"
                          "late_catalog_repair_20260829_correction3.md")}
    return ledger


def live_prediction_events(view):
    """Live-side entry: post-2026-08-29 prediction features use ONLY a
    separately receipted issue-time view; the late-repair calibration
    snapshot refuses typed."""
    role = (view or {}).get("temporal_role")
    if role == "CALIBRATION_LATE_REPAIR":
        raise Refusal("MF4_CATALOG_ROLE_VIOLATION",
                      "late-repair calibration snapshot offered to a "
                      "live prediction")
    if role != "ISSUE_TIME_VIEW":
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND", repr(role))
    if not view.get("issue_time_receipt"):
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND",
                      "issue-time view lacks its receipt reference")
    return view["events"]
