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
import w2_mf4_catalog_acquire_grassmann as ACQ
from w2_mf4_catalog_acquire_grassmann import (
    Refusal, TEMPORAL_ROLE_POLICY, _strict_loads,
    calibration_snapshot_role_guard, query_contract)

SNAPSHOT_SCHEMA = "geospec-mf4-calibration-catalog-snapshot-v1"
RECEIPT_SCHEMA = "geospec-mf4-catalog-acquisition-receipt-v1"

# Closed keysets (codex 0024Z Gate-2 quarantine repair): the receipt
# and snapshot admit EXACTLY the fire's registered fields -- any
# missing or injected key refuses.
RECEIPT_KEYSET = frozenset((
    "schema", "fired_utc", "attempts", "snapshot_file",
    "snapshot_sha256", "canonical_event_table_sha256",
    "authorization", "authorization_content",
    "acquisition_code_identity", "fault_segments_identity",
    "query_contract_sha256"))
SNAPSHOT_KEYSET = frozenset((
    "schema", "temporal_role", "temporal_role_policy", "amendment",
    "lane_status", "query_contract", "query_contract_sha256",
    "canonical_event_table", "canonical_event_table_sha256",
    "region_membership", "events_by_region_counts", "authorization",
    "authorization_content", "acquisition_code_identity",
    "fault_segments_identity"))
AUTH_CONTENT_KEYSET = frozenset((
    "schema", "public_head_commit", "public_head_tree",
    "module_git_blob_sha256", "query_contract_sha256", "codex_pass",
    "owner_fire_go", "output_target_must_be_absent"))
ROW_KEYSET = frozenset(("id", "lat", "lon", "mag", "time_ms",
                        "time_utc"))

# Codex 0257Z blocker 1: the RESULT COMMIT is part of the trust root.
# This later reviewed commit binds the earlier acquisition commit --
# no self-reference. The loader authenticates caller-supplied bytes
# against the bytes reopened from this committed public history
# before ANY other check.
CATALOG_COMMIT = "4893e63281e52d5a8e9d5047fe2aa2f445cd0dc4"
CATALOG_COMMIT_PARENT = "f636c234e0957ea7719091072bc91591f3fe9570"
CATALOG_SNAPSHOT_REL = ("docs/f2g_window2_execution/"
                        "mf4_catalog_snapshot/catalog_snapshot_v1.json")
CATALOG_RECEIPT_REL = ("docs/f2g_window2_execution/"
                       "mf4_catalog_snapshot/acquisition_receipt_v1.json")
CATALOG_SNAPSHOT_SHA256 = ("490c407796209a513995a9012911a1e37648256"
                           "22d4e33d355c983a13dbbb7f3")
CATALOG_RECEIPT_SHA256 = ("054002dd617b174e859645fce5c34a9adbc5b5c8"
                          "024a0eed54f55757ae29c9ff")
CATALOG_SNAPSHOT_BLOB_OID = "4c87f45f49662143a3189ea5eefc698f7c67e0d8"
CATALOG_RECEIPT_BLOB_OID = "55993672fa7f4a69215c2822c448496e2a415363"


def authenticate_result_bytes(snapshot_bytes, receipt_bytes):
    """Codex 0257Z blocker 1: a genuine authorization chain must not
    be replayable into forged result bytes. The caller-supplied pair
    must be BYTE-IDENTICAL to the pair reopened from the registered
    acquisition commit on the trusted public ref; that commit must be
    the registered descendant of the authorized head. Refuses typed
    BEFORE any other loader check."""
    if not isinstance(snapshot_bytes, (bytes, bytearray)) \
            or not isinstance(receipt_bytes, (bytes, bytearray)):
        raise Refusal("MF4_CATALOG_RECEIPT_UNBOUND",
                      "loader requires snapshot BYTES and receipt "
                      "BYTES, not objects")
    if _sha(bytes(snapshot_bytes)) != CATALOG_SNAPSHOT_SHA256 \
            or _sha(bytes(receipt_bytes)) != CATALOG_RECEIPT_SHA256:
        raise Refusal("MF4_CATALOG_RESULT_UNAUTHENTICATED",
                      "caller bytes are not the registered committed "
                      "acquisition pair")
    if not ACQ._is_ancestor(ACQ.REPO, CATALOG_COMMIT, "origin/master"):
        raise Refusal("MF4_CATALOG_RESULT_UNAUTHENTICATED",
                      "registered acquisition commit is not on the "
                      "trusted public ref")
    try:
        parent = ACQ._git(ACQ.REPO, "rev-parse",
                          CATALOG_COMMIT + "^").decode().strip()
    except (Refusal, Exception):                        # noqa: BLE001
        raise Refusal("MF4_CATALOG_RESULT_UNAUTHENTICATED",
                      "acquisition commit parent unreadable")
    if parent != CATALOG_COMMIT_PARENT:
        raise Refusal("MF4_CATALOG_RESULT_UNAUTHENTICATED",
                      "acquisition commit does not descend from the "
                      "authorized head")
    for rel, oid, sha, caller in (
            (CATALOG_SNAPSHOT_REL, CATALOG_SNAPSHOT_BLOB_OID,
             CATALOG_SNAPSHOT_SHA256, snapshot_bytes),
            (CATALOG_RECEIPT_REL, CATALOG_RECEIPT_BLOB_OID,
             CATALOG_RECEIPT_SHA256, receipt_bytes)):
        try:
            got_oid = ACQ._git(ACQ.REPO, "rev-parse",
                               f"{CATALOG_COMMIT}:{rel}") \
                .decode().strip()
            blob = ACQ._git(ACQ.REPO, "show",
                            f"{CATALOG_COMMIT}:{rel}")
        except (Refusal, Exception):                    # noqa: BLE001
            raise Refusal("MF4_CATALOG_RESULT_UNAUTHENTICATED",
                          f"committed result bytes unreadable: {rel}")
        if got_oid != oid or _sha(blob) != sha \
                or blob != bytes(caller):
            raise Refusal("MF4_CATALOG_RESULT_UNAUTHENTICATED",
                          f"reopened committed bytes diverge: {rel}")


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


def _num_finite(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and v not in (float("inf"), float("-inf")))


def _row_schema(table):
    """Codex 0024Z quarantine repair: every canonical row carries
    EXACTLY the registered fields with the registered types."""
    for i, ev in enumerate(table):
        if not isinstance(ev, dict) or set(ev) != ROW_KEYSET:
            raise Refusal("MF4_CATALOG_ROW_SCHEMA",
                          f"row {i}: keyset diverges from registered")
        if not isinstance(ev["id"], str) or not ev["id"]:
            raise Refusal("MF4_CATALOG_ROW_SCHEMA", f"row {i}: id")
        for k in ("lat", "lon", "mag"):
            if not _num_finite(ev[k]):
                raise Refusal("MF4_CATALOG_ROW_SCHEMA",
                              f"row {i}: {k} {ev[k]!r}")
        if not isinstance(ev["time_ms"], int) \
                or isinstance(ev["time_ms"], bool):
            raise Refusal("MF4_CATALOG_ROW_SCHEMA",
                          f"row {i}: time_ms {ev['time_ms']!r}")
        if not isinstance(ev["time_utc"], str) or not ev["time_utc"]:
            raise Refusal("MF4_CATALOG_ROW_SCHEMA",
                          f"row {i}: time_utc")


def verify_acquisition_trust_anchor(snap, rec):
    """Codex 0024Z Gate-2 quarantine repair, option 2: reopen and
    fully re-verify the committed fire-authorization chain and bind
    the RECOMPUTED identity to both snapshot and receipt. A mutually
    self-issued pair -- internally digest-consistent but with no
    committed pass/go chain -- refuses here. Committed records (the
    codex pass, the owner go, the public geospec tip) are the trust
    anchor; the embedded authorization_content is only the claim
    being re-verified."""
    ac, ac2 = snap.get("authorization_content"), \
        rec.get("authorization_content")
    if not isinstance(ac, dict) or ac != ac2:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "authorization_content absent or diverges "
                      "between snapshot and receipt")
    if set(ac) != AUTH_CONTENT_KEYSET:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "authorization_content keyset diverges")
    if ac.get("schema") != ACQ.AUTH_SCHEMA:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      f"authorization schema {ac.get('schema')!r}")
    _, contract_sha = query_contract()
    if ac.get("query_contract_sha256") != contract_sha:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "authorization contract pin diverges from "
                      "recompute")

    def _reachable(commit):
        try:
            got = ACQ._git(ACQ.FRAMEWORK_REPO, "merge-base", commit,
                           "origin/main").decode().strip()
            return got == ACQ._git(ACQ.FRAMEWORK_REPO, "rev-parse",
                                   commit).decode().strip()
        except (Refusal, Exception):                    # noqa: BLE001
            return False
    cp = ac.get("codex_pass") or {}
    go = ac.get("owner_fire_go") or {}
    for src, k in ((cp, "framework_commit"), (cp, "file"),
                   (go, "source_framework_commit"),
                   (go, "source_file")):
        if not isinstance(src.get(k), str) or not src.get(k):
            raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                          f"authorization chain pointer missing {k}")
    if not _reachable(cp["framework_commit"]):
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "pass commit unreachable from origin/main")
    try:
        pass_bytes = ACQ._git(ACQ.FRAMEWORK_REPO, "show",
                              f"{cp['framework_commit']}:{cp['file']}")
    except (Refusal, Exception):                        # noqa: BLE001
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "pass record unreadable from committed history")
    try:
        pr = _strict_loads(pass_bytes)
    except Exception:                                   # noqa: BLE001
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "committed pass record is not strict JSON")
    if not isinstance(pr, dict) or pr.get("verdict") != "PRE_HTTP_PASS":
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "committed pass verdict is not PRE_HTTP_PASS")
    if pr.get("module_git_blob_sha256") \
            != ac.get("module_git_blob_sha256") \
            or pr.get("query_contract_sha256") != contract_sha \
            or pr.get("result_tree") != ac.get("public_head_tree"):
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "committed pass identities diverge from the "
                      "authorization claim")
    if go["source_framework_commit"] == cp["framework_commit"]:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "go commit equals pass commit")
    if not _reachable(go["source_framework_commit"]) \
            or not ACQ._is_ancestor(ACQ.FRAMEWORK_REPO,
                                    cp["framework_commit"],
                                    go["source_framework_commit"]):
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "go commit unreachable or does not descend "
                      "from the pass commit")
    try:
        go_bytes = ACQ._git(ACQ.FRAMEWORK_REPO, "show",
                            f"{go['source_framework_commit']}:"
                            f"{go['source_file']}")
        gr = _strict_loads(go_bytes)
    except (Refusal, Exception):                        # noqa: BLE001
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "committed go record unreadable or not strict "
                      "JSON")
    if not isinstance(gr, dict) or gr.get("verdict") != "OWNER_FIRE_GO":
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "committed go verdict is not OWNER_FIRE_GO")
    if gr.get("pass_framework_commit") != cp["framework_commit"] \
            or gr.get("public_head_commit") \
            != ac.get("public_head_commit") \
            or gr.get("public_head_tree") \
            != ac.get("public_head_tree") \
            or gr.get("scope") != ACQ.SCOPE_LITERAL:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "committed go bindings diverge from the "
                      "authorization claim")
    # geospec side: the authorized head must be committed public
    # history with the exact claimed tree, and the module blob AT
    # that commit must be the pinned acquisition code
    head = ac["public_head_commit"]
    if not ACQ._is_ancestor(ACQ.REPO, head, "origin/master"):
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "authorized head is not public geospec "
                      "history")
    try:
        tree = ACQ._git(ACQ.REPO, "rev-parse",
                        head + "^{tree}").decode().strip()
        blob = ACQ._git(ACQ.REPO, "show",
                        f"{head}:{ACQ.MODULE_REL}")
    except (Refusal, Exception):                        # noqa: BLE001
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "authorized head/tree/blob unreadable from "
                      "public geospec history")
    if tree != ac["public_head_tree"]:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "authorized tree diverges from committed head")
    if _sha(blob) != ac["module_git_blob_sha256"]:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "module blob at the authorized head diverges "
                      "from the pinned identity")
    aci = rec.get("acquisition_code_identity") or {}
    if aci.get("git_blob_sha256") != ac["module_git_blob_sha256"]:
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "receipt code identity diverges from the "
                      "anchored module blob")
    if snap.get("fault_segments_identity") \
            != rec.get("fault_segments_identity") \
            or not isinstance(rec.get("fault_segments_identity"),
                              dict):
        raise Refusal("MF4_CATALOG_TRUST_ANCHOR",
                      "fault_segments identity absent or diverges")
    return {"pass_framework_commit": cp["framework_commit"],
            "go_framework_commit": go["source_framework_commit"],
            "public_head_commit": head,
            "public_head_tree": ac["public_head_tree"],
            "module_git_blob_sha256": ac["module_git_blob_sha256"]}


def verify_snapshot_semantics(snap, rec):
    """Codex 0257Z blocker 2: deterministic reconciliation of the
    acquisition RESULT against the registered contract -- commit
    authentication proves the bytes are the committed pair; this
    stage proves a committed pair is semantically well-formed before
    it becomes calibration material."""
    bboxes, _ = ACQ.build_bboxes()
    attempts = rec.get("attempts")
    counts = snap.get("events_by_region_counts")
    membership = snap.get("region_membership")
    table = snap.get("canonical_event_table")
    if not isinstance(attempts, dict) \
            or set(attempts) != set(ACQ.ADMITTED):
        raise Refusal("MF4_CATALOG_SEMANTICS",
                      "attempts do not cover exactly the 13 "
                      "registered regions")
    if not isinstance(counts, dict) \
            or set(counts) != set(ACQ.ADMITTED):
        raise Refusal("MF4_CATALOG_SEMANTICS",
                      "region counts do not cover exactly the 13 "
                      "registered regions")
    if not isinstance(membership, dict):
        raise Refusal("MF4_CATALOG_SEMANTICS",
                      "region membership absent")
    for region in ACQ.ADMITTED:
        att = attempts[region]
        url, _params = ACQ.query_url(bboxes[region]["bbox"])
        if not isinstance(att, dict) \
                or att.get("region") != region \
                or att.get("requested_url") != url \
                or att.get("bbox") != bboxes[region]["bbox"] \
                or att.get("carrier") != bboxes[region]["carrier"] \
                or att.get("http_status") != 200 \
                or att.get("parse_result") != "OK" \
                or att.get("event_count") != counts.get(region):
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"attempt record for {region} diverges "
                          "from the registered contract/result")
    ids = [ev["id"] for ev in table]
    if len(set(ids)) != len(ids):
        raise Refusal("MF4_CATALOG_SEMANTICS",
                      "canonical table ids are not unique")
    t_lo = int(dt.datetime(2025, 10, 11,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)
    t_hi = int(dt.datetime(2026, 8, 28,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)
    prev = None
    for ev in table:
        key = (ev["time_ms"], ev["id"])
        if prev is not None and key < prev:
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"table order violation at {ev['id']}")
        prev = key
        if not (ACQ.MINMAG <= ev["mag"]):
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"{ev['id']}: magnitude below registered "
                          "threshold")
        if not (t_lo <= ev["time_ms"] < t_hi):
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"{ev['id']}: time outside registered "
                          "window")
        t = dt.datetime.fromtimestamp(ev["time_ms"] / 1000.0,
                                      dt.timezone.utc)
        rendered = t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        if ev["time_utc"] != rendered:
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"{ev['id']}: time_utc is not the exact "
                          "millisecond rendering of time_ms")
        regions = membership.get(ev["id"])
        if not isinstance(regions, list) or not regions \
                or regions != sorted(set(regions)) \
                or any(r not in bboxes for r in regions):
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"{ev['id']}: membership regions invalid")
        for r in regions:
            bb = bboxes[r]["bbox"]
            if not (bb["min_lat"] <= ev["lat"] <= bb["max_lat"]
                    and bb["min_lon"] <= ev["lon"] <= bb["max_lon"]):
                raise Refusal("MF4_CATALOG_SEMANTICS",
                              f"{ev['id']}: outside listed region "
                              f"bbox {r}")
    if set(membership) != set(ids):
        raise Refusal("MF4_CATALOG_SEMANTICS",
                      "membership keys diverge from table ids")
    for region in ACQ.ADMITTED:
        recount = sum(1 for regs in membership.values()
                      if region in regs)
        if recount != counts[region]:
            raise Refusal("MF4_CATALOG_SEMANTICS",
                          f"{region}: count {counts[region]} != "
                          f"membership recompute {recount}")
    if sum(counts.values()) != sum(len(v)
                                   for v in membership.values()):
        raise Refusal("MF4_CATALOG_SEMANTICS",
                      "regional count total diverges from membership "
                      "cardinality (undisclosed dedup drift)")


def load_verified_snapshot(snapshot_bytes, receipt_bytes):
    """The ONLY sanctioned loader for calibration catalog material:
    the caller bytes are first AUTHENTICATED against the registered
    committed acquisition pair (codex 0257Z blocker 1), then the full
    receipt/table/role/chain/semantics validation runs. Every check
    refuses typed BEFORE any conversion or calibration."""
    authenticate_result_bytes(snapshot_bytes, receipt_bytes)
    return _validate_pair(snapshot_bytes, receipt_bytes)


def _validate_pair(snapshot_bytes, receipt_bytes):
    """Post-authentication validation stage. NOT a sanctioned entry
    point -- consumers go through load_verified_snapshot (which
    authenticates first); this seam exists so the mutation locks can
    exercise every deeper check with fixture pairs that could never
    pass byte authentication."""
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
    # codex 0024Z quarantine repair: closed keysets, named snapshot
    # path, strict fired UTC, registered row schema, and the
    # committed-chain trust anchor
    if set(rec) != RECEIPT_KEYSET:
        raise Refusal("MF4_CATALOG_RECEIPT_KEYSET",
                      f"receipt keyset diverges: "
                      f"{sorted(set(rec) ^ RECEIPT_KEYSET)[:4]}")
    if set(snap) != SNAPSHOT_KEYSET:
        raise Refusal("MF4_CATALOG_SNAPSHOT_KEYSET",
                      f"snapshot keyset diverges: "
                      f"{sorted(set(snap) ^ SNAPSHOT_KEYSET)[:4]}")
    if rec.get("snapshot_file") != "catalog_snapshot_v1.json":
        raise Refusal("MF4_CATALOG_RECEIPT_BINDING",
                      f"receipt names snapshot file "
                      f"{rec.get('snapshot_file')!r}")
    try:
        dt.datetime.strptime(rec.get("fired_utc", ""),
                             "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        raise Refusal("MF4_CATALOG_RECEIPT_BINDING",
                      f"fired_utc {rec.get('fired_utc')!r} is not "
                      "strict UTC")
    _row_schema(snap["canonical_event_table"])
    anchor = verify_acquisition_trust_anchor(snap, rec)
    verify_snapshot_semantics(snap, rec)
    return snap, {"snapshot_sha256": snap_sha,
                  "canonical_event_table_sha256": dig,
                  "receipt_schema": rec["schema"],
                  "authorization": sa,
                  "trust_anchor": anchor}


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
        "trust_anchor": ident["trust_anchor"],
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
