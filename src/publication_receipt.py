#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6 §1 publication receipt — the SERVER-stamped hit-clock (GeoSpec Forward Plan P2 item 1).

Implements the contract fixed by `tests/test_publication_receipt_redkats_cayley.py` (cayley, geospec 6280b1c),
UNEDITED (the Iberia decouple applied to GeoSpec: cayley authors the red bar, grassmann implements to it).

Why a server stamp: the hit-clock (when an alarm actually became publicly available) must be bound to a durable,
SERVER-side deployment record — a GitHub Pages build / Actions run whose `created_at` the client cannot forge. A
git commit timestamp is client-controlled and therefore insufficient; a client-stamped receipt is *worse than
none* because it looks durable. A day without a schema-valid server receipt is INELIGIBLE for hit credit, and an
availability earlier than the server stamp is NEVER synthesized — absence degrades conservatively to the
23:59:59Z ceiling of the day, hit-ineligible, never to any earlier time.

Interface (consumed by run_and_publish + the R4 prospective scorer):
    build_publication_receipt(artifact_paths, commit_sha, deployment) -> receipt dict
    verify_publication_receipt(receipt, artifact_bytes)               -> True | raise ValueError
    alarm_available_at_utc(day_iso, receipt|None)                     -> ISO-8601 UTC str
    day_eligible_for_hit(day_record)                                  -> bool
"""
import datetime
import hashlib

SCHEMA = "geospec-publication-receipt-v1"

# The R6 §1 gate: ONLY a source that names a recognized SERVER deployment API grants the hit-clock. This is an
# allowlist, not a denylist — an unknown/arbitrary source is not *proven* server-side, so it is refused
# conservatively (a denylist would admit `source="anything"` as durable, exactly the "looks durable but isn't"
# failure R6 §1 exists to prevent). The named client-side sources (git-commit-timestamp, local-clock, "",
# missing) are simply the canonical non-server cases. Extend this set deliberately when a new *server* deployment
# API is genuinely wired.
_SERVER_SOURCES = frozenset({"github-pages-build", "github-actions-run"})


def _parse_utc(ts):
    """Parse an ISO-8601 UTC timestamp; raise ValueError on empty/non-string/unparseable/non-UTC. UTC means a
    trailing `Z` or an explicit `+00:00` offset — a naive or non-zero-offset stamp is refused."""
    if not isinstance(ts, str) or not ts:
        raise ValueError(f"server created_at must be a non-empty ISO-8601 UTC string, got {ts!r}")
    norm = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    try:
        dt = datetime.datetime.fromisoformat(norm)
    except ValueError as exc:
        raise ValueError(f"server created_at {ts!r} is not parseable ISO-8601: {exc}") from exc
    if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"server created_at {ts!r} must be UTC (trailing Z or +00:00)")
    return dt


def _validate_deployment(deployment):
    """Enforce that `deployment` is a SERVER-side record and return its canonical {id, created_at, source}.
    Raises ValueError if it is not a dict, lacks a non-empty id, carries a client-side/unknown/missing source,
    or has an empty/missing/unparseable/non-UTC created_at."""
    if not isinstance(deployment, dict):
        raise ValueError("deployment must be a server-side record dict")
    dep_id = deployment.get("id")
    if not (isinstance(dep_id, str) and dep_id):
        raise ValueError("deployment.id must be a non-empty server deployment id")
    source = deployment.get("source")
    if source not in _SERVER_SOURCES:
        raise ValueError(
            f"deployment.source {source!r} does not name a recognized server API "
            f"{sorted(_SERVER_SOURCES)} — a client-stamped receipt "
            "(git-commit-timestamp / local-clock / '' / missing) is refused: it looks durable but is not")
    _parse_utc(deployment.get("created_at"))            # raises on empty / missing / unparseable / non-UTC
    return {"id": dep_id, "created_at": deployment["created_at"], "source": source}


def build_publication_receipt(artifact_paths, commit_sha, deployment):
    """Build a durable publication receipt binding {artifact sha256 hashes, commit SHA, server deployment record}.

    artifact_paths: {repo_relpath: abs_path} — every file is read and sha256-hashed into receipt.artifact_hashes.
    deployment: a SERVER-side record; a client-side / unknown / incomplete deployment raises ValueError (refuse to
    build — better no receipt than a forgeable one). `built_utc` is the (informational) build wall-clock; the
    AUTHORITATIVE hit-clock is deployment.created_at.
    """
    dep = _validate_deployment(deployment)              # fail-closed BEFORE hashing anything
    if not (isinstance(commit_sha, str) and commit_sha):
        raise ValueError("commit_sha must be a non-empty string")
    if not (isinstance(artifact_paths, dict) and artifact_paths):
        raise ValueError("artifact_paths must be a non-empty {repo_relpath: abs_path} mapping")
    artifact_hashes = {}
    for relpath, abs_path in artifact_paths.items():
        with open(abs_path, "rb") as fh:
            artifact_hashes[relpath] = hashlib.sha256(fh.read()).hexdigest()
    return {
        "schema": SCHEMA,
        "artifact_hashes": artifact_hashes,
        "commit_sha": commit_sha,
        "deployment": dep,
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _validate_receipt_structure(receipt):
    """Structural + server-side validity of a receipt (NO byte re-hash — that is verify's job). Raises ValueError
    on any defect. Returns the receipt on success."""
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be a dict")
    if receipt.get("schema") != SCHEMA:
        raise ValueError(f"receipt schema {receipt.get('schema')!r} != {SCHEMA!r}")
    ah = receipt.get("artifact_hashes")
    if not (isinstance(ah, dict) and ah
            and all(isinstance(k, str) and isinstance(v, str) and v for k, v in ah.items())):
        raise ValueError("receipt.artifact_hashes must be a non-empty {relpath: sha256hex} mapping")
    if not (isinstance(receipt.get("commit_sha"), str) and receipt.get("commit_sha")):
        raise ValueError("receipt.commit_sha must be a non-empty string")
    if not (isinstance(receipt.get("built_utc"), str) and receipt.get("built_utc")):
        raise ValueError("receipt.built_utc must be a non-empty string")
    _validate_deployment(receipt.get("deployment"))     # server source + parseable UTC created_at
    return receipt


def _is_valid_server_receipt(receipt):
    """Boolean form of _validate_receipt_structure — True iff `receipt` is a schema-valid server-side receipt."""
    try:
        _validate_receipt_structure(receipt)
        return True
    except ValueError:
        return False


def verify_publication_receipt(receipt, artifact_bytes):
    """Re-verify a receipt against the actual artifact bytes. Returns True, or raises ValueError on any
    structural defect, artifact-set mismatch (missing/extra), or hash mismatch."""
    _validate_receipt_structure(receipt)
    if not isinstance(artifact_bytes, dict):
        raise ValueError("artifact_bytes must be a {repo_relpath: bytes} mapping")
    recorded = receipt["artifact_hashes"]
    if set(recorded) != set(artifact_bytes):
        missing = sorted(set(recorded) - set(artifact_bytes))
        extra = sorted(set(artifact_bytes) - set(recorded))
        raise ValueError(f"artifact set mismatch: missing bytes for {missing}, unexpected bytes for {extra}")
    for relpath, want in recorded.items():
        got = hashlib.sha256(artifact_bytes[relpath]).hexdigest()
        if got != want:
            raise ValueError(f"artifact hash mismatch for {relpath}: recomputed {got} != receipt {want}")
    return True


def alarm_available_at_utc(day_iso, receipt):
    """When alarms published on `day_iso` (YYYY-MM-DD) became publicly available.

    A schema-valid server receipt => EXACTLY its server stamp (deployment.created_at) — never adjusted earlier.
    No receipt (or a receipt that is not a valid server record) => the conservative 23:59:59Z ceiling of the day,
    NEVER any earlier value. Availability is thus never synthesized to an unprovable earlier time.
    """
    if receipt is not None and _is_valid_server_receipt(receipt):
        return receipt["deployment"]["created_at"]
    return f"{day_iso}T23:59:59Z"


def day_eligible_for_hit(day_record):
    """True IFF day_record carries a schema-valid server-side publication receipt. The R4 prospective scorer uses
    this: receipt-less (or client-stamped) days may still be scored for FALSE-ALARM accounting but can never earn
    HIT credit."""
    if not isinstance(day_record, dict):
        return False
    return _is_valid_server_receipt(day_record.get("publication_receipt"))
