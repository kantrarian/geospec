#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6 §1 publication receipt — REV 3: verify-then-admit, carrier-bound day, sealed standing.

Implements the contract fixed by `tests/test_publication_receipt_redkats_cayley.py` (cayley, geospec 216780a),
UNEDITED, under codex's rev-2/rev-3 WORKS-WITH-FIX findings (the decouple).

Standing requires VERIFICATION, not structure (codex B1). `day_eligible_for_hit` is True ONLY for a typed
`VerifiedReceipt`, which is SEALED behind admission: `admit_receipt` is the only minting path. Admission (a)
re-hashes every recorded artifact from INDEPENDENTLY loaded bytes (git-object seam), (b) binds the day by the
REOPENED canonical carrier's `["date"]` (not the mutable receipt field), and (c) reopens the named GitHub Pages
server record and matches id/status=built/no-error/commit/completion-stamp (codex B2). Availability is the
COMPLETION stamp `updated_at` (codex HIGH clock, per docs/CORRECTION_2026-08-07_receipt_availability_completion_stamp.md).
Everything fails closed; no synthetic deployments, no fallbacks, no backfill.
"""
import datetime
import hashlib
import json
import re

SCHEMA = "geospec-publication-receipt-v2"
MANDATORY_ARTIFACTS = ("docs/ensemble_latest.json", "docs/data.csv")           # alarm carrier + scoring carrier
ARTIFACT_ALLOWLIST = MANDATORY_ARTIFACTS + ("docs/validated_events.json",
                                            "docs/r4_prospective_record.json", "docs/r5_daily.json")
_CARRIER = "docs/ensemble_latest.json"                                          # binds the day via its ["date"]

# Only a recognized SERVER deployment API label is a build-time sanity gate; the AUTHORITATIVE check is the
# server-record reopen in admit_receipt (a label alone is not an attestation — codex B2).
_SERVER_SOURCES = frozenset({"github-pages-build", "github-actions-run"})
_DEPLOYMENT_KEYS = frozenset({"id", "api_url", "status", "error", "created_at", "updated_at", "source"})
_RECEIPT_KEYS = frozenset({"schema", "day", "artifact_hashes", "commit_sha", "deployment",
                           "availability_utc", "built_utc"})
_40HEX = re.compile(r"[0-9a-f]{40}\Z")
_64HEX = re.compile(r"[0-9a-f]{64}\Z")
_MINT_TOKEN = object()                                                          # module-private admission token


def _api_url_for(build_id):
    return f"https://api.github.com/repos/kantrarian/geospec/pages/builds/{build_id}"


def _parse_day(s):
    if not isinstance(s, str):
        raise ValueError(f"day must be a YYYY-MM-DD string, got {s!r}")
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()                     # strict; "08/05/2026" raises


def _parse_utc(ts):
    if not isinstance(ts, str) or not ts:
        raise ValueError(f"timestamp must be a non-empty ISO-8601 UTC string, got {ts!r}")
    norm = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    dt = datetime.datetime.fromisoformat(norm)                                  # raises ValueError on garbage
    if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"timestamp {ts!r} must be UTC (Z / +00:00)")
    return dt


def _require_commit(sha):
    if not (isinstance(sha, str) and _40HEX.match(sha)):
        raise ValueError(f"commit_sha must be lowercase 40-hex, got {sha!r}")


def _validate_deployment(dep):
    """Exact server-deployment record: EXACT keyset, built + error-free, created<=updated, allowlisted source,
    api_url pinned to this repo AND consistent with id. Raises ValueError on any defect. Returns the record."""
    if not isinstance(dep, dict) or frozenset(dep) != _DEPLOYMENT_KEYS:
        raise ValueError("deployment must carry EXACTLY {id,api_url,status,error,created_at,updated_at,source}")
    if dep["status"] != "built":
        raise ValueError(f"deployment.status {dep['status']!r} != 'built'")
    if dep["error"] not in (None, ""):
        raise ValueError(f"deployment.error must be empty, got {dep['error']!r}")
    if dep["source"] not in _SERVER_SOURCES:
        raise ValueError(f"deployment.source {dep['source']!r} not a recognized server API {sorted(_SERVER_SOURCES)}")
    if not (isinstance(dep["id"], str) and dep["id"]):
        raise ValueError("deployment.id must be a non-empty string")
    if dep["api_url"] != _api_url_for(dep["id"]):
        raise ValueError("deployment.api_url must be the pinned kantrarian/geospec builds/<id> URL")
    if _parse_utc(dep["created_at"]) > _parse_utc(dep["updated_at"]):
        raise ValueError("deployment.created_at must be <= updated_at")
    return dep


def build_publication_receipt(day, artifact_paths, commit_sha, deployment):
    """Build a schema-v2 receipt binding {day, artifact sha256s, commit, server deployment}. The day is bound at
    build by the canonical carrier's ["date"]; availability = the deployment COMPLETION stamp (updated_at). Fails
    closed on a bad day/commit, a non-server/incomplete deployment, or an artifact set that violates the
    mandatory/allowlist policy or whose carrier date != day."""
    _parse_day(day)
    _require_commit(commit_sha)
    _validate_deployment(deployment)
    if not isinstance(artifact_paths, dict) or not artifact_paths:
        raise ValueError("artifact_paths must be a non-empty {repo_relpath: abs_path} mapping")
    for rel in artifact_paths:
        if rel not in ARTIFACT_ALLOWLIST:
            raise ValueError(f"artifact {rel!r} is not in the allowlist {ARTIFACT_ALLOWLIST}")
    for m in MANDATORY_ARTIFACTS:
        if m not in artifact_paths:
            raise ValueError(f"mandatory carrier {m!r} missing from artifact_paths")
    artifact_hashes = {}
    for rel, path in artifact_paths.items():
        with open(path, "rb") as fh:
            artifact_hashes[rel] = hashlib.sha256(fh.read()).hexdigest()
    with open(artifact_paths[_CARRIER], "rb") as fh:
        carrier = json.loads(fh.read().decode("utf-8"))
    if carrier.get("date") != day:
        raise ValueError(f"carrier {_CARRIER} date {carrier.get('date')!r} != day {day!r}")
    return {
        "schema": SCHEMA,
        "day": day,
        "artifact_hashes": artifact_hashes,
        "commit_sha": commit_sha,
        "deployment": {k: deployment[k] for k in _DEPLOYMENT_KEYS},
        "availability_utc": deployment["updated_at"],
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def verify_publication_receipt(receipt, artifact_bytes):
    """Byte binding (rev-1 role): recompute every recorded hash from the supplied bytes; exact keyset. True or
    ValueError."""
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be a dict")
    ah = receipt.get("artifact_hashes")
    if not (isinstance(ah, dict) and ah) or not isinstance(artifact_bytes, dict):
        raise ValueError("artifact_hashes and artifact_bytes must be non-empty dicts")
    if set(ah) != set(artifact_bytes):
        raise ValueError(f"artifact set mismatch: hashes {sorted(ah)} vs bytes {sorted(artifact_bytes)}")
    for rel, want in ah.items():
        if hashlib.sha256(artifact_bytes[rel]).hexdigest() != want:
            raise ValueError(f"artifact hash mismatch for {rel}")
    return True


class VerifiedReceipt:
    """A receipt that has PASSED admission — the only standing-bearing type. SEALED: direct construction raises;
    instances exist only via `admit_receipt` (which supplies the module-private mint token)."""
    __slots__ = ("day", "availability_utc", "receipt", "_minted")

    def __init__(self, day, availability_utc, receipt, _token=None):
        if _token is not _MINT_TOKEN:
            raise ValueError("VerifiedReceipt is sealed — mint only via admit_receipt()")
        self.day = day
        self.availability_utc = availability_utc
        self.receipt = receipt
        self._minted = True


def _is_minted(x):
    return isinstance(x, VerifiedReceipt) and getattr(x, "_minted", False) is True


def admit_receipt(receipt, day, artifact_loader, server_record_loader):
    """Mint a VerifiedReceipt IFF every fail-closed check passes; else raise ValueError. `artifact_loader(commit,
    relpath)->bytes` and `server_record_loader(api_url)->dict` are the independent evidence seams (production:
    git cat-file blob, gh api). No check may be skipped; a raise from a loader is a fail-closed rejection."""
    # structure + exact keyset + schema
    if not isinstance(receipt, dict) or frozenset(receipt) != _RECEIPT_KEYS:
        raise ValueError("receipt must carry EXACTLY the schema-v2 keyset")
    if receipt["schema"] != SCHEMA:
        raise ValueError(f"receipt schema {receipt['schema']!r} != {SCHEMA!r}")
    # day: the receipt field must match the request (carrier re-binds it below)
    _parse_day(day)
    _parse_day(receipt["day"])
    if receipt["day"] != day:
        raise ValueError(f"receipt.day {receipt['day']!r} != requested day {day!r}")
    _require_commit(receipt["commit_sha"])
    # deployment + availability == completion stamp
    dep = _validate_deployment(receipt["deployment"])
    if receipt["availability_utc"] != dep["updated_at"]:
        raise ValueError("receipt.availability_utc != deployment.updated_at (completion stamp)")
    _parse_utc(receipt["availability_utc"])
    # artifact policy: non-empty, allowlist, mandatory carriers, lowercase-64hex
    ah = receipt["artifact_hashes"]
    if not (isinstance(ah, dict) and ah):
        raise ValueError("artifact_hashes must be a non-empty mapping (a receipt must attest something)")
    for rel, h in ah.items():
        if rel not in ARTIFACT_ALLOWLIST:
            raise ValueError(f"artifact {rel!r} not in allowlist")
        if not (isinstance(h, str) and _64HEX.match(h)):
            raise ValueError(f"artifact hash for {rel!r} is not lowercase 64-hex")
    for m in MANDATORY_ARTIFACTS:
        if m not in ah:
            raise ValueError(f"mandatory carrier {m!r} missing from receipt")
    # re-hash EVERY recorded artifact from independently loaded bytes
    loaded = {}
    for rel, want in ah.items():
        try:
            data = artifact_loader(receipt["commit_sha"], rel)
        except Exception as exc:
            raise ValueError(f"artifact {rel!r} not loadable: {exc}")
        if not isinstance(data, (bytes, bytearray)) or hashlib.sha256(data).hexdigest() != want:
            raise ValueError(f"artifact {rel!r} bytes do not match the recorded hash")
        loaded[rel] = bytes(data)
    # the CARRIER binds the day — reopen and parse ["date"]; a mutable receipt field never binds it
    try:
        carrier = json.loads(loaded[_CARRIER].decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"carrier {_CARRIER} not parseable JSON: {exc}")
    if carrier.get("date") != receipt["day"] or carrier.get("date") != day:
        raise ValueError(f"carrier date {carrier.get('date')!r} != receipt.day/requested day")
    # reopen the named server record and match it
    try:
        rec = server_record_loader(dep["api_url"])
    except Exception as exc:
        raise ValueError(f"server record not reopenable at {dep['api_url']}: {exc}")
    if not isinstance(rec, dict):
        raise ValueError("server record must be a dict")
    if rec.get("id") != dep["id"]:
        raise ValueError("server record id != deployment.id")
    if rec.get("status") != "built":
        raise ValueError("server record status != 'built'")
    if rec.get("error") not in (None, ""):
        raise ValueError("server record carries an error")
    if rec.get("commit") != receipt["commit_sha"]:
        raise ValueError("server record commit != receipt.commit_sha")
    if _parse_utc(rec.get("created_at")) > _parse_utc(rec.get("updated_at")):
        raise ValueError("server record created_at > updated_at")
    if rec.get("updated_at") != receipt["availability_utc"] or rec.get("updated_at") != dep["updated_at"]:
        raise ValueError("server record updated_at != availability_utc/deployment.updated_at")
    return VerifiedReceipt(receipt["day"], receipt["availability_utc"], receipt, _token=_MINT_TOKEN)


def day_eligible_for_hit(x):
    """True IFF x is an ADMISSION-minted VerifiedReceipt. Every dict (valid-looking or forged) is False; a
    directly-constructed instance cannot exist (construction raises) and would be rejected anyway."""
    return _is_minted(x)


def alarm_available_at_utc(day, verified):
    """A minted VerifiedReceipt => its completion-stamp availability EXACTLY (before OR after the ceiling). None
    (or anything not admission-minted) => the conservative `{day}T23:59:59Z` ceiling, never earlier."""
    if _is_minted(verified):
        return verified.availability_utc
    return f"{day}T23:59:59Z"
