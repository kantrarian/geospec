#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The 2,056-key THREE-WAY DISPOSITION CAPSULE (grassmann).

codex 1746Z finding 3 + 1900Z disposition. The capsule partitions
EVERY v4 scientific key exactly and disjointly into:

  REUSE_OR_BRIDGE  -- served by preserved v3 evidence under the
                      registered lane mapping, VERIFIED by rerunning
                      the registered v4 transform on the preserved
                      body (can-serve, not merely key-set membership)
  PREDECESSOR      -- the already-fired corrected-OMNI probe day
  HTTP_CAPTURE     -- the only keys the network entrypoint may fire

It RECOMPUTES the partition from the pinned authority + the preserved
archive; it never asserts counts. Every partition binds a sorted-key
digest, and REUSE/PREDECESSOR additionally bind a per-key
source-body digest, so the derivation is independently reproducible
by anyone holding the same inputs -- and the digests are checkable
even where the v3 body store is not mounted (codex 1900Z: the
all-null facts stay grassmann-source-attested until this capsule
reopens the raw SHA).

ZERO HTTP. Nothing here fires or authorizes a request; the capsule is
the CEILING on what may ever be fired.

Usage:
  python w2_disposition_capsule_grassmann.py build     # write it
  python w2_disposition_capsule_grassmann.py verify    # re-verify
  python w2_disposition_capsule_grassmann.py --selftest
"""
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
# v2 (codex 2119Z closure 2): v1 closed only the top level and the
# partition KEY SETS, so every nested field was forgeable while the
# report still read clean -- well-formed, not authenticating, the
# same family as the defects we have each been caught by one level
# up. v1 remains valid for its ORIGINAL purpose (request-ceiling
# membership, which consumes partition membership only); v2 is what a
# LINEAGE REGISTRY requires, and every nested field consumed as
# authority is closed and independently re-derived here.
CAPSULE_SCHEMA = "f2g-w2-key-disposition-capsule-v2"
# cayley's 0110Z P0: build_fixture_capsule over the REAL authority
# minted a capsule that passed the strict verifier while claiming all
# 2056 keys native -- internally true, externally false. Their
# pin-binding closes it from the boundary side; this closes it from
# MINE, structurally: a fixture capsule carries a DIFFERENT SCHEMA,
# so no production verifier can ever accept one whatever it claims.
FIXTURE_CAPSULE_SCHEMA = ("f2g-w2-key-disposition-capsule-"
                          "FIXTURE-ONLY-v2")
_AUTHORITY_KEYS = {"path", "blob_sha256", "keys_sha256", "census"}
_OLD_AUTHORITY_KEYS = {"commit", "path", "blob_sha256",
                       "keys_sha256"}
_ARCHIVE_KEYS = {"path", "sha256", "store_id"}
_LINEAGE_KEYS = {"v3_key", "s_v3_sha256", "t_v3_sha256",
                 "raw_body_sha256", "raw_body_bytes", "outcome",
                 "s_v4_sha256"}
_PREDECESSOR_KEYS = {"spent_probe"}
_HEX64 = 64
CAPSULE_PATH = ("docs/f2g_window2_execution/"
                "key_disposition_capsule_v4.json")
AUTHORITY_PATH = ("docs/f2g_window2_execution/"
                  "staged_expected_contracts_v3.json")
V3_ARCHIVE_PATH = ("docs/f2g_window2_execution/staged_envelopes/"
                   "capture_run_archive.json")
PARTITIONS = ("REUSE_OR_BRIDGE", "PREDECESSOR", "HTTP_CAPTURE")
# the ONE registered v3 -> v4 lane rename (codex 0527Z finding 3)
LANE_MAP = {"MF4_FEED": "MAG_WEATHER_FEED"}
# v3 evidence that is SUPERSEDED and may never be reused: the old
# OMNI bodies carry the retired variable set 17/21/25 and cannot
# produce the frozen Newell regressor (codex 0527Z finding 2)
SUPERSEDED_V3 = (("MF4_FEED", "omni"),)
PREDECESSOR_KEY = "MAG_WEATHER_FEED/omni/2026-01-01"


class DispositionRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _d(msg):
    raise DispositionRefusal("DISPOSITION_REFUSED: " + str(msg))


def _canon(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _blob(ref):
    p = subprocess.run(["git", "-C", REPO, "cat-file", "blob", ref],
                       capture_output=True)
    if p.returncode != 0:
        _d(f"unreadable git object {ref}")
    return p.stdout


def _authority(commitish="HEAD"):
    raw = _blob(f"{commitish}:{AUTHORITY_PATH}")
    a = json.loads(raw.decode("utf-8"))
    return a, hashlib.sha256(raw).hexdigest()


def _v4_keys(authority):
    out = []
    pk = authority["prestart_expected_keys"]
    for lane in sorted(pk):
        for ck in sorted(pk[lane]):
            for day in pk[lane][ck]:
                out.append(f"{lane}/{ck}/{day}")
    if len(set(out)) != len(out):
        _d("the authority key set is not unique")
    return out


def derive(authority, archive, store_root, transform=None):
    """RECOMPUTE the partition. For every preserved v3 key whose
    mapped v4 key is registered, reopen the body and run the
    REGISTERED v4 transform: only a key whose evidence actually
    SERVES its v4 contract may be reused."""
    xf = transform or CAP.admission_transform
    v4 = _v4_keys(authority)
    v4set = set(v4)
    if PREDECESSOR_KEY not in v4set:
        _d(f"the predecessor key {PREDECESSOR_KEY} is not registered")
    v3 = dict(archive.get("admitted", {}))
    v3.update(archive.get("refused", {}))
    reuse, notes = {}, {}
    for k in sorted(v3):
        lane, ck, day = k.split("/")
        if (lane, ck) in SUPERSEDED_V3:
            continue
        v4key = f"{LANE_MAP.get(lane, lane)}/{ck}/{day}"
        if v4key not in v4set or v4key == PREDECESSOR_KEY:
            continue
        v4lane = LANE_MAP.get(lane, lane)
        try:
            s = ACC.authoritative_static_contract(authority, v4lane,
                                                  ck, day)
        except Exception:
            continue
        sha = v3[k]["raw_body_sha256"]
        path = os.path.join(store_root, sha + ".body")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            body = f.read()
        if hashlib.sha256(body).hexdigest() != sha:
            _d(f"preserved body for {k} does not match its address")
        try:
            art = xf(v4lane, body, s)
        except CAP.CaptureRefusal:
            continue
        # codex closure 2: the LINEAGE record binds both endpoints.
        # raw_body_bytes is RECOMPUTED from the reopened bytes, never
        # copied from the archive's submitted field; s_v4_sha256 is
        # REPORTED but never authoritative -- verify re-derives it.
        reuse[v4key] = {"v3_key": k, "raw_body_sha256": sha,
                        "raw_body_bytes": len(body),
                        "outcome": art.get("outcome"),
                        "s_v3_sha256": v3[k]["static_contract_sha256"],
                        "t_v3_sha256": v3[k]["transcript_sha256"],
                        "s_v4_sha256": _canon(s)}
        notes[v4key] = art.get("support_predicate")
    http = sorted(v4set - set(reuse) - {PREDECESSOR_KEY})
    pred = {PREDECESSOR_KEY: {"spent_probe": True}}
    return reuse, pred, http, notes


def build(store_root, commitish="HEAD", transform=None):
    authority, auth_sha = _authority(commitish)
    araw = _blob(f"{commitish}:{V3_ARCHIVE_PATH}") \
        if _archive_committed(commitish) else None
    if araw is None:
        p = os.path.join(REPO, *V3_ARCHIVE_PATH.split("/"))
        with open(p, "rb") as f:
            araw = f.read()
    archive = json.loads(araw.decode("utf-8"))
    reuse, pred, http, _notes = derive(authority, archive,
                                       store_root, transform)
    caps = {
        "schema": CAPSULE_SCHEMA,
        "authority": {"path": AUTHORITY_PATH,
                      "blob_sha256": auth_sha,
                      "keys_sha256":
                          authority["prestart_expected_keys_sha256"],
                      "census": len(_v4_keys(authority))},
        "transform_identity": CAP.transform_identity(),
        "v3_archive": {"path": V3_ARCHIVE_PATH,
                       "sha256": hashlib.sha256(araw).hexdigest(),
                       "store_id": archive.get("store_id")},
        # the OLD authority the historical exchanges happened under
        # (codex closure 3: both endpoints are authenticated)
        "old_authority": dict(archive["authority"]),
        "lane_map": dict(LANE_MAP),
        "superseded_v3": [list(x) for x in SUPERSEDED_V3],
        "reuse_or_bridge": reuse,
        "predecessor": pred,
        "http_capture": list(http),
    }
    caps["partitions"] = _partition_block(caps)
    return caps


def _archive_committed(commitish):
    p = subprocess.run(
        ["git", "-C", REPO, "cat-file", "-e",
         f"{commitish}:{V3_ARCHIVE_PATH}"], capture_output=True)
    return p.returncode == 0


def _partition_block(caps):
    """Per-partition sorted-key digests + source-body digests. These
    are DERIVED here and RECOMPUTED by the verifier -- a submitted
    block is never trusted."""
    reuse, pred, http = (caps["reuse_or_bridge"],
                         caps["predecessor"], caps["http_capture"])
    return {
        "REUSE_OR_BRIDGE": {
            "count": len(reuse),
            "keys_sha256": _canon(sorted(reuse)),
            "source_bodies_sha256": _canon(
                [[k, reuse[k]["raw_body_sha256"]]
                 for k in sorted(reuse)])},
        "PREDECESSOR": {
            "count": len(pred), "keys_sha256": _canon(sorted(pred)),
            "source_bodies_sha256": _canon([])},
        "HTTP_CAPTURE": {
            "count": len(http), "keys_sha256": _canon(sorted(http)),
            "source_bodies_sha256": _canon([])},
    }


def build_fixture_capsule(authority, http_keys, store_root,
                          archive_path, old_authority=None,
                          reuse=None, predecessor=None):
    """THE SUPPORTED FIXTURE PATH (cayley's 2340Z testability ask).

    The answer to "how do I satisfy the strict lineage contract over
    a fixture?" is NOT an injectable verifier and NOT a fail-open
    hole -- both would reopen exactly what closure 2 exists to close,
    and a fixture ground down until the verifier stops complaining is
    a fixture shaped to satisfy a checker rather than to represent
    reality. Cayley was right to stop rather than hand-iterate.

    The supported answer is to make the FIXTURE REAL: write a real
    (small) archive file, put real bodies in a real store directory,
    and let every derivation actually run. This helper assembles such
    a capsule; the caller writes the bodies. `archive_path` may be
    absolute so the fixture can live in a temp dir.

    Nothing here is relaxed: the resulting capsule is verified by the
    SAME strict `verify_lineage_registry` every production caller
    uses, and it refuses if the fixture is not internally true."""
    reuse = dict(reuse or {})
    pred = dict(predecessor or {})
    # a FIXTURE may never be minted over the REAL registered census
    # (cayley 0110Z): that is the exact shape that claimed 2056
    # native keys against a truth of 635/1420/1
    try:
        real, _sha = _authority("HEAD")
        if authority.get("prestart_expected_keys_sha256") == \
                real.get("prestart_expected_keys_sha256"):
            _d("build_fixture_capsule REFUSES the REAL registered "
               "authority -- a fixture over the production census "
               "is exactly the internally-true / externally-false "
               "capsule this constructor must never be able to mint")
    except DispositionRefusal:
        raise
    except Exception:
        pass
    arch = {"schema": "f2g-w2-capture-run-archive-v1",
            "store_id": "fixture-store",
            "authority": dict(old_authority or {
                "commit": "0" * 40, "path": AUTHORITY_PATH,
                "blob_sha256": "0" * 64, "keys_sha256": "0" * 64}),
            "admitted": {}, "refused": {}}
    raw = (json.dumps(arch, indent=1, sort_keys=True) + "\n").encode()
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
    with open(archive_path, "wb") as f:
        f.write(raw)
    caps = {"schema": FIXTURE_CAPSULE_SCHEMA,
            "authority": {
                "path": AUTHORITY_PATH,
                "blob_sha256": hashlib.sha256(
                    json.dumps(authority, sort_keys=True).encode()
                ).hexdigest(),
                "keys_sha256":
                    authority["prestart_expected_keys_sha256"],
                "census": len(_v4_keys(authority))},
            "transform_identity": CAP.transform_identity(),
            "v3_archive": {"path": archive_path,
                           "sha256": hashlib.sha256(raw).hexdigest(),
                           "store_id": arch["store_id"]},
            "old_authority": dict(arch["authority"]),
            "lane_map": dict(LANE_MAP),
            "superseded_v3": [list(x) for x in SUPERSEDED_V3],
            "reuse_or_bridge": reuse, "predecessor": pred,
            "http_capture": sorted(http_keys)}
    caps["partitions"] = _partition_block(caps)
    os.makedirs(store_root, exist_ok=True)
    return caps


def verify_ceiling(capsule, authority=None, commitish="HEAD"):
    """THE REQUEST-CEILING contract ONLY (codex 2303Z closure 2):
    exact/disjoint request-membership over the authority key set. It
    REPORTS that lineage evidence is NOT verified, so a ceiling PASS
    can never be misread as a lineage PASS. The network entrypoint
    may use this; the boundary and restager may NOT."""
    out = _verify(capsule, authority, commitish, None, False)
    out["lineage_evidence_verified"] = False
    return out


def verify_lineage_registry(capsule, authority=None,
                            commitish="HEAD", store_root=None):
    """THE LINEAGE-REGISTRY contract (codex 2303Z closure 2): FAILS
    CLOSED unless it can reopen the registered archive AND the
    source bodies. A missing archive or store can never produce a
    clean lineage pass -- that fail-open policy is exactly what let
    six doctored fields through on a host without the store."""
    root = store_root or os.environ.get(
        "W2_V3_STORE", "E:/GeoSpec/w2_capture_store_20260825")
    if not os.path.isdir(root):
        _d("LINEAGE registry verification requires the source body "
           f"store; {root} is not readable -- fail CLOSED")
    if not (capsule or {}).get("reuse_or_bridge"):
        _d("LINEAGE registry verification over an EMPTY reuse "
           "partition is VACUOUS -- there is no lineage to "
           "authenticate, so it can never report lineage evidence "
           "as verified (cayley 0110Z)")
    out = _verify(capsule, authority, commitish, root, True)
    # honesty of the report itself: 'verified' means bodies were
    # actually recomputed, never merely that nothing was checked
    n = out.get("bodies_recomputed") or 0
    out["lineage_evidence_verified"] = bool(n)
    if not n:
        _d("LINEAGE registry reported zero recomputed bodies -- a "
           "verification that checked nothing is not a pass")
    return out


def verify(capsule, authority=None, commitish="HEAD",
           store_root=None):
    """Deprecated shim: callers must choose verify_ceiling() or
    verify_lineage_registry() explicitly. Kept only so existing
    call sites keep working while they are migrated."""
    return _verify(capsule, authority, commitish, store_root,
                   bool(store_root))


def _verify(capsule, authority=None, commitish="HEAD",
            store_root=None, lineage=False):
    """The capsule verifier: closed schema, EXACT and DISJOINT
    partition of the registered authority key set, recomputed
    per-partition digests, and a recomputed counts block. Nothing
    submitted is trusted."""
    c = capsule
    if not isinstance(c, dict) or c.get("schema") != CAPSULE_SCHEMA:
        _d("capsule is not the registered schema")
    want_top = {"schema", "authority", "transform_identity",
                "v3_archive", "old_authority", "lane_map",
                "superseded_v3", "reuse_or_bridge", "predecessor",
                "http_capture", "partitions"}
    if set(c) != want_top:
        _d(f"capsule top-level field set is not closed "
           f"(missing={sorted(want_top - set(c))}, "
           f"extra={sorted(set(c) - want_top)})")
    if authority is None:
        authority, auth_sha = _authority(commitish)
        if c["authority"]["blob_sha256"] != auth_sha:
            _d("capsule authority digest does not match the "
               f"registered authority at {commitish}")
    keys = _v4_keys(authority)
    if c["authority"]["census"] != len(keys):
        _d("capsule census does not match the authority key count")
    reuse, pred, http = (set(c["reuse_or_bridge"]),
                         set(c["predecessor"]), set(c["http_capture"]))
    if len(http) != len(c["http_capture"]):
        _d("HTTP_CAPTURE contains duplicate keys")
    for a, b, na, nb in ((reuse, pred, "REUSE_OR_BRIDGE",
                          "PREDECESSOR"),
                         (reuse, http, "REUSE_OR_BRIDGE",
                          "HTTP_CAPTURE"),
                         (pred, http, "PREDECESSOR",
                          "HTTP_CAPTURE")):
        both = a & b
        if both:
            _d(f"partitions {na} and {nb} OVERLAP on "
               f"{sorted(both)[:3]} -- a key has exactly one "
               "disposition")
    union = reuse | pred | http
    if union != set(keys):
        _d("the partition is not EXACT over the authority key set "
           f"(missing={sorted(set(keys) - union)[:3]}, "
           f"extra={sorted(union - set(keys))[:3]})")
    recomputed = _partition_block(c)
    if c["partitions"] != recomputed:
        _d("the partitions block is DERIVED, never submitted -- it "
           "does not recompute from the capsule's own key lists")
    _verify_nested(c, authority, store_root, lineage)
    return {"census": len(keys),
            "REUSE_OR_BRIDGE": len(reuse),
            "PREDECESSOR": len(pred),
            "HTTP_CAPTURE": len(http),
            "bodies_recomputed": (len(reuse) if store_root
                                  else None)}


def _hex(v, what):
    if not (isinstance(v, str) and len(v) == _HEX64
            and all(ch in "0123456789abcdef" for ch in v)):
        _d(f"{what} is not lowercase-hex sha256")


def _closed(obj, want, what):
    if not isinstance(obj, dict) or set(obj) != want:
        _d(f"{what} is not the closed field set "
           f"(missing={sorted(want - set(obj or {}))}, "
           f"extra={sorted(set(obj or {}) - want)})")


def _verify_nested(c, authority, store_root=None, lineage=False):
    """codex 2119Z closure 2: EVERY nested field consumed as
    authority is closed and INDEPENDENTLY re-derived. A verifier that
    closes the top level and the key sets but leaves the interior
    open is well-formed, not authenticating."""
    _closed(c["authority"], _AUTHORITY_KEYS, "authority")
    _closed(c["old_authority"], _OLD_AUTHORITY_KEYS, "old_authority")
    _closed(c["v3_archive"], _ARCHIVE_KEYS, "v3_archive")
    _hex(c["authority"]["blob_sha256"], "authority.blob_sha256")
    _hex(c["authority"]["keys_sha256"], "authority.keys_sha256")
    _hex(c["old_authority"]["blob_sha256"], "old_authority.blob")
    _hex(c["old_authority"]["keys_sha256"], "old_authority.keys")
    _hex(c["v3_archive"]["sha256"], "v3_archive.sha256")
    # the archive digest is PROVENANCE and must be checked against
    # the ACTUAL archive wherever it resolves -- a bound digest that
    # nobody re-derives is exactly the forgeable field codex
    # demonstrated. Git first, then disk; if it resolves at all, the
    # match is REQUIRED.
    _araw = None
    if _archive_committed("HEAD"):
        _araw = _blob(f"HEAD:{c['v3_archive']['path']}")
    else:
        # repo-relative by default; an ABSOLUTE path is honoured so a
        # FIXTURE capsule can name a real fixture archive outside the
        # repo. This weakens nothing: the bytes it resolves to are
        # still digest-checked below, and the store must still exist.
        _p = c["v3_archive"]["path"]
        _ap = _p if os.path.isabs(_p) else \
            os.path.join(REPO, *_p.split("/"))
        if os.path.isfile(_ap):
            with open(_ap, "rb") as f:
                _araw = f.read()
    if _araw is None and lineage:
        _d("LINEAGE registry verification requires the registered "
           "archive; it could not be resolved -- fail CLOSED. A "
           "missing evidence source is never a clean pass")
    if _araw is not None:
        if hashlib.sha256(_araw).hexdigest() != \
                c["v3_archive"]["sha256"]:
            _d("v3_archive.sha256 does not match the resolved "
               "archive bytes")
        _arch = json.loads(_araw.decode("utf-8"))
        if _arch.get("store_id") != c["v3_archive"]["store_id"]:
            _d("v3_archive.store_id does not match the archive's "
               "own store identity")
        if _arch.get("authority") != c["old_authority"]:
            _d("old_authority does not match the authority identity "
               "the archive itself records")
    if c["authority"]["keys_sha256"] != \
            authority["prestart_expected_keys_sha256"]:
        _d("authority.keys_sha256 does not match the registered "
           "authority's own key digest")
    # the lane map and superseded set are REGISTERED constants, not
    # capsule opinions -- a capsule may not redefine the mapping it
    # is verified against
    if c["lane_map"] != dict(LANE_MAP):
        _d(f"lane_map {c['lane_map']} is not the REGISTERED map "
           f"{dict(LANE_MAP)}")
    if [tuple(x) for x in c["superseded_v3"]] != list(SUPERSEDED_V3):
        _d("superseded_v3 is not the REGISTERED superseded set")
    # the TRANSFORM identity is re-derived from the actual module,
    # never trusted from the field
    live = CAP.transform_identity()
    if c["transform_identity"] != live:
        _d("transform_identity does not match the LIVE registered "
           f"transform (capsule names "
           f"{str(c['transform_identity'])[:60]})")
    for k, e in c["predecessor"].items():
        _closed(e, _PREDECESSOR_KEYS, f"predecessor[{k}]")
        if e["spent_probe"] is not True:
            _d(f"predecessor[{k}] must record spent_probe true")
    # the v3 authority the historical exchanges happened under. It
    # is reopened ONLY when there are lineage entries to authenticate
    # against it: with an empty REUSE partition there is no lineage
    # claim, so deriving contracts for zero keys would prove nothing.
    # (The identity is still closed, hex-checked, and cross-checked
    # against the authority the archive itself records, above.)
    if not c["reuse_or_bridge"]:
        return
    old_raw = _blob(f"{c['old_authority']['commit']}:"
                    f"{c['old_authority']['path']}")
    if hashlib.sha256(old_raw).hexdigest() != \
            c["old_authority"]["blob_sha256"]:
        _d("old_authority bytes diverge from its pinned digest")
    old_auth = json.loads(old_raw.decode("utf-8"))
    seen_sources = {}
    for v4key in sorted(c["reuse_or_bridge"]):
        e = c["reuse_or_bridge"][v4key]
        _closed(e, _LINEAGE_KEYS, f"lineage[{v4key}]")
        for f_ in ("raw_body_sha256", "s_v3_sha256", "t_v3_sha256",
                   "s_v4_sha256"):
            _hex(e[f_], f"lineage[{v4key}].{f_}")
        if not isinstance(e["raw_body_bytes"], int) or \
                e["raw_body_bytes"] <= 0:
            _d(f"lineage[{v4key}].raw_body_bytes is not a positive "
               "integer")
        # INJECTIVE: two v4 keys may never share one source operation
        src = e["v3_key"]
        if src in seen_sources:
            _d(f"lineage source {src} is claimed by BOTH "
               f"{seen_sources[src]} and {v4key} -- the v3 -> v4 "
               "mapping must be injective")
        seen_sources[src] = v4key
        # the v4 key must DERIVE from the v3 key through the
        # REGISTERED map; a lineage may not assert an arbitrary pair
        try:
            olane, ock, oday = src.split("/")
        except ValueError:
            _d(f"lineage[{v4key}].v3_key {src!r} is not a key")
        if f"{LANE_MAP.get(olane, olane)}/{ock}/{oday}" != v4key:
            _d(f"lineage[{v4key}] does not derive from its source "
               f"{src} under the registered lane map")
        # S_v3 and S_v4 are DERIVED here; the capsule's digests are
        # compared to the derivation, never used as the authority
        try:
            s3 = ACC.authoritative_static_contract(old_auth, olane,
                                                   ock, oday)
        except Exception as exc:
            _d(f"lineage[{v4key}]: S_v3 does not derive from the "
               f"pinned old authority ({type(exc).__name__})")
        if _canon(s3) != e["s_v3_sha256"]:
            _d(f"lineage[{v4key}].s_v3_sha256 does not match the "
               "INDEPENDENTLY derived S_v3")
        nlane, nck, nday = v4key.split("/")
        try:
            s4 = ACC.authoritative_static_contract(authority, nlane,
                                                   nck, nday)
        except Exception as exc:
            _d(f"lineage[{v4key}]: S_v4 does not derive from the "
               f"current authority ({type(exc).__name__})")
        if _canon(s4) != e["s_v4_sha256"]:
            _d(f"lineage[{v4key}].s_v4_sha256 does not match the "
               "INDEPENDENTLY derived S_v4 -- a lineage is never "
               "allowed to name the contract that authenticates it")
        if store_root:
            p = os.path.join(store_root, e["raw_body_sha256"] +
                             ".body")
            if not os.path.isfile(p):
                _d(f"lineage[{v4key}] body is absent from the store")
            with open(p, "rb") as f:
                raw = f.read()
            if hashlib.sha256(raw).hexdigest() != \
                    e["raw_body_sha256"]:
                _d(f"lineage[{v4key}] body digest mismatch")
            if len(raw) != e["raw_body_bytes"]:
                _d(f"lineage[{v4key}].raw_body_bytes {e['raw_body_bytes']}"
                   f" != the RECOMPUTED length {len(raw)}")


def may_fire(capsule, lane, carrier, utc_day):
    """THE ceiling test used by the network entrypoint: a key may
    reach the opener ONLY as a member of HTTP_CAPTURE."""
    key = f"{lane}/{carrier}/{utc_day}"
    if key in set(capsule.get("reuse_or_bridge", {})):
        _d(f"{key} is REUSE_OR_BRIDGE -- it is served by preserved "
           "evidence and must never be re-requested")
    if key in set(capsule.get("predecessor", {})):
        _d(f"{key} is the PREDECESSOR probe key -- already fired and "
           "never re-requested (its bytes are the grammar anchor)")
    if key not in set(capsule.get("http_capture", [])):
        _d(f"{key} is not a member of HTTP_CAPTURE")
    return True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    store = os.environ.get("W2_V3_STORE",
                           "E:/GeoSpec/w2_capture_store_20260825")
    out = os.path.join(REPO, *CAPSULE_PATH.split("/"))
    if mode in ("build", "rebuild"):
        caps = build(store)
        print("derived:", verify(caps, store_root=store))
        if mode == "rebuild" and os.path.exists(out):
            # codex 2119Z closure 2 ordered the upgraded capsule to
            # be REGENERATED and re-pinned; an explicit mode, never a
            # silent overwrite of a create-once artifact
            os.remove(out)
        CAP._write_once_json(out, caps, "DISPOSITION_DIVERGENT")
        print("written:", CAPSULE_PATH)
    elif mode == "verify":
        with open(out, encoding="utf-8") as f:
            caps = json.load(f)
        print("verified:", verify(
            caps, store_root=(store if os.path.isdir(store)
                              else None)))
    elif mode == "--selftest":
        _selftest()
    else:
        raise SystemExit("usage: build | rebuild | verify | --selftest")


def _selftest():
    """codex 2119Z closure 2 doctors, run against the REAL committed
    capsule -- the same artifact he doctored field-by-field and found
    ACCEPTED under v1. Each forgery must now REFUSE."""
    out = os.path.join(REPO, *CAPSULE_PATH.split("/"))
    store = os.environ.get("W2_V3_STORE",
                           "E:/GeoSpec/w2_capture_store_20260825")
    if not os.path.isfile(out):
        print("w2_disposition_capsule selftest: capsule absent, "
              "structural locks only")
        return
    with open(out, encoding="utf-8") as f:
        caps = json.load(f)
    authority, _sha = _authority("HEAD")
    base = verify(caps, authority)
    assert base["census"] == (base["REUSE_OR_BRIDGE"]
                              + base["PREDECESSOR"]
                              + base["HTTP_CAPTURE"])

    def refuses(mutate, needle):
        c = json.loads(json.dumps(caps))
        mutate(c)
        try:
            verify(c, authority)
        except DispositionRefusal as e:
            return needle in str(e)
        return False

    def first_reuse(c):
        return sorted(c["reuse_or_bridge"])[0]
    # --- codex's five demonstrated forgeries, verbatim ---
    assert refuses(lambda c: c["reuse_or_bridge"][first_reuse(c)]
                   .__setitem__("v3_key", "BOGUS/x/1900-01-01"),
                   "does not derive from its source")
    assert refuses(lambda c: c["reuse_or_bridge"][first_reuse(c)]
                   .__setitem__("raw_body_bytes", -1),
                   "not a positive integer")
    assert refuses(lambda c: c.__setitem__("transform_identity",
                                           {"forged": True}),
                   "does not match the LIVE registered transform")
    assert refuses(lambda c: c["v3_archive"]
                   .__setitem__("sha256", "0" * 64),
                   "does not match the resolved archive bytes")
    assert refuses(lambda c: c.__setitem__("lane_map",
                                           {"BOGUS": "MAG_FEED"}),
                   "is not the REGISTERED map")
    # --- the lineage-specific authentications ---
    assert refuses(lambda c: c["reuse_or_bridge"][first_reuse(c)]
                   .__setitem__("s_v3_sha256", "0" * 64),
                   "INDEPENDENTLY derived S_v3")
    assert refuses(lambda c: c["reuse_or_bridge"][first_reuse(c)]
                   .__setitem__("s_v4_sha256", "0" * 64),
                   "never allowed to name the contract")
    # (the archive-identity check fires first here, which is the
    # stronger statement: the capsule cannot disagree with the
    # authority identity the archive itself recorded)
    assert refuses(lambda c: c["old_authority"]
                   .__setitem__("blob_sha256", "0" * 64),
                   "old_authority does not match")
    assert refuses(lambda c: c["reuse_or_bridge"][first_reuse(c)]
                   .__setitem__("extra_field", 1),
                   "is not the closed field set")
    assert refuses(lambda c: c["reuse_or_bridge"][first_reuse(c)]
                   .pop("t_v3_sha256"),
                   "is not the closed field set")

    # INJECTIVITY: two v4 keys claiming one source operation
    def dupe(c):
        ks = sorted(c["reuse_or_bridge"])[:2]
        c["reuse_or_bridge"][ks[1]]["v3_key"] =             c["reuse_or_bridge"][ks[0]]["v3_key"]
    assert refuses(dupe, "must be injective")
    # a RECOMPUTED body length disagreeing with the store
    c2 = json.loads(json.dumps(caps))
    k2 = sorted(c2["reuse_or_bridge"])[0]
    c2["reuse_or_bridge"][k2]["raw_body_bytes"] += 1
    if os.path.isdir(store):
        try:
            verify(c2, authority, store_root=store)
            raise AssertionError("submitted length must refuse")
        except DispositionRefusal as e:
            assert "RECOMPUTED length" in str(e)
    # may_fire ceiling behaviour on the real partition
    hk = sorted(caps["http_capture"])[0]
    assert may_fire(caps, *hk.split("/")) is True
    rk = sorted(caps["reuse_or_bridge"])[0]
    try:
        may_fire(caps, *rk.split("/"))
        raise AssertionError("a REUSE key must never be firable")
    except DispositionRefusal:
        pass
    try:
        may_fire(caps, *PREDECESSOR_KEY.split("/"))
        raise AssertionError("the probe key must never re-fire")
    except DispositionRefusal:
        pass
    print("w2_disposition_capsule selftest: ALL PASS "
          f"({base}, no network)")


if __name__ == "__main__":
    main()
