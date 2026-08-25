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
CAPSULE_SCHEMA = "f2g-w2-key-disposition-capsule-v1"
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
        reuse[v4key] = {"v3_key": k, "raw_body_sha256": sha,
                        "raw_body_bytes": v3[k]["raw_body_bytes"],
                        "outcome": art.get("outcome")}
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


def verify(capsule, authority=None, commitish="HEAD"):
    """The capsule verifier: closed schema, EXACT and DISJOINT
    partition of the registered authority key set, recomputed
    per-partition digests, and a recomputed counts block. Nothing
    submitted is trusted."""
    c = capsule
    if not isinstance(c, dict) or c.get("schema") != CAPSULE_SCHEMA:
        _d("capsule is not the registered schema")
    want_top = {"schema", "authority", "transform_identity",
                "v3_archive", "lane_map", "superseded_v3",
                "reuse_or_bridge", "predecessor", "http_capture",
                "partitions"}
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
    return {"census": len(keys),
            "REUSE_OR_BRIDGE": len(reuse),
            "PREDECESSOR": len(pred),
            "HTTP_CAPTURE": len(http)}


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
    if mode == "build":
        caps = build(store)
        print("derived:", verify(caps))
        CAP._write_once_json(out, caps, "DISPOSITION_DIVERGENT")
        print("written:", CAPSULE_PATH)
    elif mode == "verify":
        with open(out, encoding="utf-8") as f:
            caps = json.load(f)
        print("verified:", verify(caps))
    elif mode == "--selftest":
        _selftest()
    else:
        raise SystemExit("usage: build | verify | --selftest")


def _selftest():
    """Closed-predicate locks over a synthetic authority/archive --
    no store, no network, no repo state required."""
    auth_keys = {"MAG_FEED": {"frn": ["2026-01-01", "2026-01-02"]},
                 "MAG_WEATHER_FEED": {"omni": ["2026-01-01"]}}
    authority = {"prestart_expected_keys": auth_keys,
                 "prestart_expected_keys_sha256": _canon(auth_keys)}
    caps = {"schema": CAPSULE_SCHEMA,
            "authority": {"path": AUTHORITY_PATH,
                          "blob_sha256": "a" * 64,
                          "keys_sha256": _canon(auth_keys),
                          "census": 3},
            "transform_identity": {"module": "kat"},
            "v3_archive": {"path": V3_ARCHIVE_PATH,
                           "sha256": "b" * 64, "store_id": "kat"},
            "lane_map": dict(LANE_MAP),
            "superseded_v3": [list(x) for x in SUPERSEDED_V3],
            "reuse_or_bridge": {
                "MAG_FEED/frn/2026-01-01": {
                    "v3_key": "MAG_FEED/frn/2026-01-01",
                    "raw_body_sha256": "c" * 64,
                    "raw_body_bytes": 10, "outcome": "ADMITTED"}},
            "predecessor": {PREDECESSOR_KEY: {"spent_probe": True}},
            "http_capture": ["MAG_FEED/frn/2026-01-02"]}
    caps["partitions"] = _partition_block(caps)
    assert verify(caps, authority) == {
        "census": 3, "REUSE_OR_BRIDGE": 1, "PREDECESSOR": 1,
        "HTTP_CAPTURE": 1}

    def refuses(fn, needle):
        try:
            fn()
            return False
        except DispositionRefusal as e:
            return needle in str(e)

    def mut(**over):
        c = json.loads(json.dumps(caps))
        c.update(over)
        return c
    # the ceiling test: only HTTP_CAPTURE may reach the opener
    assert may_fire(caps, "MAG_FEED", "frn", "2026-01-02") is True
    assert refuses(lambda: may_fire(caps, "MAG_FEED", "frn",
                                    "2026-01-01"),
                   "is REUSE_OR_BRIDGE")
    assert refuses(lambda: may_fire(caps, "MAG_WEATHER_FEED", "omni",
                                    "2026-01-01"),
                   "is the PREDECESSOR probe key")
    assert refuses(lambda: may_fire(caps, "MAG_FEED", "frn",
                                    "2026-09-09"),
                   "not a member of HTTP_CAPTURE")
    # overlap, inexactness, submitted counts, closure
    ov = json.loads(json.dumps(caps))
    ov["http_capture"].append("MAG_FEED/frn/2026-01-01")
    ov["partitions"] = _partition_block(ov)
    assert refuses(lambda: verify(ov, authority), "OVERLAP")
    sh = json.loads(json.dumps(caps))
    sh["http_capture"] = []
    sh["partitions"] = _partition_block(sh)
    assert refuses(lambda: verify(sh, authority), "not EXACT")
    ct = json.loads(json.dumps(caps))
    ct["partitions"]["HTTP_CAPTURE"]["count"] = 99
    assert refuses(lambda: verify(ct, authority),
                   "DERIVED, never submitted")
    dg = json.loads(json.dumps(caps))
    dg["partitions"]["REUSE_OR_BRIDGE"]["source_bodies_sha256"] = \
        "0" * 64
    assert refuses(lambda: verify(dg, authority),
                   "DERIVED, never submitted")
    assert refuses(lambda: verify(mut(extra_field=1), authority),
                   "field set is not closed")
    assert refuses(lambda: verify(mut(schema="other"), authority),
                   "not the registered schema")
    cen = mut(authority=dict(caps["authority"], census=999))
    assert refuses(lambda: verify(cen, authority), "census")
    dup = json.loads(json.dumps(caps))
    dup["http_capture"] = ["MAG_FEED/frn/2026-01-02"] * 2
    dup["partitions"] = _partition_block(dup)
    assert refuses(lambda: verify(dup, authority), "duplicate keys")
    print("w2_disposition_capsule selftest: ALL PASS (no network)")


if __name__ == "__main__":
    main()
