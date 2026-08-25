#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RESTAGED_LINEAGE -- the two-leg join for v3-evidence-served keys
(grassmann). codex 2119Z closures 1 and 3; finding-4 Option 2.

A restaged key is NOT a native capture and must never be dressed as
one. Its truth has two DIFFERENT types and they stay two legs:

  LEG 1 -- HISTORICAL EXCHANGE (immutable): derive `S_v3` from the
    PINNED v3 authority, verify the ORIGINAL `T_v3` verbatim against
    it, and reopen the body at `T_v3`'s own content address. That
    exchange happened under `S_v3` and nothing may rewrite it.
  LEG 2 -- CURRENT DERIVATION (recomputable): derive `S_v4` from the
    PINNED current authority and recompute the artifact from that
    same reopened body through the REGISTERED v4 transform.

What is deliberately NOT done: emitting a native envelope whose
static half is v4 while its receipt projects `T_v3`. That would
assert the false relation `T_v3 -> S_v4` -- history the exchange
never had. The original transcript bytes and digest are preserved
exactly; this record sits BESIDE them.

Everything consumed as authority is reopened from manifest pins and
RE-DERIVED (closure 3). A caller-supplied artifact, lineage,
authority or transform identity is ignored in favour of the pinned
derivation, or refuses.

ZERO HTTP: this module never opens a socket.
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
import w2_producer_grassmann as PROD
import w2_disposition_capsule_grassmann as DISP

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESTAGE_SCHEMA = "f2g-w2-restage-lineage-v1"
JOIN_KIND = "RESTAGED_LINEAGE"
RESTAGE_KEYS = {
    "schema", "join_kind", "v3_key", "v4_key", "old_authority",
    "new_authority", "s_v3_sha256", "s_v4_sha256", "t_v3_sha256",
    "raw_body_sha256", "raw_body_bytes", "transform_identity",
    "artifact_sha256", "outcome"}
_AUTH_KEYS = {"commit", "path", "blob_sha256", "keys_sha256"}


class RestageRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _r(msg):
    raise RestageRefusal("RESTAGE_REFUSED: " + str(msg))


def _blob(repo, ref):
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob", ref],
                       capture_output=True)
    if p.returncode != 0:
        _r(f"unreadable git object {ref}")
    return p.stdout


def _closed(obj, want, what):
    if not isinstance(obj, dict) or set(obj) != want:
        _r(f"{what} is not the closed field set "
           f"(missing={sorted(want - set(obj or {}))}, "
           f"extra={sorted(set(obj or {}) - want)})")


def _hex(v, what):
    if not (isinstance(v, str) and len(v) == 64
            and all(c in "0123456789abcdef" for c in v)):
        _r(f"{what} is not lowercase-hex sha256")


def _derive_pair(repo, record):
    """Reopen BOTH pinned authorities and derive BOTH contracts.
    Neither digest in the record is used as authority: they are
    compared to what the pins actually produce."""
    out = {}
    for side, akey, key in (("v3", "old_authority", "v3_key"),
                            ("v4", "new_authority", "v4_key")):
        a = record[akey]
        raw = _blob(repo, f"{a['commit']}:{a['path']}")
        if hashlib.sha256(raw).hexdigest() != a["blob_sha256"]:
            _r(f"{akey} bytes diverge from its pinned digest")
        auth = json.loads(raw.decode("utf-8"))
        if auth.get("prestart_expected_keys_sha256") != \
                a["keys_sha256"]:
            _r(f"{akey}.keys_sha256 does not match the authority's "
               "own key digest")
        lane, ck, day = record[key].split("/")
        try:
            s = ACC.authoritative_static_contract(auth, lane, ck, day)
        except Exception as exc:
            _r(f"S_{side} does not derive from {akey} for "
               f"{record[key]} ({type(exc).__name__}: {exc})")
        out[side] = (auth, s, lane, ck, day)
    return out


def verify_restage_lineage(repo, record, transcript, raw_body):
    """THE two-leg verifier (closures 1 + 3). `transcript` is the
    ORIGINAL T_v3 reopened verbatim from the v3 staged tree and
    `raw_body` the bytes reopened at its content address -- both are
    re-checked here rather than trusted."""
    rec = record
    _closed(rec, RESTAGE_KEYS, "restage record")
    if rec.get("schema") != RESTAGE_SCHEMA:
        _r("record is not the registered restage schema")
    if rec.get("join_kind") != JOIN_KIND:
        _r(f"join_kind must be {JOIN_KIND} -- a restaged key is "
           "never reported as a native capture")
    _closed(rec["old_authority"], _AUTH_KEYS, "old_authority")
    _closed(rec["new_authority"], _AUTH_KEYS, "new_authority")
    for f_ in ("s_v3_sha256", "s_v4_sha256", "t_v3_sha256",
               "raw_body_sha256", "artifact_sha256"):
        _hex(rec[f_], f"record.{f_}")
    # the v4 key must DERIVE from the v3 key through the REGISTERED
    # map -- a record may not assert an arbitrary pair
    olane, ock, oday = rec["v3_key"].split("/")
    if f"{DISP.LANE_MAP.get(olane, olane)}/{ock}/{oday}" != \
            rec["v4_key"]:
        _r(f"v4_key {rec['v4_key']} does not derive from v3_key "
           f"{rec['v3_key']} under the registered lane map")
    derived = _derive_pair(repo, rec)
    _auth3, s3, _l3, _c3, _d3 = derived["v3"]
    _auth4, s4, lane4, _c4, _d4 = derived["v4"]
    if PROD._canon_digest(s3) != rec["s_v3_sha256"]:
        _r("s_v3_sha256 does not match the INDEPENDENTLY derived "
           "S_v3")
    if PROD._canon_digest(s4) != rec["s_v4_sha256"]:
        _r("s_v4_sha256 does not match the INDEPENDENTLY derived "
           "S_v4 -- a lineage never names the contract that "
           "authenticates it")
    # ---- LEG 1: the historical exchange, verified verbatim ----
    if PROD._canon_digest(transcript) != rec["t_v3_sha256"]:
        _r("the reopened transcript is not the one this lineage "
           "names (digest mismatch) -- the original T is immutable")
    got = hashlib.sha256(raw_body).hexdigest()
    if got != rec["raw_body_sha256"] or \
            len(raw_body) != rec["raw_body_bytes"]:
        _r("the reopened body does not match the lineage's content "
           "address / recomputed length")
    try:
        PROD.verify_transcript(transcript, s3, raw_body=raw_body)
    except Exception as exc:
        _r(f"LEG 1 refused: the original T_v3 does not verify "
           f"against the derived S_v3 ({type(exc).__name__}: {exc})")
    # ---- LEG 2: the current derivation, RECOMPUTED ----
    live = CAP.transform_identity()
    if rec["transform_identity"] != live:
        _r("transform_identity does not match the LIVE registered "
           "transform")
    try:
        art = CAP.admission_transform(lane4, raw_body, s4)
    except CAP.CaptureRefusal as exc:
        _r(f"LEG 2 refused: the preserved body does not serve the "
           f"v4 contract ({exc})")
    if PROD._canon_digest(art) != rec["artifact_sha256"]:
        _r("artifact_sha256 does not match the RECOMPUTED v4 "
           "artifact -- the artifact is derived here, never taken "
           "from the record")
    if art.get("outcome") != rec["outcome"]:
        _r("outcome does not match the recomputed artifact")
    return {"join_kind": JOIN_KIND, "v4_key": rec["v4_key"],
            "v3_key": rec["v3_key"], "outcome": art.get("outcome"),
            "artifact_sha256": rec["artifact_sha256"]}


def build_restage_lineage(repo, v3_key, v4_key, old_authority,
                          new_authority, transcript, raw_body):
    """Assemble the record by DERIVING every binding, then prove it
    through the verifier before returning it."""
    lane4 = v4_key.split("/")[0]
    rec = {"schema": RESTAGE_SCHEMA, "join_kind": JOIN_KIND,
           "v3_key": v3_key, "v4_key": v4_key,
           "old_authority": dict(old_authority),
           "new_authority": dict(new_authority),
           "t_v3_sha256": PROD._canon_digest(transcript),
           "raw_body_sha256": hashlib.sha256(raw_body).hexdigest(),
           "raw_body_bytes": len(raw_body),
           "transform_identity": CAP.transform_identity()}
    derived = _derive_pair(repo, dict(rec, s_v3_sha256="0" * 64,
                                      s_v4_sha256="0" * 64,
                                      artifact_sha256="0" * 64,
                                      outcome=None))
    rec["s_v3_sha256"] = PROD._canon_digest(derived["v3"][1])
    rec["s_v4_sha256"] = PROD._canon_digest(derived["v4"][1])
    art = CAP.admission_transform(lane4, raw_body, derived["v4"][1])
    rec["artifact_sha256"] = PROD._canon_digest(art)
    rec["outcome"] = art.get("outcome")
    verify_restage_lineage(repo, rec, transcript, raw_body)
    return rec, art


def _selftest():
    """Against REAL preserved evidence: one reuse key from the
    committed capsule, restaged through both legs, then doctored."""
    caps_p = os.path.join(REPO, *DISP.CAPSULE_PATH.split("/"))
    store = os.environ.get("W2_V3_STORE",
                           "E:/GeoSpec/w2_capture_store_20260825")
    v3_staged = os.path.join(REPO, "docs", "f2g_window2_execution",
                             "staged_envelopes")
    if not (os.path.isfile(caps_p) and os.path.isdir(store)):
        print("w2_restage_lineage selftest: inputs absent, skipped")
        return
    with open(caps_p, encoding="utf-8") as f:
        caps = json.load(f)
    v4_key = sorted(caps["reuse_or_bridge"])[0]
    ent = caps["reuse_or_bridge"][v4_key]
    v3_key = ent["v3_key"]
    lane, ck, day = v3_key.split("/")
    stem = f"{lane.lower()}_{ck}_{day}"
    with open(os.path.join(v3_staged, stem + ".transcript.json"),
              encoding="utf-8") as f:
        t3 = json.load(f)
    with open(os.path.join(store, ent["raw_body_sha256"] + ".body"),
              "rb") as f:
        body = f.read()
    new_auth = {"commit": subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"],
        capture_output=True).stdout.decode().strip(),
        "path": DISP.AUTHORITY_PATH,
        "blob_sha256": caps["authority"]["blob_sha256"],
        "keys_sha256": caps["authority"]["keys_sha256"]}
    rec, art = build_restage_lineage(
        REPO, v3_key, v4_key, caps["old_authority"], new_auth, t3,
        body)
    out = verify_restage_lineage(REPO, rec, t3, body)
    assert out["join_kind"] == JOIN_KIND
    assert out["v4_key"] == v4_key and out["v3_key"] == v3_key

    def refuses(mut, needle, tr=None, bd=None):
        r2 = json.loads(json.dumps(rec))
        mut(r2)
        try:
            verify_restage_lineage(REPO, r2,
                                   t3 if tr is None else tr,
                                   body if bd is None else bd)
        except RestageRefusal as e:
            return needle in str(e)
        return False
    # a restaged key may never be reported as a native capture
    assert refuses(lambda r: r.__setitem__("join_kind", "NATIVE"),
                   "never reported as a native capture")
    # the artifact is RECOMPUTED, never taken from the record
    assert refuses(lambda r: r.__setitem__("artifact_sha256",
                                           "0" * 64),
                   "RECOMPUTED v4 artifact")
    # both contracts are re-derived from their pins
    assert refuses(lambda r: r.__setitem__("s_v3_sha256", "0" * 64),
                   "INDEPENDENTLY derived S_v3")
    assert refuses(lambda r: r.__setitem__("s_v4_sha256", "0" * 64),
                   "never names the contract that authenticates it")
    # an arbitrary v3 -> v4 pair refuses
    assert refuses(lambda r: r.__setitem__("v3_key",
                                           "MAG_FEED/frn/1999-01-01"),
                   "does not derive from v3_key")
    # a SYNTHETIC transcript cannot stand in for the original
    t_fake = dict(t3, request_start_utc="2020-01-01T00:00:00Z")
    assert refuses(lambda r: None, "the original T is immutable",
                   tr=t_fake)
    # swapped body refuses at the content address
    assert refuses(lambda r: None, "content address",
                   bd=body + b"x")
    # a forged transform identity refuses
    assert refuses(lambda r: r.__setitem__("transform_identity",
                                           {"forged": True}),
                   "LIVE registered transform")
    # closure: extra or missing field
    assert refuses(lambda r: r.__setitem__("extra", 1),
                   "closed field set")
    assert refuses(lambda r: r.pop("t_v3_sha256"),
                   "closed field set")
    # a wrong old-authority pin refuses before any leg passes
    assert refuses(lambda r: r["old_authority"].__setitem__(
        "blob_sha256", "0" * 64), "diverge from its pinned digest")
    print(f"w2_restage_lineage selftest: ALL PASS ({v4_key} via "
          f"{v3_key}, outcome={out['outcome']}, no network)")


if __name__ == "__main__":
    _selftest()
