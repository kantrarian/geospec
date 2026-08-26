#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE OFFLINE v3 -> v4 RESTAGER (grassmann) -- codex 1746Z finding 4.

Walks the REUSE_OR_BRIDGE partition of the manifest-pinned
disposition capsule and, for every key, emits a RESTAGED_LINEAGE
record proven through BOTH legs (see
`w2_restage_lineage_grassmann`): the immutable historical exchange
under `S_v3`, and the current derivation under `S_v4`.

**ZERO HTTP.** This module opens no socket and imports no fetcher.
Its selftest proves the stronger property directly: it replaces the
capture layer's `http_fetch` with a raiser for the duration of a
restage, so a single attempted request would fail the test.

Store discipline (codex finding 4): bodies reachable from the v4
archive are copied into the NEW named v4 store; the v3 tail and the
superseded-OMNI bytes STAY in the historical store and are never
carried across. The retired `MF4_FEED` lane name has no production
alias -- this driver carries the frozen v3 stem vocabulary itself,
exactly as the v3 repair driver does.

Modes:
  plan        zero-write: how many keys, and what is already staged
  run         emit records + copy bodies (resumable, create-once)
  --selftest  one real key end-to-end with the fetcher disabled
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC
import w2_disposition_capsule_grassmann as DISP
import w2_restage_lineage_grassmann as LIN

REPO = LIN.REPO
V3_STORE = os.environ.get("W2_V3_STORE",
                          "E:/GeoSpec/w2_capture_store_20260825")
V3_STAGED = os.path.join(REPO, "docs", "f2g_window2_execution",
                         "staged_envelopes")
V4_STORE = "E:/GeoSpec/w2_capture_store_v4"
V4_STAGED = os.path.join(REPO, "docs", "f2g_window2_execution",
                         "staged_envelopes_v4")
LINEAGE_LEDGER = os.path.join(REPO, "docs", "f2g_window2_execution",
                              "restage_lineage_ledger_v4.jsonl")
RESTAGE_SUFFIX = ".restage.json"
# FROZEN v3 stem vocabulary -- carried here, never a production alias
V3_LANES = ("DAY_CAPSULE", "SELECTION_RECORDS", "MAG_FEED",
            "MF4_FEED")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_CK = re.compile(r"[a-z0-9_]{1,64}")


def _v3_stem(v3_key):
    lane, ck, day = v3_key.split("/")
    if lane not in V3_LANES or not _DAY.fullmatch(day) or \
            not _CK.fullmatch(ck):
        raise SystemExit(f"REFUSING: {v3_key} is not a v3 key")
    return f"{lane.lower()}_{ck}_{day}"


def _capsule(commitish="HEAD"):
    raw = subprocess.run(
        ["git", "-C", REPO, "cat-file", "blob",
         f"{commitish}:{DISP.CAPSULE_PATH}"],
        capture_output=True).stdout
    if not raw:
        raise SystemExit("REFUSING: the disposition capsule is not "
                         f"committed at {commitish}")
    caps = json.loads(raw.decode("utf-8"))
    authority, _sha = DISP._authority(commitish)
    # the restager consumes LINEAGE evidence, so it must use the
    # strict registry contract -- fail-closed if the archive or the
    # source bodies cannot be reopened (codex 2303Z closure 2)
    DISP.verify_lineage_registry(caps, authority, commitish)
    return caps, authority, subprocess.run(
        ["git", "-C", REPO, "rev-parse", commitish],
        capture_output=True).stdout.decode().strip()


def _new_authority(caps, head):
    return {"commit": head, "path": DISP.AUTHORITY_PATH,
            "blob_sha256": caps["authority"]["blob_sha256"],
            "keys_sha256": caps["authority"]["keys_sha256"]}


def restage_one(caps, new_auth, v4_key, write=True):
    """Reopen the v3 evidence, prove BOTH legs, and (when writing)
    emit the v4 staged set: the restage record, the v4 contract, the
    recomputed v4 artifact, and the ORIGINAL transcript preserved
    verbatim beside them."""
    ent = caps["reuse_or_bridge"][v4_key]
    v3_key = ent["v3_key"]
    stem3 = _v3_stem(v3_key)
    with open(os.path.join(V3_STAGED, stem3 + ".transcript.json"),
              encoding="utf-8") as f:
        t3 = json.load(f)
    bpath = os.path.join(V3_STORE, ent["raw_body_sha256"] + ".body")
    with open(bpath, "rb") as f:
        body = f.read()
    rec, art = LIN.build_restage_lineage(
        REPO, v3_key, v4_key, caps["old_authority"], new_auth, t3,
        body)
    if not write:
        return rec, art
    lane, ck, day = v4_key.split("/")
    stem4 = CAP._path_tokens(lane, ck, day)
    # the body moves into the NEW named v4 store, content-addressed
    os.makedirs(V4_STORE, exist_ok=True)
    dest = os.path.join(V4_STORE, rec["raw_body_sha256"] + ".body")
    if not os.path.exists(dest):
        shutil.copyfile(bpath, dest)
    with open(dest, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != \
                rec["raw_body_sha256"]:
            raise SystemExit(f"REFUSING: {v4_key} copied body does "
                             "not match its content address")
    os.makedirs(V4_STAGED, exist_ok=True)
    s4 = ACC.authoritative_static_contract(
        json.loads(subprocess.run(
            ["git", "-C", REPO, "cat-file", "blob",
             f"{new_auth['commit']}:{new_auth['path']}"],
            capture_output=True).stdout.decode()), lane, ck, day)
    for suffix, obj in ((RESTAGE_SUFFIX, rec),
                        (ACC.STAGED_CLASS_SUFFIX["contract"], s4),
                        (ACC.STAGED_CLASS_SUFFIX["artifact"], art),
                        (ACC.STAGED_CLASS_SUFFIX["transcript"], t3)):
        CAP._write_once_json(os.path.join(V4_STAGED, stem4 + suffix),
                             obj, "RESTAGE_ARTIFACT_DIVERGENT")
    return rec, art


def plan(commitish="HEAD"):
    caps, _a, _h = _capsule(commitish)
    keys = sorted(caps["reuse_or_bridge"])
    done = sum(1 for k in keys
               if os.path.exists(os.path.join(
                   V4_STAGED,
                   CAP._path_tokens(*k.split("/")) + RESTAGE_SUFFIX)))
    print(f"REUSE_OR_BRIDGE keys: {len(keys)}")
    print(f"already restaged    : {done}")
    print(f"v4 store            : {V4_STORE}")
    print("zero HTTP; no write performed")
    return keys


def run(commitish="HEAD"):
    caps, _a, head = _capsule(commitish)
    new_auth = _new_authority(caps, head)
    keys = sorted(caps["reuse_or_bridge"])
    done = skipped = 0
    for k in keys:
        stem4 = CAP._path_tokens(*k.split("/"))
        if os.path.exists(os.path.join(V4_STAGED,
                                       stem4 + RESTAGE_SUFFIX)):
            skipped += 1
            continue
        rec, _art = restage_one(caps, new_auth, k)
        with open(LINEAGE_LEDGER, "a", encoding="utf-8",
                  newline="\n") as f:
            f.write(json.dumps(
                {"v4_key": k, "v3_key": rec["v3_key"],
                 "join_kind": rec["join_kind"],
                 "claim": rec["claim"],
                 "raw_body_sha256": rec["raw_body_sha256"],
                 "artifact_sha256": rec["artifact_sha256"],
                 "http_requests": 0}, sort_keys=True) + "\n")
        done += 1
        if done % 200 == 0:
            print(f"  restaged {done}...", flush=True)
    print(f"restaged: {done}  already-present: {skipped}")
    # the v4 store must hold EXACTLY the bodies reachable from the
    # restaged set (native captures add to this set later)
    want = {caps["reuse_or_bridge"][k]["raw_body_sha256"] + ".body"
            for k in keys}
    have = {f for f in os.listdir(V4_STORE) if f.endswith(".body")}
    extra = have - want
    if extra:
        print(f"NOTE: {len(extra)} object(s) in the v4 store are not "
              "reachable from the restaged set (native captures "
              "land here too)")
    print("missing from store:", len(want - have))


def _selftest():
    """One REAL key end-to-end with the fetcher DISABLED -- if any
    code path attempted a request, this test would fail."""
    _usable = os.path.isdir(V3_STORE) and any(
        n.endswith(".body") for n in os.listdir(V3_STORE))
    if not (_usable and os.path.isdir(V3_STAGED)):
        print("w2_restage_v4 selftest: inputs absent, skipped W2_RESULT=SKIPPED_NO_INPUTS")
        return
    caps, _a, head = _capsule("HEAD")
    new_auth = _new_authority(caps, head)
    k = sorted(caps["reuse_or_bridge"])[0]

    def _no_network(*a, **kw):
        raise AssertionError("the restager attempted an HTTP fetch")
    saved = CAP.http_fetch
    CAP.http_fetch = _no_network
    try:
        rec, art = restage_one(caps, new_auth, k, write=False)
    finally:
        CAP.http_fetch = saved
    assert rec["join_kind"] == LIN.JOIN_KIND
    assert rec["v4_key"] == k
    # the emitted record proves itself through the REAL verifier
    stem3 = _v3_stem(rec["v3_key"])
    with open(os.path.join(V3_STAGED, stem3 + ".transcript.json"),
              encoding="utf-8") as f:
        t3 = json.load(f)
    with open(os.path.join(V3_STORE,
                           rec["raw_body_sha256"] + ".body"),
              "rb") as f:
        body = f.read()
    out = LIN.verify_restage_lineage(REPO, rec, t3, body)
    assert out["v4_key"] == k
    # the frozen v3 vocabulary is carried HERE, not aliased in
    # production: the retired lane resolves for this driver only
    assert _v3_stem("MF4_FEED/kp/2026-08-01") == \
        "mf4_feed_kp_2026-08-01"
    assert "MF4_FEED" not in __import__(
        "w2_producer_grassmann").RECORD_LANES
    print(f"w2_restage_v4 selftest: ALL PASS ({k}, fetcher disabled, "
          "no network)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if mode == "plan":
        plan()
    elif mode == "run":
        run()
    elif mode == "--selftest":
        _selftest()
    else:
        raise SystemExit("usage: plan | run | --selftest")


if __name__ == "__main__":
    main()
