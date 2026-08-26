#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE STRICT POST-MANIFEST RESTAGE VERIFICATION BATCH (grassmann).

codex 0410Z fix 1 + fix 2 (outer half) + fix 4 (strict mode).

The manifest-owned verifier was landed but UNUSED -- and an unused
safe door does not close the old load-bearing one. This is the
entrypoint that makes it load-bearing: it walks EVERY restaged key
through `verify_restage_lineage_pinned` against a named manifest
commit and emits the post-manifest verification receipt.

Ordering (codex's two-record sequence): this runs AFTER the single
manifest regeneration. Its receipt is a DOWNSTREAM operation record
whose authority is the manifest it names -- it is not an input the
manifest must recursively pin, which is why a summary pinned inside
that manifest could never honestly contain this result.

STRICT BY CONSTRUCTION: a missing capsule, store, record, transcript
or body is a TYPED FAILURE. There is no skip path and no green exit
on absent evidence -- that was the defect that let me report a skip
as a green selftest.

ZERO HTTP.

Usage:
  python w2_restage_verify_batch_grassmann.py <manifest-commit>
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
import w2_disposition_capsule_grassmann as DISP
import w2_restage_lineage_grassmann as LIN
import w2_restage_v4_grassmann as RES

REPO = LIN.REPO
RECEIPT_SCHEMA = "f2g-w2-postmanifest-verification-receipt-v1"


class BatchRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _b(msg):
    raise BatchRefusal("RESTAGE_BATCH_REFUSED: " + str(msg))


def _require_valid_manifest(manifest_commit):
    """codex 0410Z fix 2 (outer half): establish that the object at
    `manifest_commit` IS a valid closed prestart manifest BEFORE any
    child pin is opened. Without this the caller chooses the commit
    and nothing checks what lives there."""
    import f2g_execution_manifest_verifier_cayley as EMV
    p = subprocess.run(["git", "-C", REPO, "rev-parse",
                        f"{manifest_commit}^{{commit}}"],
                       capture_output=True)
    full = p.stdout.decode().strip()
    if p.returncode != 0 or len(full) != 40:
        _b(f"{manifest_commit!r} does not resolve to a commit")
    try:
        verdict = EMV.verify(REPO, full, prestart=True)
    except Exception as exc:
        _b(f"the execution-manifest verifier refused at {full[:12]} "
           f"({type(exc).__name__}: {exc})")
    if verdict.get("verdict") != "PASS":
        _b(f"the execution manifest at {full[:12]} is not a prestart "
           f"PASS (verdict={verdict.get('verdict')!r})")
    open_slots = [n for n, s in
                  (verdict.get("slots") or {}).items()
                  if isinstance(s, dict) and s.get("status") == "OPEN"]
    if open_slots:
        _b(f"the manifest at {full[:12]} still has OPEN slots "
           f"{sorted(open_slots)} -- a post-manifest verification "
           "runs only over a closed manifest")
    return full, verdict


def _require(path, what):
    if not os.path.exists(path):
        _b(f"{what} is ABSENT at {path} -- strict mode: a missing "
           "input is a typed failure, never a skip and never green")
    return path


def run(manifest_commit, store_root=None):
    full, _verdict = _require_valid_manifest(manifest_commit)
    caps_raw = subprocess.run(
        ["git", "-C", REPO, "cat-file", "blob",
         f"{full}:{DISP.CAPSULE_PATH}"], capture_output=True).stdout
    if not caps_raw:
        _b(f"the disposition capsule is not present at {full[:12]}")
    caps = json.loads(caps_raw.decode("utf-8"))
    keys = sorted(caps.get("reuse_or_bridge") or {})
    if not keys:
        _b("the registered capsule has an EMPTY reuse partition -- "
           "there is nothing to verify and that is not a pass")
    store = store_root or RES.V3_STORE
    _require(store, "the source body store")
    _require(RES.V4_STAGED, "the v4 staged tree")
    attempted = verified = 0
    v4keys, v3keys, tset, bset, aset = [], [], [], [], []
    outcomes = {}
    for k in keys:
        attempted += 1
        lane, ck, day = k.split("/")
        stem = CAP._path_tokens(lane, ck, day)
        rp = _require(os.path.join(RES.V4_STAGED,
                                   stem + RES.RESTAGE_SUFFIX),
                      f"the restage record for {k}")
        tp = _require(os.path.join(RES.V4_STAGED, stem +
                                   ".transcript.json"),
                      f"the preserved transcript for {k}")
        with open(rp, encoding="utf-8") as f:
            rec = json.load(f)
        with open(tp, encoding="utf-8") as f:
            t3 = json.load(f)
        bp = _require(os.path.join(store,
                                   rec["raw_body_sha256"] + ".body"),
                      f"the source body for {k}")
        with open(bp, "rb") as f:
            body = f.read()
        try:
            out = LIN.verify_restage_lineage_pinned(
                REPO, full, rec, t3, body, store_root=store)
        except Exception as exc:
            _b(f"{k} FAILED manifest-owned verification "
               f"({type(exc).__name__}: {str(exc)[:160]})")
        verified += 1
        v4keys.append(k)
        v3keys.append(rec["v3_key"])
        tset.append(rec["t_v3_sha256"])
        bset.append(rec["raw_body_sha256"])
        aset.append(rec["artifact_sha256"])
        outcomes[out["outcome"]] = outcomes.get(out["outcome"], 0) + 1
    if attempted != verified:
        _b("attempted != verified")

    def dg(xs):
        return hashlib.sha256(json.dumps(
            sorted(xs), separators=(",", ":")).encode()).hexdigest()
    return {"schema": RECEIPT_SCHEMA,
            "manifest_commit": full,
            "capsule_sha256": hashlib.sha256(caps_raw).hexdigest(),
            "transform_identity": CAP.transform_identity(),
            "attempted": attempted, "verified": verified,
            "v4_key_digest": dg(v4keys), "v3_key_digest": dg(v3keys),
            "original_t_digest_set": dg(tset),
            "body_digest_set": dg(bset),
            "artifact_digest_set": dg(aset),
            "outcomes": dict(sorted(outcomes.items())),
            "http_requests": 0,
            "interpreter": sys.version.split()[0],
            "claim_scope": "MANIFEST_OWNED_RESTAGE_VERIFICATION",
            "authorizes": "NOTHING"}


def _selftest():
    """STRICT: this asserts the refusal contract, which is what can
    be established before the final manifest exists. It never exits
    green on absent evidence."""
    def refuses(fn, needle):
        try:
            fn()
            return False
        except BatchRefusal as e:
            return needle in str(e)
    assert refuses(lambda: run("not-a-real-commit"),
                   "does not resolve")
    # the CURRENT manifest still has OPEN slots by design, so the
    # batch must refuse against it -- proving it cannot be run early
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                          capture_output=True).stdout.decode().strip()
    assert refuses(lambda: run(head),
                   "OPEN slots") or refuses(lambda: run(head),
                                            "not a prestart PASS")
    # the OLD record-owned API cannot produce this receipt: it has no
    # manifest input at all, so no call to it can yield a
    # manifest-owned claim (codex 0410Z fix 1 doctor)
    import inspect
    old = inspect.signature(LIN.verify_restage_lineage).parameters
    assert "manifest_commit" not in old
    new = inspect.signature(
        LIN.verify_restage_lineage_pinned).parameters
    assert "manifest_commit" in new
    print("w2_restage_verify_batch selftest: ALL PASS "
          "(refusal contract only; no manifest-owned PASS exists "
          "before the single regeneration)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return
    if len(sys.argv) < 2:
        raise SystemExit("usage: <manifest-commit> | --selftest")
    print(json.dumps(run(sys.argv[1]), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
