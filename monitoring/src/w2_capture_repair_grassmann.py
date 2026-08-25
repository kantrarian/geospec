#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OFFLINE capture-run repair + the closed two-part archive
(grassmann), per codex postflight rulings 4 and 5 (2026-08-25T0527Z).

**ZERO NEW HTTP.** This driver never opens a socket. Refetching the
preserved bodies would discard the exact evidence that found the
defect (codex ruling 4), so every repair replays bytes already in the
named store, located by their TRANSCRIPT-BOUND content addresses.

`replay` (ruling 4): for every ledger key whose refusal was the Kp
status-vocabulary class, reopen the preserved transcript T, resolve
its raw_body_sha256 in the store, derive S from the frozen authority,
re-verify T against S + body through the REAL producer gate, rebuild
the artifact through the REPAIRED transform, and create the missing
record/contract/artifact create-once -- byte-identical to what the
original capture would have written had the vocabulary been right.
Appends a repair ledger binding each old refusal event to the
repaired transform identity.

`archive` (ruling 5): build ONE closed archive over ALL authority
keys, partitioned exactly into ADMITTED and REFUSED, every entry
bound to its transcript/body, every refused entry carrying a
RECOMPUTED typed refusal code, and verify it -- the union accounts
for every body in the store (no orphans) while only the admitted
partition is reachable by the boundary verifier.

Usage:
  python w2_capture_repair_grassmann.py replay
  python w2_capture_repair_grassmann.py archive
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC
import w2_producer_grassmann as PROD
import w2_capture_run_grassmann as RUN

REPO = RUN.REPO
STAGED_DIR = RUN.STAGED_DIR
STORE = RUN.STORE_PHYSICAL
LEDGER = RUN.LEDGER
REPAIR_LEDGER = os.path.join(
    REPO, "docs", "f2g_window2_execution",
    "capture_repair_ledger_20260825.jsonl")
ARCHIVE_PATH = os.path.join(STAGED_DIR,
                            "capture_run_archive.json")
# the exact refusal class the 2026-08-25 amendment repairs
KP_VOCAB_MARK = "is not in the registered GFZ vocabulary"


def _read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _body(sha):
    p = os.path.join(STORE, sha + ".body")
    with open(p, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise SystemExit(f"REFUSING: store body {sha[:12]} corrupt")
    return raw


def _stem(key):
    lane, ck, day = key.split("/")
    return CAP._path_tokens(lane, ck, day), lane, ck, day


def _cls(stem, cls):
    return os.path.join(STAGED_DIR,
                        stem + ACC.STAGED_CLASS_SUFFIX[cls])


def _ledger():
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def replay(authority):
    rows = _ledger()
    targets = [r for r in rows if r["status"] == "REFUSED"
               and KP_VOCAB_MARK in r.get("refusal", "")]
    print(f"vocabulary-class refusals to replay: {len(targets)}")
    ident = CAP.transform_identity()
    done = skipped = 0
    for r in targets:
        key = r["key"]
        stem, lane, ck, day = _stem(key)
        if os.path.exists(_cls(stem, "record")):
            skipped += 1
            continue
        t = _read(_cls(stem, "transcript"))
        body = _body(t["raw_body_sha256"])
        s = ACC.authoritative_static_contract(authority, lane, ck,
                                              day)
        # the preserved transcript must still bind S and these bytes
        PROD.verify_transcript(t, s, raw_body=body)
        artifact = CAP.admission_transform(lane, body, s)
        record = PROD.build_envelope_record(
            lane=s["lane"], carrier=s["carrier"],
            utc_day=s["utc_day"], raw_body=body,
            source=dict(s["source"]), endpoint=s["endpoint"],
            request_params=dict(s["request_params"]), transcript=t,
            cutoff=s["cutoff"],
            operation_params=dict(s["operation_params"]),
            expected_keys=list(s["expected_keys"]),
            artifact=artifact)
        # the FULL five-map join through the real gate, as at capture
        PROD.verify_staged_day_set(
            {day: record}, {day: body}, {day: artifact}, {day: s},
            {day: t}, [day], ck, lane)
        CAP._write_once_json(_cls(stem, "contract"), s,
                             "CAPTURE_RECORD_DIVERGENT")
        CAP._write_once_json(_cls(stem, "artifact"), artifact,
                             "CAPTURE_RECORD_DIVERGENT")
        CAP._write_once_json(_cls(stem, "record"), record,
                             "CAPTURE_RECORD_DIVERGENT")
        with open(REPAIR_LEDGER, "a", encoding="utf-8",
                  newline="\n") as f:
            f.write(json.dumps({
                "key": key, "repair": "kp-status-vocabulary-20260825",
                "original_refusal": r.get("refusal", "")[:300],
                "original_seq": r["seq"],
                "raw_body_sha256": t["raw_body_sha256"],
                "transcript_sha256": PROD._canon_digest(t),
                "output_sha256": record["output_sha256"],
                "definitive_intervals":
                    artifact.get("definitive_intervals"),
                "transform_identity": ident,
                "http_requests": 0}, sort_keys=True) + "\n")
        done += 1
    print(f"replayed: {done}  already-present: {skipped}")
    return done


def archive(authority):
    keys = [f"{l}/{c}/{d}" for (l, c, d) in RUN._keys(authority)]
    admitted, refused = {}, {}
    for key in keys:
        stem, lane, ck, day = _stem(key)
        t = _read(_cls(stem, "transcript"))
        common = {"lane": lane, "carrier": ck, "utc_day": day,
                  "static_contract_sha256":
                      t["static_contract_sha256"],
                  "transcript_sha256": PROD._canon_digest(t),
                  "raw_body_sha256": t["raw_body_sha256"],
                  "raw_body_bytes": t["raw_body_bytes"]}
        rp = _cls(stem, "record")
        if os.path.exists(rp):
            admitted[key] = dict(
                common, output_sha256=_read(rp)["output_sha256"])
            continue
        # RECOMPUTE the typed refusal from the preserved body
        s = ACC.authoritative_static_contract(authority, lane, ck,
                                              day)
        try:
            CAP.admission_transform(lane, _body(
                t["raw_body_sha256"]), s)
            raise SystemExit(
                f"REFUSING: {key} has no record but its preserved "
                "body now transforms cleanly -- run replay first")
        except CAP.CaptureRefusal as exc:
            msg = str(exc)
            refused[key] = dict(
                common, refusal_code=msg.split(":", 1)[0].strip(),
                refusal_detail=msg.split(":", 1)[1].strip()[:400])
    auth_id = _read(_cls(_stem(keys[0])[0], "transcript"))["authority"]
    arch = CAP.build_capture_run_archive(
        RUN.STORE_ID, RUN.STORE_ROOT, auth_id, admitted, refused)
    desc = {"schema": CAP.STORE_DESCRIPTOR_SCHEMA,
            "store_id": RUN.STORE_ID, "store_root": RUN.STORE_ROOT,
            "physical_root": STORE}
    out = CAP.verify_capture_run_archive(
        arch, desc, authority["prestart_expected_keys"])
    CAP._write_once_json(ARCHIVE_PATH, arch,
                         "CAPTURE_RECORD_DIVERGENT")
    print("archive verified:", out)
    codes = {}
    for e in refused.values():
        codes[e["refusal_detail"][:46]] = \
            codes.get(e["refusal_detail"][:46], 0) + 1
    for k in sorted(codes, key=lambda x: -codes[x]):
        print(f"  refused x{codes[k]}: {k}")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "archive"
    authority = RUN._authority()
    if mode == "replay":
        replay(authority)
    elif mode == "archive":
        archive(authority)
    else:
        raise SystemExit("usage: replay | archive")


if __name__ == "__main__":
    main()
