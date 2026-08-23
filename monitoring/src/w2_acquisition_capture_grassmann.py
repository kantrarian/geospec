#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 ACQUISITION-CAPTURE harness (grassmann, s4t lane) -- the
SUPPLEMENTARY artifact under the staged-envelope trust boundary
(producer_boundary amendment v1 + grassmann 1406Z/1523Z rulings,
codex 1400Z ruling 2).

CLAIM CEILING (the amendment's, restated): acquisition correctness
BEFORE the staged bytes is receipt-attested, not source-code-attested.
This module is NOT load-bearing for the producer_boundary slot bind --
the envelope RECORDS are. It exists so captures are mechanical,
uniform, and receipt-complete: one fetch -> exact bytes to the staging
tree -> one closed envelope record built through the producer REV 5
surface (content recomputed at build) -> record written for the
in-repo staged_envelopes tree.

Capture SPECS carry the science (endpoint, params, lane, carrier,
day); the harness carries none. Every record binds the exact request,
the HTTP receipt (status + selected headers + fired UTC), the raw body
content address, and the produced-artifact digest -- reopenable and
recomputable per the amendment.

The selftest performs NO network I/O: a registered fake opener serves
byte fixtures; the round trip is proven through the REAL producer
verification surface (verify_envelope_record + verify_staged_day_set).
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_producer_grassmann as PROD

CAPTURE_TIMEOUT_S = 120
RECEIPT_HEADERS = ("content-type", "content-length", "date", "server")


class CaptureRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _utc_now_z():
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def http_fetch(url, opener=None, timeout=CAPTURE_TIMEOUT_S):
    """One GET -> (body bytes, fetch evidence). `opener` is injectable
    so the selftest never touches the network (it returns (status,
    headers, body, effective_url)); production passes None and uses
    urllib. Non-200 refuses typed -- an error body is never staged as
    data. Evidence records the REQUEST-START and RESPONSE-COMPLETE
    instants and the EFFECTIVE post-redirect URL (codex 1843Z item
    3)."""
    started = _utc_now_z()
    if opener is None:
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                status = getattr(r, "status", r.getcode())
                headers = {k.lower(): v for k, v in r.headers.items()}
                body = r.read()
                effective = r.geturl()
        except Exception as exc:
            raise CaptureRefusal(
                f"CAPTURE_FETCH_FAILED: {url} -> "
                f"{type(exc).__name__}: {exc}")
    else:
        status, headers, body, effective = opener(url)
    completed = _utc_now_z()
    evidence = {"requested_url": str(url),
                "effective_url": str(effective),
                "request_start_utc": started,
                "response_complete_utc": completed,
                "http_status": int(status),
                "headers": {k: str(headers.get(k)) for k in
                            RECEIPT_HEADERS if k in headers}}
    if int(status) != 200:
        raise CaptureRefusal(
            f"CAPTURE_HTTP_STATUS: {url} -> {status}")
    if not body:
        raise CaptureRefusal(f"CAPTURE_EMPTY_BODY: {url}")
    return bytes(body), evidence


def write_body(staging_dir, body):
    """Content-addressed raw-body write: <sha256>.body, immutable --
    an existing file must already match its address byte-for-byte."""
    sha = hashlib.sha256(body).hexdigest()
    os.makedirs(staging_dir, exist_ok=True)
    path = os.path.join(staging_dir, f"{sha}.body")
    if os.path.exists(path):
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != sha:
                raise CaptureRefusal(
                    f"CAPTURE_STAGING_CORRUPT: {path} does not match "
                    "its content address")
        return path, sha
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, path)
    return path, sha


def static_contract_of(spec):
    """The closed pre-capture static contract (S) carried by a spec --
    source identity ONLY (kind, ref); content digests are derived,
    never registered (codex 1843Z item 1)."""
    return {"schema": PROD.STATIC_CONTRACT_SCHEMA,
            "lane": str(spec["lane"]),
            "carrier": str(spec["carrier"]),
            "utc_day": str(spec["utc_day"]),
            "source": {"kind": str(spec["source"]["kind"]),
                       "ref": str(spec["source"]["ref"])},
            "endpoint": str(spec["endpoint"]),
            "request_params": dict(spec["request_params"]),
            "cutoff": str(spec["cutoff"]),
            "operation_params": dict(spec["operation_params"]),
            "expected_keys": sorted(str(k) for k in
                                    spec["expected_keys"])}


def capture_day(spec, staging_dir, records_dir, transcripts_dir,
                artifact_builder, opener=None):
    """One (lane, carrier, day) capture under the codex 1843Z S/T/E
    design: derive S from the spec -> fetch -> stage exact bytes ->
    WRITE AND REOPEN the closed transcript T (before any record
    exists) -> build the produced artifact -> build E through the
    producer REV 6 surface (E's dynamic seam = projection(T)) ->
    verify the full S/T/E join through the REAL day-set gate -> write
    the record. Returns (record_path, transcript_path, record,
    transcript). No credentials enter any descriptor."""
    want = {"lane", "carrier", "utc_day", "endpoint",
            "request_params", "source", "cutoff", "operation_params",
            "expected_keys"}
    if not isinstance(spec, dict) or set(spec) != want:
        got = set(spec) if isinstance(spec, dict) else None
        raise CaptureRefusal(
            f"CAPTURE_SPEC_NOT_CLOSED: missing="
            f"{sorted(want - got) if got else '?'} unknown="
            f"{sorted(got - want) if got else '?'}")
    if not isinstance(spec["source"], dict) or \
            set(spec["source"]) != {"kind", "ref"}:
        raise CaptureRefusal(
            "CAPTURE_SPEC_NOT_CLOSED: spec source carries identity "
            "only (kind, ref)")
    s = static_contract_of(spec)
    url = PROD.requested_url_of(s["endpoint"], s["request_params"])
    body, ev = http_fetch(url, opener=opener)
    body_path, sha = write_body(staging_dir, body)
    transcript = {"schema": PROD.TRANSCRIPT_SCHEMA,
                  "lane": s["lane"], "carrier": s["carrier"],
                  "utc_day": s["utc_day"],
                  "static_contract_sha256": PROD._canon_digest(s),
                  "requested_url": ev["requested_url"],
                  "effective_url": ev["effective_url"],
                  "request_start_utc": ev["request_start_utc"],
                  "response_complete_utc":
                      ev["response_complete_utc"],
                  "http_status": ev["http_status"],
                  "headers": dict(ev["headers"]),
                  "raw_body_sha256": sha,
                  "raw_body_bytes": len(body)}
    os.makedirs(transcripts_dir, exist_ok=True)
    stem = (f"{s['lane'].lower()}_{s['carrier']}_"
            f"{s['utc_day']}")
    t_path = os.path.join(transcripts_dir,
                          f"{stem}.transcript.json")
    with open(t_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(transcript, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(t_path, encoding="utf-8") as f:
        transcript = json.load(f)          # reopen before E (item 3)
    PROD.verify_transcript(transcript, s, raw_body=body)
    artifact = artifact_builder(body)
    record = PROD.build_envelope_record(
        lane=s["lane"], carrier=s["carrier"], utc_day=s["utc_day"],
        raw_body=body, source=dict(s["source"]),
        endpoint=s["endpoint"],
        request_params=dict(s["request_params"]),
        transcript=transcript, cutoff=s["cutoff"],
        operation_params=dict(s["operation_params"]),
        expected_keys=list(s["expected_keys"]), artifact=artifact)
    # the full join through the REAL gate before anything is returned
    PROD.verify_staged_day_set(
        {s["utc_day"]: record}, {s["utc_day"]: body},
        {s["utc_day"]: artifact}, {s["utc_day"]: s},
        {s["utc_day"]: transcript}, [s["utc_day"]], s["carrier"],
        s["lane"])
    os.makedirs(records_dir, exist_ok=True)
    rec_path = os.path.join(records_dir, f"{stem}.record.json")
    with open(rec_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    return rec_path, t_path, record, transcript


# --------------------------------------------------- body-store inventory
INVENTORY_SCHEMA = "f2g-w2-staged-body-inventory-v1"


def build_staged_body_inventory(store_id, store_root, entries):
    """codex 1843Z item 4: the closed in-repo inventory of the
    external (s4t) content-addressed body store. `entries` =
    {(lane, carrier, day) key string 'lane/carrier/day':
    {'sha256', 'bytes'}}; paths are DERIVED (<sha256>.body) so no
    entry can point outside the store."""
    objects = {}
    for key in sorted(entries):
        e = entries[key]
        objects[key] = {"path": f"{e['sha256']}.body",
                        "sha256": str(e["sha256"]),
                        "bytes": int(e["bytes"])}
    return {"schema": INVENTORY_SCHEMA, "store_id": str(store_id),
            "store_root": str(store_root), "objects": objects}


def verify_staged_body_inventory(inventory, store_dir):
    """Reopen EVERY object from the named store: missing, extra,
    path-escaping, or content-mismatched objects refuse typed; an
    unavailable store is a refusal, never a PASS."""
    if not isinstance(inventory, dict) or \
            inventory.get("schema") != INVENTORY_SCHEMA or \
            set(inventory) != {"schema", "store_id", "store_root",
                               "objects"}:
        raise CaptureRefusal("CAPTURE_INVENTORY_NOT_CLOSED")
    if not os.path.isdir(store_dir):
        raise CaptureRefusal(
            f"CAPTURE_STORE_UNAVAILABLE: {store_dir}")
    seen = set()
    for key, obj in sorted(inventory["objects"].items()):
        if set(obj) != {"path", "sha256", "bytes"}:
            raise CaptureRefusal(
                f"CAPTURE_INVENTORY_NOT_CLOSED: {key}")
        if obj["path"] != f"{obj['sha256']}.body" or \
                os.path.basename(obj["path"]) != obj["path"]:
            raise CaptureRefusal(
                f"CAPTURE_INVENTORY_PATH_ESCAPE: {key} -> "
                f"{obj['path']}")
        p = os.path.join(store_dir, obj["path"])
        if not os.path.isfile(p):
            raise CaptureRefusal(
                f"CAPTURE_INVENTORY_OBJECT_MISSING: {key}")
        with open(p, "rb") as f:
            raw = f.read()
        if hashlib.sha256(raw).hexdigest() != obj["sha256"] or \
                len(raw) != obj["bytes"]:
            raise CaptureRefusal(
                f"CAPTURE_INVENTORY_OBJECT_MISMATCH: {key}")
        seen.add(obj["path"])
    extra = {f for f in os.listdir(store_dir)
             if f.endswith(".body")} - seen
    if extra:
        raise CaptureRefusal(
            f"CAPTURE_INVENTORY_EXTRA_OBJECTS: {sorted(extra)[:4]}")
    return {"objects_verified": len(seen)}


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="w2_capture_kat_")
    staging = os.path.join(root, "staging")
    records = os.path.join(root, "records")
    transcripts = os.path.join(root, "transcripts")

    FIX = {"https://kat.example/fdsn?cha=HHZ&net=UW":
           (200, {"content-type": "text/plain"}, b"kat-body-1",
            "https://edge.example/fdsn-final"),
           "https://kat.example/err?d=1":
           (503, {"content-type": "text/html"}, b"oops",
            "https://kat.example/err?d=1"),
           "https://kat.example/empty?d=1":
           (200, {}, b"", "https://kat.example/empty?d=1")}

    def opener(url):
        if url not in FIX:
            raise AssertionError(f"unexpected url {url}")
        return FIX[url]

    def refuses(fn, code):
        try:
            fn()
            return False
        except CaptureRefusal as e:
            return str(e).startswith(code)

    def spec(day="2026-08-20", **over):
        s = {"lane": "DAY_CAPSULE", "carrier": "cascadia",
             "utc_day": day, "endpoint": "https://kat.example/fdsn",
             "request_params": {"net": "UW", "cha": "HHZ"},
             "source": {"kind": "fdsn-availability",
                        "ref": "https://kat.example/fdsn"},
             "cutoff": "2026-08-25",
             "operation_params": {"carrier": "cascadia", "day": day},
             "expected_keys": [day]}
        s.update(over)
        return s

    def builder(body):
        return {"n_bytes": len(body)}

    # round trip: capture -> staged bytes + TRANSCRIPT (written and
    # reopened before E) -> record -> and capture_day itself already
    # re-verifies the full S/T/E join through the REAL gate
    rp, tp, rec, tr = capture_day(spec(), staging, records,
                                  transcripts, builder,
                                  opener=opener)
    body = b"kat-body-1"
    assert rec["raw_body_sha256"] == hashlib.sha256(body).hexdigest()
    assert rec["source"]["sha256"] == rec["raw_body_sha256"]
    assert rec["receipt"]["http_status"] == 200
    assert rec["receipt"]["effective_url"] == \
        "https://edge.example/fdsn-final"
    assert rec["receipt"]["transcript_sha256"] == \
        PROD._canon_digest(tr)
    assert rec["capture_time_utc"] == tr["response_complete_utc"]
    assert tr["static_contract_sha256"] == \
        PROD._canon_digest(static_contract_of(spec()))
    staged = os.path.join(staging,
                          rec["raw_body_sha256"] + ".body")
    with open(staged, "rb") as f:
        assert f.read() == body
    # reopened artifacts round-trip through the REAL join again
    with open(rp, encoding="utf-8") as f:
        rec_reload = json.load(f)
    with open(tp, encoding="utf-8") as f:
        tr_reload = json.load(f)
    assert rec_reload == rec and tr_reload == tr
    out = PROD.verify_staged_day_set(
        {"2026-08-20": rec_reload}, {"2026-08-20": body},
        {"2026-08-20": builder(body)},
        {"2026-08-20": static_contract_of(spec())},
        {"2026-08-20": tr_reload},
        ["2026-08-20"], "cascadia", "DAY_CAPSULE")
    assert set(out) == {"2026-08-20"}

    # immutability: re-capture reuses the identical address; a
    # corrupted staged file refuses
    rp2, tp2, rec2, tr2 = capture_day(spec(), staging, records,
                                      transcripts, builder,
                                      opener=opener)
    assert rec2["raw_body_sha256"] == rec["raw_body_sha256"]
    with open(staged, "wb") as f:
        f.write(b"tampered")
    assert refuses(lambda: write_body(staging, body),
                   "CAPTURE_STAGING_CORRUPT")
    with open(staged, "wb") as f:
        f.write(body)                     # restore

    # doctors: non-200, empty body, non-closed spec, sha-carrying spec
    assert refuses(lambda: capture_day(
        spec(endpoint="https://kat.example/err",
             request_params={"d": "1"}), staging, records,
        transcripts, builder, opener=opener), "CAPTURE_HTTP_STATUS")
    assert refuses(lambda: capture_day(
        spec(endpoint="https://kat.example/empty",
             request_params={"d": "1"}), staging, records,
        transcripts, builder, opener=opener), "CAPTURE_EMPTY_BODY")
    bad = spec()
    del bad["cutoff"]
    assert refuses(lambda: capture_day(bad, staging, records,
                                       transcripts, builder,
                                       opener=opener),
                   "CAPTURE_SPEC_NOT_CLOSED")
    bad2 = spec()
    bad2["surprise"] = 1
    assert refuses(lambda: capture_day(bad2, staging, records,
                                       transcripts, builder,
                                       opener=opener),
                   "CAPTURE_SPEC_NOT_CLOSED")
    assert refuses(lambda: capture_day(
        spec(source={"kind": "k", "ref": "r", "sha256": "0" * 64}),
        staging, records, transcripts, builder, opener=opener),
        "CAPTURE_SPEC_NOT_CLOSED")

    # --- the body-store inventory (codex 1843Z item 4) ---
    inv = build_staged_body_inventory(
        "s4t-kat", "kat://store",
        {"DAY_CAPSULE/cascadia/2026-08-20":
         {"sha256": rec["raw_body_sha256"],
          "bytes": rec["raw_body_bytes"]}})
    assert verify_staged_body_inventory(inv, staging) == \
        {"objects_verified": 1}
    # unavailable store / missing object / mismatch / extra / escape
    assert refuses(
        lambda: verify_staged_body_inventory(
            inv, os.path.join(root, "nope")),
        "CAPTURE_STORE_UNAVAILABLE")
    inv_missing = json.loads(json.dumps(inv))
    inv_missing["objects"]["x/y/z"] = {"path": "ab" * 32 + ".body",
                                       "sha256": "ab" * 32,
                                       "bytes": 1}
    assert refuses(
        lambda: verify_staged_body_inventory(inv_missing, staging),
        "CAPTURE_INVENTORY_OBJECT_MISSING")
    with open(staged, "wb") as f:
        f.write(b"tampered")
    assert refuses(
        lambda: verify_staged_body_inventory(inv, staging),
        "CAPTURE_INVENTORY_OBJECT_MISMATCH")
    with open(staged, "wb") as f:
        f.write(body)
    stray = os.path.join(staging, "ff" * 32 + ".body")
    with open(stray, "wb") as f:
        f.write(b"stray")
    assert refuses(
        lambda: verify_staged_body_inventory(inv, staging),
        "CAPTURE_INVENTORY_EXTRA_OBJECTS")
    os.remove(stray)
    inv_esc = json.loads(json.dumps(inv))
    k = next(iter(inv_esc["objects"]))
    inv_esc["objects"][k]["path"] = "..\\" + \
        inv_esc["objects"][k]["path"]
    assert refuses(
        lambda: verify_staged_body_inventory(inv_esc, staging),
        "CAPTURE_INVENTORY_PATH_ESCAPE")

    print("w2_acquisition_capture selftest: ALL PASS (no network)")


if __name__ == "__main__":
    _selftest()
