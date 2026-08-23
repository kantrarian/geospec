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
    """One GET -> (body bytes, receipt). `opener` is injectable so the
    selftest never touches the network; production passes None and
    uses urllib. Non-200 refuses typed -- an error body is never
    staged as data."""
    fired = _utc_now_z()
    if opener is None:
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                status = getattr(r, "status", r.getcode())
                headers = {k.lower(): v for k, v in r.headers.items()}
                body = r.read()
        except Exception as exc:
            raise CaptureRefusal(
                f"CAPTURE_FETCH_FAILED: {url} -> "
                f"{type(exc).__name__}: {exc}")
    else:
        status, headers, body = opener(url)
    receipt = {"requested_url": str(url), "fired_utc": fired,
               "http_status": int(status),
               "headers": {k: str(headers.get(k)) for k in
                           RECEIPT_HEADERS if k in headers}}
    if int(status) != 200:
        raise CaptureRefusal(
            f"CAPTURE_HTTP_STATUS: {url} -> {status}")
    if not body:
        raise CaptureRefusal(f"CAPTURE_EMPTY_BODY: {url}")
    return bytes(body), receipt


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


def capture_day(spec, staging_dir, records_dir, artifact_builder,
                opener=None):
    """One (lane, carrier, day) capture: fetch -> stage exact bytes ->
    build the produced artifact via `artifact_builder(body)` (a
    producer REV 2/3 transform returning the lane artifact) -> build
    the closed envelope record through the producer REV 5 surface ->
    write the record. Returns (record_path, record).

    spec (closed): {"lane","carrier","utc_day","endpoint",
    "request_params","source","cutoff","operation_params",
    "expected_keys"} -- the url is endpoint + params, recorded
    verbatim."""
    want = {"lane", "carrier", "utc_day", "endpoint",
            "request_params", "source", "cutoff", "operation_params",
            "expected_keys"}
    if not isinstance(spec, dict) or set(spec) != want:
        got = set(spec) if isinstance(spec, dict) else None
        raise CaptureRefusal(
            f"CAPTURE_SPEC_NOT_CLOSED: missing="
            f"{sorted(want - got) if got else '?'} unknown="
            f"{sorted(got - want) if got else '?'}")
    from urllib.parse import urlencode
    url = spec["endpoint"]
    if spec["request_params"]:
        url = url + "?" + urlencode(sorted(
            spec["request_params"].items()))
    body, receipt = http_fetch(url, opener=opener)
    body_path, sha = write_body(staging_dir, body)
    artifact = artifact_builder(body)
    record = PROD.build_envelope_record(
        lane=spec["lane"], carrier=spec["carrier"],
        utc_day=spec["utc_day"], raw_body=body,
        source=dict(spec["source"], sha256=sha),
        endpoint=spec["endpoint"],
        request_params=dict(spec["request_params"]),
        receipt=receipt, capture_time_utc=receipt["fired_utc"],
        cutoff=spec["cutoff"],
        operation_params=dict(spec["operation_params"]),
        expected_keys=list(spec["expected_keys"]),
        artifact=artifact)
    os.makedirs(records_dir, exist_ok=True)
    rec_path = os.path.join(
        records_dir,
        f"{spec['lane'].lower()}_{spec['carrier']}_"
        f"{spec['utc_day']}.record.json")
    with open(rec_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    return rec_path, record


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="w2_capture_kat_")
    staging = os.path.join(root, "staging")
    records = os.path.join(root, "records")

    FIX = {"https://kat.example/fdsn?cha=HHZ&net=UW":
           (200, {"content-type": "text/plain"}, b"kat-body-1"),
           "https://kat.example/err?d=1":
           (503, {"content-type": "text/html"}, b"oops"),
           "https://kat.example/empty?d=1":
           (200, {}, b"")}

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
                        "ref": "https://kat.example/fdsn",
                        "sha256": "0" * 64},
             "cutoff": "2026-08-25",
             "operation_params": {"carrier": "cascadia", "day": day},
             "expected_keys": [day]}
        s.update(over)
        return s

    def builder(body):
        return {"n_bytes": len(body)}

    # round trip: capture -> staged bytes -> record -> REAL producer
    # verification incl the mandatory-content day-set gate
    rp, rec = capture_day(spec(), staging, records, builder,
                          opener=opener)
    body = b"kat-body-1"
    assert rec["raw_body_sha256"] == hashlib.sha256(body).hexdigest()
    assert rec["source"]["sha256"] == rec["raw_body_sha256"]
    assert rec["receipt"]["http_status"] == 200
    staged = os.path.join(staging,
                          rec["raw_body_sha256"] + ".body")
    with open(staged, "rb") as f:
        assert f.read() == body
    with open(rp, encoding="utf-8") as f:
        rec_reload = json.load(f)
    assert rec_reload == rec
    PROD.verify_envelope_record(rec_reload, raw_body=body,
                                artifact=builder(body))
    contract = {k: rec[k] for k in ("source", "endpoint",
                                    "request_params", "receipt",
                                    "capture_time_utc", "cutoff",
                                    "operation_params",
                                    "expected_keys")}
    out = PROD.verify_staged_day_set(
        {"2026-08-20": rec_reload}, {"2026-08-20": body},
        {"2026-08-20": builder(body)}, {"2026-08-20": contract},
        ["2026-08-20"], "cascadia", "DAY_CAPSULE")
    assert set(out) == {"2026-08-20"}

    # immutability: re-capture reuses the identical address; a
    # corrupted staged file refuses
    rp2, rec2 = capture_day(spec(), staging, records, builder,
                            opener=opener)
    assert rec2["raw_body_sha256"] == rec["raw_body_sha256"]
    with open(staged, "wb") as f:
        f.write(b"tampered")
    assert refuses(lambda: write_body(staging, body),
                   "CAPTURE_STAGING_CORRUPT")
    with open(staged, "wb") as f:
        f.write(body)                     # restore

    # doctors: non-200, empty body, non-closed spec
    assert refuses(lambda: capture_day(
        spec(endpoint="https://kat.example/err",
             request_params={"d": "1"}), staging, records, builder,
        opener=opener), "CAPTURE_HTTP_STATUS")
    assert refuses(lambda: capture_day(
        spec(endpoint="https://kat.example/empty",
             request_params={"d": "1"}), staging, records, builder,
        opener=opener), "CAPTURE_EMPTY_BODY")
    bad = spec()
    del bad["cutoff"]
    assert refuses(lambda: capture_day(bad, staging, records,
                                       builder, opener=opener),
                   "CAPTURE_SPEC_NOT_CLOSED")
    bad2 = spec()
    bad2["surprise"] = 1
    assert refuses(lambda: capture_day(bad2, staging, records,
                                       builder, opener=opener),
                   "CAPTURE_SPEC_NOT_CLOSED")

    print("w2_acquisition_capture selftest: ALL PASS (no network)")


if __name__ == "__main__":
    _selftest()
