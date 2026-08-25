#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Phase-B bounded 1,794-key RAW CAPTURE RUN driver (grassmann).

AUTHORIZATION: codex 2026-08-25T02:06:35Z "PASS -- PHASE A CLOSED"
at manifest 77bca1f (framework note 2026-08-25-0206): "Grassmann is
authorized to execute the frozen, bounded 1,794-key capture run."
Fired on asylum's explicit in-session go 2026-08-25 ("fire the
capture run"). That close authorizes ONLY the raw capture run: no
boundary bind/admission claim, no Tier-S/Tier-C, no PRESTART, no
owner seal, no scoring/value opening, no promotion, no publication,
no scientific conclusion. Lambda_geo remains INCONCLUSIVE.

DISCIPLINE (typed-refusal, one request per key):
- every key goes through the PRODUCTION entrypoint
  `capture_authorized` at the REVIEWED manifest commit -- the full
  authority admission chain (accrual_impl BOUND slot pin, byte
  verification, closed-schema/census/reproducer validation) runs for
  every key; the artifact is built by the registered
  `admission_transform` (no caller builder is passed);
- EXACTLY ONE HTTP request per key per driver invocation: any
  refusal (transport, HTTP status, empty body, transform/cadence/
  numeric gate) is recorded TYPED in the run ledger and the run
  moves on -- no retry, no fallback, no synthetic data;
- RESUMABLE: a key whose staged record already exists is SKIPPED
  (write-once semantics make a same-key recapture divergence-refuse
  anyway); a rerun attempts only never-captured keys, one fresh
  single request each, each attempt ledgered;
- per-host politeness pacing (>= PACING_S seconds between requests
  to the same host);
- verified TLS everywhere via certifi (the kp attempt-2 precedent:
  the local trust store fails for kp.gfz-potsdam.de);
- the authority reproducer (the pinned generator build) is computed
  ONCE and served per-key as a deep copy -- the same computation the
  per-key default would repeat 1,794 times.

Bytes -> the NAMED content-addressed store (STORE physical root);
records + transcripts -> the in-repo staged_envelopes tree; contract
and artifact class files are written per captured key in the staged
four-class format. The inventory + store descriptor are written at
run end from the staged records on disk.

Usage:
  python w2_capture_run_grassmann.py plan   # census + preflight only
  python w2_capture_run_grassmann.py run    # execute (resumable)
"""
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC
import w2_expected_contracts_gen_cayley as GEN

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
MANIFEST_COMMIT = "77bca1f57f612cee2047a68e23fc1c7f5b77b3c5"
AUTHORITY_PATH = ("docs/f2g_window2_execution/"
                  "staged_expected_contracts_v3.json")
AUTHORITY_SHA = ("b5b0a61a4edcdac2bfcd8a819f2e118a"
                 "81c492947a0b6167ffd07ef09338967e")
STORE_PHYSICAL = "E:/GeoSpec/w2_capture_store_20260825"
STORE_ID = "s4t-w2-capture-20260825"
STORE_ROOT = "s4t://geospec/w2/capture_20260825"
STAGED_DIR = os.path.join(REPO, "docs", "f2g_window2_execution",
                          "staged_envelopes")
LEDGER = os.path.join(REPO, "docs", "f2g_window2_execution",
                      "capture_run_ledger_20260825.jsonl")
PACING_S = 1.0
TIMEOUT_S = 90
UA = "geospec-w2-capture/1.0 (kantrarian/geospec window-2)"

_last_by_host = {}


def _paced_verified_opener(url):
    """One request, verified TLS, per-host pacing. HTTP errors return
    their status tuple so the capture layer refuses TYPED
    (CAPTURE_HTTP_STATUS); error bodies are never staged."""
    from urllib.parse import urlsplit
    host = urlsplit(url).netloc
    now = time.monotonic()
    wait = _last_by_host.get(host, 0) + PACING_S - now
    if wait > 0:
        time.sleep(wait)
    _last_by_host[host] = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    ctx = None
    if url.startswith("https://"):
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S,
                                    context=ctx) as r:
            return (getattr(r, "status", r.getcode()),
                    {k.lower(): v for k, v in r.headers.items()},
                    r.read(), r.geturl())
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return (exc.code, {k.lower(): v for k, v in
                           (exc.headers or {}).items()},
                b"", getattr(exc, "url", url))


def _authority():
    """The FROZEN authority from the COMMITTED git blob at the
    reviewed manifest commit -- working copies EOL-convert, the
    frozen digest binds the blob (cayley finding-4 precedent)."""
    import subprocess
    p = subprocess.run(
        ["git", "-C", REPO, "cat-file", "blob",
         f"{MANIFEST_COMMIT}:{AUTHORITY_PATH}"],
        capture_output=True)
    if p.returncode != 0:
        raise SystemExit("REFUSING: authority blob unreadable at "
                         "the reviewed manifest commit")
    raw = p.stdout
    got = hashlib.sha256(raw).hexdigest()
    if got != AUTHORITY_SHA:
        raise SystemExit(
            f"REFUSING: authority bytes {got[:12]} diverge from the "
            f"frozen digest {AUTHORITY_SHA[:12]}")
    return json.loads(raw.decode("utf-8"))


def _keys(authority):
    out = []
    for lane in sorted(authority["prestart_expected_keys"]):
        for ck in sorted(authority["prestart_expected_keys"][lane]):
            for day in authority["prestart_expected_keys"][lane][ck]:
                out.append((lane, ck, day))
    return out


def _ledger_append(entry):
    with open(LEDGER, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _staged_class_path(stem, cls):
    return os.path.join(STAGED_DIR,
                        stem + ACC.STAGED_CLASS_SUFFIX[cls])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    authority = _authority()
    keys = _keys(authority)
    n_lane = {}
    for lane, ck, day in keys:
        n_lane[lane] = n_lane.get(lane, 0) + 1
    print(f"census: {len(keys)} keys {n_lane}")
    if len(keys) != 1794:
        raise SystemExit("REFUSING: census != 1794")
    done = sum(
        1 for lane, ck, day in keys
        if os.path.exists(_staged_class_path(
            CAP._path_tokens(lane, ck, day), "record")))
    print(f"already staged (resume-skip): {done}")
    if mode != "run":
        print("plan only -- no request fired")
        return
    os.makedirs(STORE_PHYSICAL, exist_ok=True)
    os.makedirs(STAGED_DIR, exist_ok=True)
    # ONE pinned-generator reproduction served per-key as a copy
    build_once = GEN.build(REPO)
    repro_raw = json.dumps(build_once, sort_keys=True)

    def reproducer():
        return json.loads(repro_raw)
    counts = {"CAPTURED": 0, "SKIPPED": 0, "REFUSED": 0, "ERROR": 0}
    t0 = time.monotonic()
    for i, (lane, ck, day) in enumerate(keys):
        stem = CAP._path_tokens(lane, ck, day)
        if os.path.exists(_staged_class_path(stem, "record")):
            counts["SKIPPED"] += 1
            continue
        entry = {"key": f"{lane}/{ck}/{day}", "seq": i}
        try:
            rp, tp, rec, tr = CAP.capture_authorized(
                REPO, MANIFEST_COMMIT, AUTHORITY_PATH, lane, ck,
                day, STORE_PHYSICAL, STAGED_DIR, STAGED_DIR, None,
                opener=_paced_verified_opener,
                authority_reproducer=reproducer)
            s = ACC.authoritative_static_contract(authority, lane,
                                                  ck, day)
            body_path = os.path.join(
                STORE_PHYSICAL, rec["raw_body_sha256"] + ".body")
            with open(body_path, "rb") as f:
                body = f.read()
            artifact = CAP.admission_transform(lane, body, s)
            CAP._write_once_json(_staged_class_path(stem, "contract"),
                                 s, "CAPTURE_RECORD_DIVERGENT")
            CAP._write_once_json(_staged_class_path(stem, "artifact"),
                                 artifact, "CAPTURE_RECORD_DIVERGENT")
            entry.update(status="CAPTURED",
                         raw_body_sha256=rec["raw_body_sha256"],
                         raw_body_bytes=rec["raw_body_bytes"],
                         capture_time_utc=rec["capture_time_utc"])
            counts["CAPTURED"] += 1
        except CAP.CaptureRefusal as exc:
            entry.update(status="REFUSED", refusal=str(exc)[:600])
            counts["REFUSED"] += 1
        except Exception as exc:
            entry.update(status="ERROR",
                         error=f"{type(exc).__name__}: "
                               f"{str(exc)[:500]}")
            counts["ERROR"] += 1
        _ledger_append(entry)
        if (i + 1) % 100 == 0:
            el = time.monotonic() - t0
            print(f"[{i + 1}/{len(keys)}] {counts} "
                  f"({el / 60:.1f} min)", flush=True)
    # inventory + descriptor from the staged records on disk
    entries = {}
    for lane, ck, day in keys:
        stem = CAP._path_tokens(lane, ck, day)
        rp = _staged_class_path(stem, "record")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                rec = json.load(f)
            entries[f"{lane}/{ck}/{day}"] = {
                "sha256": rec["raw_body_sha256"],
                "bytes": rec["raw_body_bytes"]}
    inv = CAP.build_staged_body_inventory(STORE_ID, STORE_ROOT,
                                          entries)
    desc = {"schema": CAP.STORE_DESCRIPTOR_SCHEMA,
            "store_id": STORE_ID, "store_root": STORE_ROOT,
            "physical_root": STORE_PHYSICAL}
    CAP._write_once_json(
        os.path.join(STAGED_DIR, ACC.STAGED_INVENTORY_BASENAME),
        inv, "CAPTURE_RECORD_DIVERGENT")
    CAP._write_once_json(
        os.path.join(STAGED_DIR, ACC.STORE_DESCRIPTOR_BASENAME),
        desc, "CAPTURE_RECORD_DIVERGENT")
    print("inventory objects:", len(entries))
    print("verify_staged_body_inventory:",
          CAP.verify_staged_body_inventory(inv, desc))
    print("FINAL:", counts)


if __name__ == "__main__":
    main()
