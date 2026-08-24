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


def http_fetch(url, opener=None, timeout=CAPTURE_TIMEOUT_S,
               clock=None):
    """One GET -> (body bytes, fetch evidence). `opener` is injectable
    so the selftest never touches the network (it returns (status,
    headers, body, effective_url)); production passes None and uses
    urllib. `clock` is injectable (canonical-Z callable) so the
    write-once recapture semantics are deterministically testable.
    Non-200 refuses typed -- an error body is never staged as data.
    Evidence records the REQUEST-START and RESPONSE-COMPLETE instants
    and the EFFECTIVE post-redirect URL (codex 1843Z item 3)."""
    clk = clock or _utc_now_z
    started = clk()
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
    completed = clk()
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


def _write_once_json(path, obj, divergent_code):
    """codex 2015Z item 3 + 2235Z item 4: audit carriers are
    WRITE-ONCE with an atomic NO-REPLACE publication -- a unique
    same-directory temp is published via os.link (create-once: raises
    if the destination exists; never overwrites). The losing racer
    reopens the winner: identical canonical bytes are reused,
    divergent bytes refuse typed. Exactly one publication can ever
    win; the first bytes remain intact."""
    def _reopen_or_refuse():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if PROD._canon_digest(existing) != PROD._canon_digest(obj):
            raise CaptureRefusal(
                f"{divergent_code}: {os.path.basename(path)} already "
                "exists with divergent content")
        return existing

    if os.path.exists(path):
        return _reopen_or_refuse()
    # codex 0130Z item-4 residual: the temp must be unique PER CALL
    # (a per-PID temp lets a second THREAD mutate the already-linked
    # inode); mkstemp gives an exclusive unique same-directory file
    import tempfile as _tf
    fd, tmp = _tf.mkstemp(dir=os.path.dirname(path) or ".",
                          suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8",
                       newline="\n") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)        # atomic create-once, no replace
        except FileExistsError:
            return _reopen_or_refuse()
        # post-link verification: the published destination must equal
        # THIS caller's canonical bytes (any mutation via a shared
        # inode surfaces here), exactly as the losing path checks
        return _reopen_or_refuse()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


_CARRIER_TOKEN = None


def _path_tokens(lane, carrier, utc_day):
    """codex 2015Z item 3 + 2235Z item 5: validate every
    path-deriving token BEFORE any path exists -- lane from the
    registered enum, carrier a STRICT STRING against a closed grammar
    via fullmatch (a $-anchored match admits a trailing newline into
    an audit path), day canonical."""
    global _CARRIER_TOKEN
    if _CARRIER_TOKEN is None:
        import re
        _CARRIER_TOKEN = re.compile(r"[a-z0-9_]{1,64}")
    if lane not in PROD.RECORD_LANES:
        raise CaptureRefusal(f"CAPTURE_PATH_TOKEN_INVALID: lane "
                             f"{lane!r}")
    if not isinstance(carrier, str) \
            or not _CARRIER_TOKEN.fullmatch(carrier):
        raise CaptureRefusal(f"CAPTURE_PATH_TOKEN_INVALID: carrier "
                             f"{carrier!r}")
    try:
        PROD._canon_day(utc_day, "utc_day")
    except PROD.ProducerRefusal:
        raise CaptureRefusal(f"CAPTURE_PATH_TOKEN_INVALID: day "
                             f"{utc_day!r}")
    return f"{lane.lower()}_{carrier}_{utc_day}"


# codex 0349Z item 1: the FIXTURE sentinel authority identity. It is
# schema-valid (so fixture transcripts/records close) but names no
# real authority -- admission comparison against the pinned authority
# identity refuses it, so nothing captured through the fixture path
# can ever be admitted.
FIXTURE_AUTHORITY_ID = {"commit": "0" * 40,
                        "path": "fixture://unbound-authority",
                        "blob_sha256": "0" * 64,
                        "keys_sha256": "0" * 64}


def capture_day(spec, staging_dir, records_dir, transcripts_dir,
                artifact_builder, opener=None, clock=None,
                authority_id=None):
    """FIXTURE/INJECTION path (codex 0349Z item 1: the free-spec
    production path is REMOVED -- production captures go through
    capture_authorized, whose only input is the manifest-pinned
    authority identity; this entrypoint remains for fixtures and is
    marked by the sentinel authority identity admission refuses).
    One (lane, carrier, day) capture under the codex 1843Z S/T/E
    design: derive S from the spec -> fetch -> stage exact bytes ->
    WRITE AND REOPEN the closed transcript T (write-once; a divergent
    recapture refuses) -> build the produced artifact -> build E
    through the producer surface (E's dynamic seam = projection(T)) ->
    verify the full S/T/E join through the REAL day-set gate -> write
    the record (write-once). Returns (record_path, transcript_path,
    record, transcript). No credentials enter any descriptor."""
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
    stem = _path_tokens(spec["lane"], spec["carrier"],
                        spec["utc_day"])
    auth_id = dict(authority_id or FIXTURE_AUTHORITY_ID)
    PROD._validate_authority_id(auth_id, "CAPTURE_AUTHORITY_INVALID",
                                refusal_cls=CaptureRefusal)
    s = static_contract_of(spec)
    url = PROD.requested_url_of(s["endpoint"], s["request_params"])
    body, ev = http_fetch(url, opener=opener, clock=clock)
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
                  "raw_body_bytes": len(body),
                  "authority": auth_id}
    os.makedirs(transcripts_dir, exist_ok=True)
    t_path = os.path.join(transcripts_dir,
                          f"{stem}.transcript.json")
    transcript = _write_once_json(t_path, transcript,
                                  "CAPTURE_TRANSCRIPT_DIVERGENT")
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
    record = _write_once_json(rec_path, record,
                              "CAPTURE_RECORD_DIVERGENT")
    return rec_path, t_path, record, transcript


# ------------------------------------------------ authorized production path
def capture_authorized(repo, authority_ref, lane, carrier, utc_day,
                       staging_dir, records_dir, transcripts_dir,
                       artifact_builder, *, opener=None, clock=None,
                       blob_reader=None, git_resolve=None):
    """THE production capture entrypoint (codex 0349Z item 1): its
    ONLY production inputs are the manifest-pinned v3 authority
    identity and the (lane, carrier, day) key. BEFORE any network
    call it: resolves the authority commit to a full 40-hex lineage;
    reopens the exact authority bytes and verifies them against the
    pinned blob digest; recomputes the key digest; requires the key to
    be REGISTERED in the authority; derives S through the instrument's
    authoritative_static_contract (an OPEN token -- the unreviewed
    class -- refuses there); and constructs the request SOLELY from S.
    Every refusal happens with ZERO network calls. The authority
    identity binds into T and E (the chronological carrier that the
    static freeze preceded the request).

    authority_ref (closed): {"commit", "path", "blob_sha256"}."""
    if not isinstance(authority_ref, dict) or \
            set(authority_ref) != {"commit", "path", "blob_sha256"}:
        raise CaptureRefusal(
            "CAPTURE_AUTHORITY_INVALID: authority_ref must be the "
            "closed {commit, path, blob_sha256} pin reference")
    if git_resolve is None:
        import subprocess

        def git_resolve(commitish):
            p = subprocess.run(
                ["git", "-C", repo, "rev-parse",
                 f"{commitish}^{{commit}}"], capture_output=True)
            full = p.stdout.decode().strip()
            if p.returncode != 0 or len(full) != 40:
                raise CaptureRefusal(
                    f"CAPTURE_AUTHORITY_INVALID: {commitish!r} does "
                    "not resolve to a 40-hex lineage")
            return full
    if blob_reader is None:
        import subprocess

        def blob_reader(commit, path):
            p = subprocess.run(
                ["git", "-C", repo, "cat-file", "blob",
                 f"{commit}:{path}"], capture_output=True)
            if p.returncode != 0:
                raise CaptureRefusal(
                    f"CAPTURE_AUTHORITY_INVALID: {path} unreadable "
                    f"at {commit}")
            return p.stdout
    full = git_resolve(authority_ref["commit"])
    if not (isinstance(full, str) and len(full) == 40
            and all(c in "0123456789abcdef" for c in full)):
        raise CaptureRefusal(
            "CAPTURE_AUTHORITY_INVALID: resolved lineage is not "
            "40-hex")
    raw = blob_reader(full, authority_ref["path"])
    got = hashlib.sha256(raw).hexdigest()
    if got != authority_ref["blob_sha256"]:
        raise CaptureRefusal(
            "CAPTURE_AUTHORITY_INVALID: authority bytes diverge from "
            f"the pinned blob ({got[:12]} != "
            f"{str(authority_ref['blob_sha256'])[:12]})")
    authority = json.loads(raw.decode("utf-8"))
    keys = authority.get("prestart_expected_keys", {})
    keys_sha = hashlib.sha256(json.dumps(
        keys, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    if keys_sha != authority.get("prestart_expected_keys_sha256"):
        raise CaptureRefusal(
            "CAPTURE_AUTHORITY_INVALID: authority key digest does "
            "not recompute")
    days = keys.get(lane, {}).get(carrier)
    if not days or str(utc_day) not in days:
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_INVALID: {lane}/{carrier}/{utc_day} "
            "is not a registered authority key (post-dated or "
            "unauthorized)")
    import w2_accrual_instrument_cayley as ACC
    try:
        s = ACC.authoritative_static_contract(authority, lane,
                                              carrier, str(utc_day))
    except Exception as exc:
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_INVALID: {exc}")
    auth_id = {"commit": full, "path": str(authority_ref["path"]),
               "blob_sha256": got, "keys_sha256": keys_sha}
    spec = {"lane": s["lane"], "carrier": s["carrier"],
            "utc_day": s["utc_day"], "endpoint": s["endpoint"],
            "request_params": dict(s["request_params"]),
            "source": dict(s["source"]), "cutoff": s["cutoff"],
            "operation_params": dict(s["operation_params"]),
            "expected_keys": list(s["expected_keys"])}
    return capture_day(spec, staging_dir, records_dir,
                       transcripts_dir, artifact_builder,
                       opener=opener, clock=clock,
                       authority_id=auth_id)


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


STORE_DESCRIPTOR_SCHEMA = "f2g-w2-store-descriptor-v1"
STORE_DESCRIPTOR_KEYS = {"schema", "store_id", "store_root",
                         "physical_root"}


def verify_staged_body_inventory(inventory, store_descriptor):
    """codex 2015Z item 1 (BLOCKER repair): the verifier binds the
    NAMED store, not a caller path. `store_descriptor` is the
    independently REGISTERED (manifest-pinned) descriptor mapping the
    logical store identity to its physical root -- there is no
    caller-controlled directory argument at all. The inventory's
    store_id AND store_root must equal the descriptor's; the physical
    root comes ONLY from the descriptor mapping. A matching digest
    proves object content, never named-store standing: correct bytes
    in the wrong store refuse. Then reopen EVERY object: missing,
    extra, path-escaping, or content-mismatched objects refuse typed;
    an unavailable store is a refusal, never a PASS. No credentials
    appear in any descriptor."""
    if not isinstance(inventory, dict) or \
            inventory.get("schema") != INVENTORY_SCHEMA or \
            set(inventory) != {"schema", "store_id", "store_root",
                               "objects"}:
        raise CaptureRefusal("CAPTURE_INVENTORY_NOT_CLOSED")
    d = store_descriptor
    if not isinstance(d, dict) or \
            d.get("schema") != STORE_DESCRIPTOR_SCHEMA or \
            set(d) != STORE_DESCRIPTOR_KEYS:
        raise CaptureRefusal("CAPTURE_STORE_DESCRIPTOR_NOT_CLOSED")
    if inventory["store_id"] != d["store_id"] or \
            inventory["store_root"] != d["store_root"]:
        raise CaptureRefusal(
            f"CAPTURE_STORE_IDENTITY_MISMATCH: inventory names "
            f"{inventory['store_id']!r}/{inventory['store_root']!r} "
            f"but the registered descriptor is {d['store_id']!r}/"
            f"{d['store_root']!r}")
    store_dir = os.path.realpath(str(d["physical_root"]))
    if not os.path.isdir(store_dir):
        raise CaptureRefusal(
            f"CAPTURE_STORE_UNAVAILABLE: {d['store_id']} -> "
            f"{store_dir}")
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
        p = os.path.realpath(os.path.join(store_dir, obj["path"]))
        if not p.startswith(store_dir + os.sep):
            raise CaptureRefusal(
                f"CAPTURE_INVENTORY_PATH_ESCAPE: {key} resolves "
                "outside the registered store")
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


def _race_worker(path, obj, code, barrier, q):
    """Module-level worker for the two-process divergent-race lock
    (codex 2235Z item 4); spawn-safe."""
    barrier.wait()
    try:
        got = _write_once_json(path, obj, code)
        q.put(("ok", got))
    except CaptureRefusal as e:
        q.put(("refused", str(e)))


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

    # a FIXED injectable clock makes write-once recapture semantics
    # deterministic (codex 2015Z item 3)
    def clock_a():
        return "2026-08-20T12:01:33Z"

    def clock_b():
        return "2026-08-20T12:05:00Z"

    # round trip: capture -> staged bytes + TRANSCRIPT (written and
    # reopened before E) -> record -> and capture_day itself already
    # re-verifies the full S/T/E join through the REAL gate
    rp, tp, rec, tr = capture_day(spec(), staging, records,
                                  transcripts, builder,
                                  opener=opener, clock=clock_a)
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

    # write-once semantics (codex 2015Z item 3): an IDENTICAL
    # recapture (same clock, same bytes) reuses the carriers...
    rp2, tp2, rec2, tr2 = capture_day(spec(), staging, records,
                                      transcripts, builder,
                                      opener=opener, clock=clock_a)
    assert rec2 == rec and tr2 == tr
    # ...a DIVERGENT recapture (different instant -> different
    # transcript bytes) refuses instead of silently overwriting
    assert refuses(lambda: capture_day(
        spec(), staging, records, transcripts, builder,
        opener=opener, clock=clock_b), "CAPTURE_TRANSCRIPT_DIVERGENT")
    with open(tp, encoding="utf-8") as f:
        assert json.load(f) == tr         # first transcript intact
    # divergent RECORD with an identical transcript: same clock but a
    # different artifact builder -> record bytes diverge -> refuse
    assert refuses(lambda: capture_day(
        spec(), staging, records, transcripts,
        lambda b: {"n_bytes": len(b), "extra": 1},
        opener=opener, clock=clock_a), "CAPTURE_RECORD_DIVERGENT")
    with open(rp, encoding="utf-8") as f:
        assert json.load(f) == rec        # first record intact
    # path-token grammar validated BEFORE any path derivation
    # (2235Z item 5: fullmatch -- trailing newline and non-string
    # carriers refuse)
    for bad_tok in (dict(carrier="Bad Carrier"),
                    dict(carrier="../escape"),
                    dict(carrier="cascadia\n"),
                    dict(carrier=123),
                    dict(lane="NOT_A_LANE"),
                    dict(utc_day="20260820"),
                    dict(utc_day="2026-08-20\n")):
        sp = spec()
        sp.update(bad_tok)
        sp["operation_params"] = {"carrier": sp["carrier"],
                                  "day": sp["utc_day"]}
        sp["expected_keys"] = [sp["utc_day"]]
        assert refuses(lambda sp=sp: capture_day(
            sp, staging, records, transcripts, builder,
            opener=opener, clock=clock_a),
            "CAPTURE_PATH_TOKEN_INVALID"), bad_tok
    # staging immutability: a corrupted staged file refuses
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

    # --- capture_authorized (codex 0349Z item 1): the production
    # path's only input is the pinned authority identity; every
    # refusal fires with ZERO network calls (opener counter) ---
    counted = {"n": 0}

    def copener(url):
        counted["n"] += 1
        return opener(url)
    kat_keys = {"SELECTION_RECORDS": {"cascadia": ["2026-08-20"]}}
    kat_auth = {"schema": "f2g-w2-expected-contracts-v2",
                "prestart_expected_keys": kat_keys,
                "prestart_expected_keys_sha256": hashlib.sha256(
                    json.dumps(kat_keys, sort_keys=True,
                               separators=(",", ":")).encode()
                ).hexdigest(),
                "static_layer": {"SELECTION_RECORDS": {"carriers": {
                    "cascadia": {
                        "static_contract_template": {
                            "source": {"kind": "fdsn-availability",
                                       "ref": ("https://kat.example/"
                                               "fdsn")},
                            "endpoint": "https://kat.example/fdsn",
                            "request_params": {"net": "UW",
                                               "cha": "HHZ"},
                            "operation_params": {
                                "carrier": "cascadia",
                                "day": "{day}"}},
                        "cutoff": "2026-08-25"}}}},
                "dynamic_layer": {}, "digests": {},
                "provenance": {"generator": "kat"}}
    auth_raw = json.dumps(kat_auth, indent=1,
                          sort_keys=True).encode()
    a_ref = {"commit": "kat-auth",
             "path": ("docs/f2g_window2_execution/"
                      "staged_expected_contracts_v3.json"),
             "blob_sha256": hashlib.sha256(auth_raw).hexdigest()}

    def a_resolve(c):
        if c == "kat-auth":
            return "b" * 40
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_INVALID: {c!r} does not resolve")

    def a_reader(commit, path):
        if (commit, path) == ("b" * 40, a_ref["path"]):
            return auth_raw
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_INVALID: {path} unreadable at "
            f"{commit}")
    rp3, tp3, rec3, tr3 = capture_authorized(
        ".", a_ref, "SELECTION_RECORDS", "cascadia", "2026-08-20",
        staging, records, transcripts, builder, opener=copener,
        clock=clock_a, blob_reader=a_reader, git_resolve=a_resolve)
    assert counted["n"] == 1
    assert rec3["receipt"]["authority"] == {
        "commit": "b" * 40, "path": a_ref["path"],
        "blob_sha256": a_ref["blob_sha256"],
        "keys_sha256": kat_auth["prestart_expected_keys_sha256"]}
    assert tr3["authority"] == rec3["receipt"]["authority"]
    # ZERO-NETWORK refusal battery
    base_n = counted["n"]

    def a_call(ref=a_ref, lane="SELECTION_RECORDS", ck="cascadia",
               day="2026-08-20", reader=a_reader,
               resolve=a_resolve):
        return capture_authorized(
            ".", ref, lane, ck, day, staging, records, transcripts,
            builder, opener=copener, clock=clock_a,
            blob_reader=reader, git_resolve=resolve)
    # post-dated / unauthorized key
    assert refuses(lambda: a_call(day="2026-09-30"),
                   "CAPTURE_AUTHORITY_INVALID")
    # wrong lineage
    assert refuses(lambda: a_call(ref=dict(a_ref, commit="other")),
                   "CAPTURE_AUTHORITY_INVALID")
    # divergent authority bytes
    assert refuses(lambda: a_call(ref=dict(a_ref,
                                           blob_sha256="0" * 64)),
                   "CAPTURE_AUTHORITY_INVALID")
    # OPEN token in the consumed template (the unreviewed class)
    open_auth = json.loads(auth_raw.decode())
    open_auth["static_layer"]["SELECTION_RECORDS"]["carriers"][
        "cascadia"]["static_contract_template"]["endpoint"] = \
        "OPEN_REVIEW_ROUND"
    open_raw = json.dumps(open_auth, indent=1,
                          sort_keys=True).encode()
    open_ref = {"commit": "kat-auth", "path": a_ref["path"],
                "blob_sha256": hashlib.sha256(open_raw).hexdigest()}
    assert refuses(lambda: a_call(
        ref=open_ref,
        reader=lambda c, p: open_raw),
        "CAPTURE_AUTHORITY_INVALID")
    # forged authority key digest
    forged_auth = json.loads(auth_raw.decode())
    forged_auth["prestart_expected_keys_sha256"] = "attested"
    f_raw = json.dumps(forged_auth, indent=1,
                       sort_keys=True).encode()
    f_ref = {"commit": "kat-auth", "path": a_ref["path"],
             "blob_sha256": hashlib.sha256(f_raw).hexdigest()}
    assert refuses(lambda: a_call(ref=f_ref,
                                  reader=lambda c, p: f_raw),
                   "CAPTURE_AUTHORITY_INVALID")
    # unclosed ref
    assert refuses(lambda: a_call(ref={"commit": "kat-auth"}),
                   "CAPTURE_AUTHORITY_INVALID")
    assert counted["n"] == base_n     # zero network on every refusal

    # --- the body-store inventory (codex 1843Z item 4 + 2015Z item
    # 1: the NAMED-store binding -- no caller path argument exists) ---
    def desc(**over):
        d = {"schema": STORE_DESCRIPTOR_SCHEMA,
             "store_id": "s4t-kat", "store_root": "kat://store",
             "physical_root": staging}
        d.update(over)
        return d
    inv = build_staged_body_inventory(
        "s4t-kat", "kat://store",
        {"DAY_CAPSULE/cascadia/2026-08-20":
         {"sha256": rec["raw_body_sha256"],
          "bytes": rec["raw_body_bytes"]}})
    assert verify_staged_body_inventory(inv, desc()) == \
        {"objects_verified": 1}
    # the codex 2015Z repro: correct bytes in the WRONG store refuse
    # (the registered descriptor names another store identity)
    assert refuses(
        lambda: verify_staged_body_inventory(
            inv, desc(store_id="OFFICIAL_S4T",
                      store_root="s4t://official/window2")),
        "CAPTURE_STORE_IDENTITY_MISMATCH")
    assert refuses(
        lambda: verify_staged_body_inventory(inv, {"free": "dict"}),
        "CAPTURE_STORE_DESCRIPTOR_NOT_CLOSED")
    # unavailable store / missing object / mismatch / extra / escape
    assert refuses(
        lambda: verify_staged_body_inventory(
            inv, desc(physical_root=os.path.join(root, "nope"))),
        "CAPTURE_STORE_UNAVAILABLE")
    inv_missing = json.loads(json.dumps(inv))
    inv_missing["objects"]["x/y/z"] = {"path": "ab" * 32 + ".body",
                                       "sha256": "ab" * 32,
                                       "bytes": 1}
    assert refuses(
        lambda: verify_staged_body_inventory(inv_missing, desc()),
        "CAPTURE_INVENTORY_OBJECT_MISSING")
    with open(staged, "wb") as f:
        f.write(b"tampered")
    assert refuses(
        lambda: verify_staged_body_inventory(inv, desc()),
        "CAPTURE_INVENTORY_OBJECT_MISMATCH")
    with open(staged, "wb") as f:
        f.write(body)
    stray = os.path.join(staging, "ff" * 32 + ".body")
    with open(stray, "wb") as f:
        f.write(b"stray")
    assert refuses(
        lambda: verify_staged_body_inventory(inv, desc()),
        "CAPTURE_INVENTORY_EXTRA_OBJECTS")
    os.remove(stray)
    inv_esc = json.loads(json.dumps(inv))
    k = next(iter(inv_esc["objects"]))
    inv_esc["objects"][k]["path"] = "..\\" + \
        inv_esc["objects"][k]["path"]
    assert refuses(
        lambda: verify_staged_body_inventory(inv_esc, desc()),
        "CAPTURE_INVENTORY_PATH_ESCAPE")

    # --- the REAL two-process divergent race (2235Z item 4): exactly
    # one publication wins, the loser refuses typed, the winner's
    # bytes remain intact ---
    import multiprocessing as _mp
    ctx = _mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    q = ctx.Queue()
    race_path = os.path.join(root, "race.json")
    pa = ctx.Process(target=_race_worker, args=(
        race_path, {"marker": "a"}, "CAPTURE_RECORD_DIVERGENT",
        barrier, q))
    pb = ctx.Process(target=_race_worker, args=(
        race_path, {"marker": "b"}, "CAPTURE_RECORD_DIVERGENT",
        barrier, q))
    pa.start()
    pb.start()
    res = [q.get(timeout=60), q.get(timeout=60)]
    pa.join(30)
    pb.join(30)
    kinds = sorted(k for k, _ in res)
    assert kinds == ["ok", "refused"], res
    winner = next(v for k, v in res if k == "ok")
    with open(race_path, encoding="utf-8") as f:
        final = json.load(f)
    assert final == winner and final["marker"] in ("a", "b")
    assert next(v for k, v in res if k == "refused").startswith(
        "CAPTURE_RECORD_DIVERGENT")

    # --- the codex 0130Z TWO-THREAD divergent race (same PID): the
    # per-call temp + post-link verification give exactly one winner,
    # a typed loser, intact winner bytes ---
    import threading
    t_path = os.path.join(root, "race_threads.json")
    t_barrier = threading.Barrier(2)
    t_res = []
    t_lock = threading.Lock()

    def t_worker(marker):
        t_barrier.wait()
        try:
            got = _write_once_json(t_path, {"marker": marker},
                                   "CAPTURE_RECORD_DIVERGENT")
            with t_lock:
                t_res.append(("ok", got))
        except CaptureRefusal as e:
            with t_lock:
                t_res.append(("refused", str(e)))
    th_a = threading.Thread(target=t_worker, args=("a",))
    th_b = threading.Thread(target=t_worker, args=("b",))
    th_a.start()
    th_b.start()
    th_a.join(30)
    th_b.join(30)
    t_kinds = sorted(k for k, _ in t_res)
    assert t_kinds == ["ok", "refused"], t_res
    t_winner = next(v for k, v in t_res if k == "ok")
    with open(t_path, encoding="utf-8") as f:
        assert json.load(f) == t_winner
    assert next(v for k, v in t_res if k == "refused").startswith(
        "CAPTURE_RECORD_DIVERGENT")

    print("w2_acquisition_capture selftest: ALL PASS (no network)")


if __name__ == "__main__":
    _selftest()
