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

REV 7 (codex freeze-review findings 1+3): (1) capture_authorized
locates the authority pin ONLY in the AUTHORITY_SLOT ('accrual_impl')
slot of the execution manifest -- a same-path pin in any other slot is
a wrong-slot pin and refuses UNADMITTED before network; (3) the
REGISTERED production lane-transform dispatcher `admission_transform`
lands here: selection presence (closed 17-column FDSN rows, registered
station set, THE exact epoch-overlap predicate), MAG minute series
(USGS ws / INTERMAGNET GIN, provider-null refusal), MF4 (GFZ Kp / the
OMNIWeb CGI listing with per-carrier registered fill sentinels). The
production capture path builds its artifact through this dispatcher;
a caller-supplied builder can cross-check but never substitute.

REV 8 (codex end-to-end findings 1-3): (1) cascadia routes through
w2_cascadia.registry_for_day ITSELF -- canonical NET.STA identity at
the required day-START instant, frozen location precedence, and
same-location overlap refusal; no second semantic implementation.
(2) MAG/MF4 scientific gates: canonical UTC instants (fullmatch), the
registered cadences as EXACT unique ordered grids (minute 1440/1441;
OMNIWeb full 1440; GFZ 8x3h at 00..21), finite-real values only
(booleans/strings/NaN/Inf refuse; None is the registered missingness
state, counted per channel), the registered GFZ status vocabulary
(unregistered tokens refuse; 'prov' registered from the provider's
definitive/provisional dichotomy, only 'def' probe-evidenced and only
'def' counts definitive). (3) capture_authorized requires the
authority slot to be a closed BOUND mapping BEFORE any pin read -- a
pin sitting in an OPEN slot never authorizes capture.
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
EXEC_MANIFEST_PATH = ("docs/f2g_window2_execution/"
                      "execution_manifest.json")
# codex freeze-review finding 1: the ONE manifest slot whose pins may
# carry the capture authority. producer_boundary is OPEN until the
# final boundary bind (an OPEN slot cannot carry pins), so admission
# reads the authority from the already-BOUND accrual_impl slot ONLY --
# a same-path pin in any other slot is a wrong-slot pin and refuses.
AUTHORITY_SLOT = "accrual_impl"


def capture_authorized(repo, manifest_commit, authority_path, lane,
                       carrier, utc_day, staging_dir, records_dir,
                       transcripts_dir, artifact_builder, *,
                       opener=None, clock=None, blob_reader=None,
                       git_resolve=None, authority_reproducer=None):
    """THE production capture entrypoint (codex 0349Z item 1 + 1328Z
    item 3): its ONLY production inputs are the REVIEWED manifest
    commit, the registered authority path, and the (lane, carrier,
    day) key. BEFORE any network call it: resolves the manifest
    commit to a full 40-hex lineage; reopens the EXECUTION MANIFEST
    at that commit and locates the authority pin INTERNALLY (a merely
    committed authority -- codex's fresh-repo evil-endpoint probe --
    has no pin and refuses here); reopens the pinned authority bytes
    and verifies them against the PIN's blob digest; runs the FULL
    closed-schema / digest / census / pinned-reproducer authority
    validation; requires the key REGISTERED; derives S through
    authoritative_static_contract (an OPEN token -- the unreviewed
    class -- refuses there); and constructs the request SOLELY from
    S. Every refusal happens with ZERO network calls. The authority
    identity binds into T and E."""
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
    mc_full = git_resolve(manifest_commit)
    if not (isinstance(mc_full, str) and len(mc_full) == 40
            and all(c in "0123456789abcdef" for c in mc_full)):
        raise CaptureRefusal(
            "CAPTURE_AUTHORITY_INVALID: manifest lineage is not "
            "40-hex")
    manifest = json.loads(blob_reader(
        mc_full, EXEC_MANIFEST_PATH).decode("utf-8"))
    pin = None
    slot = manifest.get("slots", {}).get(AUTHORITY_SLOT)
    # codex end-to-end finding 3: the authority slot must be a CLOSED
    # BOUND mapping BEFORE any pin is read -- a pin sitting in an
    # OPEN (or malformed) slot never authorizes capture
    if not isinstance(slot, dict) or \
            slot.get("status") != "BOUND" or \
            not isinstance(slot.get("pins"), list):
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_UNADMITTED: the {AUTHORITY_SLOT!r} "
            f"slot of the execution manifest at {mc_full[:12]} is "
            "not a BOUND closed mapping -- a pin in an OPEN slot "
            "never authorizes capture")
    for p in slot["pins"]:
        if isinstance(p, dict) and \
                p.get("path") == str(authority_path):
            pin = p
    if pin is None:
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_UNADMITTED: {authority_path} is not "
            f"a pin of the {AUTHORITY_SLOT!r} slot of the execution "
            f"manifest at {mc_full[:12]} -- a committed or wrong-"
            "slot authority is never an ADMITTED authority")
    pin_commit = git_resolve(pin["commit"])
    raw = blob_reader(pin_commit, str(authority_path))
    got = hashlib.sha256(raw).hexdigest()
    if got != pin.get("blob_sha256"):
        raise CaptureRefusal(
            "CAPTURE_AUTHORITY_INVALID: authority bytes diverge "
            f"from the MANIFEST pin ({got[:12]} != "
            f"{str(pin.get('blob_sha256'))[:12]})")
    authority = json.loads(raw.decode("utf-8"))
    # the FULL closed-authority validation (schema / recomputed key
    # digest / census / pinned-reproducer reproduction) -- codex
    # 1328Z item 3: never a subset
    import w2_accrual_instrument_cayley as ACC
    try:
        ACC._validate_expected_keys_authority(
            repo, authority, reproducer=authority_reproducer)
    except Exception as exc:
        raise CaptureRefusal(f"CAPTURE_AUTHORITY_INVALID: {exc}")
    return _capture_from_validated_authority(
        repo, authority,
        {"commit": pin_commit, "path": str(authority_path),
         "blob_sha256": got,
         "keys_sha256": authority["prestart_expected_keys_sha256"]},
        lane, carrier, utc_day, staging_dir, records_dir,
        transcripts_dir, artifact_builder, opener=opener,
        clock=clock)


def _capture_from_validated_authority(repo, authority, auth_id,
                                      lane, carrier, utc_day,
                                      staging_dir, records_dir,
                                      transcripts_dir,
                                      artifact_builder, *,
                                      opener=None, clock=None):
    keys = authority.get("prestart_expected_keys", {})
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
    spec = {"lane": s["lane"], "carrier": s["carrier"],
            "utc_day": s["utc_day"], "endpoint": s["endpoint"],
            "request_params": dict(s["request_params"]),
            "source": dict(s["source"]), "cutoff": s["cutoff"],
            "operation_params": dict(s["operation_params"]),
            "expected_keys": list(s["expected_keys"])}

    # codex freeze-review finding 3: the PRODUCTION artifact is built
    # by the registered admission transform -- the same callable the
    # boundary recomputes through. A caller-supplied builder can
    # cross-check but never substitute: any divergence refuses.
    def _registered_builder(body, _s=spec, _caller=artifact_builder):
        art = admission_transform(_s["lane"], body, _s)
        if _caller is not None:
            theirs = _caller(body)
            if PROD._canon_digest(theirs) != PROD._canon_digest(art):
                raise CaptureRefusal(
                    "CAPTURE_ARTIFACT_DIVERGENT: the caller-supplied "
                    "builder product diverges from the registered "
                    "admission transform -- the registered transform "
                    "is the only production artifact authority")
        return art
    return capture_day(spec, staging_dir, records_dir,
                       transcripts_dir, _registered_builder,
                       opener=opener, clock=clock,
                       authority_id=auth_id)


def capture_with_authority_ref_fixture(
        repo, authority_ref, lane, carrier, utc_day, staging_dir,
        records_dir, transcripts_dir, artifact_builder, *,
        opener=None, clock=None, blob_reader=None,
        git_resolve=None):
    """EXPLICITLY-NAMED FIXTURE HELPER (codex 1328Z item 3): the raw
    authority-ref path -- accepts a caller {commit, path,
    blob_sha256} WITHOUT manifest admission or the full authority
    validation. Never the production entry; retained only so KATs can
    stage partial-validity fixtures."""
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
# ---- the registered production lane-transform dispatcher (codex
# freeze-review finding 3): admission artifacts are COMPUTED by
# committed code from the exact staged bytes + the AUTHORITATIVE S --
# never by a caller callback, never digest-only. verify_staged_boundary
# recomputes every admitted artifact through this same callable. ----

ARTIFACT_SCHEMA = "f2g-w2-admission-artifact-v1"
_FDSN_COLS = 17
# per-carrier registered fill sentinels (probe-evidenced: parser notes
# v1 pinned 9999.99/99999.9/999.99 for omni vars 17/21/25; SYM/H
# minute fill is 99999)
_OMNIWEB_FILL = {"omni": ("9999.99", "99999.9", "999.99"),
                 "sym_h": ("99999",)}


def _xerr(msg):
    raise CaptureRefusal("ADMISSION_TRANSFORM_REFUSED: " + str(msg))


def _xf_text(raw_body):
    if not isinstance(raw_body, (bytes, bytearray)) or not raw_body:
        _xerr("raw body must be non-empty bytes")
    try:
        return bytes(raw_body).decode("utf-8")
    except UnicodeDecodeError:
        _xerr("body is not valid UTF-8")


def _xf_day_window(day):
    from datetime import datetime, timedelta
    try:
        d0 = datetime.fromisoformat(str(day) + "T00:00:00")
    except (TypeError, ValueError):
        _xerr(f"contract day {day!r} is not canonical YYYY-MM-DD")
    d1 = d0 + timedelta(days=1)
    return d0, d1, d1.strftime("%Y-%m-%d")


def _xf_instant(raw, what):
    """Provider epoch instants are UTC; a trailing Z is normalized so
    naive/aware comparison can never raise instead of refusing."""
    from datetime import datetime
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        _xerr(f"unparseable {what} instant {raw!r}")


def _xf_selection(raw_body, s):
    """SELECTION_RECORDS: closed 17-column FDSN station-text rows ->
    per-day station presence. Registered-list carriers (socal via
    operation_params.registered_station_filter; istanbul/turkey via
    the request 'sta' list) retain STA identity and the registered
    OVERLAP predicate ([start, end) intersects [day, day_next); an
    absent EndTime is open-ended) -- the 1647Z parser contract.
    CASCADIA (the bbox multi-network carrier) uses the FROZEN carrier
    identity and time frame (codex end-to-end finding 1): rows parse
    into the frozen epoch capsule and run through
    w2_cascadia.registry_for_day ITSELF -- canonical NET.STA
    identity, activity at the required day-START instant, blank ->
    '00' -> lexicographic location precedence, same-location overlap
    refusal. No second semantic implementation exists. Outside-
    station rows are retained in the raw evidence and can never enter
    the presence set. Malformed rows, unregistered networks/channels,
    and duplicate epochs refuse."""
    text = _xf_text(raw_body)
    day0, day1, _ = _xf_day_window(s["utc_day"])
    rp = dict(s.get("request_params") or {})
    nets = {n.strip() for n in str(rp.get("net", "")).split(",")}
    chans = {c.strip() for c in str(rp.get("cha", "")).split(",")}
    op = dict(s.get("operation_params") or {})
    if "registered_station_filter" in op:
        registered = [x.strip() for x in
                      str(op["registered_station_filter"]).split(",")]
    elif "sta" in rp:
        registered = [x.strip() for x in str(rp["sta"]).split(",")]
    else:
        registered = None
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#"):
        _xerr("selection body lacks the FDSN text header")
    rows = [ln for ln in lines[1:]
            if ln.strip() and not ln.startswith("#")]
    if not rows:
        _xerr("selection body carries zero data rows")
    seen_epochs = set()
    parsed = []
    for ln in rows:
        f = ln.split("|")
        if len(f) != _FDSN_COLS:
            _xerr(f"malformed FDSN row ({len(f)} columns != "
                  f"{_FDSN_COLS}): {ln[:60]!r}")
        net, sta, loc, cha = (f[0].strip(), f[1].strip(),
                              f[2].strip(), f[3].strip())
        start_s, end_s = f[15].strip(), f[16].strip()
        if net not in nets:
            _xerr(f"row network {net!r} is not a registered request "
                  "network")
        if cha not in chans:
            _xerr(f"row channel {cha!r} is not a registered request "
                  "channel")
        ep_start = _xf_instant(start_s, "epoch StartTime")
        ep_end = _xf_instant(end_s, "epoch EndTime") if end_s \
            else None
        ekey = (net, sta, loc, cha, start_s)
        if ekey in seen_epochs:
            _xerr(f"duplicate channel epoch {net}.{sta}.{loc}.{cha} "
                  f"@ {start_s}")
        seen_epochs.add(ekey)
        parsed.append({"network": net, "station": sta,
                       "location": loc, "channel": cha,
                       "lat_raw": f[4].strip(),
                       "lon_raw": f[5].strip(),
                       "epoch_start": ep_start,
                       "epoch_end": ep_end})
    if s.get("carrier") == "cascadia":
        if registered is not None:
            _xerr("cascadia is the bbox carrier; a registered "
                  "station list is not part of its frozen identity")
        import w2_cascadia as CASC
        epochs = []
        for r in parsed:
            try:
                lat = float(r["lat_raw"])
                lon = float(r["lon_raw"])
            except ValueError:
                _xerr("malformed FDSN coordinate for "
                      f"{r['network']}.{r['station']}")
            epochs.append({"network": r["network"],
                           "station": r["station"],
                           "location": r["location"],
                           "channel": r["channel"],
                           "latitude": lat, "longitude": lon,
                           "epoch_start": r["epoch_start"],
                           "epoch_end": r["epoch_end"]})
        try:
            reg_rows = CASC.registry_for_day(s["utc_day"],
                                             epochs=epochs)
        except (CASC.EpochOverlapError,
                CASC.RegistryInputInvalid) as exc:
            _xerr(f"frozen cascadia registry refusal: {exc}")
        present = sorted(r["id"] for r in reg_rows)
        return {"schema": ARTIFACT_SCHEMA, "lane": s["lane"],
                "carrier": s["carrier"], "utc_day": s["utc_day"],
                "kind": "fdsn-station-presence",
                "identity": "net.sta-registry-day-start",
                "registered_stations": None,
                "present_stations": present,
                "absent_stations": [],
                "data_rows": len(rows),
                "outside_station_rows_excluded": 0}
    outside = 0
    present = set()
    for r in parsed:
        if registered is not None and r["station"] not in registered:
            outside += 1
            continue
        if r["epoch_start"] < day1 and (r["epoch_end"] is None
                                        or r["epoch_end"] > day0):
            present.add(r["station"])
    return {"schema": ARTIFACT_SCHEMA, "lane": s["lane"],
            "carrier": s["carrier"], "utc_day": s["utc_day"],
            "kind": "fdsn-station-presence",
            "identity": "sta-overlap",
            "registered_stations": (sorted(registered)
                                    if registered is not None
                                    else None),
            "present_stations": sorted(present),
            "absent_stations": (sorted(set(registered) - present)
                                if registered is not None else []),
            "data_rows": len(rows),
            "outside_station_rows_excluded": outside}


# the registered GFZ Kp status vocabulary + definitive policy: only
# 'def' counts definitive; 'prov' is the provider's provisional state
# (registered from the GFZ definitive/provisional dichotomy; only
# 'def' is probe-evidenced -- an unregistered token REFUSES rather
# than admitting)
KP_STATUS_VOCAB = ("def", "prov")
_CANON_TS = None


def _xf_grid_instant(ts_raw, what):
    """Canonical UTC instant grammar (fullmatch) -> parsed naive
    instant. Fractional seconds 1-6 digits allowed; anything else
    refuses."""
    global _CANON_TS
    if _CANON_TS is None:
        import re
        _CANON_TS = re.compile(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z")
    ts = str(ts_raw)
    if not isinstance(ts_raw, str) or not _CANON_TS.fullmatch(ts):
        _xerr(f"{what}: non-canonical UTC instant {ts_raw!r}")
    from datetime import datetime
    return datetime.fromisoformat(ts[:-1])


def _xf_time_grid(instants, day, step_minutes, counts, what):
    """The registered cadence gate: the instant list must BE the
    exact unique ordered grid day0 + i*step for i in 0..N-1 with N in
    `counts` -- duplicates, gaps, out-of-order, off-grid fractionals,
    and wrong-day instants all refuse as cadence violations."""
    from datetime import timedelta
    day0, _, _ = _xf_day_window(day)
    if not isinstance(instants, list) or not instants:
        _xerr(f"{what}: zero samples")
    if len(instants) not in counts:
        _xerr(f"{what}: {len(instants)} samples violates the "
              f"registered cadence (allowed {sorted(counts)})")
    for i, t in enumerate(instants):
        got = _xf_grid_instant(t, what)
        want = day0 + timedelta(minutes=step_minutes * i)
        if got != want:
            _xerr(f"{what}: instant {t!r} at index {i} violates the "
                  f"registered cadence (expected "
                  f"{want.isoformat()}Z)")


def _xf_numeric_or_null(v, what):
    """A scientific sample is None (the registered missingness state)
    or a FINITE real -- booleans, strings, NaN, and Inf refuse."""
    import math
    if v is None:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)) \
            or not math.isfinite(v):
        _xerr(f"{what}: nonnumeric or nonfinite sample {v!r}")


def _xf_mag(raw_body, s):
    """MAG_FEED minute series (USGS ws JSON / INTERMAGNET GIN JSON):
    structural validation + the registered minute cadence (exact
    unique ordered grid; 1440 samples, or 1441 with the inclusive
    day-next terminal) + finite-real values (None = the registered
    missingness state, counted per channel) + provider-null refusal.
    A series with ZERO definitive samples never admits."""
    kind = (s.get("source") or {}).get("kind")
    if kind not in ("usgs-geomag-ws-minute",
                    "intermagnet-gin-minute"):
        _xerr(f"unregistered MAG source kind {kind!r}")
    doc_text = _xf_text(raw_body)
    try:
        doc = json.loads(doc_text)
    except ValueError:
        _xerr("MAG body is not valid JSON")
    day = s["utc_day"]
    rp = dict(s.get("request_params") or {})

    def _series(times, comps, observatory, what):
        _xf_time_grid(times, day, 1, (1440, 1441), what)
        nulls = {}
        definitive = [False] * len(times)
        for cid in sorted(comps):
            vals = comps[cid]
            if not isinstance(vals, list) or \
                    len(vals) != len(times):
                _xerr(f"channel {cid} length diverges from the "
                      f"time grid ({what})")
            for v in vals:
                _xf_numeric_or_null(v, f"{what} channel {cid}")
            nulls[cid] = sum(1 for v in vals if v is None)
            for i, v in enumerate(vals):
                if v is not None:
                    definitive[i] = True
        if not nulls:
            _xerr(f"{what}: zero channels")
        n_def = sum(definitive)
        if n_def == 0:
            _xerr("provider-null MAG series (zero definitive "
                  "samples)")
        return {"schema": ARTIFACT_SCHEMA, "lane": s["lane"],
                "carrier": s["carrier"], "utc_day": day,
                "kind": "mag-minute-series",
                "observatory": observatory,
                "samples": len(times), "channels": sorted(nulls),
                "null_by_channel": {k: nulls[k] for k in
                                    sorted(nulls)},
                "definitive_samples": n_def}
    if kind == "usgs-geomag-ws-minute":
        if not isinstance(doc, dict) or \
                not isinstance(doc.get("values"), list):
            _xerr("USGS MAG body lacks the values channel list")
        iaga = ((((doc.get("metadata") or {}).get("intermagnet")
                  or {}).get("imo") or {}).get("iaga_code"))
        if iaga != rp.get("id"):
            _xerr(f"observatory {iaga!r} diverges from the "
                  f"registered id {rp.get('id')!r}")
        comps = {}
        for ch in doc["values"]:
            if not isinstance(ch, dict) or \
                    not isinstance(ch.get("values"), list):
                _xerr("USGS MAG channel entry is not closed")
            comps[str(ch.get("id"))] = ch["values"]
        return _series(doc.get("times"), comps, iaga, "USGS MAG")
    if not isinstance(doc, dict):
        _xerr("GIN MAG body is not a JSON object")
    comps = {k: v for k, v in doc.items()
             if k != "datetime" and isinstance(v, list)}
    if not comps:
        _xerr("GIN MAG body carries zero component arrays")
    return _series(doc.get("datetime"), comps,
                   rp.get("observatoryIagaCode"), "GIN MAG")


def _xf_mf4(raw_body, s):
    """MF4_FEED: GFZ Kp JSON (eight three-hour intervals, definitive
    counted by status) or OMNIWeb high-res CGI listings (minute rows
    bound to the registered day by YYYY+DOY; per-carrier registered
    fill sentinels counted; an all-sentinel listing refuses)."""
    kind = (s.get("source") or {}).get("kind")
    day = s["utc_day"]
    if kind == "gfz-kp-json":
        import math
        try:
            doc = json.loads(_xf_text(raw_body))
        except ValueError:
            _xerr("Kp body is not valid JSON")
        kp, dts, st = (doc.get("Kp"), doc.get("datetime"),
                       doc.get("status"))
        if not (isinstance(kp, list) and isinstance(dts, list)
                and isinstance(st, list)):
            _xerr("Kp body lacks the Kp/datetime/status arrays")
        if not (len(kp) == len(dts) == len(st)) or not kp:
            _xerr("Kp arrays are empty or length-divergent")
        # the registered GFZ cadence: EXACTLY eight three-hour
        # intervals at 00..21 on the registered day, unique ordered
        _xf_time_grid(dts, day, 180, (8,), "Kp intervals")
        for v in kp:
            if isinstance(v, bool) or \
                    not isinstance(v, (int, float)) or \
                    not math.isfinite(v) or not (0.0 <= v <= 9.0):
                _xerr(f"Kp value {v!r} is not a finite index value "
                      "in [0, 9]")
        for x in st:
            if x not in KP_STATUS_VOCAB:
                _xerr(f"Kp status {x!r} is not in the registered "
                      f"GFZ vocabulary {KP_STATUS_VOCAB}")
        n_def = sum(1 for x in st if x == "def")
        return {"schema": ARTIFACT_SCHEMA, "lane": s["lane"],
                "carrier": s["carrier"], "utc_day": day,
                "kind": "kp-intervals", "intervals": len(kp),
                "status_counts": {v: st.count(v)
                                  for v in KP_STATUS_VOCAB
                                  if v in st},
                "definitive_intervals": n_def}
    if kind == "omniweb-highres-cgi":
        from datetime import datetime
        carrier = s["carrier"]
        if carrier not in _OMNIWEB_FILL:
            _xerr(f"no registered fill-sentinel set for OMNIWeb "
                  f"carrier {carrier!r}")
        fills = set(_OMNIWEB_FILL[carrier])
        rp = dict(s.get("request_params") or {})
        v = rp.get("vars")
        n_vars = len(v) if isinstance(v, (list, tuple)) else 1
        want_doy = datetime.fromisoformat(
            day + "T00:00:00").timetuple().tm_yday
        want_year = int(day[:4])
        text = _xf_text(raw_body)
        rows = []
        in_data = False
        for ln in text.splitlines():
            t = ln.strip()
            if t.startswith("YYYY DOY HR MN"):
                in_data = True
                continue
            if not in_data:
                continue
            if not t or t.startswith("<"):
                in_data = False
                continue
            tok = t.split()
            if len(tok) != 4 + n_vars:
                _xerr(f"malformed OMNIWeb data row ({len(tok)} "
                      f"tokens != {4 + n_vars}): {t[:60]!r}")
            try:
                yy, doy, hh, mn = (int(tok[0]), int(tok[1]),
                                   int(tok[2]), int(tok[3]))
            except ValueError:
                _xerr(f"malformed OMNIWeb time tokens: {t[:60]!r}")
            if yy != want_year or doy != want_doy:
                _xerr(f"OMNIWeb row {yy}/{doy:03d} is outside the "
                      f"registered day {day} (DOY {want_doy:03d})")
            # the registered minute cadence: row i must sit at
            # EXACTLY minute i -- duplicates, gaps, and out-of-order
            # rows all violate the grid
            i = len(rows)
            if (hh, mn) != (i // 60, i % 60):
                _xerr(f"OMNIWeb row {hh:02d}:{mn:02d} at index {i} "
                      "violates the registered minute cadence "
                      f"(expected {i // 60:02d}:{i % 60:02d})")
            rows.append(tok[4:])
        if len(rows) != 1440:
            _xerr(f"OMNIWeb listing carries {len(rows)} rows -- the "
                  "registered cadence is the full 1440-minute grid")
        import math
        fill_by_col = [0] * n_vars
        n_def = 0
        for vals in rows:
            all_fill = True
            for i, x in enumerate(vals):
                if x in fills:
                    fill_by_col[i] += 1
                    continue
                try:
                    fv = float(x)
                except ValueError:
                    _xerr("nonnumeric OMNIWeb value token "
                          f"{x!r} (not a registered fill sentinel)")
                if not math.isfinite(fv):
                    _xerr(f"nonfinite OMNIWeb value token {x!r}")
                all_fill = False
            if not all_fill:
                n_def += 1
        if n_def == 0:
            _xerr("all OMNIWeb samples are provider fill sentinels "
                  "(zero definitive samples)")
        return {"schema": ARTIFACT_SCHEMA, "lane": s["lane"],
                "carrier": s["carrier"], "utc_day": day,
                "kind": "omniweb-minute-listing", "samples":
                    len(rows), "value_columns": n_vars,
                "fill_by_column": list(fill_by_col),
                "definitive_samples": n_def}
    _xerr(f"unregistered MF4 source kind {kind!r}")


def admission_transform(lane, raw_body, static_contract):
    """THE registered production lane-transform dispatcher (codex
    freeze-review finding 3; the accrual boundary's fail-closed
    default). Routes on (lane, source.kind) from the AUTHORITATIVE
    static contract -- an unregistered pair refuses, so no body class
    the freeze never reviewed can be admitted."""
    if not isinstance(static_contract, dict):
        _xerr("static contract must be the authoritative S mapping")
    if static_contract.get("lane") != lane:
        _xerr(f"dispatch lane {lane!r} diverges from the "
              f"authoritative S lane "
              f"{static_contract.get('lane')!r}")
    kind = (static_contract.get("source") or {}).get("kind")
    if lane == "SELECTION_RECORDS":
        if kind != "fdsn-station-channel":
            _xerr(f"unregistered SELECTION source kind {kind!r}")
        return _xf_selection(raw_body, static_contract)
    if lane == "MAG_FEED":
        return _xf_mag(raw_body, static_contract)
    if lane == "MF4_FEED":
        return _xf_mf4(raw_body, static_contract)
    _xerr(f"unregistered capture lane {lane!r}")


def _selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="w2_capture_kat_")
    staging = os.path.join(root, "staging")
    records = os.path.join(root, "records")
    transcripts = os.path.join(root, "transcripts")

    FDSN_HDR = ("#Network|Station|Location|Channel|Latitude|"
                "Longitude|Elevation|Depth|Azimuth|Dip|Instrument|"
                "Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|"
                "EndTime")
    FDSN_KAT_BODY = "\n".join([
        FDSN_HDR,
        "UW|KAT1||HHZ|0|0|0|0|0|-90|kat|1|1|m/s|100|"
        "2020-01-01T00:00:00|",
        "UW|KAT2||HHZ|0|0|0|0|0|-90|kat|1|1|m/s|100|"
        "2019-01-01T00:00:00|2021-01-01T00:00:00",
    ]).encode() + b"\n"
    FIX = {"https://kat.example/fdsn?cha=HHZ&net=UW":
           (200, {"content-type": "text/plain"}, b"kat-body-1",
            "https://edge.example/fdsn-final"),
           "https://kat.example/fdsn2?cha=HHZ&net=UW":
           (200, {"content-type": "text/plain"}, FDSN_KAT_BODY,
            "https://kat.example/fdsn2?cha=HHZ&net=UW"),
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

    # --- capture_authorized (codex 0349Z item 1 + 1328Z item 3):
    # the production path takes the MANIFEST commit + the registered
    # authority path; every refusal fires with ZERO network calls ---
    counted = {"n": 0}

    def copener(url):
        counted["n"] += 1
        return opener(url)
    kat_keys = {"SELECTION_RECORDS": {"cascadia": ["2026-08-20"]},
                "MAG_FEED": {"frn": ["2026-08-20"]},
                "MF4_FEED": {"mf4drv": ["2026-08-20"]}}

    def kat_template(lane, ck):
        # registered (lane, source.kind) pairs so the PRODUCTION
        # admission transform (freeze finding 3) routes the fixture
        kinds = {"SELECTION_RECORDS": "fdsn-station-channel",
                 "MAG_FEED": "usgs-geomag-ws-minute",
                 "MF4_FEED": "gfz-kp-json"}
        return {"source": {"kind": kinds[lane],
                           "ref": "https://kat.example/fdsn2"},
                "endpoint": "https://kat.example/fdsn2",
                "request_params": {"net": "UW", "cha": "HHZ"},
                "operation_params": {"carrier": ck, "day": "{day}"}}
    kat_auth = {"schema": "f2g-w2-expected-contracts-v3",
                "template_token_vocabulary": ["{day}", "{day_next}", "{day_compact}"],
                "prestart_expected_keys": kat_keys,
                "prestart_expected_keys_sha256": hashlib.sha256(
                    json.dumps(kat_keys, sort_keys=True,
                               separators=(",", ":")).encode()
                ).hexdigest(),
                "static_layer": {
                    lane: {"carriers": {ck: {
                        "static_contract_template":
                            kat_template(lane, ck),
                        "cutoff": "2026-08-25"}}}
                    for lane, cks in kat_keys.items()
                    for ck in cks},
                "dynamic_layer": {}, "digests": {},
                "provenance": {"generator": "kat"}}
    auth_raw = json.dumps(kat_auth, indent=1,
                          sort_keys=True).encode()
    AUTH_PATH = ("docs/f2g_window2_execution/"
                 "staged_expected_contracts_v3.json")
    MAN_C = "e" * 40
    PIN_C = "b" * 40
    kat_man = {"slots": {AUTHORITY_SLOT: {
        "status": "BOUND",
        "pins": [{"path": AUTH_PATH, "commit": "kat-auth",
                  "blob_sha256": hashlib.sha256(auth_raw)
                  .hexdigest()}]}}}
    man_raw = json.dumps(kat_man).encode()

    def m_resolve(c):
        if c == "kat-man":
            return MAN_C
        if c == "kat-auth":
            return PIN_C
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_INVALID: {c!r} does not resolve")

    def m_reader(commit, path):
        if (commit, path) == (MAN_C, EXEC_MANIFEST_PATH):
            return man_raw
        if (commit, path) == (PIN_C, AUTH_PATH):
            return auth_raw
        raise CaptureRefusal(
            f"CAPTURE_AUTHORITY_INVALID: {path} unreadable at "
            f"{commit}")

    def a_repro():
        return json.loads(auth_raw.decode())
    rp3, tp3, rec3, tr3 = capture_authorized(
        ".", "kat-man", AUTH_PATH, "SELECTION_RECORDS", "cascadia",
        "2026-08-20", staging, records, transcripts, None,
        opener=copener, clock=clock_a, blob_reader=m_reader,
        git_resolve=m_resolve, authority_reproducer=a_repro)
    assert counted["n"] == 1
    assert rec3["receipt"]["authority"] == {
        "commit": PIN_C, "path": AUTH_PATH,
        "blob_sha256": hashlib.sha256(auth_raw).hexdigest(),
        "keys_sha256": kat_auth["prestart_expected_keys_sha256"]}
    assert tr3["authority"] == rec3["receipt"]["authority"]
    # freeze finding 3: the production artifact IS the registered
    # transform's product (record binds it via output_sha256) --
    # overlap-present KAT1 in, epoch-closed KAT2 out
    import w2_accrual_instrument_cayley as _ACC
    s_kat = _ACC.authoritative_static_contract(
        kat_auth, "SELECTION_RECORDS", "cascadia", "2026-08-20")
    art3 = admission_transform("SELECTION_RECORDS", FDSN_KAT_BODY,
                               s_kat)
    # fixture carrier 'cascadia' routes through the FROZEN registry:
    # canonical NET.STA identity at the day-start instant (KAT2's
    # epoch closed 2021 -> absent)
    assert art3["present_stations"] == ["UW.KAT1"]
    assert art3["identity"] == "net.sta-registry-day-start"
    assert art3["absent_stations"] == []
    assert art3["registered_stations"] is None
    assert art3["data_rows"] == 2
    assert rec3["output_sha256"] == PROD._canon_digest(art3)
    # ZERO-NETWORK refusal battery (production path)
    base_n = counted["n"]

    def a_call(mc="kat-man", path=AUTH_PATH,
               lane="SELECTION_RECORDS", ck="cascadia",
               day="2026-08-20", reader=m_reader,
               resolve=m_resolve, repro=a_repro, abuilder=None):
        return capture_authorized(
            ".", mc, path, lane, ck, day, staging, records,
            transcripts, abuilder, opener=copener, clock=clock_a,
            blob_reader=reader, git_resolve=resolve,
            authority_reproducer=repro)
    # freeze finding 1: a same-path pin in ANY OTHER slot is a
    # wrong-slot pin and refuses UNADMITTED before network
    wrongslot_man = {"slots": {"producer_boundary":
                               kat_man["slots"][AUTHORITY_SLOT]}}
    ws_raw = json.dumps(wrongslot_man).encode()
    assert refuses(lambda: a_call(
        reader=lambda c, p: ws_raw if p == EXEC_MANIFEST_PATH
        else m_reader(c, p)), "CAPTURE_AUTHORITY_UNADMITTED")
    # codex end-to-end finding 3: an OPEN slot STILL carrying the
    # reviewed pin never authorizes capture -- status is checked
    # BEFORE any pin is read, zero network
    open_slot_man = json.loads(man_raw.decode())
    open_slot_man["slots"][AUTHORITY_SLOT]["status"] = "OPEN"
    osm_raw = json.dumps(open_slot_man).encode()
    assert refuses(lambda: a_call(
        reader=lambda c, p: osm_raw if p == EXEC_MANIFEST_PATH
        else m_reader(c, p)), "CAPTURE_AUTHORITY_UNADMITTED")
    # ...and a slot with no status key at all is not a closed
    # BOUND mapping either
    nostatus_man = json.loads(man_raw.decode())
    del nostatus_man["slots"][AUTHORITY_SLOT]["status"]
    ns_raw = json.dumps(nostatus_man).encode()
    assert refuses(lambda: a_call(
        reader=lambda c, p: ns_raw if p == EXEC_MANIFEST_PATH
        else m_reader(c, p)), "CAPTURE_AUTHORITY_UNADMITTED")
    # the codex fresh-repo probe: a COMMITTED but UNPINNED authority
    # (wrong path / no manifest pin) refuses UNADMITTED
    assert refuses(lambda: a_call(path="docs/other/evil_auth.json"),
                   "CAPTURE_AUTHORITY_UNADMITTED")
    # post-dated / unauthorized key
    assert refuses(lambda: a_call(day="2026-09-30"),
                   "CAPTURE_AUTHORITY_INVALID")
    # wrong manifest lineage
    assert refuses(lambda: a_call(mc="nope"),
                   "CAPTURE_AUTHORITY_INVALID")
    # divergent authority bytes vs the MANIFEST pin
    tampered_man = json.loads(man_raw.decode())
    tampered_man["slots"][AUTHORITY_SLOT]["pins"][0][
        "blob_sha256"] = "0" * 64
    t_raw = json.dumps(tampered_man).encode()
    assert refuses(lambda: a_call(
        reader=lambda c, p: t_raw if p == EXEC_MANIFEST_PATH
        else m_reader(c, p)), "CAPTURE_AUTHORITY_INVALID")
    # malformed authority schema (full validation, never a subset)
    mal = json.loads(auth_raw.decode())
    del mal["static_layer"]
    mal_raw = json.dumps(mal).encode()
    mal_man = json.loads(man_raw.decode())
    mal_man["slots"][AUTHORITY_SLOT]["pins"][0][
        "blob_sha256"] = hashlib.sha256(mal_raw).hexdigest()
    mal_man_raw = json.dumps(mal_man).encode()

    def mal_reader(c, p):
        if p == EXEC_MANIFEST_PATH:
            return mal_man_raw
        if p == AUTH_PATH:
            return mal_raw
        return m_reader(c, p)
    assert refuses(lambda: a_call(reader=mal_reader,
                                  repro=lambda: json.loads(
                                      mal_raw.decode())),
                   "CAPTURE_AUTHORITY_INVALID")
    # non-reproducing authority
    assert refuses(lambda: a_call(
        repro=lambda: dict(a_repro(),
                           provenance={"generator": "other"})),
        "CAPTURE_AUTHORITY_INVALID")
    # wrong census (the production reproducer path)
    assert refuses(lambda: a_call(repro=None),
                   "CAPTURE_AUTHORITY_INVALID")
    # OPEN token in the consumed template (the unreviewed class)
    open_auth = json.loads(auth_raw.decode())
    open_auth["static_layer"]["SELECTION_RECORDS"]["carriers"][
        "cascadia"]["static_contract_template"]["endpoint"] = \
        "OPEN_REVIEW_ROUND"
    open_raw = json.dumps(open_auth).encode()
    open_man = json.loads(man_raw.decode())
    open_man["slots"][AUTHORITY_SLOT]["pins"][0][
        "blob_sha256"] = hashlib.sha256(open_raw).hexdigest()
    open_man_raw = json.dumps(open_man).encode()

    def open_reader(c, p):
        if p == EXEC_MANIFEST_PATH:
            return open_man_raw
        if p == AUTH_PATH:
            return open_raw
        return m_reader(c, p)
    assert refuses(lambda: a_call(
        reader=open_reader,
        repro=lambda: json.loads(open_raw.decode())),
        "CAPTURE_AUTHORITY_INVALID")
    assert counted["n"] == base_n     # zero network on every refusal
    # freeze finding 3: a caller-supplied builder whose product
    # diverges from the registered transform refuses (this one DOES
    # reach the network; the transcript reuses write-once bytes)
    assert refuses(lambda: a_call(abuilder=builder),
                   "CAPTURE_ARTIFACT_DIVERGENT")
    # ...while a caller builder that MATCHES the transform admits
    rp3b, _, rec3b, _ = a_call(
        abuilder=lambda b: admission_transform(
            "SELECTION_RECORDS", b, s_kat))
    assert rec3b == rec3
    # the explicitly-named FIXTURE helper still carries the ref path
    # (partial validity, never production)
    a_ref = {"commit": "kat-auth", "path": AUTH_PATH,
             "blob_sha256": hashlib.sha256(auth_raw).hexdigest()}
    _, _, rec4, _ = capture_with_authority_ref_fixture(
        ".", a_ref, "SELECTION_RECORDS", "cascadia", "2026-08-20",
        staging, records, transcripts,
        lambda b: admission_transform("SELECTION_RECORDS", b,
                                      s_kat),
        opener=copener, clock=clock_a,
        blob_reader=lambda c, p: auth_raw,
        git_resolve=lambda c: PIN_C)
    assert rec4 == rec3               # write-once reuse, same bytes

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
          "bytes": rec["raw_body_bytes"]},
         "SELECTION_RECORDS/cascadia/2026-08-20":
         {"sha256": rec3["raw_body_sha256"],
          "bytes": rec3["raw_body_bytes"]}})
    assert verify_staged_body_inventory(inv, desc()) == \
        {"objects_verified": 2}
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

    # --- the registered admission-transform battery (freeze finding
    # 3): per-lane positives + the malformed/wrong-net/wrong-channel/
    # duplicate-epoch/partial-day/provider-null/fill-sentinel doctors
    def xrefuses(fn, needle):
        try:
            fn()
            return False
        except CaptureRefusal as e:
            return str(e).startswith("ADMISSION_TRANSFORM_REFUSED") \
                and needle in str(e)

    def sel_s(**over):
        s = {"lane": "SELECTION_RECORDS", "carrier": "katsel",
             "utc_day": "2026-08-20",
             "endpoint": "https://kat.example/f",
             "request_params": {"net": "CI", "cha": "HHZ",
                                "sta": "AAA,BBB"},
             "source": {"kind": "fdsn-station-channel",
                        "ref": "kat://f"},
             "cutoff": "2026-08-25",
             "operation_params": {"carrier": "katsel",
                                  "day": "2026-08-20"},
             "expected_keys": ["2026-08-20"]}
        s.update(over)
        return s

    def sel_body(*rows):
        return ("\n".join((FDSN_HDR,) + rows) + "\n").encode()

    def sel_row(sta, start, end, net="CI", cha="HHZ"):
        return (f"{net}|{sta}||{cha}|0|0|0|0|0|-90|kat|1|1|m/s|100|"
                f"{start}|{end}")
    # positive: AAA present (open epoch), BBB absent (epoch closed
    # before the day), CCC outside the registered set and excluded
    art = admission_transform("SELECTION_RECORDS", sel_body(
        sel_row("AAA", "2020-01-01T00:00:00", ""),
        sel_row("BBB", "2019-01-01T00:00:00", "2021-01-01T00:00:00"),
        sel_row("CCC", "2020-01-01T00:00:00", "")), sel_s())
    assert art["present_stations"] == ["AAA"]
    assert art["absent_stations"] == ["BBB"]
    assert art["registered_stations"] == ["AAA", "BBB"]
    assert art["outside_station_rows_excluded"] == 1
    # partial-day epoch: overlap [day, day_next) counts PRESENT --
    # the epoch ends mid-day and still intersects
    art_p = admission_transform("SELECTION_RECORDS", sel_body(
        sel_row("AAA", "2020-01-01T00:00:00",
                "2026-08-20T06:00:00")), sel_s())
    assert art_p["present_stations"] == ["AAA"]
    # boundary-exact epoch: ends AT day start -> [start, end) does
    # NOT intersect -> absent
    art_b = admission_transform("SELECTION_RECORDS", sel_body(
        sel_row("AAA", "2020-01-01T00:00:00",
                "2026-08-20T00:00:00")), sel_s())
    assert art_b["present_stations"] == []
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body("CI|AAA|broken"), sel_s()),
        "malformed FDSN row")
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(
            sel_row("AAA", "2020-01-01T00:00:00", "", net="XX")),
        sel_s()), "not a registered request network")
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(
            sel_row("AAA", "2020-01-01T00:00:00", "", cha="BHZ")),
        sel_s()), "not a registered request channel")
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(
            sel_row("AAA", "2020-01-01T00:00:00", ""),
            sel_row("AAA", "2020-01-01T00:00:00", "")), sel_s()),
        "duplicate channel epoch")
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", b"no header\n", sel_s()),
        "lacks the FDSN text header")
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(), sel_s()),
        "zero data rows")
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(sel_row(
            "AAA", "2020-01-01T00:00:00", "")),
        sel_s(source={"kind": "evil", "ref": "kat://f"})),
        "unregistered SELECTION source kind")

    # --- cascadia: the FROZEN carrier identity + time frame (codex
    # end-to-end finding 1) through w2_cascadia.registry_for_day ---
    def casc_s(**over):
        s = sel_s(carrier="cascadia",
                  request_params={"net": "UW,CC,CN", "cha": "HHZ"},
                  operation_params={"carrier": "cascadia",
                                    "day": "2026-08-20"})
        s.update(over)
        return s

    def crow(net, sta, loc, start, end, cha="HHZ"):
        return (f"{net}|{sta}|{loc}|{cha}|1.0|2.0|0|0|0|-90|kat|1|1|"
                f"m/s|100|{start}|{end}")
    # NET.STA identity (cross-network collision stays distinct);
    # day-START instant (a later-that-day starter is ABSENT -- the
    # exact class behind codex's nine extra stations); simultaneous
    # blank/'00' locations resolve blank-first without refusal
    art_c = admission_transform("SELECTION_RECORDS", sel_body(
        crow("UW", "ABC", "", "2020-01-01T00:00:00", ""),
        crow("CC", "ABC", "", "2020-01-01T00:00:00", ""),
        crow("CC", "LATE", "", "2026-08-20T10:00:00", ""),
        crow("CN", "PREC", "", "2019-01-01T00:00:00", ""),
        crow("CN", "PREC", "00", "2018-01-01T00:00:00", "")),
        casc_s())
    assert art_c["present_stations"] == ["CC.ABC", "CN.PREC",
                                         "UW.ABC"]
    assert art_c["identity"] == "net.sta-registry-day-start"
    # TOUT: the blank epoch ends exactly as '00' opens at day start
    # -- the dead blank epoch is never selected
    art_t = admission_transform("SELECTION_RECORDS", sel_body(
        crow("UW", "TOUT", "", "2019-01-01T00:00:00",
             "2026-08-20T00:00:00"),
        crow("UW", "TOUT", "00", "2026-08-20T00:00:00", "")),
        casc_s())
    assert art_t["present_stations"] == ["UW.TOUT"]
    # RER: three adjacent epochs -- exactly the day-start one active
    art_r = admission_transform("SELECTION_RECORDS", sel_body(
        crow("UW", "RER", "", "2018-01-01T00:00:00",
             "2019-01-01T00:00:00"),
        crow("UW", "RER", "", "2019-01-01T00:00:00",
             "2027-01-01T00:00:00"),
        crow("UW", "RER", "", "2027-01-01T00:00:00", "")), casc_s())
    assert art_r["present_stations"] == ["UW.RER"]
    # same-location overlap refuses THROUGH the frozen registry
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(
            crow("UW", "OVL", "", "2019-01-01T00:00:00", ""),
            crow("UW", "OVL", "", "2020-01-01T00:00:00", "")),
        casc_s()), "frozen cascadia registry refusal")
    # a registered station list contradicts the frozen bbox identity
    assert xrefuses(lambda: admission_transform(
        "SELECTION_RECORDS", sel_body(
            crow("UW", "ABC", "", "2020-01-01T00:00:00", "")),
        casc_s(request_params={"net": "UW,CC,CN", "cha": "HHZ",
                               "sta": "ABC"})),
        "not part of its frozen identity")

    def mag_s(**over):
        s = {"lane": "MAG_FEED", "carrier": "katmag",
             "utc_day": "2026-08-20",
             "endpoint": "https://kat.example/m",
             "request_params": {"id": "KAT"},
             "source": {"kind": "usgs-geomag-ws-minute",
                        "ref": "kat://m"},
             "cutoff": "2026-08-25",
             "operation_params": {"carrier": "katmag",
                                  "day": "2026-08-20"},
             "expected_keys": ["2026-08-20"]}
        s.update(over)
        return s

    def grid_times(n=1440, day="2026-08-20"):
        from datetime import datetime as _dt, timedelta as _td
        d0 = _dt.fromisoformat(day + "T00:00:00")
        return [(d0 + _td(minutes=i)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z") for i in range(n)]

    def usgs_body(vals, iaga="KAT", times=None):
        times = times if times is not None else grid_times(len(vals))
        return json.dumps({
            "type": "Timeseries",
            "metadata": {"intermagnet": {"imo": {"iaga_code": iaga}}},
            "times": times,
            "values": [{"id": "X", "values": vals}]}).encode()
    full = [1.0] * 1439 + [None]
    art_m = admission_transform("MAG_FEED", usgs_body(full), mag_s())
    assert art_m["samples"] == 1440
    assert art_m["definitive_samples"] == 1439
    assert art_m["null_by_channel"] == {"X": 1}
    # the inclusive day-next terminal (the USGS 1441 shape) admits
    art_m2 = admission_transform("MAG_FEED",
                                 usgs_body([2.0] * 1441), mag_s())
    assert art_m2["samples"] == 1441
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([None] * 1440), mag_s()),
        "provider-null MAG series")
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1440, iaga="EVIL"), mag_s()),
        "diverges from the registered id")
    # codex end-to-end finding 2, exact repro 1: one timestamp +
    # values=["NOT_A_NUMBER"] must refuse (cadence gate)
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body(["NOT_A_NUMBER"]), mag_s()),
        "violates the registered cadence")
    # ...and a full-grid nonnumeric string refuses at the value gate
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1439 + ["NOT_A_NUMBER"]),
        mag_s()), "nonnumeric or nonfinite sample")
    # nonfinite (NaN survives JSON round-trip) refuses
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1439 + [float("nan")]),
        mag_s()), "nonnumeric or nonfinite sample")
    # cadence doctors: duplicate, out-of-order, missing slot,
    # wrong-day grid, non-canonical instant
    t_dup = grid_times()
    t_dup[5] = t_dup[4]
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1440, times=t_dup), mag_s()),
        "violates the registered cadence")
    t_ooo = grid_times()
    t_ooo[7], t_ooo[8] = t_ooo[8], t_ooo[7]
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1440, times=t_ooo), mag_s()),
        "violates the registered cadence")
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1439), mag_s()),
        "violates the registered cadence")
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1440,
                              times=grid_times(day="2026-08-21")),
        mag_s()), "violates the registered cadence")
    t_bad = grid_times()
    t_bad[0] = "2026-08-20 00:00:00"
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", usgs_body([1.0] * 1440, times=t_bad), mag_s()),
        "non-canonical UTC instant")
    gin_raw = json.dumps({
        "datetime": grid_times(), "@info": {},
        "H": [1.0] * 1440,
        "Z": [None] + [2.0] * 1439}).encode()
    art_g = admission_transform("MAG_FEED", gin_raw, mag_s(
        source={"kind": "intermagnet-gin-minute", "ref": "kat://g"},
        request_params={"observatoryIagaCode": "KAT"}))
    assert art_g["samples"] == 1440
    assert art_g["definitive_samples"] == 1440
    assert art_g["channels"] == ["H", "Z"]
    assert art_g["null_by_channel"] == {"H": 0, "Z": 1}
    assert xrefuses(lambda: admission_transform(
        "MAG_FEED", json.dumps(
            {"datetime": grid_times(),
             "H": [None] * 1440}).encode(),
        mag_s(source={"kind": "intermagnet-gin-minute",
                      "ref": "kat://g"})), "provider-null")

    def mf4_s(carrier="kp", kind="gfz-kp-json", **over):
        s = {"lane": "MF4_FEED", "carrier": carrier,
             "utc_day": "2026-08-20",
             "endpoint": "https://kat.example/k",
             "request_params": {"start": "2026-08-20T00:00:00Z"},
             "source": {"kind": kind, "ref": "kat://k"},
             "cutoff": "2026-08-25",
             "operation_params": {"carrier": carrier,
                                  "day": "2026-08-20"},
             "expected_keys": ["2026-08-20"]}
        s.update(over)
        return s

    def kp_body(vals=None, status=None, dts=None):
        return json.dumps({
            "Kp": vals if vals is not None else [1.0] * 8,
            "status": status if status is not None
            else ["def"] * 8,
            "datetime": dts if dts is not None
            else ["2026-08-20T%02d:00:00Z" % (3 * i)
                  for i in range(8)]}).encode()
    art_k = admission_transform("MF4_FEED", kp_body(
        status=["def"] * 7 + ["prov"]), mf4_s())
    assert art_k["intervals"] == 8
    assert art_k["definitive_intervals"] == 7
    assert art_k["status_counts"] == {"def": 7, "prov": 1}
    # codex end-to-end finding 2, exact repro 3: an unregistered
    # status token refuses instead of admitting
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", kp_body(status=["NOT_REGISTERED"] * 8),
        mf4_s()), "not in the registered GFZ vocabulary")
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", kp_body(vals=[11.0] + [1.0] * 7), mf4_s()),
        "not a finite index value")
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", kp_body(vals=[float("nan")] + [1.0] * 7),
        mf4_s()), "not a finite index value")
    # the registered 8x3h cadence: short, duplicated, out-of-order,
    # and wrong-day interval lists all refuse
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", json.dumps(
            {"Kp": [1.0], "status": ["def"],
             "datetime": ["2026-08-20T00:00:00Z"]}).encode(),
        mf4_s()), "violates the registered cadence")
    kp_dts = ["2026-08-20T%02d:00:00Z" % (3 * i) for i in range(8)]
    kp_dup = list(kp_dts)
    kp_dup[3] = kp_dup[2]
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", kp_body(dts=kp_dup), mf4_s()),
        "violates the registered cadence")
    kp_ooo = list(kp_dts)
    kp_ooo[0], kp_ooo[1] = kp_ooo[1], kp_ooo[0]
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", kp_body(dts=kp_ooo), mf4_s()),
        "violates the registered cadence")
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", kp_body(dts=["2026-08-21T%02d:00:00Z" % (3 * i)
                                 for i in range(8)]), mf4_s()),
        "violates the registered cadence")

    def omni_body(make=None, n=1440,
                  header="YYYY DOY HR MN      1       2      3 "):
        lines = ["<HTML><pre>Selected parameters:", header]
        for i in range(n):
            hh, mn = divmod(i, 60)
            vals = make(i) if make else "  -4.68   556.3   0.95"
            lines.append(f"2026 232 {hh:2d} {mn:2d} {vals}")
        lines.append("</pre></HTML>")
        return ("\n".join(lines) + "\n").encode()
    art_o = admission_transform("MF4_FEED", omni_body(
        make=lambda i: ("9999.99 99999.9 999.99" if i == 1
                        else "  -4.68   556.3   0.95")), mf4_s(
        carrier="omni", kind="omniweb-highres-cgi",
        request_params={"vars": ["17", "21", "25"]}))
    assert art_o["samples"] == 1440
    assert art_o["definitive_samples"] == 1439
    assert art_o["fill_by_column"] == [1, 1, 1]
    # codex end-to-end finding 2, exact repro 2: nonnumeric value
    # tokens refuse instead of counting as definitive
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", omni_body(
            make=lambda i: ("NOT_A_NUMBER ALSO_BAD STILL_BAD"
                            if i == 0
                            else "  -4.68   556.3   0.95")), mf4_s(
            carrier="omni", kind="omniweb-highres-cgi",
            request_params={"vars": ["17", "21", "25"]})),
        "nonnumeric OMNIWeb value token")
    sym_s = mf4_s(carrier="sym_h", kind="omniweb-highres-cgi",
                  request_params={"vars": "41"})
    art_sym = admission_transform("MF4_FEED", omni_body(
        make=lambda i: "  -28", header="YYYY DOY HR MN    1 "),
        sym_s)
    assert art_sym["samples"] == 1440
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", omni_body(make=lambda i: "99999",
                              header="YYYY DOY HR MN    1 "),
        sym_s), "provider fill sentinels")
    # cadence doctors: short listing and a duplicated minute row
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", omni_body(make=lambda i: "  -28", n=1439,
                              header="YYYY DOY HR MN    1 "),
        sym_s), "full 1440-minute grid")
    dup_lines = omni_body(make=lambda i: "  -28",
                          header="YYYY DOY HR MN    1 "
                          ).decode().splitlines()
    dup_lines[10] = dup_lines[9]      # minute 7 repeats minute 6
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", ("\n".join(dup_lines) + "\n").encode(), sym_s),
        "violates the registered minute cadence")
    omni_txt2 = ("<HTML><pre>Selected parameters:\n"
                 "YYYY DOY HR MN      1       2      3 \n"
                 "2026 232  0  0   -4.68   556.3   0.95\n"
                 "</pre></HTML>\n").encode()
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", omni_txt2, mf4_s(
            carrier="omni", kind="omniweb-highres-cgi",
            request_params={"vars": ["17", "21"]})),
        "malformed OMNIWeb data row")
    wrong_day_omni = ("<pre>\nYYYY DOY HR MN    1 \n"
                      "2026 001  0  0   -28\n</pre>\n").encode()
    assert xrefuses(lambda: admission_transform(
        "MF4_FEED", wrong_day_omni, sym_s),
        "outside the registered day")
    assert xrefuses(lambda: admission_transform(
        "DAY_CAPSULE", b"x", sel_s(lane="DAY_CAPSULE")),
        "unregistered capture lane")

    print("w2_acquisition_capture selftest: ALL PASS (no network)")


if __name__ == "__main__":
    _selftest()
