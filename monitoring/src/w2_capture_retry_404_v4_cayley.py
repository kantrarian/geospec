#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXACT-KEY 404 RETRY, one shot (cayley) -- codex 1345Z contract.

WHAT THIS IS
------------
Asylum authorized (grassmann `68f40f81`, in-session) ONE exact-key
retry of the single transport-404 refusal from the terminal 635-run:

    MAG_FEED/new/2026-03-23      original seq 81, CAPTURE_HTTP_STATUS 404

`run()` is FORBIDDEN for this: it takes no key filter and, measured
against the frozen tree, would fire 213 logical captures -- the one
authorized key plus the 212 VIC keys codex ruled zero-HTTP. This module
is the dedicated non-generic successor codex specified: the target is a
SOURCE CONSTANT, there is no target argument, no exclusion list, no
plan iteration, and record existence is never enumeration. The 212 VIC
keys cannot enter the call graph.

ONE SHOT, CONSUMED BY DISPATCH
------------------------------
Immediately before the opener can execute, the dispatch record is
created atomically. Its existence consumes the authorization: every
later invocation refuses BEFORE network. A dispatch with no result is
`RETRY_INDETERMINATE_AFTER_DISPATCH` -- not permission to try again;
only a new explicit owner authorization reopens that state.

THE FROZEN 635-LINE TERMINAL LEDGER IS NEVER TOUCHED. The retry writes
create-once artifacts under its own registered paths and appends its
one-line index to a SEPARATE retry ledger.

OUTCOME CONTRACT (codex 1345Z sections 3-4)
-------------------------------------------
- 200 + registered transform passes: the ordinary four v4 classes are
  published create-once; the scientific objects remain plain
  NATIVE_V4_CAPTURE -- the operation result records that this native
  capture was the authorized retry. Store re-freeze / inventory rebuild
  is grassmann's step and is named as REQUIRED-NEXT in the result.
- transport 404 / non-200 / connection failure: a closed transport-only
  result -- status, URLs, request/response instants, allowlisted
  headers, zero scientific body, one logical opener call. It closes the
  RETRY AUDIT only; it is not a transcript, record, contract, artifact
  or inventory member, and the boundary stays 2,055/2,056.
- transport OK, transform refuses: body preserved in the store,
  TRANSFORM_REFUSED result, NO partial class set published. Any repair
  is zero-HTTP and separately reviewed; this authorization is spent.

A second 404 is a realistic outcome, not an edge case (grassmann
1353Z): this module is built to close the audit cleanly on it.
"""
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC
import w2_producer_grassmann as PROD
import w2_capture_run_v4_grassmann as RUN4

REPO = RUN4.REPO

# ---- THE TARGET IS A SOURCE CONSTANT. There is no target argument. --
TARGET_LANE = "MAG_FEED"
TARGET_CARRIER = "new"
TARGET_DAY = "2026-03-23"
TARGET_KEY = f"{TARGET_LANE}/{TARGET_CARRIER}/{TARGET_DAY}"
ORIGINAL_SEQ = 81
# the frozen terminal evidence this retry is bound to (grassmann 1353Z
# verified all of these 7/7 against the frozen ledger on devildog)
TERMINAL_LEDGER_SHA256 = ("a01bb3259aada3f9fee1d01998579f2e5f2e5280"
                          "024382d9ac50d7c2d23956ad")
# canonical digest of the one original REFUSED entry:
# json.dumps(entry, sort_keys=True, separators=(",", ":")), UTF-8
ORIGINAL_ENTRY_SHA256 = ("6657385255cff94c5cb167c618ccc1041eb74cab"
                         "029febed62ac55219fdb68f4")
# the URL that was ACTUALLY requested (taken from the terminal ledger's
# own refusal string, not reconstructed): the derived static contract
# must reproduce exactly this, or fire refuses.
REGISTERED_REQUEST_URL = ("https://geomag.usgs.gov/ws/data/"
                          "?endtime=2026-03-24T00%3A00%3A00Z"
                          "&format=json&id=NEW&sampling_period=60"
                          "&starttime=2026-03-23T00%3A00%3A00Z")
MAX_LOGICAL_HTTP_OPERATIONS = 1
OWNER_AUTH_REF = "inbox 68f40f81 (asylum in-session, one exact-key "\
                 "retry outside the 635 ceiling)"
CONTRACT_REF = "codex 1345Z exact-key retry contract"

RETRY_DIR = os.path.join(REPO, "docs", "f2g_window2_execution",
                         "retry_404_v4")
DISPATCH_BASENAME = "mag_feed_new_2026-03-23.dispatch.json"
RESULT_BASENAME = "mag_feed_new_2026-03-23.result.json"
RETRY_LEDGER = os.path.join(REPO, "docs", "f2g_window2_execution",
                            "capture_retry_ledger_v4.jsonl")
# codex 1906Z finding 2: the evidence allowlist is DEFINED FROM the
# capture authority -- a divergent local list stranded a spend when a
# real response carried content-length (transcript receipted it, the
# evidence dropped it, the join refused its own live capture).
HEADER_ALLOWLIST = tuple(CAP.RECEIPT_HEADERS)


class RetryRefusal(SystemExit):
    """Typed, fail-closed. The code leads the message."""


def _refuse(code, detail):
    raise RetryRefusal(f"{code}: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _norm_source_sha256(raw):
    """codex 1906Z finding 1: the manifest pins LF blobs while a
    normal Windows checkout materializes CRLF, so raw disk bytes and
    the pinned blob live in different byte domains. This module uses
    the SAME normalization rule the ACC live allowlist and the
    manifest verifier already apply (ACC._norm_py), so one Python
    source has one authoritative digest on every host."""
    return _sha(ACC._norm_py(raw))


def _canon_entry_digest(entry):
    """grassmann 1353Z serialization, stated there explicitly so a
    differently-serialised digest fails at AUTHORING time, not fire
    time: sort_keys, separators (",", ":"), UTF-8."""
    return _sha(json.dumps(entry, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"))


def _resolve_commit(commitish):
    """codex 1929Z primitive 1: <commitish>^{commit} is resolved
    EXACTLY ONCE at the entrypoint to a full lowercase 40-hex object
    ID, and only that value flows through precheck, manifest/capsule/
    module reopening, capture and the dispatch record. A moving ref
    can therefore never make two reads concern different commits, and
    the persisted operation identity is a stable Git lineage."""
    import subprocess
    pr = subprocess.run(
        ["git", "-C", REPO, "rev-parse",
         f"{commitish}^{{commit}}"], capture_output=True)
    full = pr.stdout.decode().strip().lower()
    if pr.returncode != 0 or len(full) != 40 or \
            any(c not in "0123456789abcdef" for c in full):
        _refuse("RETRY_MANIFEST_UNRESOLVABLE",
                f"{commitish!r} does not resolve to a commit")
    return full


def _production_io():
    """The production wiring. fire() from the CLI ALWAYS uses this;
    the _kat_io injection point exists for the selftest bars ONLY and
    is never reachable from the command line."""
    stem = CAP._path_tokens(TARGET_LANE, TARGET_CARRIER, TARGET_DAY)
    return {
        "ledger_path": RUN4.LEDGER,
        "staged_dir": RUN4.STAGED_DIR,
        "store_dir": RUN4.STORE_PHYSICAL,
        "retry_dir": RETRY_DIR,
        "dispatch_path": os.path.join(RETRY_DIR, DISPATCH_BASENAME),
        "result_path": os.path.join(RETRY_DIR, RESULT_BASENAME),
        "retry_ledger": RETRY_LEDGER,
        "attempt_dir": os.path.join(RETRY_DIR, "attempt_local"),
        "stem": stem,
        "allowlist_fn": ACC.runtime_allowlist_check,
        # residual 4: the boundary-typed opener; the runner's opener
        # stays untouched with the historical 635 evidence
        "opener": _retry_production_opener,
        # production capture entry: the pinned-resolving entrypoint.
        # The selftest substitutes the registered fixture path
        # (capture_day), exactly as the window-2 bar's positive cases
        # do -- capture_authorized cannot run green pre-landing since
        # it resolves this module from a BOUND pin.
        "capture_fn": None,
        "expect_ledger_sha": TERMINAL_LEDGER_SHA256,
        "expect_entry_sha": ORIGINAL_ENTRY_SHA256,
        "expect_url": REGISTERED_REQUEST_URL,
    }


def _read_terminal_ledger(io):
    path = io["ledger_path"]
    if not os.path.isfile(path):
        _refuse("RETRY_TERMINAL_LEDGER_ABSENT", path)
    with open(path, "rb") as f:
        raw = f.read()
    if _sha(raw) != io["expect_ledger_sha"]:
        _refuse("RETRY_TERMINAL_LEDGER_DIGEST_DIVERGENT",
                f"{_sha(raw)[:16]} != {io['expect_ledger_sha'][:16]} "
                "-- the frozen 635-line ledger is not the bound bytes")
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines()
            if x.strip()]
    if len(rows) != 635:
        _refuse("RETRY_TERMINAL_LEDGER_COUNT", str(len(rows)))
    return rows


def precheck(manifest_commit, io):
    """codex 1345Z checks 1-5, all BEFORE any opener is reachable.
    Returns the context fire() needs. Fail-closed throughout."""
    # (1) executed disk bytes are a current BOUND pin + allowlist PASS
    _me = "monitoring/src/" + os.path.basename(
        os.path.abspath(__file__))
    try:
        allow = io["allowlist_fn"](REPO, manifest_commit)
    except Exception as exc:
        _refuse("RETRY_ALLOWLIST_REFUSED",
                f"{type(exc).__name__}: {str(exc)[:200]}")
    checked = {p for _s, p in (allow.get("pins") or ())}
    if _me not in checked:
        _refuse("RETRY_MODULE_UNBOUND",
                f"{_me} is not among the BOUND pins checked against "
                f"executed disk bytes at {manifest_commit} -- a "
                "network-spending module may not execute unpinned")
    # (2) authority + capsule close normally; 635-key HTTP partition;
    # the target is authorized
    st = RUN4.capsule_pin_status(manifest_commit)
    if not st["runnable"]:
        _refuse("RETRY_CAPSULE_NOT_RUNNABLE", str(st["reason"]))
    authority, capsule, keys, counts, csha = RUN4.load_plan(
        st["pin_commit"])
    if csha != st["pinned_blob_sha256"]:
        _refuse("RETRY_CAPSULE_PIN_DIVERGENT", "reopened capsule does "
                "not match the pinned digest")
    if len(keys) != 635:
        _refuse("RETRY_PLAN_NOT_635", str(len(keys)))
    if TARGET_KEY not in set(keys):
        _refuse("RETRY_TARGET_NOT_AUTHORIZED",
                f"{TARGET_KEY} is not in HTTP_CAPTURE")
    # (3) the immutable terminal ledger binds exactly one original 404
    rows = _read_terminal_ledger(io)
    hits = [r for r in rows if r.get("key") == TARGET_KEY]
    if len(hits) != 1:
        _refuse("RETRY_ORIGINAL_ENTRY_COUNT",
                f"{len(hits)} entries for {TARGET_KEY}; exactly one "
                "is required")
    entry = hits[0]
    if entry.get("seq") != ORIGINAL_SEQ:
        _refuse("RETRY_ORIGINAL_SEQ_DIVERGENT",
                f"{entry.get('seq')} != {ORIGINAL_SEQ}")
    if entry.get("status") != "REFUSED" or \
            "404" not in str(entry.get("error",
                                       entry.get("refusal", ""))):
        _refuse("RETRY_ORIGINAL_NOT_A_404",
                json.dumps(entry)[:200])
    got = _canon_entry_digest(entry)
    if got != io["expect_entry_sha"]:
        _refuse("RETRY_ORIGINAL_ENTRY_DIGEST_DIVERGENT",
                f"{got[:16]} != {io['expect_entry_sha'][:16]}")
    # (4) authorization unspent; no ordinary class already published
    if os.path.exists(io["dispatch_path"]):
        if os.path.exists(io["result_path"]):
            _refuse("RETRY_ALREADY_EXECUTED",
                    "dispatch and result both exist; the one "
                    "authorization is spent")
        _refuse("RETRY_INDETERMINATE_AFTER_DISPATCH",
                "a dispatch record exists with no result; this is NOT "
                "permission to try again -- only a new explicit owner "
                "authorization reopens it")
    if os.path.exists(io["result_path"]):
        _refuse("RETRY_EVIDENCE_INCONSISTENT",
                "a result exists with no dispatch")
    if os.path.exists(io["retry_ledger"]):
        with open(io["retry_ledger"], encoding="utf-8") as f:
            if any(TARGET_KEY in x for x in f if x.strip()):
                _refuse("RETRY_ALREADY_LEDGERED", TARGET_KEY)
    for cls, suf in ACC.STAGED_CLASS_SUFFIX.items():
        p = os.path.join(io["staged_dir"], io["stem"] + suf)
        if os.path.exists(p):
            _refuse("RETRY_TARGET_CLASS_ALREADY_PUBLISHED",
                    f"{cls} exists at {p}")
    # (5) the derived static contract reproduces the registered URL
    s = ACC.authoritative_static_contract(
        authority, TARGET_LANE, TARGET_CARRIER, TARGET_DAY)
    url = PROD.requested_url_of(s["endpoint"], s["request_params"])
    if url != io["expect_url"]:
        _refuse("RETRY_URL_DIVERGENT",
                f"derived {url!r} != registered {io['expect_url']!r}")
    return {"authority": authority, "capsule_sha": csha,
            "pin_commit": st["pin_commit"], "static": s,
            "original_entry": entry, "rows": rows}


class RetryTransportError(Exception):
    """codex 1705Z repair 4 + 1830Z residual 4: raised ONLY around
    the actual urlopen/response-read boundary, and the ONLY exception
    fire() maps to a transport outcome. URL parsing, pacing, request
    construction, certificate lookup and SSL-context construction are
    LOCAL failures and must never be reported as external transport
    causes."""

    def __init__(self, msg, evidence):
        super().__init__(msg)
        self.evidence = evidence


def _retry_production_opener(url):
    """The retry's OWN production opener (codex 1830Z residual 4).
    It mirrors the runner's pacing/UA/SSL construction but types the
    transport boundary exactly: only the urlopen call and the
    response read raise RetryTransportError. The runner's opener is
    NOT modified -- that file is accepted byte-for-byte and belongs
    to the historical 635 evidence; codex may later prefer folding
    this boundary back into the runner, which is a separate reviewed
    change.

    An HTTP error status is a RESPONSE, not a transport failure: it
    returns (code, {}, b"", url) exactly as the runner's opener does.
    """
    import ssl as _ssl
    import urllib.error as _uerr
    import urllib.request as _ureq
    from urllib.parse import urlsplit
    host = urlsplit(url).netloc
    wait = RUN4._last_by_host.get(host, 0) + RUN4.PACING_S         - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    RUN4._last_by_host[host] = time.monotonic()
    req = _ureq.Request(url, headers={"User-Agent": RUN4.UA})
    ctx = None
    if url.startswith("https://"):
        import certifi
        ctx = _ssl.create_default_context(cafile=certifi.where())
        ctx.check_hostname = True
        ctx.verify_mode = _ssl.CERT_REQUIRED
    # ---- the transport boundary starts HERE ----
    t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with _ureq.urlopen(req, timeout=RUN4.TIMEOUT_S,
                           context=ctx) as r:
            return (getattr(r, "status", r.getcode()),
                    {k.lower(): v for k, v in r.headers.items()},
                    r.read(), r.geturl())
    except _uerr.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return (exc.code, {}, b"", getattr(exc, "url", url))
    except Exception as exc:
        raise RetryTransportError(
            f"{type(exc).__name__}: {str(exc)[:200]}",
            {"requested_url": url, "request_start_utc": t0,
             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    # ---- the transport boundary ends HERE ----


TRANSPORT_RECEIPT_BASENAME_SUFFIX = ".transport_receipt.json"


def _transport_receipt_path(io):
    return os.path.join(io["attempt_dir"],
                        io["stem"] + TRANSPORT_RECEIPT_BASENAME_SUFFIX)


def _read_transport_receipt(io, dispatch=None):
    """CLAIM BOUNDARY (codex 2017Z finding 3, Option A -- asylum may
    instead choose Option B, external anchoring/signing, which is an
    owner-level evidence-design decision): this receipt is a
    CREATE-ONCE LOCAL CONSISTENCY JOIN, not tamper evidence against
    an actor who can coherently rewrite the packet and its
    self-digests. Evidentiary tamper resistance begins only once the
    completed packet is externally committed/anchored. Within that
    stated scope, the receipt is closed, self-digested, and bound to
    its dispatch/attempt identity."""
    rp = _transport_receipt_path(io)
    if not os.path.isfile(rp):
        return None
    try:
        with open(rp, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _refuse("RETRY_TRANSPORT_RECEIPT_DEFORMED",
                f"the transport receipt is unreadable "
                f"({type(exc).__name__})")
    if not isinstance(rec, dict) or set(rec) != {
            "schema", "kind", "dispatch_sha256", "attempt_id",
            "evidence", "receipt_sha256"} or \
            rec.get("schema") != "f2g-w2-retry-transport-receipt-v1" \
            or rec.get("kind") not in ("response", "boundary_error"):
        _refuse("RETRY_TRANSPORT_RECEIPT_DEFORMED",
                "the transport receipt is not the closed object")
    if _canon_entry_digest({k: v for k, v in rec.items()
                            if k != "receipt_sha256"}) != \
            rec.get("receipt_sha256"):
        _refuse("RETRY_TRANSPORT_RECEIPT_DEFORMED",
                "the transport receipt self-digest does not "
                "recompute")
    if dispatch is not None:
        if rec.get("dispatch_sha256") != \
                dispatch.get("dispatch_sha256") or \
                rec.get("attempt_id") != dispatch.get("attempt_id"):
            _refuse("RETRY_TRANSPORT_RECEIPT_DEFORMED",
                    "the transport receipt does not bind this "
                    "dispatch/attempt identity")
    return rec


def _validate_transport_value(tv, io, kind, what):
    """codex 1929Z primitive 2: UNCONDITIONAL value checks for every
    transport projection, independent of whether a transcript exists.
    kind: "response" (OK keyset) or "boundary_error"."""
    if kind == "response":
        if not isinstance(tv, dict) or set(tv) != TRANSPORT_OK_KEYS:
            _refuse(what, "transport keyset is not the closed opener "
                          "response evidence")
        if not isinstance(tv.get("status"), int):
            _refuse(what, "transport status is untyped")
        if tv.get("requested_url") != io["expect_url"]:
            _refuse(what, "transport requested URL is not the "
                          "registered URL")
        if not isinstance(tv.get("effective_url"), str) or \
                not tv["effective_url"]:
            _refuse(what, "transport effective URL is not a nonempty "
                          "string")
        hh = tv.get("headers")
        if not isinstance(hh, dict) or any(
                not isinstance(k, str) or k != k.lower()
                or k not in HEADER_ALLOWLIST
                or not isinstance(v, str)
                for k, v in hh.items()):
            _refuse(what, "transport headers are outside the "
                          "lowercase receipt allowlist or untyped")
        if not isinstance(tv.get("body_bytes_seen"), int) or \
                tv["body_bytes_seen"] < 0:
            _refuse(what, "transport body count is not a "
                          "nonnegative integer")
        if not _is_canonical_utc(tv.get("request_start_utc")) or \
                not _is_canonical_utc(
                    tv.get("response_complete_utc")):
            _refuse(what, "transport instants are not canonical")
        _require_phase_order(what, tv["request_start_utc"],
                             tv["response_complete_utc"])
    else:
        if not isinstance(tv, dict) or set(tv) != TRANSPORT_ERR_KEYS:
            _refuse(what, "transport keyset is not the closed "
                          "boundary-error evidence")
        if tv.get("requested_url") != io["expect_url"]:
            _refuse(what, "transport requested URL is not the "
                          "registered URL")
        if not _is_canonical_utc(tv.get("request_start_utc")):
            _refuse(what, "transport instant is not canonical")
        ev = tv.get("error")
        if not isinstance(ev, str) or not ev or len(ev) > 300:
            _refuse(what, "transport error is empty or unbounded")


def _validate_transport_member(tv, io, kind, what, *,
                               transcript=None, dispatch=None):
    """The ONE transport authority chain (codex 1929Z primitive 2):
    the attempt-local transport RECEIPT is mandatory for every
    opener_calls=1 member; the member's transport must equal the
    receipt's evidence field-for-field; unconditional value checks
    always run; and where a transcript exists its receipted headers
    and projection must equal the evidence EXACTLY -- no evidence-only
    keys, no subset fallback."""
    receipt = _read_transport_receipt(io, dispatch=dispatch)
    if receipt is None:
        _refuse(what, "no attempt-local transport receipt exists -- "
                      "a transport claim without its receipt is "
                      "unverifiable")
    if receipt["kind"] != kind:
        _refuse(what, f"transport receipt kind {receipt['kind']!r} "
                      f"does not match the member's phase {kind!r}")
    if tv != receipt["evidence"]:
        _refuse(what, "the member's transport does not equal the "
                      "attempt-local transport receipt "
                      "field-for-field")
    _validate_transport_value(tv, io, kind, what)
    if transcript is not None and kind == "response":
        if tv.get("requested_url") != transcript.get("requested_url") \
                or tv.get("effective_url") != \
                transcript.get("effective_url") \
                or tv.get("status") != transcript.get("http_status") \
                or tv.get("body_bytes_seen") != \
                transcript.get("raw_body_bytes"):
            _refuse(what, "transport projection does not equal the "
                          "authenticated transcript")
        if tv.get("headers") != {
                str(k).lower(): str(v) for k, v in
                (transcript.get("headers") or {}).items()}:
            _refuse(what, "transport headers do not equal the "
                          "transcript's receipted headers exactly")


class _CountingOpener:
    """Hard ceiling of ONE logical opener call, plus safe transport
    evidence capture for the transport-only result."""

    def __init__(self, inner, io=None, dispatch=None):
        self.inner = inner
        self.io = io
        self.dispatch = dispatch or {}
        self.calls = 0
        self.evidence = None

    def _receipt(self, kind, evidence):
        if self.io is None:
            return
        rec = {"schema": "f2g-w2-retry-transport-receipt-v1",
               "kind": kind,
               "dispatch_sha256":
                   self.dispatch.get("dispatch_sha256"),
               "attempt_id": self.dispatch.get("attempt_id"),
               "evidence": evidence}
        rec["receipt_sha256"] = _canon_entry_digest(
            {k: v for k, v in rec.items() if k != "receipt_sha256"})
        _write_json_create_once(_transport_receipt_path(self.io), rec)

    def __call__(self, url):
        self.calls += 1
        if self.calls > MAX_LOGICAL_HTTP_OPERATIONS:
            raise AssertionError(
                "RETRY_OPENER_CEILING: a second opener call was "
                "attempted; the ceiling is 1")
        t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            status, headers, body, eff = self.inner(url)
        except RetryTransportError as exc:
            # already typed at the urlopen boundary; attach counting
            # context, RECEIPT the boundary error (codex 1929Z
            # primitive 2: every transport outcome leaves its
            # create-once attempt-local receipt before raising), and
            # re-raise. Any OTHER exception from the opener is a
            # LOCAL failure and propagates untyped.
            if self.evidence is None:
                self.evidence = dict(exc.evidence)
            self._receipt("boundary_error", self.evidence)
            raise
        t1 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.evidence = {
            "requested_url": url, "effective_url": eff,
            "status": int(status),
            "request_start_utc": t0, "response_complete_utc": t1,
            "headers": {str(k).lower(): str(v)
                        for k, v in (headers or {}).items()
                        if str(k).lower() in HEADER_ALLOWLIST},
            "body_bytes_seen": len(body or b"")}
        # primitive 2: the response receipt is written BEFORE the
        # body is returned to any consumer
        self._receipt("response", self.evidence)
        return status, headers, body, eff


# codex 1705Z repair 3: CLOSED record schemas. Exact key sets, so a
# record cannot smuggle or drop fields and still authenticate.
DISPATCH_FIELDS = frozenset({
    "schema", "key", "owner_authorization", "contract",
    "original_ledger", "manifest_commit", "manifest_blob_sha256",
    "capsule_pin_commit",
    "capsule_sha256", "attempt_id", "executed_code", "store",
    "expected_classes", "registered_request_url",
    "max_logical_http_operations", "vic_http_operations",
    "dispatched_utc", "dispatch_sha256"})
PREPARED_FIELDS = frozenset({
    "schema", "key", "dispatch_sha256", "outcome",
    "class_canon_sha256", "opener_calls", "transport",
    "terminal_ledger_sha256_recomputed", "terminal_ledger_unchanged",
    "completed_utc", "prepared_sha256"})
RESULT_OUTCOMES = frozenset({
    "CAPTURED_ADMITTED", "TRANSPORT_REFUSED_NON_200",
    "TRANSPORT_ERROR", "TRANSFORM_REFUSED", "CAPTURE_REFUSED",
    "CAPTURE_REFUSED_AFTER_TRANSPORT",
    "INTERNAL_ERROR_AFTER_DISPATCH",
    "INTERNAL_ERROR_AFTER_TRANSPORT"})


def _is_hex64(v):
    return isinstance(v, str) and len(v) == 64 and \
        all(c in "0123456789abcdef" for c in v)


_UTC_RE = None


def _parse_canonical_utc(v):
    """codex 1929Z primitive 3: strptime alone accepts non-padded
    directives (2026-9-30...), so canonical means PARSE + EXACT
    ROUND-TRIP. Returns the parsed datetime, or None."""
    if not isinstance(v, str):
        return None
    import datetime
    try:
        dt = datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    if dt.strftime("%Y-%m-%dT%H:%M:%SZ") != v:
        return None
    return dt


def _is_canonical_utc(v):
    return _parse_canonical_utc(v) is not None


def _require_phase_order(what, *instants):
    """dispatch <= request-start <= response-complete <= completion,
    over the phases that exist -- compared as PARSED datetimes, never
    as strings (codex 1929Z: "2026-10..." < "2026-9..." under string
    order)."""
    seq = []
    for x in instants:
        if x is None:
            continue
        dt = _parse_canonical_utc(x)
        if dt is None:
            _refuse(what, f"phase instant {x!r} is not canonical")
        seq.append((dt, x))
    for (da, xa), (db, xb) in zip(seq, seq[1:]):
        if da > db:
            _refuse(what, f"phase instants out of order: {xa!r} > "
                          f"{xb!r}")


# the exact transport-evidence keysets _CountingOpener produces
TRANSPORT_OK_KEYS = frozenset({
    "requested_url", "effective_url", "status", "request_start_utc",
    "response_complete_utc", "headers", "body_bytes_seen"})
TRANSPORT_ERR_KEYS = frozenset({
    "requested_url", "request_start_utc", "error"})
RESULT_FIELDS = frozenset({
    "schema", "key", "dispatch_sha256", "outcome", "refusal",
    "opener_calls", "transport", "scientific",
    "terminal_ledger_sha256_recomputed", "terminal_ledger_unchanged",
    "completed_utc", "result_sha256"})
MODULE_PATH = "monitoring/src/w2_capture_retry_404_v4_cayley.py"
REQUIRED_NEXT_TEXT = ("store re-freeze + inventory rebuild "
                      "(grassmann) before any boundary claim")


def _require_current_ledger(io):
    with open(io["ledger_path"], "rb") as f:
        cur = _sha(f.read())
    if cur != io["expect_ledger_sha"]:
        _refuse("RETRY_TERMINAL_LEDGER_CHANGED",
                f"current ledger bytes recompute {cur[:16]}, the "
                f"registered digest is "
                f"{io['expect_ledger_sha'][:16]}")


def _attempt_obj(io, cls):
    ap = os.path.join(io["attempt_dir"],
                      io["stem"] + ACC.STAGED_CLASS_SUFFIX[cls])
    if not os.path.isfile(ap):
        return None
    with open(ap, encoding="utf-8") as f:
        return json.load(f)


def _validate_dispatch_semantics(d, io):
    """codex 1845Z residual 3: every nested dispatch value JOINS to
    the registered operation identity -- a self-digest seals content,
    it never chooses it."""
    if d.get("owner_authorization") != OWNER_AUTH_REF or \
            d.get("contract") != CONTRACT_REF:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "owner-authorization/contract text is not the "
                "registered text")
    ol = d.get("original_ledger")
    want_ol = {"path": ACC.CAPTURE_LEDGER_PATH,
               "sha256": io["expect_ledger_sha"],
               "seq": ORIGINAL_SEQ,
               "entry_sha256": io["expect_entry_sha"]}
    if ol != want_ol:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "original-ledger binding does not equal the "
                "registered path/digest/seq/entry values")
    if d.get("registered_request_url") != io["expect_url"]:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "request URL is not the registered URL")
    if d.get("store") != {"id": RUN4.STORE_ID,
                          "root": RUN4.STORE_ROOT}:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "store identity is not the registered store")
    if d.get("max_logical_http_operations") != 1 or \
            d.get("vic_http_operations") != 0:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "the one-shot ceiling invariants do not hold")
    want_cls = sorted(io["stem"] + suf
                      for suf in ACC.STAGED_CLASS_SUFFIX.values())
    if sorted(d.get("expected_classes", ())) != want_cls:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "expected classes are not exactly the four target "
                "stems")
    def _is_hex40(v):
        return isinstance(v, str) and len(v) == 40 and \
            all(c in "0123456789abcdef" for c in v)
    # codex 2017Z finding 2: the PERSISTED representation must be the
    # immutable lineage itself -- exact lowercase 40-hex that
    # resolves to ITSELF. A re-digested mutable alias ("HEAD") can
    # never re-enter the operation record after authoring.
    mc = d.get("manifest_commit")
    if not _is_hex40(mc):
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "manifest_commit is not exact lowercase 40-hex -- a "
                "mutable ref is never a persisted operation "
                "identity")
    if _resolve_commit(mc) != mc:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "manifest_commit does not resolve to itself")
    if not _is_hex64(d.get("capsule_sha256")) or \
            not _is_hex40(d.get("capsule_pin_commit")):
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "capsule pin grammar is not exact lowercase hex")
    aid = d.get("attempt_id")
    if not isinstance(aid, str) or len(aid) != 32 or \
            any(c not in "0123456789abcdef" for c in aid):
        _refuse("RETRY_DISPATCH_SEMANTICS", "attempt id grammar")
    if not _is_canonical_utc(d.get("dispatched_utc")):
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "dispatched_utc is not a canonical UTC instant")
    ec = d.get("executed_code")
    if not isinstance(ec, dict) or ec.get("path") != MODULE_PATH or \
            not _is_hex64(ec.get("disk_sha256")):
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "executed-code path/digest is not the registered "
                "module")
    # reopen the manifest pin at the dispatch's manifest commit and
    # require the executed-code binding to match it
    # codex 1906Z finding 4: the manifest is reopened FAIL-CLOSED --
    # an unresolvable commit refuses; it never downgrades to the
    # fixture-only unpinned form. The dispatch also binds the full
    # manifest object identity.
    try:
        man_raw = RUN4._blob(d.get("manifest_commit"),
                             CAP.EXEC_MANIFEST_PATH)
    except SystemExit:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "the dispatch manifest commit does not resolve -- "
                "public validation fails closed, never downgrades")
    if _sha(man_raw) != d.get("manifest_blob_sha256"):
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "manifest_blob_sha256 does not recompute from the "
                "reopened manifest bytes")
    with open(os.path.abspath(__file__), "rb") as f:
        cur_norm = _norm_source_sha256(f.read())
    if ec.get("disk_sha256") != cur_norm:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "executed-code digest does not equal the current "
                "normalized module source")
    pin = _module_pin_binding(d.get("manifest_commit"))
    if pin.get("pin_commit") is not None:
        if not _is_hex40(str(ec.get("pin_commit") or "")) or \
                ec.get("pin_commit") != pin["pin_commit"] or \
                ec.get("pin_blob_sha256") != pin["pin_blob_sha256"]:
            _refuse("RETRY_DISPATCH_SEMANTICS",
                    "executed-code pin binding does not equal the "
                    "reopened manifest pin in exact hex form")
        if cur_norm != pin["pin_blob_sha256"]:
            _refuse("RETRY_DISPATCH_SEMANTICS",
                    "current normalized module source diverges from "
                    "the reopened manifest pin")
    else:
        # fixtures ONLY: the unpinned form is permitted solely under
        # an explicit internal KAT context supplied out-of-band --
        # never inferred from the (attacker-controllable) dispatch
        if not io.get("_kat_allow_unpinned"):
            _refuse("RETRY_DISPATCH_SEMANTICS",
                    "the module is not pinned at the dispatch "
                    "manifest and no internal KAT context permits "
                    "the unpinned form")
        if set(ec) != {"path", "disk_sha256", "pin_commit",
                       "pin_blob_sha256", "note"} or \
                ec.get("pin_commit") is not None:
            _refuse("RETRY_DISPATCH_SEMANTICS",
                    "executed-code keyset is not the closed "
                    "not-yet-pinned form")
    # codex 1906Z finding 4: capsule identity is REOPENED and joined,
    # not grammar-checked
    try:
        st = RUN4.capsule_pin_status(d.get("manifest_commit"))
    except SystemExit:
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "capsule pin status does not resolve at the "
                "dispatch manifest")
    if st.get("pin_commit") != d.get("capsule_pin_commit") or \
            st.get("pinned_blob_sha256") != d.get("capsule_sha256"):
        _refuse("RETRY_DISPATCH_SEMANTICS",
                "capsule pin binding does not equal the reopened "
                "registered capsule status")


def _validate_prepared_semantics(pr, io, dispatch=None,
                                 transcript=None, record=None):
    """codex 1845Z residual 1: the prepared projection JOINS to the
    authenticated transcript/record, the registered ledger digest and
    the CURRENT ledger bytes -- shape alone seals nothing."""
    if pr.get("outcome") != "CAPTURED_ADMITTED":
        _refuse("RETRY_PREPARED_SEMANTICS",
                f"outcome {pr.get('outcome')!r}; the only permitted "
                "prepared outcome is CAPTURED_ADMITTED")
    cm = pr.get("class_canon_sha256")
    if not isinstance(cm, dict) or \
            set(cm) != set(ACC.STAGED_CLASS_SUFFIX) or \
            not all(_is_hex64(v) for v in cm.values()):
        _refuse("RETRY_PREPARED_SEMANTICS",
                "class map is not exactly the four 64-hex classes")
    if pr.get("opener_calls") != 1:
        _refuse("RETRY_PREPARED_SEMANTICS",
                "a prepared admitted outcome requires exactly one "
                "opener call")
    if transcript is None:
        transcript = _attempt_obj(io, "transcript")
    if record is None:
        record = _attempt_obj(io, "record")
    if transcript is not None and \
            cm.get("transcript") != PROD._canon_digest(transcript):
        _refuse("RETRY_PREPARED_SEMANTICS",
                "class map transcript digest does not equal the "
                "attempt transcript")
    if transcript is None:
        _refuse("RETRY_PREPARED_SEMANTICS",
                "an admitted prepared record requires its "
                "authenticated attempt transcript")
    _validate_transport_member(pr.get("transport"), io, "response",
                               "RETRY_PREPARED_SEMANTICS",
                               transcript=transcript,
                               dispatch=dispatch)
    if record is not None and pr["transport"].get(
            "body_bytes_seen") != record.get("raw_body_bytes"):
        _refuse("RETRY_PREPARED_SEMANTICS",
                "transport body bytes do not equal the "
                "authenticated record")
    if pr.get("terminal_ledger_sha256_recomputed") != \
            io["expect_ledger_sha"] or \
            pr.get("terminal_ledger_unchanged") is not True:
        _refuse("RETRY_PREPARED_SEMANTICS",
                "prepared ledger binding is not the registered "
                "digest with unchanged=true")
    _require_current_ledger(io)
    if not _is_canonical_utc(pr.get("completed_utc")):
        _refuse("RETRY_PREPARED_SEMANTICS",
                "completed_utc is not a canonical UTC instant")
    # primitive 3: the FULL phase order on the admitted path --
    # dispatched <= request-start <= response-complete <= completion
    _require_phase_order(
        "RETRY_PREPARED_SEMANTICS",
        dispatch.get("dispatched_utc") if dispatch else None,
        pr["transport"].get("request_start_utc"),
        pr["transport"].get("response_complete_utc"),
        pr["completed_utc"])
    _require_phase_order(
        "RETRY_PREPARED_SEMANTICS",
        transcript.get("response_complete_utc"),
        pr["completed_utc"])
    if dispatch is not None and \
            pr.get("dispatch_sha256") != dispatch.get(
                "dispatch_sha256"):
        _refuse("RETRY_PREPARED_SEMANTICS",
                "prepared does not bind this dispatch")


def _expected_admitted_result(io, dispatch, prepared, record):
    """codex 1845Z residual 2: ONE canonical admitted-result
    projection derived from authenticated dispatch + prepared +
    attempt record. An admitted result must equal it field for
    field."""
    body = {
        "schema": "f2g-w2-retry-404-result-v1",
        "key": TARGET_KEY,
        "dispatch_sha256": dispatch["dispatch_sha256"],
        "outcome": "CAPTURED_ADMITTED",
        "refusal": None,
        "opener_calls": prepared["opener_calls"],
        "transport": prepared["transport"],
        "scientific": {
            "classes_published": dict(
                prepared["class_canon_sha256"]),
            "raw_body_sha256": record.get("raw_body_sha256"),
            "proof_kind": "NATIVE_V4_CAPTURE",
            "required_next": REQUIRED_NEXT_TEXT},
        "terminal_ledger_sha256_recomputed":
            prepared["terminal_ledger_sha256_recomputed"],
        "terminal_ledger_unchanged":
            prepared["terminal_ledger_unchanged"],
        "completed_utc": prepared["completed_utc"]}
    body["result_sha256"] = _canon_entry_digest(
        {k: v for k, v in body.items() if k != "result_sha256"})
    return body


def _validate_result_semantics(r, io, dispatch=None, prepared=None,
                               record=None):
    """codex 1845Z residual 4: every enum member is a complete closed
    sum type -- exact nested keys, opener count, transport
    presence/shape, scientific contract, typed refusal, exact ledger
    binding and canonical completion, per outcome."""
    oc = r.get("outcome")
    if oc not in RESULT_OUTCOMES:
        _refuse("RETRY_RESULT_SEMANTICS", f"unknown outcome {oc!r}")
    if r.get("terminal_ledger_sha256_recomputed") != \
            io["expect_ledger_sha"] or \
            r.get("terminal_ledger_unchanged") is not True:
        _refuse("RETRY_RESULT_SEMANTICS",
                "result ledger binding is not the registered digest "
                "with unchanged=true")
    if not _is_canonical_utc(r.get("completed_utc")):
        _refuse("RETRY_RESULT_SEMANTICS",
                "completed_utc is not a canonical UTC instant")
    sci = r.get("scientific")
    if oc == "CAPTURED_ADMITTED":
        if dispatch is None or prepared is None or record is None:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "an admitted result cannot be validated without "
                    "its dispatch/prepared/attempt context")
        want = _expected_admitted_result(io, dispatch, prepared,
                                         record)
        if r != want:
            diffs = sorted(k for k in set(r) | set(want)
                           if r.get(k) != want.get(k))
            _refuse("RETRY_RESULT_SEMANTICS",
                    "admitted result does not equal the canonical "
                    f"projection (fields: {diffs[:4]})")
        return
    if not isinstance(r.get("refusal"), str) or not r["refusal"]:
        _refuse("RETRY_RESULT_SEMANTICS",
                f"{oc} requires a nonempty typed refusal")
    d_utc = dispatch.get("dispatched_utc") if dispatch else None
    if oc == "TRANSPORT_REFUSED_NON_200":
        tv = r.get("transport")
        if r.get("opener_calls") != 1:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a non-200 refusal requires exactly one opener "
                    "call")
        _validate_transport_member(tv, io, "response",
                                   "RETRY_RESULT_SEMANTICS",
                                   dispatch=dispatch)
        if tv.get("status") == 200:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a non-200 refusal cannot carry status 200")
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            tv["request_start_utc"],
                            tv["response_complete_utc"],
                            r["completed_utc"])
        if not r["refusal"].endswith(f"-> {tv['status']}") or \
                tv["requested_url"] not in r["refusal"] or \
                "CAPTURE_HTTP_STATUS" not in r["refusal"]:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "the typed refusal does not parse to the same "
                    "status/URL the evidence carries")
        if sci != {}:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transport refusal carries no scientific "
                    "content")
    elif oc == "TRANSPORT_ERROR":
        tv = r.get("transport")
        if r.get("opener_calls") != 1:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transport error requires exactly one opener "
                    "call")
        _validate_transport_member(tv, io, "boundary_error",
                                   "RETRY_RESULT_SEMANTICS",
                                   dispatch=dispatch)
        # primitive 2: the refusal binds the ONE receipted error
        if r["refusal"] != tv.get("error"):
            _refuse("RETRY_RESULT_SEMANTICS",
                    "the refusal does not equal the receipted "
                    "boundary error")
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            tv["request_start_utc"],
                            r["completed_utc"])
        if sci != {}:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transport error carries no scientific "
                    "content")
    elif oc == "TRANSFORM_REFUSED":
        # the transcript EXISTS by construction on this path (it is
        # written before the transform runs) -- its absence refuses
        a_tr = _attempt_obj(io, "transcript")
        if a_tr is None:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transform refusal requires its authenticated "
                    "attempt transcript")
        tv = r.get("transport")
        if r.get("opener_calls") != 1:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transform refusal follows exactly one "
                    "successful transport call")
        _validate_transport_member(tv, io, "response",
                                   "RETRY_RESULT_SEMANTICS",
                                   transcript=a_tr,
                                   dispatch=dispatch)
        if not isinstance(sci, dict) or \
                set(sci) != {"raw_body_preserved", "note"} or \
                not _is_hex64(sci.get("raw_body_preserved")) or \
                not sci.get("note"):
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transform refusal binds exactly one "
                    "preserved-body digest and a note, no classes")
        if sci["raw_body_preserved"] != a_tr.get("raw_body_sha256"):
            _refuse("RETRY_RESULT_SEMANTICS",
                    "the preserved body digest does not equal the "
                    "authenticated attempt transcript")
        if "ADMISSION_TRANSFORM_REFUSED" not in r["refusal"]:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a transform refusal must carry the typed "
                    "transform code")
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            tv.get("request_start_utc"),
                            tv.get("response_complete_utc"),
                            r["completed_utc"])
    elif oc == "CAPTURE_REFUSED_AFTER_TRANSPORT":
        a_tr = _attempt_obj(io, "transcript")
        tv = r.get("transport")
        if r.get("opener_calls") != 1 or sci != {}:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a post-transport capture refusal binds one "
                    "opener call and no scientific content")
        _validate_transport_member(tv, io, "response",
                                   "RETRY_RESULT_SEMANTICS",
                                   transcript=a_tr,
                                   dispatch=dispatch)
        _code = r["refusal"].split(":", 1)[0]
        if not _code.startswith(("CAPTURE_", "PRESTART_",
                                 "REFUSING")):
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a capture refusal must carry the capture "
                    "layer's typed vocabulary")
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            tv.get("request_start_utc"),
                            tv.get("response_complete_utc"),
                            r["completed_utc"])
    elif oc == "INTERNAL_ERROR_AFTER_TRANSPORT":
        a_tr = _attempt_obj(io, "transcript")
        tv = r.get("transport")
        if r.get("opener_calls") != 1 or sci != {}:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a post-transport internal error binds one "
                    "opener call and no scientific content")
        _validate_transport_member(tv, io, "response",
                                   "RETRY_RESULT_SEMANTICS",
                                   transcript=a_tr,
                                   dispatch=dispatch)
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            tv.get("request_start_utc"),
                            tv.get("response_complete_utc"),
                            r["completed_utc"])
    elif oc == "INTERNAL_ERROR_AFTER_DISPATCH":
        if r.get("transport") is not None:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a pre-transport internal error never carries "
                    "transport evidence")
        if r.get("opener_calls") not in (0, 1) or sci != {}:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "an internal error binds its opener count and "
                    "no scientific content")
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            r["completed_utc"])
    elif oc == "CAPTURE_REFUSED":
        # the pre-opener capture-phase refusal class
        if r.get("opener_calls") != 0 or \
                r.get("transport") is not None or sci != {}:
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a capture refusal is pre-opener: zero opener "
                    "calls, no transport, no scientific content")
        _code = r["refusal"].split(":", 1)[0]
        if not _code.startswith(("CAPTURE_", "PRESTART_",
                                 "REFUSING")):
            _refuse("RETRY_RESULT_SEMANTICS",
                    "a capture refusal must carry the capture "
                    "layer's typed vocabulary")
        _require_phase_order("RETRY_RESULT_SEMANTICS", d_utc,
                            r["completed_utc"])


def _validate_record(obj, fields, digest_field, schema, what):
    """Closed schema + recomputed self-digest, or a typed refusal.
    The self-digest is over the record minus its digest field, under
    the same canonical serialization every record here uses."""
    if not isinstance(obj, dict) or set(obj) != fields:
        got = sorted(set(obj) ^ fields) if isinstance(obj, dict) \
            else type(obj).__name__
        _refuse("RETRY_RECORD_NOT_CLOSED", f"{what}: {got}")
    if obj.get("schema") != schema:
        _refuse("RETRY_RECORD_SCHEMA", f"{what}: {obj.get('schema')}")
    body = {k: v for k, v in obj.items() if k != digest_field}
    if _canon_entry_digest(body) != obj.get(digest_field):
        _refuse("RETRY_RECORD_SELF_DIGEST",
                f"{what}: {digest_field} does not recompute")
    if obj.get("key") != TARGET_KEY:
        _refuse("RETRY_RECORD_WRONG_KEY", f"{what}: {obj.get('key')}")


def _write_json_create_once(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return CAP._write_once_json(path, obj, "RETRY_EVIDENCE_DIVERGENT")


def _create_exclusive_json(path, obj):
    """codex 1547Z repair 3: the DISPATCH create is exclusive -- the
    single os.link winner continues and EVERY FileExistsError refuses,
    even for byte-identical content. Two same-second callers built
    identical dispatch bytes and _write_once_json's identical-reuse
    rule let both proceed to the opener; the ceiling was per process,
    not per authorization. Exclusive-create makes dispatch existence
    consume the authorization for every caller but one."""
    import tempfile as _tf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = _tf.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)     # atomic create-once, no replace
        except FileExistsError:
            _refuse("RETRY_ALREADY_DISPATCHED",
                    "another dispatch won the exclusive create; this "
                    "caller must not proceed even if its bytes are "
                    "identical -- the authorization is consumed by "
                    "EXISTENCE, not by content")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _module_pin_binding(manifest_commit):
    """The retry module's OWN manifest pin (commit + blob), bound into
    the dispatch alongside the executed disk digest -- codex 1547Z
    repair 3. Fail-open here would let the dispatch under-bind, so a
    missing pin refuses (production requires the module BOUND anyway
    via precheck's allowlist)."""
    me = "monitoring/src/w2_capture_retry_404_v4_cayley.py"
    try:
        raw = RUN4._blob(manifest_commit, CAP.EXEC_MANIFEST_PATH)
        man = json.loads(raw.decode("utf-8"))
        for slot in man.get("slots", {}).values():
            for q in slot.get("pins", ()):
                if q.get("path") == me:
                    return {"pin_commit": q.get("commit"),
                            "pin_blob_sha256": q.get("blob_sha256")}
    except SystemExit:
        pass
    return {"pin_commit": None, "pin_blob_sha256": None,
            "note": "module not yet pinned at this manifest (KAT "
                    "fixtures only; production precheck refuses "
                    "RETRY_MODULE_UNBOUND first)"}


def fire(manifest_commit, _kat_io=None):
    # primitive 1: single resolution, first line -- everything below
    # sees only the immutable 40-hex object ID
    manifest_commit = _resolve_commit(manifest_commit)
    io = _production_io()
    if _kat_io is not None:
        io.update(_kat_io)
    ctx = precheck(manifest_commit, io)

    # ---- dispatch: created atomically BEFORE the opener is reachable.
    # Existence consumes the authorization.
    dispatch = {
        "schema": "f2g-w2-retry-404-dispatch-v1",
        "key": TARGET_KEY,
        "owner_authorization": OWNER_AUTH_REF,
        "contract": CONTRACT_REF,
        "original_ledger": {
            "path": "docs/f2g_window2_execution/"
                    "capture_run_ledger_v4.jsonl",
            "sha256": io["expect_ledger_sha"],
            "seq": ORIGINAL_SEQ,
            "entry_sha256": io["expect_entry_sha"]},
        "manifest_commit": manifest_commit,
        "manifest_blob_sha256": _sha(RUN4._blob(
            manifest_commit, CAP.EXEC_MANIFEST_PATH)),
        "capsule_pin_commit": ctx["pin_commit"],
        "capsule_sha256": ctx["capsule_sha"],
        "attempt_id": os.urandom(16).hex(),
        "executed_code": {
            "path": "monitoring/src/"
                    "w2_capture_retry_404_v4_cayley.py",
            # NORMALIZED source digest (finding 1) -- CRLF and LF
            # checkouts of the same pinned blob agree
            "disk_sha256": _norm_source_sha256(
                open(os.path.abspath(__file__), "rb").read()),
            **_module_pin_binding(manifest_commit)},
        "store": {"id": RUN4.STORE_ID, "root": RUN4.STORE_ROOT},
        "expected_classes": [io["stem"] + suf for suf in
                             ACC.STAGED_CLASS_SUFFIX.values()],
        "registered_request_url": io["expect_url"],
        "max_logical_http_operations": MAX_LOGICAL_HTTP_OPERATIONS,
        "vic_http_operations": 0,
        "dispatched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())}
    dispatch["dispatch_sha256"] = _canon_entry_digest(
        {k: v for k, v in dispatch.items() if k != "dispatch_sha256"})
    _create_exclusive_json(io["dispatch_path"], dispatch)

    opener = _CountingOpener(io["opener"], io, dispatch)
    os.makedirs(io["attempt_dir"], exist_ok=True)
    outcome, sci, refusal, prepared = None, {}, None, None
    capture_fn = io["capture_fn"]
    stem = io["stem"]
    prepared_path = os.path.join(io["retry_dir"],
                                 stem + ".prepared.json")

    def _phase():
        """codex 2017Z finding 1: the member is derived from PHASE,
        never from lexical nesting. DISPATCHED -> TRANSPORT_RECEIPTED
        (the mandatory receipt exists) -> PREPARED_DURABLE (the
        prepared create-once write succeeded)."""
        if os.path.exists(prepared_path):
            return "PREPARED_DURABLE"
        if os.path.isfile(_transport_receipt_path(io)):
            return "TRANSPORT_RECEIPTED"
        return "DISPATCHED"

    try:
        # ---- capture (transport + producer gate) ------------------
        if capture_fn is None:
            _rp, _tp, rec, tr = CAP.capture_authorized(
                REPO, manifest_commit, RUN4.AUTHORITY_PATH,
                TARGET_LANE, TARGET_CARRIER, TARGET_DAY,
                io["store_dir"], io["attempt_dir"], io["attempt_dir"],
                None, opener=opener,
                authority_reproducer=None)
        else:
            _rp, _tp, rec, tr = capture_fn(io, opener)
        # ---- post-response scientific phase -----------------------
        with open(os.path.join(
                io["store_dir"],
                rec["raw_body_sha256"] + ".body"), "rb") as f:
            body = f.read()
        s = ctx["static"]
        # the transform handler spans the TRANSFORM CALL ONLY
        # (codex 1830Z residual 3)
        art = None
        try:
            art = CAP.admission_transform(TARGET_LANE, body, s)
        except CAP.CaptureRefusal as exc:
            refusal = str(exc)
            outcome = "TRANSFORM_REFUSED"
            sci = {"raw_body_preserved": rec["raw_body_sha256"],
                   "note": "body persisted in the named store; NO "
                           "partial class set published; any repair "
                           "is zero-HTTP and separately reviewed -- "
                           "this authorization is spent"}
        if art is not None:
            for cls, obj in (("contract", s), ("artifact", art),
                             ("record", rec), ("transcript", tr)):
                _write_json_create_once(
                    os.path.join(
                        io["attempt_dir"],
                        stem + ACC.STAGED_CLASS_SUFFIX[cls]),
                    obj)
            # the PREPARED record: trust root + the complete
            # operation projection measured at this instant
            # (codex 1705Z repair 2 + 1830Z residual 1)
            with open(io["ledger_path"], "rb") as f:
                _led = _sha(f.read())
            prepared = {
                "schema": "f2g-w2-retry-prepared-v1",
                "key": TARGET_KEY,
                "dispatch_sha256": dispatch["dispatch_sha256"],
                "outcome": "CAPTURED_ADMITTED",
                "class_canon_sha256": {
                    cls: PROD._canon_digest(obj)
                    for cls, obj in
                    (("contract", s), ("artifact", art),
                     ("record", rec), ("transcript", tr))},
                "opener_calls": opener.calls,
                "transport": opener.evidence,
                "terminal_ledger_sha256_recomputed": _led,
                "terminal_ledger_unchanged":
                    _led == io["expect_ledger_sha"],
                "completed_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            prepared["prepared_sha256"] = _canon_entry_digest(
                {k: v for k, v in prepared.items()
                 if k != "prepared_sha256"})
            _validate_prepared_semantics(prepared, io,
                                         dispatch=dispatch,
                                         transcript=tr,
                                         record=rec)
            _write_json_create_once(prepared_path, prepared)
            # once PREPARED_DURABLE, a publication failure is a
            # RESUMABLE typed state -- never a contradictory
            # terminal result (codex 1830Z residual 3)
            try:
                pub = finalize_publication(io, prepared)
            except BaseException as exc:
                _refuse(
                    "RETRY_PUBLICATION_BLOCKED_AFTER_PREPARED",
                    "publication failed after the prepared "
                    f"record was minted ({type(exc).__name__}: "
                    f"{str(exc)[:200]}); no terminal result is "
                    "written and the state is resumable via "
                    "finalize()")
            sci = {"classes_published": pub,
                   "raw_body_sha256": rec["raw_body_sha256"],
                   "proof_kind": "NATIVE_V4_CAPTURE",
                   "required_next": "store re-freeze + inventory "
                                    "rebuild (grassmann) before "
                                    "any boundary claim"}
            outcome = "CAPTURED_ADMITTED"
    except (RetryRefusal, SystemExit):
        raise
    except AssertionError:
        raise
    except CAP.CaptureRefusal as exc:
        refusal = str(exc)
        # classify by the EXACT leading typed code and by phase
        # (codex 1906Z finding 3)
        _code = refusal.split(":", 1)[0]
        if _code == "CAPTURE_HTTP_STATUS":
            outcome = "TRANSPORT_REFUSED_NON_200"
        elif "ADMISSION_TRANSFORM_REFUSED" in refusal:
            # production capture_authorized runs the registered
            # transform before returning (codex 1547Z repair 5A)
            outcome = "TRANSFORM_REFUSED"
            _stem_t = io["stem"] + ACC.STAGED_CLASS_SUFFIX[
                "transcript"]
            _tpx = os.path.join(io["attempt_dir"], _stem_t)
            if os.path.isfile(_tpx):
                with open(_tpx, encoding="utf-8") as _f:
                    _t = json.load(_f)
                sci = {"raw_body_preserved":
                       _t.get("raw_body_sha256"),
                       "note": "body persisted before the transform "
                               "refused; NO class set published; any "
                               "repair is zero-HTTP and separately "
                               "reviewed -- this authorization is "
                               "spent"}
        elif _phase() == "TRANSPORT_RECEIPTED":
            outcome = "CAPTURE_REFUSED_AFTER_TRANSPORT"
        else:
            outcome = "CAPTURE_REFUSED"
    except RetryTransportError as exc:
        refusal = str(exc)
        outcome = "TRANSPORT_ERROR"
        if opener.evidence is None:
            opener.evidence = exc.evidence
    except Exception as exc:
        # codex 2017Z finding 1: EVERY ordinary local exception in
        # the whole post-dispatch operation terminalizes exactly once
        # by PHASE -- stored-body reopen, a non-CaptureRefusal
        # transform error, attempt-class writes and the prepared
        # write all land here. While TRANSPORT_RECEIPTED and before
        # PREPARED_DURABLE the member is INTERNAL_ERROR_AFTER_
        # TRANSPORT with the receipt preserved and no classes; a
        # post-PREPARED interruption is unreachable here (the
        # publication block above converts it to the resumable typed
        # refusal first).
        refusal = f"{type(exc).__name__}: {str(exc)[:300]}"
        if _phase() == "TRANSPORT_RECEIPTED":
            outcome = "INTERNAL_ERROR_AFTER_TRANSPORT"
            if opener.evidence is None:
                _rc = _read_transport_receipt(io)
                opener.evidence = dict(_rc["evidence"]) if _rc else None
        else:
            outcome = "INTERNAL_ERROR_AFTER_DISPATCH"

    # ---- result + one-line retry ledger index. The frozen terminal
    # ledger digest is recomputed and bound for EVERY outcome.
    with open(io["ledger_path"], "rb") as f:
        led_now = _sha(f.read())
    result = {
        "schema": "f2g-w2-retry-404-result-v1",
        "key": TARGET_KEY,
        "dispatch_sha256": dispatch["dispatch_sha256"],
        "outcome": outcome,
        "refusal": refusal,
        "opener_calls": opener.calls,
        "transport": opener.evidence,
        "scientific": sci,
        "terminal_ledger_sha256_recomputed": led_now,
        "terminal_ledger_unchanged":
            led_now == io["expect_ledger_sha"],
        # residual 1: on the admitted path the result reuses the ONE
        # completion instant the prepared record bound, so a
        # crash-finalized result and a live result carry the same
        # operation fact
        "completed_utc": (prepared["completed_utc"]
                          if outcome == "CAPTURED_ADMITTED"
                          and prepared is not None
                          else time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()))}
    result["result_sha256"] = _canon_entry_digest(
        {k: v for k, v in result.items() if k != "result_sha256"})
    # codex 1845Z: the live result passes the same semantic-join
    # layer the finalizer applies, before create-once publication
    _validate_result_semantics(
        result, io, dispatch=dispatch, prepared=prepared,
        record=(_attempt_obj(io, "record")
                if outcome == "CAPTURED_ADMITTED" else None))
    _write_json_create_once(io["result_path"], result)
    _finalize_index(io, dispatch, result)
    print(f"RETRY {outcome}: opener_calls={opener.calls}; terminal "
          f"ledger unchanged={result['terminal_ledger_unchanged']}")
    return result


def _index_entry_of(dispatch, result):
    """The one-line index, RECONSTRUCTED from dispatch+result bytes --
    the only inputs -- so a crash-lost index is recoverable and a
    hand-edited one is detectable (codex 1547Z repair 5B)."""
    return {"key": result["key"], "outcome": result["outcome"],
            "dispatch_sha256": dispatch["dispatch_sha256"],
            "result_sha256": result["result_sha256"],
            "opener_calls": result["opener_calls"],
            "http_operations_authorized":
                MAX_LOGICAL_HTTP_OPERATIONS}


def _finalize_index(io, dispatch, result):
    """Create-once / idempotent: absent -> exclusive create of the
    ONE-line closed ledger; present-identical -> no-op; divergent,
    duplicate or truncated -> typed refusal. codex 1830Z residual 2:
    the CURRENT frozen-ledger bytes must recompute to the registered
    digest before the operation index is created or accepted."""
    with open(io["ledger_path"], "rb") as f:
        _cur = _sha(f.read())
    if _cur != io["expect_ledger_sha"]:
        _refuse("RETRY_TERMINAL_LEDGER_CHANGED",
                f"current ledger bytes recompute {_cur[:16]}, the "
                f"registered digest is "
                f"{io['expect_ledger_sha'][:16]} -- no retry index "
                "may be created or accepted over a changed frozen "
                "ledger")
    want = json.dumps(_index_entry_of(dispatch, result),
                      sort_keys=True)
    path = io["retry_ledger"]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [x.strip() for x in f if x.strip()]
        if lines != [want]:
            _refuse("RETRY_INDEX_DIVERGENT",
                    f"{len(lines)} line(s) that do not equal the "
                    "reconstruction from dispatch+result bytes")
        return
    import tempfile as _tf
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = _tf.mkstemp(dir=os.path.dirname(path) or ".",
                          suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(want + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            with open(path, encoding="utf-8") as f:
                lines = [x.strip() for x in f if x.strip()]
            if lines != [want]:
                _refuse("RETRY_INDEX_DIVERGENT",
                        "a concurrent index write diverges from the "
                        "reconstruction")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def finalize_publication(io, prepared):
    """ZERO NETWORK, IDEMPOTENT, AUTHENTICATED (codex 1705Z repairs
    2+3): every attempt-local object must recompute to the canonical
    digest the PREPARED record bound BEFORE the first publication,
    the full S/T/E five-map join must pass over the reopened store
    body, and any existing completion marker must bind the same map.
    Only then are the four classes published write-once (identical
    reuse, divergent refuse) and the marker created. Nothing this
    function publishes can differ from what the prepared record
    authenticated at spend time."""
    stem = io["stem"]
    objs = {}
    for cls, suf in ACC.STAGED_CLASS_SUFFIX.items():
        ap = os.path.join(io["attempt_dir"], stem + suf)
        if not os.path.isfile(ap):
            _refuse("RETRY_ATTEMPT_EVIDENCE_INCOMPLETE",
                    f"attempt-local {cls} missing at {ap}; the "
                    "class set is published all-or-nothing from "
                    "verified attempt evidence")
        with open(ap, encoding="utf-8") as f:
            objs[cls] = json.load(f)
    want = prepared["class_canon_sha256"]
    for cls in ACC.STAGED_CLASS_SUFFIX:
        got = PROD._canon_digest(objs[cls])
        if got != want.get(cls):
            _refuse("RETRY_ATTEMPT_OBJECT_DIVERGENT",
                    f"attempt-local {cls} recomputes {got[:12]}, "
                    f"prepared bound {str(want.get(cls))[:12]}")
    # the SAME S/T/E + raw-body join admission uses, over the
    # reopened store body -- four individually-valid objects that do
    # not join must refuse before the first staged write
    rec = objs["record"]
    body_path = os.path.join(io["store_dir"],
                             str(rec.get("raw_body_sha256")) + ".body")
    if not os.path.isfile(body_path):
        _refuse("RETRY_STORE_BODY_ABSENT",
                f"no store body for the prepared record at "
                f"{body_path}")
    with open(body_path, "rb") as f:
        body = f.read()
    if _sha(body) != rec.get("raw_body_sha256"):
        _refuse("RETRY_STORE_BODY_DIVERGENT",
                "store body does not recompute to the record digest")
    day = TARGET_DAY
    try:
        PROD.verify_staged_day_set(
            {day: rec}, {day: body}, {day: objs["artifact"]},
            {day: objs["contract"]}, {day: objs["transcript"]},
            [day], TARGET_CARRIER, TARGET_LANE)
    except Exception as exc:
        _refuse("RETRY_JOIN_FAILED",
                f"the S/T/E five-map join refused the prepared set "
                f"({type(exc).__name__}: {str(exc)[:160]})")
    marker_path = os.path.join(io["retry_dir"],
                               stem + ".classes_complete.json")
    completion = {"schema": "f2g-w2-retry-classes-complete-v1",
                  "key": TARGET_KEY,
                  "class_canon_sha256": dict(want)}
    if os.path.exists(marker_path):
        with open(marker_path, encoding="utf-8") as f:
            have = json.load(f)
        # codex 1830Z residual 3B: the ENTIRE closed object must
        # match -- a right-map/wrong-schema marker previously passed
        # here and refused only after all four classes were written
        if have != completion:
            _refuse("RETRY_MARKER_DIVERGENT",
                    "an existing completion marker does not equal "
                    "the entire expected closed object "
                    "(schema/key/map)")
    pub = {}
    for cls, suf in ACC.STAGED_CLASS_SUFFIX.items():
        _write_json_create_once(
            os.path.join(io["staged_dir"], stem + suf), objs[cls])
        pub[cls] = want[cls]
    _write_json_create_once(marker_path, completion)
    return pub


def finalize(_kat_io=None):
    """ZERO NETWORK recovery entrypoint (codex 1547Z repairs 4+5B):
    from preserved dispatch/result/attempt evidence, finish whatever
    an interruption left unfinished -- class publication (only for a
    CAPTURED_ADMITTED result) and the one-line index. Never opens a
    socket, never consumes or grants authorization, refuses on any
    divergence."""
    io = _production_io()
    if _kat_io is not None:
        io.update(_kat_io)
    stem = io["stem"]
    prepared_path = os.path.join(io["retry_dir"],
                                 stem + ".prepared.json")
    if not os.path.exists(io["dispatch_path"]):
        _refuse("RETRY_NOTHING_TO_FINALIZE", "no dispatch record")
    with open(io["dispatch_path"], encoding="utf-8") as f:
        dispatch = json.load(f)
    # codex 1705Z repair 3: every record authenticates -- closed
    # schema, recomputed self-digest, exact linkage -- BEFORE any
    # publish. finalize never invents an outcome: dispatch alone is
    # INDETERMINATE; the outcome comes only from an authenticated
    # prepared or result record.
    _validate_record(dispatch, DISPATCH_FIELDS, "dispatch_sha256",
                     "f2g-w2-retry-404-dispatch-v1", "dispatch")
    _validate_dispatch_semantics(dispatch, io)
    prepared = None
    if os.path.exists(prepared_path):
        with open(prepared_path, encoding="utf-8") as f:
            prepared = json.load(f)
        _validate_record(prepared, PREPARED_FIELDS, "prepared_sha256",
                         "f2g-w2-retry-prepared-v1", "prepared")
        _validate_prepared_semantics(prepared, io, dispatch=dispatch)
        if prepared["dispatch_sha256"] != dispatch["dispatch_sha256"]:
            _refuse("RETRY_EVIDENCE_INCONSISTENT",
                    "prepared does not bind this dispatch")
    result = None
    if os.path.exists(io["result_path"]):
        with open(io["result_path"], encoding="utf-8") as f:
            result = json.load(f)
        _validate_record(result, RESULT_FIELDS, "result_sha256",
                         "f2g-w2-retry-404-result-v1", "result")
        _validate_result_semantics(
            result, io, dispatch=dispatch, prepared=prepared,
            record=(_attempt_obj(io, "record")
                    if result.get("outcome") == "CAPTURED_ADMITTED"
                    else None))
        if result["dispatch_sha256"] != dispatch["dispatch_sha256"]:
            _refuse("RETRY_EVIDENCE_INCONSISTENT",
                    "result does not bind this dispatch")
    if result is None and prepared is None:
        _refuse("RETRY_INDETERMINATE_AFTER_DISPATCH",
                "a dispatch record exists with no prepared record "
                "and no result; finalize cannot invent an outcome "
                "and this is NOT permission to retry")
    if result is not None and prepared is not None and \
            result["outcome"] != prepared["outcome"]:
        _refuse("RETRY_EVIDENCE_INCONSISTENT",
                f"result outcome {result['outcome']!r} != prepared "
                f"outcome {prepared['outcome']!r}")
    done = {"classes": None}
    outcome = (result or prepared)["outcome"]
    if outcome == "CAPTURED_ADMITTED":
        if prepared is None:
            _refuse("RETRY_EVIDENCE_INCONSISTENT",
                    "a CAPTURED_ADMITTED result has no prepared "
                    "record to authenticate the class set against")
        done["classes"] = finalize_publication(io, prepared)
    if result is None:
        # codex 1830Z residual 1: the prepared record BOUND the full
        # operation projection at spend time -- opener count,
        # transport evidence, ledger binding, completion instant.
        # The reconstructed result carries THOSE measured facts,
        # never null and never "lost". raw_body_sha256 comes from
        # the attempt record already authenticated against the
        # prepared digest map by finalize_publication above.
        rec_path = os.path.join(
            io["attempt_dir"],
            stem + ACC.STAGED_CLASS_SUFFIX["record"])
        with open(rec_path, encoding="utf-8") as f:
            _arec = json.load(f)
        result = {
            "schema": "f2g-w2-retry-404-result-v1",
            "key": TARGET_KEY,
            "dispatch_sha256": dispatch["dispatch_sha256"],
            "outcome": prepared["outcome"],
            "refusal": None,
            "opener_calls": prepared["opener_calls"],
            "transport": prepared["transport"],
            "scientific": {"classes_published": done["classes"],
                           "raw_body_sha256":
                               _arec.get("raw_body_sha256"),
                           "proof_kind": "NATIVE_V4_CAPTURE",
                           "required_next":
                               "store re-freeze + inventory rebuild "
                               "(grassmann) before any boundary "
                               "claim"},
            "terminal_ledger_sha256_recomputed":
                prepared["terminal_ledger_sha256_recomputed"],
            "terminal_ledger_unchanged":
                prepared["terminal_ledger_unchanged"],
            "completed_utc": prepared["completed_utc"]}
        result["result_sha256"] = _canon_entry_digest(
            {k: v for k, v in result.items()
             if k != "result_sha256"})
        _validate_result_semantics(
            result, io, dispatch=dispatch, prepared=prepared,
            record=_arec)
        _write_json_create_once(io["result_path"], result)
    _finalize_index(io, dispatch, result)
    done["index"] = "present"
    print(f"RETRY FINALIZE: outcome={outcome} "
          f"classes={'published' if done['classes'] else 'n/a'} "
          "index=present")
    return done


def plan(manifest_commit="HEAD"):
    """ZERO NETWORK: run every precheck and report. No dispatch is
    written; nothing is consumed."""
    manifest_commit = _resolve_commit(manifest_commit)
    io = _production_io()
    ctx = precheck(manifest_commit, io)
    print(f"target        {TARGET_KEY} (seq {ORIGINAL_SEQ})")
    print(f"authorized    {OWNER_AUTH_REF}")
    print(f"plan          635 keys, target in HTTP_CAPTURE: True")
    print(f"ledger        635 lines, digest bound, one 404 entry")
    print(f"url           derived == registered: True")
    print("no request fired")
    return ctx


# ------------------------------------------------------------------ #
# PRE-FIRE BARS (codex 1345Z "Required pre-fire bars"). Fixture io
# only ever enters through _selftest; the CLI has no injection path.
# ------------------------------------------------------------------ #
def _selftest():
    import tempfile
    import w2_no_network_grassmann as NONET
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    def refuses(fn, code):
        # matches on the TYPED CODE, whatever the carrier class --
        # publication divergence arrives as CAP.CaptureRefusal, the
        # module's own refusals as RetryRefusal/SystemExit
        try:
            fn()
        except BaseException as exc:
            return code in str(exc)
        return False

    # -- fixture terminal ledger: 635 rows whose ONLY load-bearing row
    # is the target 404, byte-frozen for digest binding
    def fixture_rows():
        rows = []
        for i in range(635):
            rows.append({"key": f"MAG_FEED/pad/2026-01-{i % 28 + 1:02d}",
                         "seq": i, "status": "CAPTURED"})
        rows[ORIGINAL_SEQ] = {
            "key": TARGET_KEY, "seq": ORIGINAL_SEQ,
            "status": "REFUSED",
            "refusal": "CAPTURE_HTTP_STATUS: 404 for "
                       + REGISTERED_REQUEST_URL}
        return rows

    def write_ledger(td, rows):
        p = os.path.join(td, "terminal_ledger.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        return p

    def kat_io(td, rows=None, **over):
        rows = fixture_rows() if rows is None else rows
        lp = write_ledger(td, rows)
        with open(lp, "rb") as f:
            lsha = _sha(f.read())
        entry = [r for r in rows if r.get("key") == TARGET_KEY]
        esha = _canon_entry_digest(entry[0]) if entry else "0" * 64
        io = {"stem": CAP._path_tokens(TARGET_LANE, TARGET_CARRIER,
                                       TARGET_DAY),
              "ledger_path": lp,
              "staged_dir": os.path.join(td, "staged"),
              "store_dir": os.path.join(td, "store"),
              "retry_dir": os.path.join(td, "retry"),
              "dispatch_path": os.path.join(td, "retry",
                                            DISPATCH_BASENAME),
              "result_path": os.path.join(td, "retry",
                                          RESULT_BASENAME),
              "retry_ledger": os.path.join(td, "retry_ledger.jsonl"),
              "attempt_dir": os.path.join(td, "attempt"),
              "allowlist_fn": lambda repo, mc: {
                  "pins": [("accrual_impl",
                            "monitoring/src/"
                            "w2_capture_retry_404_v4_cayley.py")]},
              "expect_ledger_sha": lsha,
              "expect_entry_sha": esha,
              # explicit INTERNAL KAT context (codex 1906Z finding
              # 4): only this out-of-band flag permits the unpinned
              # executed-code form; production io never sets it
              "_kat_allow_unpinned": True}
        io.update(over)
        for d in (io["staged_dir"], io["store_dir"], io["retry_dir"],
                  io["attempt_dir"]):
            os.makedirs(d, exist_ok=True)
        return io

    # -- the registered FIXTURE capture path (capture_day), exactly as
    # the window-2 bar's positive cases drive it. capture_authorized
    # cannot run green pre-landing (it requires THIS module bound).
    def capture_via_fixture(io, opener, ctx_static):
        def artifact_builder(body):
            return CAP.admission_transform(TARGET_LANE, body,
                                           ctx_static)
        spec = dict(ctx_static)
        return CAP.capture_day(
            spec, io["store_dir"], io["attempt_dir"],
            io["attempt_dir"], artifact_builder, opener=opener)

    # -- synthetic USGS body that legitimately passes the REAL
    # registered transform (shape mirrored from CAP's own selftest
    # fixture builder usgs_body(); 1440 minute samples, iaga NEW)
    def usgs_new_body():
        from datetime import datetime as _dt, timedelta as _td
        d0 = _dt.fromisoformat(TARGET_DAY + "T00:00:00")
        times = [(d0 + _td(minutes=i)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z") for i in range(1440)]
        n = 1440
        return json.dumps({
            "type": "Timeseries",
            "metadata": {"intermagnet": {"imo": {"iaga_code": "NEW"},
                                         "reported_orientation":
                                             "XYZF"}},
            "times": times,
            "values": [{"id": "X", "values": [1.0] * n},
                       {"id": "Y", "values": [2.0] * n},
                       {"id": "Z", "values": [3.0] * n},
                       {"id": "F", "values": [4.0] * n}]}).encode()

    def opener_200(url):
        # the REAL lowercase receipt headers incl content-length
        # (codex 1906Z finding 2): the transcript receipts all four,
        # and the evidence must carry them equal
        body = usgs_new_body()
        return 200, {"content-type": "application/json",
                     "content-length": str(len(body)),
                     "date": "kat", "server": "kat"}, body, url

    def opener_404(url):
        return 404, {}, b"", url

    def opener_boom(url):
        # message == evidence.error, the production boundary invariant
        raise RetryTransportError(
            "ConnectionRefusedError: connection refused",
            {"requested_url": url,
             "request_start_utc": time.strftime(
                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "error": "ConnectionRefusedError: connection refused"})

    def opener_local_boom(url):
        # a LOCAL failure inside the opener (missing CA bundle /
        # config) -- plain exception, NOT boundary-typed
        raise FileNotFoundError("local certifi CA bundle missing")

    # real authority for the REAL static contract + URL derivation
    authority, _c, _k, _n, _s5 = RUN4.load_plan("HEAD")
    s_real = ACC.authoritative_static_contract(
        authority, TARGET_LANE, TARGET_CARRIER, TARGET_DAY)
    # capture_day takes the CLOSED nine-key spec (its own contract),
    # so the full static contract is projected down to it; source is
    # identity-only {kind, ref} per the same contract
    spec_real = {k: s_real[k] for k in
                 ("lane", "carrier", "utc_day", "endpoint",
                  "request_params", "source", "cutoff",
                  "operation_params", "expected_keys")}
    spec_real["source"] = {k: s_real["source"][k]
                           for k in ("kind", "ref")}

    def spec_full_s():
        return s_real

    # ---- U1 check (5) runs fully REAL: authority -> S -> URL equals
    # the registered constant taken from the actual ledger refusal
    check("U1 derived static-contract URL == registered request URL "
          "(real authority, real derivation, zero fixture)",
          PROD.requested_url_of(s_real["endpoint"],
                                s_real["request_params"])
          == REGISTERED_REQUEST_URL)

    with NONET.no_network():
        # ---- B-refusals: every doctored precondition refuses BEFORE
        # any opener, with the opener a tripwire
        def trip(url):
            raise AssertionError("OPENER REACHED")
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td, expect_ledger_sha="0" * 64)
            check("B1 wrong terminal-ledger digest refuses pre-opener",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_TERMINAL_LEDGER_DIGEST_DIVERGENT"))
        with tempfile.TemporaryDirectory() as td:
            rows = fixture_rows()
            rows[ORIGINAL_SEQ]["seq"] = 82
            io = kat_io(td, rows=rows)
            check("B2 wrong original seq refuses pre-opener",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_ORIGINAL_SEQ_DIVERGENT"))
        with tempfile.TemporaryDirectory() as td:
            rows = fixture_rows()
            rows[ORIGINAL_SEQ + 1] = dict(
                rows[ORIGINAL_SEQ], seq=ORIGINAL_SEQ + 1)
            io = kat_io(td, rows=rows)
            check("B3 duplicate original entry refuses pre-opener",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_ORIGINAL_ENTRY_COUNT"))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td, allowlist_fn=lambda repo, mc: {
                "pins": [("accrual_impl", "monitoring/src/other.py")]})
            check("B4 unbound executed module refuses pre-opener",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_MODULE_UNBOUND"))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            with open(os.path.join(
                    io["staged_dir"],
                    CAP._path_tokens(TARGET_LANE, TARGET_CARRIER,
                                     TARGET_DAY)
                    + ".record.json"), "w") as f:
                f.write("{}")
            check("B5 pre-existing published target class refuses "
                  "pre-opener",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_TARGET_CLASS_ALREADY_PUBLISHED"))

        # ---- F1 fake 404: transport-only result, zero scientific
        # classes, dispatch consumed, terminal ledger digest unchanged
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"], i["attempt_dir"],
                    i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, s_real), opener=o)))
            staged_n = len(os.listdir(io["staged_dir"]))
            check("F1 second 404 -> transport-only result, zero "
                  "scientific classes, ledger digest unchanged",
                  r["outcome"] == "TRANSPORT_REFUSED_NON_200"
                  and r["opener_calls"] == 1 and staged_n == 0
                  and r["terminal_ledger_unchanged"] is True
                  and r["scientific"] == {},
                  f"outcome={r['outcome']} staged={staged_n}")
            # F1b the spent authorization refuses a second invocation
            check("F1b crash-free re-invocation refuses: authorization "
                  "spent",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_ALREADY_EXECUTED"))
        # ---- F2 connection failure: typed transport error
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_boom,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"], i["attempt_dir"],
                    i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, s_real), opener=o)))
            check("F2 connection failure -> typed TRANSPORT_ERROR, "
                  "no scientific classes",
                  r["outcome"] == "TRANSPORT_ERROR"
                  and len(os.listdir(io["staged_dir"])) == 0)
        # ---- F3 fake 200: one opener call, REAL transform over a
        # valid synthetic USGS NEW body, all four classes published
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"], i["attempt_dir"],
                    i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, s_real), opener=o)))
            stem = io["stem"] if "stem" in io else \
                CAP._path_tokens(TARGET_LANE, TARGET_CARRIER,
                                 TARGET_DAY)
            have = sorted(os.listdir(io["staged_dir"]))
            check("F3 fake 200 -> opener called exactly once, four "
                  "classes published, outcome CAPTURED_ADMITTED",
                  r["outcome"] == "CAPTURED_ADMITTED"
                  and r["opener_calls"] == 1 and len(have) == 4
                  and r["scientific"]["proof_kind"]
                  == "NATIVE_V4_CAPTURE",
                  f"outcome={r['outcome']} classes={have}")
            # F3b dispatch-without-result -> INDETERMINATE (constructed
            # by removing the result, keeping the dispatch)
            os.remove(io["result_path"])
            check("F3b dispatch with no result refuses "
                  "INDETERMINATE_AFTER_DISPATCH pre-opener",
                  refuses(lambda: fire("HEAD", dict(
                      io, opener=trip, capture_fn=lambda i, o: None)),
                      "RETRY_INDETERMINATE_AFTER_DISPATCH"))
        # ---- R3 the dispatch race: two barriered workers, ONE wins
        # (codex 1547Z repair 3). The loser refuses typed even though
        # its dispatch bytes could be identical; exactly one opener
        # call happens globally.
        import threading
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            calls = []

            def opener_count(url):
                calls.append(1)
                return 404, {}, b"", url
            barrier = threading.Barrier(2)
            results = [None, None]

            def worker(i):
                def gated_capture(cio, o):
                    return CAP.capture_day(
                        dict(spec_real), cio["store_dir"],
                        cio["attempt_dir"], cio["attempt_dir"],
                        lambda b: CAP.admission_transform(
                            TARGET_LANE, b, spec_full_s()),
                        opener=o)
                try:
                    barrier.wait(timeout=10)
                    results[i] = ("OK", fire("HEAD", dict(
                        io, opener=opener_count,
                        capture_fn=gated_capture))["outcome"])
                except (RetryRefusal, SystemExit) as exc:
                    results[i] = ("REFUSED", str(exc)[:60])
                except Exception as exc:  # noqa: BLE001
                    results[i] = ("ERROR",
                                  f"{type(exc).__name__}: "
                                  f"{str(exc)[:80]}")
            ts = [threading.Thread(target=worker, args=(i,))
                  for i in (0, 1)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=60)
            kinds = sorted(k for k, _v in results)
            losers = [v for k, v in results if k == "REFUSED"]
            check("R3 race: one winner, one RETRY_ALREADY_DISPATCHED "
                  "loser, exactly one opener call globally",
                  kinds == ["OK", "REFUSED"] and len(calls) == 1
                  and any("RETRY_ALREADY_DISPATCHED" in x
                          for x in losers),
                  f"results={results} calls={len(calls)}")

        # ---- R5A invalid 200: the REAL transform refuses a 200 body
        # (wrong observatory), outcome is typed TRANSFORM_REFUSED with
        # the preserved body digest, zero staged classes
        def opener_bad200(url):
            return 200, {}, b'{"values": "not-a-channel-list"}', url
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_bad200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"], i["attempt_dir"],
                    i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("R5a invalid 200 -> typed TRANSFORM_REFUSED, "
                  "preserved body digest bound, zero staged classes",
                  r["outcome"] == "TRANSFORM_REFUSED"
                  and len(os.listdir(io["staged_dir"])) == 0
                  and len(str(r["scientific"]
                              .get("raw_body_preserved", ""))) == 64,
                  f"outcome={r['outcome']} sci={r['scientific']}")

        # ---- R4 CAUSAL publication interruption (codex 1705Z
        # repair 2): a fault is injected INSIDE the real fire() path
        # after N staged writes -- not constructed post-hoc -- and
        # the real public finalize() must carry every state to the
        # same classes/marker/result/index, with the no-network
        # sentinel enforcing zero opener reach throughout.
        _g2 = globals()
        _real_writer = _g2["_write_json_create_once"]
        stem = CAP._path_tokens(TARGET_LANE, TARGET_CARRIER,
                                TARGET_DAY)
        all4 = sorted(stem + suf for suf in
                      ACC.STAGED_CLASS_SUFFIX.values())
        marker_maps = []
        ok_causal = True
        for crash_after in (0, 1, 2, 3):
            with tempfile.TemporaryDirectory() as td:
                io = kat_io(td)
                staged_seen = []

                def _crashing_writer(path, obj, _n=crash_after,
                                     _sd=io["staged_dir"]):
                    if os.path.dirname(path) == _sd:
                        if len(staged_seen) >= _n:
                            raise RuntimeError(
                                "KAT_INJECTED_CRASH mid-publication")
                        staged_seen.append(path)
                    return _real_writer(path, obj)
                _g2["_write_json_create_once"] = _crashing_writer
                crashed = False
                try:
                    fire("HEAD", dict(
                        io, opener=opener_200,
                        capture_fn=lambda i, o: CAP.capture_day(
                            dict(spec_real), i["store_dir"],
                            i["attempt_dir"], i["attempt_dir"],
                            lambda b: CAP.admission_transform(
                                TARGET_LANE, b, spec_full_s()),
                            opener=o)))
                except (RuntimeError, SystemExit) as exc:
                    # residual 3: fire converts the mid-publication
                    # crash into the typed RESUMABLE refusal and
                    # writes NO terminal result
                    crashed = ("PUBLICATION_BLOCKED_AFTER_PREPARED"
                               in str(exc))
                finally:
                    _g2["_write_json_create_once"] = _real_writer
                crashed = crashed and                     not os.path.exists(io["result_path"])
                out = finalize(dict(io))
                # residual 1: the reconstructed result preserves the
                # operation projection the prepared record bound
                with open(io["result_path"],
                          encoding="utf-8") as f:
                    _r = json.load(f)
                crashed = crashed and _r["opener_calls"] == 1                     and isinstance(_r["transport"], dict)                     and _r["transport"].get("status") == 200                     and _r["terminal_ledger_unchanged"] is True
                marker = os.path.join(io["retry_dir"],
                                      stem
                                      + ".classes_complete.json")
                with open(marker, encoding="utf-8") as f:
                    _mm = json.load(f)["class_canon_sha256"]
                # transcripts/records carry per-run clock reads, so
                # cross-state identity holds only for the
                # DETERMINISTIC classes; per-state marker==prepared
                # identity is enforced by finalize itself
                marker_maps.append(json.dumps(
                    {k: _mm[k] for k in ("contract", "artifact")},
                    sort_keys=True))
                n_idx = sum(1 for x in open(io["retry_ledger"],
                                            encoding="utf-8")
                            if x.strip())
                ok_causal = ok_causal and crashed \
                    and out["classes"] is not None \
                    and sorted(os.listdir(io["staged_dir"])) == all4 \
                    and os.path.exists(io["result_path"]) \
                    and n_idx == 1
        check("R4 causal: crash after 0/1/2/3 staged writes -> typed "
              "resumable refusal, NO terminal result; the real "
              "public finalize() completes classes+marker+result+"
              "index preserving opener_calls=1, the transport "
              "projection and the ledger binding", ok_causal)
        check("R4a all four crash states converge to identical "
              "deterministic-class digests (contract+artifact; "
              "transcript/record carry per-run clock reads)",
              len(set(marker_maps)) == 1)

        # ---- R3 negatives (codex 1705Z repair 3): finalize
        # authenticates everything before any publish
        def _mk_complete(td):
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            assert r["outcome"] == "CAPTURED_ADMITTED"
            return io

        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            # (a) doctored dispatch field, self-digest kept
            with open(io["dispatch_path"], encoding="utf-8") as f:
                d = json.load(f)
            d["contract"] = "doctored"
            with open(io["dispatch_path"], "w",
                      encoding="utf-8") as f:
                json.dump(d, f)
            check("N3a a doctored record whose self-digest no longer "
                  "recomputes refuses",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_RECORD_SELF_DIGEST"))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            # (b) prepared outcome changed (self-digest recomputed) vs
            # the standing result
            pp = os.path.join(io["retry_dir"],
                              stem + ".prepared.json")
            with open(pp, encoding="utf-8") as f:
                pr = json.load(f)
            pr["outcome"] = "TRANSFORM_REFUSED"
            pr["prepared_sha256"] = _canon_entry_digest(
                {k: v for k, v in pr.items()
                 if k != "prepared_sha256"})
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(pr, f)
            check("N3b a changed prepared outcome refuses (semantic "
                  "validator fires first; the linkage check is the "
                  "backstop)",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_PREPARED_SEMANTICS")
                  or refuses(lambda: finalize(dict(io)),
                             "RETRY_EVIDENCE_INCONSISTENT"))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            # (c) swapped attempt object: valid JSON, wrong digest
            ap = os.path.join(io["attempt_dir"],
                              stem + ACC.STAGED_CLASS_SUFFIX[
                                  "artifact"])
            with open(ap, "w", encoding="utf-8") as f:
                f.write('{"swapped": true}')
            check("N3c a swapped attempt-local object refuses against "
                  "the prepared digest map",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_ATTEMPT_OBJECT_DIVERGENT"))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            # (d) four individually-valid objects whose CROSS-OBJECT
            # join fails: doctor the attempt transcript's body digest
            # AND recompute the prepared map + self-digest over the
            # doctored set, so every per-object check passes and only
            # the S/T/E join can catch it
            tp = os.path.join(io["attempt_dir"],
                              stem + ACC.STAGED_CLASS_SUFFIX[
                                  "transcript"])
            with open(tp, encoding="utf-8") as f:
                t = json.load(f)
            t["raw_body_sha256"] = "ab" * 32
            with open(tp, "w", encoding="utf-8") as f:
                json.dump(t, f)
            pp = os.path.join(io["retry_dir"],
                              stem + ".prepared.json")
            with open(pp, encoding="utf-8") as f:
                pr = json.load(f)
            pr["class_canon_sha256"]["transcript"] = \
                PROD._canon_digest(t)
            pr["prepared_sha256"] = _canon_entry_digest(
                {k: v for k, v in pr.items()
                 if k != "prepared_sha256"})
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(pr, f)
            # crash-state: the standing result would trip the
            # projection check first; remove it so the CROSS-OBJECT
            # JOIN is the check under test
            os.remove(io["result_path"])
            os.remove(io["retry_ledger"])
            check("N3d an individually-valid set that fails the "
                  "cross-object S/T/E join refuses before any "
                  "staged write",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_JOIN_FAILED")
                  or refuses(lambda: finalize(dict(io)),
                             "RETRY_EVIDENCE_INCONSISTENT"))

        # ---- R4 (repair 4) outcome-typing controls
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _local_boom(i, o):
                raise OSError("local attempt directory failure")
            r = fire("HEAD", dict(io, opener=trip,
                                  capture_fn=_local_boom))
            check("N4a a zero-opener LOCAL failure is typed "
                  "INTERNAL_ERROR_AFTER_DISPATCH, never transport",
                  r["outcome"] == "INTERNAL_ERROR_AFTER_DISPATCH"
                  and r["opener_calls"] == 0
                  and r["transport"] is None,
                  f"outcome={r['outcome']} calls={r['opener_calls']}")
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_boom,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("N4b a one-opener CONNECTION failure is typed "
                  "TRANSPORT_ERROR with boundary evidence",
                  r["outcome"] == "TRANSPORT_ERROR"
                  and r["opener_calls"] == 1
                  and r["transport"] is not None,
                  f"outcome={r['outcome']} calls={r['opener_calls']}")
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_local_boom,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("N4c a LOCAL failure inside the opener (missing CA "
                  "bundle) is typed INTERNAL, never transport "
                  "(codex 1830Z residual 4)",
                  r["outcome"] == "INTERNAL_ERROR_AFTER_DISPATCH"
                  and r["transport"] is None,
                  f"outcome={r['outcome']}")

        # ---- N2 (codex 1830Z residual 2): the exact semantic probe
        # -- a re-digested but impossible result plus a changed
        # frozen ledger must refuse BEFORE index creation
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["outcome"] = "TRANSPORT_ERROR"
            rr["opener_calls"] = 0
            rr["transport"] = None
            rr["result_sha256"] = _canon_entry_digest(
                {k: v for k, v in rr.items()
                 if k != "result_sha256"})
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            with open(io["ledger_path"], "ab") as f:
                f.write(b'{"appended": "after result"}\n')
            os.remove(io["retry_ledger"])
            ok_n2 = refuses(lambda: finalize(dict(io)),
                            "RETRY_RESULT_SEMANTICS")                 or refuses(lambda: finalize(dict(io)),
                           "RETRY_TERMINAL_LEDGER_CHANGED")
            check("N2 a re-digested semantically impossible result "
                  "over a changed frozen ledger refuses before index "
                  "creation",
                  ok_n2 and not os.path.exists(io["retry_ledger"]))

        # ---- N3e (residual 3A causal): a typed CaptureRefusal at
        # the first staged write is a RESUMABLE publication block --
        # zero new classes, no contradictory terminal result
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _pub_refusing_writer(path, obj, _sd=io["staged_dir"]):
                if os.path.dirname(path) == _sd:
                    raise CAP.CaptureRefusal(
                        "KAT_STAGED_WRITE_CONFLICT")
                return _real_writer(path, obj)
            _g2["_write_json_create_once"] = _pub_refusing_writer
            try:
                blocked = refuses(lambda: fire("HEAD", dict(
                    io, opener=opener_200,
                    capture_fn=lambda i, o: CAP.capture_day(
                        dict(spec_real), i["store_dir"],
                        i["attempt_dir"], i["attempt_dir"],
                        lambda b: CAP.admission_transform(
                            TARGET_LANE, b, spec_full_s()),
                        opener=o))),
                    "RETRY_PUBLICATION_BLOCKED_AFTER_PREPARED")
            finally:
                _g2["_write_json_create_once"] = _real_writer
            check("N3e a typed staged-write refusal after prepared is "
                  "a RESUMABLE block: zero classes, no terminal "
                  "result, prepared outcome intact",
                  blocked
                  and len(os.listdir(io["staged_dir"])) == 0
                  and not os.path.exists(io["result_path"]))
            out = finalize(dict(io))
            check("N3f the public finalizer is the recovery path "
                  "after a publication block",
                  out["classes"] is not None
                  and os.path.exists(io["result_path"]))

        # ---- N3g (residual 3B): a right-map/WRONG-SCHEMA marker
        # refuses before any class write
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            marker = os.path.join(io["retry_dir"],
                                  stem + ".classes_complete.json")
            with open(marker, encoding="utf-8") as f:
                mk_obj = json.load(f)
            mk_obj["schema"] = "wrong-schema"
            with open(marker, "w", encoding="utf-8") as f:
                json.dump(mk_obj, f)
            for f2 in os.listdir(io["staged_dir"]):
                os.remove(os.path.join(io["staged_dir"], f2))
            ok_n3g = refuses(lambda: finalize(dict(io)),
                             "RETRY_MARKER_DIVERGENT")
            check("N3g a right-map wrong-schema marker refuses with "
                  "ZERO classes republished",
                  ok_n3g and len(os.listdir(io["staged_dir"])) == 0)

        # ---- R5B index: crash-after-result reconstruction + refusals
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"], i["attempt_dir"],
                    i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            # crash-after-result: delete the index; finalize rebuilds
            os.remove(io["retry_ledger"])
            out = finalize(dict(io))
            n = sum(1 for x in open(io["retry_ledger"],
                                    encoding="utf-8") if x.strip())
            check("R5b crash-after-result: finalize reconstructs the "
                  "one-line index from dispatch+result bytes",
                  out["index"] == "present" and n == 1)
            # idempotence: run again, still one line
            finalize(dict(io))
            n2 = sum(1 for x in open(io["retry_ledger"],
                                     encoding="utf-8") if x.strip())
            check("R5c finalize is idempotent (index stays one line)",
                  n2 == 1)
            # divergent index refuses
            with open(io["retry_ledger"], "w", encoding="utf-8",
                      newline="\n") as f:
                f.write('{"forged": true}\n')
            check("R5d a divergent index refuses typed",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_INDEX_DIVERGENT"))
            # truncated (empty) index also refuses
            with open(io["retry_ledger"], "w", encoding="utf-8") as f:
                f.write("")
            check("R5e a truncated index refuses typed",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_INDEX_DIVERGENT"))

        # ---- P1-P4 (codex 1845Z): the four exact semantic-join
        # probes, each re-digested so only the JOIN can catch it
        def _redigest(obj, field):
            obj[field] = _canon_entry_digest(
                {k: v for k, v in obj.items() if k != field})
            return obj

        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            pp = os.path.join(io["retry_dir"],
                              stem + ".prepared.json")
            with open(pp, encoding="utf-8") as f:
                pr = json.load(f)
            pr["transport"] = {"requested_url":
                               "https://evil.invalid/",
                               "status": 599}
            pr["terminal_ledger_sha256_recomputed"] = "0" * 64
            pr["completed_utc"] = "not-a-canonical-instant"
            _redigest(pr, "prepared_sha256")
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(pr, f)
            os.remove(io["result_path"])
            os.remove(io["retry_ledger"])
            n_st = len(os.listdir(io["staged_dir"]))
            ok = refuses(lambda: finalize(dict(io)),
                         "RETRY_PREPARED_SEMANTICS")
            check("P1 an evil re-digested prepared projection (599/"
                  "evil URL/zero ledger/bad instant) refuses before "
                  "class reuse, result or index",
                  ok and not os.path.exists(io["result_path"])
                  and not os.path.exists(io["retry_ledger"]))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["scientific"] = {"classes_published":
                                {"evil": "not-a-digest"},
                                "raw_body_sha256": "f" * 64,
                                "proof_kind": "FORGED_KIND"}
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            ok = refuses(lambda: finalize(dict(io)),
                         "RETRY_RESULT_SEMANTICS")
            check("P2 a forged admitted result (evil map/body/proof) "
                  "refuses against the canonical projection; no "
                  "index",
                  ok and not os.path.exists(io["retry_ledger"]))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(io["dispatch_path"], encoding="utf-8") as f:
                dd = json.load(f)
            dd["original_ledger"] = {"path": "docs/evil-ledger.jsonl",
                                     "sha256": "0" * 64,
                                     "seq": 81,
                                     "entry_sha256": "0" * 64}
            dd["registered_request_url"] = "https://evil.invalid/"
            dd["executed_code"]["path"] = "monitoring/src/evil.py"
            dd["store"] = {"id": "evil", "root": "evil://store"}
            _redigest(dd, "dispatch_sha256")
            with open(io["dispatch_path"], "w",
                      encoding="utf-8") as f:
                json.dump(dd, f)
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["dispatch_sha256"] = dd["dispatch_sha256"]
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            ok = refuses(lambda: finalize(dict(io)),
                         "RETRY_DISPATCH_SEMANTICS")
            check("P3 a rebound re-digested dispatch (evil ledger/"
                  "URL/module/store) refuses; no index",
                  ok and not os.path.exists(io["retry_ledger"]))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _capture_refuses(i, o):
                raise CAP.CaptureRefusal(
                    "CAPTURE_AUTHORITY_KAT: constructed pre-opener "
                    "capture refusal")
            r = fire("HEAD", dict(io, opener=trip,
                                  capture_fn=_capture_refuses))
            assert r["outcome"] == "CAPTURE_REFUSED"
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["opener_calls"] = 77
            rr["transport"] = {"invented": True}
            rr["scientific"] = {"classes_published":
                                {"invented": True}}
            rr["terminal_ledger_unchanged"] = False
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            ok = refuses(lambda: finalize(dict(io)),
                         "RETRY_RESULT_SEMANTICS")
            check("P4 a forged CAPTURE_REFUSED (77 openers/invented "
                  "transport/invented classes/false ledger bit) "
                  "refuses; no index",
                  ok and not os.path.exists(io["retry_ledger"]))

        # ---- one one-field negative per remaining outcome, each
        # re-digested (codex 1845Z residual 4)
        def _doctor_result(io, mutate):
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            mutate(rr)
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            return refuses(lambda: finalize(dict(io)),
                           "RETRY_RESULT_SEMANTICS")
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("E1 non-200 refusal with status doctored to 200 "
                  "refuses",
                  _doctor_result(io, lambda rr: rr["transport"]
                                 .__setitem__("status", 200)))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_boom,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("E2 transport error with nulled transport refuses",
                  _doctor_result(io, lambda rr: rr.__setitem__(
                      "transport", None)))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_bad200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("E3 transform refusal with invented classes "
                  "refuses",
                  _doctor_result(io, lambda rr: rr["scientific"]
                                 .__setitem__("classes_published",
                                              {"x": "y"})))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _local_boom2(i, o):
                raise OSError("local failure")
            fire("HEAD", dict(io, opener=trip,
                              capture_fn=_local_boom2))
            check("E4 internal error with invented transport refuses",
                  _doctor_result(io, lambda rr: rr.__setitem__(
                      "transport", {"invented": True})))

        # ---- W1 (1906Z finding 1): CRLF and LF checkouts of one
        # pinned Python blob share ONE normalized digest; a one-byte
        # semantic mutation differs
        _lf = b"x = 1\nprint(x)\n"
        _crlf = _lf.replace(b"\n", b"\r\n")
        _mut = b"x = 2\nprint(x)\n"
        check("W1 normalized source digest: CRLF == LF for the same "
              "blob; a semantic mutation differs",
              _norm_source_sha256(_lf) == _norm_source_sha256(_crlf)
              and _norm_source_sha256(_lf)
              != _norm_source_sha256(_mut))

        # ---- W2 (finding 2): a production-shaped 200 with all four
        # lowercase receipt headers -- exact transcript/evidence
        # equality and a terminal result
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(os.path.join(
                    io["attempt_dir"],
                    stem + ACC.STAGED_CLASS_SUFFIX["transcript"]),
                    encoding="utf-8") as f:
                a_tr = json.load(f)
            check("W2 content-length + the three other lowercase "
                  "receipt headers flow transcript==evidence and the "
                  "capture terminalizes",
                  r["outcome"] == "CAPTURED_ADMITTED"
                  and set(a_tr["headers"]) == set(HEADER_ALLOWLIST)
                  and r["transport"]["headers"] == a_tr["headers"],
                  f"tr={a_tr.get('headers')} "
                  f"ev={r['transport'].get('headers')}")
        # ---- W3a (finding 3A): empty body after a 200 -- a POST-
        # TRANSPORT capture refusal with a terminal result/index and
        # zero classes
        def opener_empty(url):
            return 200, {"content-type": "application/json",
                         "content-length": "0", "date": "kat",
                         "server": "kat"}, b"", url
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            r = fire("HEAD", dict(
                io, opener=opener_empty,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("W3a CAPTURE_EMPTY_BODY after one 200 -> typed "
                  "CAPTURE_REFUSED_AFTER_TRANSPORT with measured "
                  "evidence, terminal result+index, zero classes",
                  r["outcome"] == "CAPTURE_REFUSED_AFTER_TRANSPORT"
                  and r["opener_calls"] == 1
                  and r["transport"]["status"] == 200
                  and len(os.listdir(io["staged_dir"])) == 0
                  and os.path.exists(io["retry_ledger"]),
                  f"outcome={r['outcome']}")
        # ---- W3b (finding 3B): successful fetch then a local
        # OSError -- INTERNAL_ERROR_AFTER_TRANSPORT keeps the
        # measured transport
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _fetch_then_local(i, o):
                o(REGISTERED_REQUEST_URL)
                raise OSError("local staging failure after fetch")
            r = fire("HEAD", dict(io, opener=opener_200,
                                  capture_fn=_fetch_then_local))
            check("W3b one successful fetch then local OSError -> "
                  "INTERNAL_ERROR_AFTER_TRANSPORT with evidence "
                  "kept, terminal result+index, zero classes",
                  r["outcome"] == "INTERNAL_ERROR_AFTER_TRANSPORT"
                  and r["opener_calls"] == 1
                  and r["transport"]["status"] == 200
                  and len(os.listdir(io["staged_dir"])) == 0
                  and os.path.exists(io["retry_ledger"]),
                  f"outcome={r['outcome']}")

        # ---- W4 (finding 4): the exact re-digested pin-downgrade
        # probe and an independently changed capsule pin
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(io["dispatch_path"], encoding="utf-8") as f:
                dd = json.load(f)
            dd["manifest_commit"] = "attacker-unpinned-commitish"
            dd["capsule_pin_commit"] = "2" * 40
            dd["capsule_sha256"] = "2" * 64
            dd["executed_code"]["disk_sha256"] = "4" * 64
            _redigest(dd, "dispatch_sha256")
            with open(io["dispatch_path"], "w",
                      encoding="utf-8") as f:
                json.dump(dd, f)
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["dispatch_sha256"] = dd["dispatch_sha256"]
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            ok = refuses(lambda: finalize(dict(io)),
                         "RETRY_DISPATCH_SEMANTICS")
            check("W4 the re-digested pin-downgrade dispatch refuses "
                  "fail-closed (unresolvable manifest never "
                  "downgrades); no index",
                  ok and not os.path.exists(io["retry_ledger"]))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(io["dispatch_path"], encoding="utf-8") as f:
                dd = json.load(f)
            dd["capsule_pin_commit"] = "3" * 40
            _redigest(dd, "dispatch_sha256")
            with open(io["dispatch_path"], "w",
                      encoding="utf-8") as f:
                json.dump(dd, f)
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["dispatch_sha256"] = dd["dispatch_sha256"]
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            check("W4b an independently changed capsule pin refuses "
                  "against the reopened registered status; no index",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_DISPATCH_SEMANTICS")
                  and not os.path.exists(io["retry_ledger"]))

        # ---- W5 (finding 5): one-field re-digested value/time
        # negatives over live states
        def _mk_404(td):
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            return io
        with tempfile.TemporaryDirectory() as td:
            io = _mk_404(td)
            check("W5a impossible instant (9999-99-99...) refuses",
                  _doctor_result(io, lambda rr: rr["transport"]
                                 .__setitem__("request_start_utc",
                                              "9999-99-99T99:99:99Z")))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_404(td)
            check("W5b reversed phase instants refuse",
                  _doctor_result(io, lambda rr: rr.__setitem__(
                      "completed_utc", "2001-01-01T00:00:00Z")))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_404(td)
            check("W5c status/refusal mismatch (599 vs '-> 404') "
                  "refuses",
                  _doctor_result(io, lambda rr: rr["transport"]
                                 .__setitem__("status", 599)))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_404(td)
            check("W5d evil header key refuses",
                  _doctor_result(io, lambda rr: rr["transport"]
                                 ["headers"].__setitem__("evil",
                                                         "header")))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_404(td)
            check("W5e negative body count refuses",
                  _doctor_result(io, lambda rr: rr["transport"]
                                 .__setitem__("body_bytes_seen", -1)))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_bad200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            check("W5f transform-refusal body digest not equal the "
                  "attempt transcript refuses",
                  _doctor_result(io, lambda rr: rr["scientific"]
                                 .__setitem__("raw_body_preserved",
                                              "a" * 64)))

        # ---- C1x (1929Z primitive 1): single resolution at entry
        check("X1a a non-commit commitish refuses at the entry "
              "resolver",
              refuses(lambda: plan("not-a-commit"),
                      "RETRY_MANIFEST_UNRESOLVABLE"))
        _seen_mc = []
        _real_status2 = RUN4.capsule_pin_status

        def _spy_status(mc, candidate_sha=None):
            _seen_mc.append(mc)
            return _real_status2(mc, candidate_sha)
        RUN4.capsule_pin_status = _spy_status
        try:
            with tempfile.TemporaryDirectory() as td:
                io = kat_io(td)
                _al_seen = []
                _al = io["allowlist_fn"]

                def _spy_allow(repo, mc):
                    _al_seen.append(mc)
                    return _al(repo, mc)
                r = fire("HEAD", dict(
                    io, allowlist_fn=_spy_allow, opener=opener_404,
                    capture_fn=lambda i, o: CAP.capture_day(
                        dict(spec_real), i["store_dir"],
                        i["attempt_dir"], i["attempt_dir"],
                        lambda b: CAP.admission_transform(
                            TARGET_LANE, b, spec_full_s()),
                        opener=o)))
                with open(io["dispatch_path"],
                          encoding="utf-8") as f:
                    dd = json.load(f)
                _hex40 = [m for m in _seen_mc + _al_seen
                          + [dd["manifest_commit"]]]
                check("X1b fire('HEAD') resolves ONCE: every "
                      "downstream read and the persisted dispatch "
                      "carry the same immutable 40-hex object ID, "
                      "never the mutable ref text",
                      all(isinstance(m, str) and len(m) == 40
                          and m == _hex40[0] for m in _hex40)
                      and "HEAD" not in _hex40,
                      f"ids={_hex40[:3]}")
        finally:
            RUN4.capsule_pin_status = _real_status2

        # ---- C2x (primitive 2): the three exact probes + receipt
        # negatives
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_empty,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            # POST_CAPTURE_UNJOINED: garbage values, re-digested
            def _garbage(rr):
                rr["transport"]["effective_url"] = 17
                rr["transport"]["headers"]["evil"] = 3
                rr["transport"]["body_bytes_seen"] = -7
            check("X2a a post-transport capture result with garbage "
                  "evidence values refuses (unconditional value "
                  "checks + receipt equality)",
                  _doctor_result(io, _garbage))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_bad200,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            # TRANSFORM_WITHOUT_TRANSCRIPT: the transcript is now
            # REQUIRED -- deleting it must refuse, invented digest or
            # not
            os.remove(os.path.join(
                io["attempt_dir"],
                stem + ACC.STAGED_CLASS_SUFFIX["transcript"]))
            os.remove(io["retry_ledger"])
            check("X2b a transform refusal without its attempt "
                  "transcript refuses -- the join is mandatory, "
                  "never a shape-only fallback",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_RESULT_SEMANTICS"))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            os.remove(_transport_receipt_path(io))
            os.remove(io["retry_ledger"])
            check("X2c a missing transport receipt refuses every "
                  "opener_calls=1 member",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_RESULT_SEMANTICS"))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(_transport_receipt_path(io), "w",
                      encoding="utf-8") as f:
                f.write('{"forged": true}')
            os.remove(io["retry_ledger"])
            check("X2d a deformed transport receipt refuses typed",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_TRANSPORT_RECEIPT_DEFORMED"))

        # ---- C3x (primitive 3): canonical instants + parsed order
        check("X3a a non-padded instant is NOT canonical "
              "(2026-9-30T00:00:00Z)",
              not _is_canonical_utc("2026-9-30T00:00:00Z")
              and _is_canonical_utc("2026-09-30T00:00:00Z"))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_404(td)
            # Oct -> Sep reversal that STRING comparison calls
            # ordered ("2026-10..." < "2026-9...")
            def _oct_sep(rr):
                rr["transport"]["request_start_utc"] =                     "2026-10-01T00:00:00Z"
                rr["transport"]["response_complete_utc"] =                     "2026-09-30T00:00:00Z"
            check("X3b an Oct->Sep phase reversal refuses under "
                  "PARSED comparison (string order accepts it)",
                  _doctor_result(io, _oct_sep))
        with tempfile.TemporaryDirectory() as td:
            io = _mk_complete(td)
            pp = os.path.join(io["retry_dir"],
                              stem + ".prepared.json")
            with open(pp, encoding="utf-8") as f:
                pr = json.load(f)
            # a REAL reversal (a same-second capture makes a swap
            # vacuous -- self-caught): request-start moved strictly
            # AFTER response-complete, and the transport RECEIPT is
            # rewritten to match, so receipt equality holds and the
            # PARSED phase order is the check under test
            pr["transport"]["request_start_utc"] = \
                "2027-01-01T00:00:00Z"
            _redigest(pr, "prepared_sha256")
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(pr, f)
            rcp = _transport_receipt_path(io)
            with open(rcp, encoding="utf-8") as f:
                rc = json.load(f)
            rc["evidence"] = dict(pr["transport"])
            rc["receipt_sha256"] = _canon_entry_digest(
                {k: v for k, v in rc.items()
                 if k != "receipt_sha256"})
            os.remove(rcp)
            with open(rcp, "w", encoding="utf-8",
                      newline="\n") as f:
                json.dump(rc, f)
            os.remove(io["result_path"])
            os.remove(io["retry_ledger"])
            ok3c = refuses(lambda: finalize(dict(io)),
                           "RETRY_PREPARED_SEMANTICS")
            check("X3c an admitted prepared with request-after-"
                  "response refuses under PARSED phase order "
                  "(receipt equality held constant)", ok3c)
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _lb(i, o):
                raise OSError("local")
            fire("HEAD", dict(io, opener=trip, capture_fn=_lb))
            check("X3d an internal result completed before dispatch "
                  "refuses",
                  _doctor_result(io, lambda rr: rr.__setitem__(
                      "completed_utc", "2001-01-01T00:00:00Z")))

        # ---- Y1-Y4 (codex 2017Z finding 1): the four post-response
        # phase injections. Every pre-prepared local failure must
        # terminalize exactly once as INTERNAL_ERROR_AFTER_TRANSPORT
        # with result+index and zero classes.
        def _assert_terminal_internal(io, r, name):
            check(name,
                  r["outcome"] == "INTERNAL_ERROR_AFTER_TRANSPORT"
                  and r["opener_calls"] == 1
                  and isinstance(r["transport"], dict)
                  and os.path.exists(io["result_path"])
                  and os.path.exists(io["retry_ledger"])
                  and len(os.listdir(io["staged_dir"])) == 0,
                  f"outcome={r['outcome']}")
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)

            def _capture_then_eat_body(i, o):
                out = CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)
                for f2 in os.listdir(i["store_dir"]):
                    os.remove(os.path.join(i["store_dir"], f2))
                # also eat the attempt classes so the stored-body
                # reopen inside fire() is the failing stage
                for f2 in list(os.listdir(i["attempt_dir"])):
                    if not f2.endswith(
                            TRANSPORT_RECEIPT_BASENAME_SUFFIX):
                        os.remove(os.path.join(i["attempt_dir"], f2))
                return out
            r = fire("HEAD", dict(io, opener=opener_200,
                                  capture_fn=_capture_then_eat_body))
            _assert_terminal_internal(
                io, r, "Y1 stored-body reopen failure terminalizes "
                       "as INTERNAL_ERROR_AFTER_TRANSPORT with "
                       "result+index")
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            _real_xform = CAP.admission_transform

            def _local_xform(lane, body, sc):
                raise OSError("transform local failure (kat)")
            CAP.admission_transform = _local_xform
            try:
                r = fire("HEAD", dict(
                    io, opener=opener_200,
                    capture_fn=lambda i, o: CAP.capture_day(
                        dict(spec_real), i["store_dir"],
                        i["attempt_dir"], i["attempt_dir"],
                        lambda b: _real_xform(
                            TARGET_LANE, b, spec_full_s()),
                        opener=o)))
            finally:
                CAP.admission_transform = _real_xform
            _assert_terminal_internal(
                io, r, "Y2 a non-CaptureRefusal transform error "
                       "terminalizes as INTERNAL_ERROR_AFTER_"
                       "TRANSPORT")
        for _n, _name in ((0, "Y3 first attempt-class write failure "
                              "terminalizes post-transport, never "
                              "pre-transport"),
                          (99, "Y4 prepared-write failure "
                               "terminalizes post-transport with "
                               "result+index")):
            with tempfile.TemporaryDirectory() as td:
                io = kat_io(td)
                _realw = _g2["_write_json_create_once"]
                seen = []

                def _failw(path, obj, _realw=_realw, _n=_n,
                           _ad=io["attempt_dir"],
                           _rd=io["retry_dir"]):
                    tgt = (os.path.dirname(path) == _ad
                           and not path.endswith(
                               TRANSPORT_RECEIPT_BASENAME_SUFFIX))
                    prep = path.endswith(".prepared.json")
                    if (_n == 0 and tgt) or (_n == 99 and prep):
                        raise OSError("injected write failure (kat)")
                    return _realw(path, obj)
                _g2["_write_json_create_once"] = _failw
                try:
                    r = fire("HEAD", dict(
                        io, opener=opener_200,
                        capture_fn=lambda i, o: CAP.capture_day(
                            dict(spec_real), i["store_dir"],
                            i["attempt_dir"], i["attempt_dir"],
                            lambda b: CAP.admission_transform(
                                TARGET_LANE, b, spec_full_s()),
                            opener=o)))
                finally:
                    _g2["_write_json_create_once"] = _realw
                _assert_terminal_internal(io, r, _name)

        # ---- Y5 (finding 2): the re-digested mutable-alias probe
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(io["dispatch_path"], encoding="utf-8") as f:
                dd = json.load(f)
            dd["manifest_commit"] = "HEAD"
            _redigest(dd, "dispatch_sha256")
            with open(io["dispatch_path"], "w",
                      encoding="utf-8") as f:
                json.dump(dd, f)
            with open(io["result_path"], encoding="utf-8") as f:
                rr = json.load(f)
            rr["dispatch_sha256"] = dd["dispatch_sha256"]
            _redigest(rr, "result_sha256")
            with open(io["result_path"], "w",
                      encoding="utf-8") as f:
                json.dump(rr, f)
            os.remove(io["retry_ledger"])
            check("Y5 a re-digested mutable manifest alias (HEAD) "
                  "refuses in public validation; no index",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_DISPATCH_SEMANTICS")
                  and not os.path.exists(io["retry_ledger"]))

        # ---- Y6 (finding 3, Option A): receipt identity + parser
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(_transport_receipt_path(io), "w",
                      encoding="utf-8") as f:
                f.write("{")
            os.remove(io["retry_ledger"])
            check("Y6a a truncated receipt refuses TYPED, never a "
                  "raw JSONDecodeError",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_TRANSPORT_RECEIPT_DEFORMED"))
        with tempfile.TemporaryDirectory() as td:
            io = kat_io(td)
            fire("HEAD", dict(
                io, opener=opener_404,
                capture_fn=lambda i, o: CAP.capture_day(
                    dict(spec_real), i["store_dir"],
                    i["attempt_dir"], i["attempt_dir"],
                    lambda b: CAP.admission_transform(
                        TARGET_LANE, b, spec_full_s()), opener=o)))
            with open(_transport_receipt_path(io),
                      encoding="utf-8") as f:
                rc = json.load(f)
            rc["dispatch_sha256"] = "d" * 64
            rc["receipt_sha256"] = _canon_entry_digest(
                {k: v for k, v in rc.items()
                 if k != "receipt_sha256"})
            os.remove(_transport_receipt_path(io))
            with open(_transport_receipt_path(io), "w",
                      encoding="utf-8", newline="\n") as f:
                json.dump(rc, f)
            os.remove(io["retry_ledger"])
            check("Y6b a receipt bound to a different dispatch "
                  "identity refuses",
                  refuses(lambda: finalize(dict(io)),
                          "RETRY_TRANSPORT_RECEIPT_DEFORMED"))

        # ---- F4 the CLI accepts no target argument
        check("F4 the CLI has no injectable target (extra argv "
              "refuses)",
              refuses(lambda: main(["fire", "HEAD",
                                    "MAG_FEED/vic/2026-01-01"]),
                      "RETRY_USAGE"))
    print()
    if fails:
        print(f"RETRY-404 BAR FAILURES ({len(fails)}): {fails}")
        return 1
    print("RETRY-404 ONE-SHOT: ALL BARS PASS  (no network; writes "
          "confined to selftest temp dirs)")
    return 0


def main(argv):
    if not argv or argv[0] == "--selftest":
        return _selftest()
    if argv[0] == "plan":
        plan(argv[1] if len(argv) > 1 else "HEAD")
        return 0
    if argv[0] == "fire":
        if len(argv) != 2:
            _refuse("RETRY_USAGE",
                    "fire takes exactly one argument, the manifest "
                    "commit; there is NO target argument")
        fire(argv[1])
        return 0
    if argv[0] == "finalize":
        if len(argv) != 1:
            _refuse("RETRY_USAGE", "finalize takes no arguments")
        finalize()
        return 0
    _refuse("RETRY_USAGE", "plan [commitish] | fire <manifest> | "
            "finalize | --selftest")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
