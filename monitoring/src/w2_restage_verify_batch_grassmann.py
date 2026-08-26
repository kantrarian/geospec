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
import w2_no_network_grassmann as NONET

REPO = LIN.REPO
RECEIPT_SCHEMA = "f2g-w2-postmanifest-verification-receipt-v1"


class BatchRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _b(msg):
    raise BatchRefusal("RESTAGE_BATCH_REFUSED: " + str(msg))


def _validate_manifest_verdict(verdict, full):
    """codex 1400Z P2 #5: PURE, so the repaired OPEN-slots branch has
    LOAD-BEARING evidence of its own.

    The branch was previously reachable only through `run()`, whose
    live doctor asserts `not a prestart PASS` and returns before this
    point, while the injected slots_open=7 case exercises
    `_receipt_bindings()` -- a different function. So the precondition
    could be deleted with every test claimed for it still green. The
    code was right; the evidence was not.
    """
    v = verdict.get("verdict")
    if v != "PASS":
        _b(f"the execution manifest at {full[:12]} is not a prestart "
           f"PASS (verdict={v!r})")
    opened = verdict.get("slots_open")
    if not isinstance(opened, int) or isinstance(opened, bool):
        _b(f"the verifier at {full[:12]} reported no integer "
           f"slots_open ({opened!r}); a closed-manifest precondition "
           "cannot rest on a field that is absent")
    if opened:
        _b(f"the manifest at {full[:12]} still has {opened} OPEN "
           "slots -- a post-manifest verification runs only over a "
           "closed manifest")
    return opened


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
    # codex 1327Z P1, one layer further than they stated: the
    # verifier returns TOP-LEVEL `slots_open`, never a `slots`
    # mapping. This used to read (verdict.get("slots") or {}), which
    # is ALWAYS empty -- so open_slots was always [] and this
    # precondition NEVER FIRED. The selftest hid it behind an `or`
    # (refuses "OPEN slots" OR refuses "not a prestart PASS"): the
    # second branch always carried it and the dead branch was never
    # exercised. Same defect family as the substring doctors -- a
    # check that cannot fail is not a check.
    _validate_manifest_verdict(verdict, full)
    return full, verdict


def _tally(acc, claim):
    """Nest by claim kind; a status is never a raw None key."""
    kind = claim["artifact_claim_kind"]
    status = claim.get("outcome") or claim.get("support_outcome")
    acc.setdefault(kind, {})
    acc[kind][status] = acc[kind].get(status, 0) + 1
    return acc


def _require(path, what):
    if not os.path.exists(path):
        _b(f"{what} is ABSENT at {path} -- strict mode: a missing "
           "input is a typed failure, never a skip and never green")
    return path


def _receipt_bindings(verdict, allowlist):
    """codex 1327Z P1: COPY the returned facts; never restate them.

    `slots_open` was computed from `verdict.get("slots")` -- a key the
    verifier does not return -- so it was hard-zero regardless of what
    the verifier found. Injecting a distinctive result reproduced
    `injected slots_open = 7 / receipt slots_open = 0`: a receipt
    contradicting the check it claims to record, the same class as the
    `allow["checked"]` defect. Pure and side-effect free so doctors can
    drive it with deliberately distinct injected values.
    """
    v = verdict.get("verdict")
    if v != "PASS":
        _b(f"a receipt may not be assembled over verdict {v!r}")
    mode = verdict.get("mode", "prestart")
    if mode != "prestart":
        _b(f"a post-manifest receipt requires prestart mode, got "
           f"{mode!r}")
    so = verdict.get("slots_open")
    if not isinstance(so, int) or isinstance(so, bool):
        _b(f"the verifier reported no integer slots_open ({so!r})")
    if so != 0:
        _b(f"a post-manifest receipt may not record {so} OPEN slots")
    vpc = verdict.get("pins_checked")
    if not isinstance(vpc, int) or isinstance(vpc, bool) or vpc <= 0:
        _b(f"the verifier reported pins_checked={vpc!r}")
    apc = allowlist.get("pins_checked") \
        if isinstance(allowlist, dict) else None
    if not isinstance(apc, int) or isinstance(apc, bool) or apc <= 0:
        _b(f"the allowlist reported pins_checked={apc!r}")
    return {"manifest_verdict": {"verdict": v, "mode": mode,
                                 "slots_open": so,
                                 "pins_checked": vpc},
            "runtime_allowlist": {"result": "PASS",
                                  "pins_checked": apc}}


def _require_count_identity(registered, attempted, verified):
    """registered == attempted == verified, ENFORCED not printed.

    The only doctor for this used to be
    `"registered_reuse_count" in getsource(run)` -- a SOURCE-SUBSTRING
    check standing in for a RUNTIME behaviour, which is the exact
    defect class codex found in the misspelled resolver. A substring
    cannot tell you the comparison fires, or fires the right way
    round. Factored out so doctors call it with real numbers.
    """
    if not (registered == attempted == verified):
        _b(f"registered_reuse_count {registered} != attempted "
           f"{attempted} != verified {verified}")
    return registered


def _resolve_batch_preflight(full, resolver=None,
                             allowlist_check=None):
    """codex 0534Z P0: the positive path is FACTORED so doctors can
    actually EXECUTE it. My previous doctors asserted on the SOURCE
    STRING of run(), and they stayed green while the promoted
    resolver was misspelled at its DEFINITION -- the desired spelling
    appeared at the call site, nothing ever resolved the symbol, and
    both selftest calls refused in _require_valid_manifest() long
    before reaching it. A substring is not a behaviour.

    Production supplies the real defaults; there is no bypass flag.
    """
    if resolver is None:
        resolver = getattr(LIN, "resolve_pinned_bytes", None)
    if allowlist_check is None:
        import w2_accrual_instrument_cayley as _ACC
        allowlist_check = getattr(_ACC, "runtime_allowlist_check",
                                  None)
    if not callable(resolver):
        _b("the manifest-pin resolver is ABSENT or not callable -- "
           "a misspelled or missing symbol must refuse here, never "
           "surface as an AttributeError from the positive path")
    if not callable(allowlist_check):
        _b("the runtime allowlist check is ABSENT or not callable")
    # resolve the capsule through the UNIQUE MANIFEST PIN, never from
    # <commit>:path -- a smaller capsule at the manifest commit could
    # otherwise choose a smaller key universe while the manifest pins
    # the full capsule elsewhere.
    try:
        caps_raw, caps_pin = resolver(REPO, full, DISP.CAPSULE_PATH)
    except BatchRefusal:
        raise
    except Exception as exc:
        _b(f"the disposition capsule pin did not resolve at "
           f"{full[:12]} ({type(exc).__name__}: {exc})")
    caps = json.loads(caps_raw.decode("utf-8"))
    # ...and the EXECUTING tree must equal the pinned bytes, or a
    # dirty batch could emit a receipt while the manifest pins clean
    # code (the same self-authentication defect one level out)
    try:
        allow = allowlist_check(REPO, full)
    except BatchRefusal:
        raise
    except Exception as exc:
        _b(f"the runtime allowlist REFUSED at {full[:12]} "
           f"({type(exc).__name__}: {str(exc)[:160]})")
    # codex 0534Z P1: the receipt read allow["checked"], a key this
    # function never returns, so a successful walk was recorded as
    # ZERO checked pins -- a receipt contradicting the check that
    # just ran. Require the REAL key, positive, and copy it verbatim.
    pins = allow.get("pins_checked") if isinstance(allow, dict) \
        else None
    if isinstance(pins, bool) or not isinstance(pins, int) \
            or pins <= 0:
        _b(f"the runtime allowlist reported pins_checked={pins!r}; a "
           "post-manifest receipt may not rest on an allowlist walk "
           "that checked no pins")
    return caps_raw, caps_pin, caps, allow


def run(manifest_commit, store_root=None):
    """codex 1327Z P1: the sentinel now wraps the ENTIRE operation.

    It used to be entered only AFTER manifest verification, pin
    resolution, the allowlist walk and the store checks -- every one
    of which touches git and the filesystem and none of which was
    measured -- and `__exit__` was called only on the NORMAL path, so
    any typed refusal mid-run left `socket.socket` globally replaced
    by the blocked class for the rest of the process. codex forced a
    refusal after entry and observed
    `socket_restored_after_refusal = False`. A `with` block restores
    on every exit, including exceptions.
    """
    with NONET.no_network() as _net:
        return _run_measured(manifest_commit, store_root, _net)


def _run_measured(manifest_commit, store_root, _net):
    full, _verdict = _require_valid_manifest(manifest_commit)
    caps_raw, caps_pin, caps, allow = _resolve_batch_preflight(full)
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
        # codex 0445Z item 3: counts are NESTED BY CLAIM KIND and
        # never keyed by a raw outcome -- the 360 station-presence
        # artifacts have no outcome, and aggregating that None both
        # lost the distinction and CRASHED assembly (None is not
        # orderable against str).
        _tally(outcomes, out["claim"])
    if attempted != verified:
        _b("attempted != verified")
    if _net.attempts:
        _b(f"the offline batch ATTEMPTED {_net.attempts} network "
           "connection(s) -- the counter is measured, not asserted")

    def dg(xs):
        """codex 0509Z item 5: a field named *_digest_set must hash a
        SET. Hashing sorted(xs) with duplicates made two different
        multisets collide-or-differ for the wrong reason."""
        return hashlib.sha256(json.dumps(
            sorted(set(xs)), separators=(",", ":")).encode()
        ).hexdigest()

    def counts(xs):
        return {"observations": len(xs), "distinct": len(set(xs))}
    registered = _require_count_identity(len(keys), attempted,
                                        verified)
    _bind = _receipt_bindings(_verdict, allow)
    receipt = {"schema": RECEIPT_SCHEMA,
            "manifest_commit": full,
            "manifest_verdict": _bind["manifest_verdict"],
            "runtime_allowlist": _bind["runtime_allowlist"],
            "capsule_pin": {"path": DISP.CAPSULE_PATH,
                            "commit": caps_pin.get("commit"),
                            "blob_sha256": caps_pin.get(
                                "blob_sha256")},
            "registered_reuse_count": registered,
            "capsule_sha256": hashlib.sha256(caps_raw).hexdigest(),
            "transform_identity": CAP.transform_identity(),
            "attempted": attempted, "verified": verified,
            "v4_key_digest": dg(v4keys), "v3_key_digest": dg(v3keys),
            "original_t_digest_set": dg(tset),
            "original_t_counts": counts(tset),
            "body_digest_set": dg(bset),
            "body_counts": counts(bset),
            "artifact_digest_set": dg(aset),
            "artifact_counts": counts(aset),
            "v4_key_counts": counts(v4keys),
            "v3_key_counts": counts(v3keys),
            "claims": {k: dict(sorted(v.items()))
                       for k, v in sorted(outcomes.items())},
            "http_requests": _net.attempts,
            "http_counter_source": "MEASURED_SENTINEL",
            "interpreter": sys.version.split()[0],
            "claim_scope": "MANIFEST_OWNED_RESTAGE_VERIFICATION",
            "authorizes": "NOTHING"}
    return receipt


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
    # NOT an `or`: the previous form (refuses "OPEN slots" OR refuses
    # "not a prestart PASS") was always carried by the SECOND branch,
    # so the first was never exercised -- which is exactly how the
    # vacuous slots read survived. Assert the one that is true today
    # and doctor the OPEN-slots path directly, below, by injection.
    assert refuses(lambda: run(head), "not a prestart PASS")
    # the OLD record-owned API cannot produce this receipt: it has no
    # manifest input at all, so no call to it can yield a
    # manifest-owned claim (codex 0410Z fix 1 doctor)
    import inspect
    old = inspect.signature(LIN.verify_restage_lineage).parameters
    assert "manifest_commit" not in old
    new = inspect.signature(
        LIN.verify_restage_lineage_pinned).parameters
    assert "manifest_commit" in new
    # --- codex 0534Z: BEHAVIORAL doctors over the factored positive
    # preflight. The previous versions asserted on run()'s SOURCE
    # STRING and were satisfied by a call site whose symbol did not
    # exist. These EXECUTE the path. ---
    # (0) the promoted resolver must actually RESOLVE and be callable
    assert callable(LIN.resolve_pinned_bytes), \
        "the promoted pin resolver must exist under its real name"
    assert not hasattr(LIN, "resolveresolve_pinned_bytes"), \
        "the misspelled definition must be gone, not shadowed"
    FULL = {"reuse_or_bridge": {f"mag/ck/2026-01-{d:02d}": {}
                                for d in range(1, 6)}}
    _pin = {"commit": "0" * 40, "blob_sha256": "1" * 64}

    def _full_resolver(repo, commit, path):
        return (json.dumps(FULL).encode("utf-8"), dict(_pin))

    def _ok_allow(repo, commit):
        return {"manifest_commit": commit, "manifest_state": "CLOSED",
                "pins_checked": 9, "pins": []}
    # (a) a SMALLER capsule sitting at <commit>:path can never shrink
    # the key universe: the universe comes from the PIN. We prove the
    # commit-path reader is UNTOUCHED by recording every git argv the
    # preflight issues and requiring none of them to name
    # <commit>:CAPSULE_PATH.
    _seen = []
    _real_run = subprocess.run

    def _spy(cmd, *a, **kw):
        _seen.append(list(cmd) if isinstance(cmd, (list, tuple))
                     else [str(cmd)])
        return _real_run(cmd, *a, **kw)
    subprocess.run = _spy
    try:
        _craw, _cpin, _caps, _allow = _resolve_batch_preflight(
            "f" * 40, resolver=_full_resolver,
            allowlist_check=_ok_allow)
    finally:
        subprocess.run = _real_run
    assert sorted(_caps["reuse_or_bridge"]) == \
        sorted(FULL["reuse_or_bridge"]), \
        "the key universe must be the FULL pinned set"
    _forbidden = f"{'f' * 40}:{DISP.CAPSULE_PATH}"
    assert not any(_forbidden in str(tok) for c in _seen
                   for tok in c), \
        "the batch must never read the capsule from <commit>:path"
    assert _allow["pins_checked"] == 9
    # (b) a fake allowlist that RAISES on a dirty pinned path must
    # refuse in preflight, before any receipt assembly

    def _dirty_allow(repo, commit):
        raise RuntimeError("RUNTIME_ALLOWLIST_VIOLATION: "
                           "[('accrual_impl', 'x.py', 'aa!=bb')]")
    assert refuses(lambda: _resolve_batch_preflight(
        "f" * 40, resolver=_full_resolver,
        allowlist_check=_dirty_allow), "runtime allowlist REFUSED")
    # (c) an allowlist that checked NOTHING cannot support a receipt
    assert refuses(lambda: _resolve_batch_preflight(
        "f" * 40, resolver=_full_resolver,
        allowlist_check=lambda r, c: {"pins_checked": 0, "pins": []}),
        "pins_checked=0")
    # (d) a missing/misspelled resolver REFUSES rather than raising
    # AttributeError out of the positive path. The needle is the
    # guard's OWN wording: "not callable" alone was ALSO matched by
    # Python's incidental "'str' object is not callable" once the
    # guard was deleted, so that doctor passed a live mutant -- the
    # same accidental-substring failure as the defect being repaired.
    assert refuses(lambda: _resolve_batch_preflight(
        "f" * 40, resolver="not-callable",
        allowlist_check=_ok_allow),
        "the manifest-pin resolver is ABSENT")
    # (e) the count identity is EXECUTED, not read out of source
    assert _require_count_identity(1420, 1420, 1420) == 1420
    for _r, _a, _v in ((1420, 1419, 1419), (1420, 1420, 1419),
                       (1419, 1420, 1420), (0, 0, 1)):
        assert refuses(
            lambda r=_r, a=_a, v=_v: _require_count_identity(r, a, v),
            "registered_reuse_count"), (_r, _a, _v)

    # --- codex 1400Z P2 #5: the OPEN-slots precondition, DIRECTLY ---
    assert _validate_manifest_verdict(
        {"verdict": "PASS", "slots_open": 0, "pins_checked": 5},
        "a" * 40) == 0
    # the exact inconsistent shape: a PASS carrying open slots
    assert refuses(lambda: _validate_manifest_verdict(
        {"verdict": "PASS", "slots_open": 7}, "a" * 40),
        "still has 7 OPEN slots")
    for bad in ({"verdict": "PASS"},
                {"verdict": "PASS", "slots_open": True},
                {"verdict": "PASS", "slots_open": "0"},
                {"verdict": "PASS", "slots_open": None}):
        assert refuses(lambda b=bad: _validate_manifest_verdict(
            b, "a" * 40), "no integer slots_open"), bad
    assert refuses(lambda: _validate_manifest_verdict(
        {"verdict": "REFUSE", "slots_open": 0}, "a" * 40),
        "not a prestart PASS")

    # --- codex 1327Z P1 #3: receipt bindings are COPIED, doctored
    # with deliberately DISTINCT injected values so a hard-coded or
    # wrong-key field cannot pass ---
    good_v = {"verdict": "PASS", "mode": "prestart",
              "slots_open": 0, "pins_checked": 31}
    good_a = {"pins_checked": 17, "pins": []}
    b = _receipt_bindings(good_v, good_a)
    # 31 and 17 are distinct from each other and from any literal the
    # old code could have restated
    assert b["manifest_verdict"] == {
        "verdict": "PASS", "mode": "prestart", "slots_open": 0,
        "pins_checked": 31}, b
    assert b["runtime_allowlist"] == {"result": "PASS",
                                      "pins_checked": 17}, b
    # the exact case codex reproduced: injected slots_open = 7 must
    # NOT be recorded as 0
    assert refuses(lambda: _receipt_bindings(
        dict(good_v, slots_open=7), good_a), "7 OPEN slots")
    assert refuses(lambda: _receipt_bindings(
        dict(good_v, verdict="REFUSE"), good_a), "may not be assembled")
    assert refuses(lambda: _receipt_bindings(
        dict(good_v, mode="poststart"), good_a), "requires prestart")
    assert refuses(lambda: _receipt_bindings(
        dict(good_v, pins_checked=0), good_a), "pins_checked=0")
    for bad in ({"pins_checked": 0}, {"pins_checked": None}, {},
                {"checked": [1, 2, 3]}):
        assert refuses(lambda bad=bad: _receipt_bindings(good_v, bad),
                       "allowlist reported pins_checked")
    # a verifier that reports no integer slots_open cannot be copied
    for miss in ({"verdict": "PASS", "mode": "prestart",
                  "pins_checked": 3},
                 dict(good_v, slots_open=True)):
        assert refuses(lambda m=miss: _receipt_bindings(m, good_a),
                       "no integer slots_open")

    # --- codex 1327Z P1 #4: the sentinel is whole-operation and
    # EXCEPTION-SAFE ---
    import socket as _sock
    _orig_sock, _orig_conn = _sock.socket, _sock.create_connection
    # (a) a typed refusal mid-run must still restore BOTH hooks --
    # codex observed socket_restored_after_refusal = False
    assert refuses(lambda: run("not-a-real-commit"), "does not resolve")
    assert _sock.socket is _orig_sock, \
        "socket.socket was left globally replaced after a refusal"
    assert _sock.create_connection is _orig_conn, \
        "create_connection was left globally replaced after a refusal"
    # ...and after a refusal raised DEEPER in, past preflight
    assert refuses(lambda: run(head), "not a prestart PASS")
    assert _sock.socket is _orig_sock and \
        _sock.create_connection is _orig_conn
    # (b) a PREFLIGHT connect attempt is blocked, COUNTED, and cannot
    # emit a receipt -- preflight now runs inside the sentinel
    def _connecting_resolver(repo, commit, path):
        _sock.create_connection(("example.invalid", 80))
        raise AssertionError("the connect should have been blocked")
    with NONET.no_network() as _probe:
        assert _probe.attempts == 0
        assert refuses(lambda: _resolve_batch_preflight(
            "f" * 40, resolver=_connecting_resolver,
            allowlist_check=lambda r, c: {"pins_checked": 1}),
            "did not resolve")
        assert _probe.attempts == 1, _probe.attempts
    assert _sock.socket is _orig_sock and \
        _sock.create_connection is _orig_conn

    # codex 0445Z item 3: the exact 360 + 1060 mixed composition
    # must ASSEMBLE DETERMINISTICALLY before the evidence-host run
    mixed = {}
    for _ in range(1057):
        _tally(mixed, {"artifact_claim_kind": "SUPPORT_SERIES",
                       "outcome": "ADMITTED"})
    for _ in range(3):
        _tally(mixed, {"artifact_claim_kind": "SUPPORT_SERIES",
                       "outcome": "ADMITTED_ABSENCE"})
    for _ in range(360):
        _tally(mixed, {"artifact_claim_kind": "STATION_PRESENCE",
                       "support_outcome": "NOT_APPLICABLE"})
    assembled = {k: dict(sorted(v.items()))
                 for k, v in sorted(mixed.items())}
    assert assembled == {
        "STATION_PRESENCE": {"NOT_APPLICABLE": 360},
        "SUPPORT_SERIES": {"ADMITTED": 1057,
                           "ADMITTED_ABSENCE": 3}}, assembled
    assert sum(sum(v.values()) for v in assembled.values()) == 1420
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
