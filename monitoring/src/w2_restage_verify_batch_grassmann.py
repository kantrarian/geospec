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
    _net = NONET.no_network()
    _net.__enter__()
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
    _net.__exit__()
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
    # codex 0534Z P1: populate these from the reports that were
    # actually returned. Hard-coded literals restate the happy path
    # instead of recording it -- the receipt would have said PASS /
    # prestart / 0 no matter what the verifier found.
    _slots = _verdict.get("slots") or {}
    _open = [n for n, sl in _slots.items()
             if isinstance(sl, dict) and sl.get("status") == "OPEN"]
    receipt = {"schema": RECEIPT_SCHEMA,
            "manifest_commit": full,
            "manifest_verdict": {
                "verdict": _verdict.get("verdict"),
                "mode": _verdict.get("mode", "prestart"),
                "slots_open": len(_open),
                "pins_checked": _verdict.get("pins_checked")},
            "runtime_allowlist": {
                "result": "PASS",
                "pins_checked": allow["pins_checked"]},
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
    # codex 0534Z: the emitted counts must EQUAL the injected reports
    if receipt["runtime_allowlist"]["pins_checked"] != \
            allow["pins_checked"]:
        _b("the receipt's allowlist pins_checked does not equal the "
           "allowlist report it claims to record")
    if receipt["manifest_verdict"]["verdict"] != \
            _verdict.get("verdict"):
        _b("the receipt's manifest verdict does not equal the "
           "verifier verdict it claims to record")
    if receipt["manifest_verdict"]["slots_open"] != 0:
        _b("a post-manifest receipt may not record OPEN slots")
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
