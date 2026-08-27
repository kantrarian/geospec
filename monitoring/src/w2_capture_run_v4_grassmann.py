#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE SUCCESSOR v4 CAPTURE RUNNER (grassmann) -- codex 1746Z
finding 2.

A SEPARATE successor runner. The historical v3 driver
(`w2_capture_run_grassmann.py`) is deliberately NOT mutated: it is
the executable record of the 1,794-key run and stays pinned to its
own manifest, authority, census and retired lane vocabulary.

This runner fires ONLY the keys the manifest-pinned disposition
capsule places in `HTTP_CAPTURE`. It does not compute that set and
cannot widen it: the ceiling lives in the production entrypoint
(`capture_authorized`), which reopens the pinned capsule itself, so
even a defect in this loop cannot reach the opener with an
unauthorized key.

Modes:
  plan [commitish]   ZERO NETWORK. Enumerate the closed HTTP
                     partition from the committed capsule, verify it
                     against the registered authority, and report the
                     exact count and per-carrier breakdown.
  run <manifest>     Fire the plan against the REVIEWED manifest
                     commit. Refuses unless the capsule is pinned
                     there. Resumable; one request per key per
                     invocation; typed refusals ledgered.

Firing additionally requires, and this runner does not grant:
  - codex's capture-readiness PASS, and
  - an in-session owner go on this host.
"""
import hashlib
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC
import w2_disposition_capsule_grassmann as DISP

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
AUTHORITY_PATH = DISP.AUTHORITY_PATH
# a NEW named v4 store and ledger -- the v3 store is historical
# evidence and is never written to by this runner
STORE_PHYSICAL = "E:/GeoSpec/w2_capture_store_v4"
STORE_ID = "s4t-w2-capture-v4"
STORE_ROOT = "s4t://geospec/w2/capture_v4"
STAGED_DIR = os.path.join(REPO, "docs", "f2g_window2_execution",
                          "staged_envelopes_v4")
LEDGER = os.path.join(REPO, "docs", "f2g_window2_execution",
                      "capture_run_ledger_v4.jsonl")
PACING_S = 1.0
TIMEOUT_S = 90
UA = "geospec-w2-capture/1.0 (kantrarian/geospec window-2)"

_last_by_host = {}


def _blob(commitish, path):
    p = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                        f"{commitish}:{path}"], capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"REFUSING: {path} unreadable at {commitish}")
    return p.stdout


def _paced_verified_opener(url):
    from urllib.parse import urlsplit
    host = urlsplit(url).netloc
    wait = _last_by_host.get(host, 0) + PACING_S - time.monotonic()
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
        return (exc.code, {}, b"", getattr(exc, "url", url))


def load_plan(commitish="HEAD"):
    """ZERO NETWORK. Reopen the committed capsule + authority and
    return the closed HTTP partition, verified."""
    araw = _blob(commitish, AUTHORITY_PATH)
    authority = json.loads(araw.decode("utf-8"))
    craw = _blob(commitish, DISP.CAPSULE_PATH)
    capsule = json.loads(craw.decode("utf-8"))
    if capsule["authority"]["blob_sha256"] != \
            hashlib.sha256(araw).hexdigest():
        raise SystemExit(
            "REFUSING: the capsule does not bind the authority at "
            f"{commitish}")
    # the runner plans REQUESTS, so the ceiling contract is the
    # right one; it reports that lineage evidence is unverified
    counts = DISP.verify_ceiling(capsule, authority)
    keys = sorted(capsule["http_capture"])
    if len(keys) != counts["HTTP_CAPTURE"]:
        raise SystemExit("REFUSING: plan size diverges from the "
                         "verified partition")
    return authority, capsule, keys, counts, \
        hashlib.sha256(craw).hexdigest()


def capsule_pin_status(commitish, candidate_sha=None):
    """codex 0110Z P0-1: the old `_capsule_is_pinned` returned True
    merely because a pin with that PATH existed. It never checked the
    slot status, never resolved the pin's commit, never recomputed the
    pinned blob, and never compared it to the capsule the runner had
    just enumerated.

    That is the nominal-vs-actual binding again, and after Step 2 it
    would have been actively dangerous: `plan` could print a clean 635
    and `pinned=True` from a NEW candidate capsule while
    capture_authorized -- which correctly consumes the capsule FROM
    THE PIN -- still saw the OLD one and refused every key. A path is
    not an object.

    Returns a typed status; never a bare bool.
    """
    out = {"slot": CAP.AUTHORITY_SLOT, "path": DISP.CAPSULE_PATH,
           "slot_bound": False, "pin_count": 0, "pin_commit": None,
           "pinned_blob_sha256": None, "pin_recomputes": False,
           "candidate_sha256": candidate_sha, "current_pin": False,
           "runnable": False, "reason": None}

    def no(why):
        out["reason"] = why
        return out
    try:
        man = json.loads(_blob(
            commitish, CAP.EXEC_MANIFEST_PATH).decode("utf-8"))
    except Exception as exc:
        return no(f"the execution manifest is unreadable at "
                  f"{commitish} ({type(exc).__name__})")
    slot = man.get("slots", {}).get(CAP.AUTHORITY_SLOT) or {}
    out["slot_bound"] = slot.get("status") == "BOUND"
    if not out["slot_bound"]:
        return no(f"slot {CAP.AUTHORITY_SLOT} is "
                  f"{slot.get('status')!r}, not BOUND -- an OPEN slot "
                  "pins nothing that can serve as a ceiling")
    pins = [p for p in (slot.get("pins") or ())
            if isinstance(p, dict) and p.get("path") == DISP.CAPSULE_PATH]
    out["pin_count"] = len(pins)
    if len(pins) != 1:
        return no(f"{len(pins)} capsule pins at {DISP.CAPSULE_PATH}; "
                  "exactly one is required")
    pin = pins[0]
    out["pin_commit"] = pin.get("commit")
    out["pinned_blob_sha256"] = pin.get("blob_sha256")
    try:
        raw = _blob(pin["commit"], DISP.CAPSULE_PATH)
    except Exception as exc:
        return no(f"the pinned capsule commit "
                  f"{str(pin.get('commit'))[:12]} does not resolve "
                  f"({type(exc).__name__})")
    got = hashlib.sha256(raw).hexdigest()
    out["pin_recomputes"] = got == pin.get("blob_sha256")
    if not out["pin_recomputes"]:
        return no(f"the pinned capsule blob does not recompute: pin "
                  f"says {str(pin.get('blob_sha256'))[:12]}, "
                  f"{str(pin.get('commit'))[:12]} gives {got[:12]}")
    if candidate_sha is not None and candidate_sha != got:
        out["current_pin"] = False
        return no(f"the manifest pins capsule {got[:12]} but this "
                  f"plan enumerated {candidate_sha[:12]} -- the "
                  "entrypoint would consume the PINNED bytes and "
                  "refuse every key in this plan")
    out["current_pin"] = True
    out["runnable"] = True
    return out


def plan(commitish="HEAD"):
    import collections
    _a, _c, keys, counts, csha = load_plan(commitish)
    print(f"capsule {csha[:16]} @ {commitish}")
    print("partition:", counts)
    print("PLAN (HTTP_CAPTURE only):", len(keys))
    for lc, n in sorted(collections.Counter(
            "/".join(k.split("/")[:2]) for k in keys).items()):
        print(f"  {lc:32s} {n}")
    print("first:", keys[0])
    print("last :", keys[-1])
    st = capsule_pin_status(commitish, candidate_sha=csha)
    print(f"CURRENT_PIN={str(st['current_pin']).lower()}  "
          f"{'RUNNABLE' if st['runnable'] else 'NOT_RUNNABLE'}")
    print(f"  slot_bound={st['slot_bound']} pins={st['pin_count']} "
          f"pin_commit={str(st['pin_commit'])[:12]} "
          f"pin_blob={str(st['pinned_blob_sha256'])[:12]} "
          f"recomputes={st['pin_recomputes']}")
    print(f"  candidate={str(st['candidate_sha256'])[:12]}")
    if not st["runnable"]:
        print("  -> this plan enumerates a verified CANDIDATE, not a "
              "runnable ceiling:")
        print("     " + str(st["reason"]))
    print("no request fired")
    return keys


def run(manifest_commit):
    # codex 0110Z P0-1(3): enumerate from the VERIFIED MANIFEST PIN so
    # this loop and capture_authorized() consume ONE object. Reading
    # the capsule at the named descendant while the entrypoint reads
    # it from the pin is two objects wearing one name.
    # codex 0151Z P0-1(3): the runner must bind ITS OWN executed
    # bytes before anything else. A pin that is never checked against
    # executed disk bytes is nominal, not load-bearing -- and this
    # executable is the one that would spend the 635.
    import w2_accrual_instrument_cayley as ACC
    _me = "monitoring/src/" + os.path.basename(__file__)
    try:
        _allow = ACC.runtime_allowlist_check(REPO, manifest_commit)
    except Exception as exc:
        raise SystemExit(
            "REFUSING: the runtime allowlist refused at "
            f"{manifest_commit} ({type(exc).__name__}: "
            f"{str(exc)[:200]})")
    _checked = {p for _s, p in (_allow.get("pins") or ())}
    if _me not in _checked:
        raise SystemExit(
            f"REFUSING: {_me} is not among the BOUND pins checked "
            f"against executed disk bytes at {manifest_commit} "
            f"({len(_checked)} checked) -- the executable that would "
            "spend the ceiling is not itself bound")
    _probe = load_plan(manifest_commit)[4]
    st = capsule_pin_status(manifest_commit, candidate_sha=_probe)
    if not st["runnable"]:
        raise SystemExit(
            "REFUSING: no runnable capsule ceiling at "
            f"{manifest_commit} -- {st['reason']}")
    authority, capsule, keys, counts, csha = load_plan(
        st["pin_commit"])
    if csha != st["pinned_blob_sha256"]:
        raise SystemExit(
            "REFUSING: the capsule reopened from the pin does not "
            "match the pinned digest")
    print("plan:", len(keys), "keys; capsule", csha[:16])
    os.makedirs(STORE_PHYSICAL, exist_ok=True)
    os.makedirs(STAGED_DIR, exist_ok=True)
    import w2_expected_contracts_gen_cayley as GEN
    repro_raw = json.dumps(GEN.build(REPO), sort_keys=True)

    def reproducer():
        return json.loads(repro_raw)
    tally = {"CAPTURED": 0, "SKIPPED": 0, "REFUSED": 0, "ERROR": 0}
    t0 = time.monotonic()
    for i, key in enumerate(keys):
        lane, ck, day = key.split("/")
        stem = CAP._path_tokens(lane, ck, day)
        rp = os.path.join(STAGED_DIR,
                          stem + ACC.STAGED_CLASS_SUFFIX["record"])
        if os.path.exists(rp):
            tally["SKIPPED"] += 1
            continue
        entry = {"key": key, "seq": i}
        try:
            _rp, _tp, rec, _tr = CAP.capture_authorized(
                REPO, manifest_commit, AUTHORITY_PATH, lane, ck, day,
                STORE_PHYSICAL, STAGED_DIR, STAGED_DIR, None,
                opener=_paced_verified_opener,
                authority_reproducer=reproducer)
            s = ACC.authoritative_static_contract(authority, lane,
                                                  ck, day)
            with open(os.path.join(
                    STORE_PHYSICAL,
                    rec["raw_body_sha256"] + ".body"), "rb") as f:
                body = f.read()
            art = CAP.admission_transform(lane, body, s)
            for cls, obj in (("contract", s), ("artifact", art)):
                CAP._write_once_json(
                    os.path.join(STAGED_DIR,
                                 stem + ACC.STAGED_CLASS_SUFFIX[cls]),
                    obj, "CAPTURE_RECORD_DIVERGENT")
            entry.update(status="CAPTURED",
                         raw_body_sha256=rec["raw_body_sha256"],
                         raw_body_bytes=rec["raw_body_bytes"],
                         outcome=art.get("outcome"),
                         capture_time_utc=rec["capture_time_utc"])
            tally["CAPTURED"] += 1
        except CAP.CaptureRefusal as exc:
            entry.update(status="REFUSED", refusal=str(exc)[:600])
            tally["REFUSED"] += 1
        except Exception as exc:
            entry.update(status="ERROR",
                         error=f"{type(exc).__name__}: "
                               f"{str(exc)[:500]}")
            tally["ERROR"] += 1
        with open(LEDGER, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(keys)}] {tally} "
                  f"({(time.monotonic() - t0) / 60:.1f} min)",
                  flush=True)
    print("FINAL:", tally)


def _selftest():
    """Zero-network locks: the plan is EXACTLY the capsule's closed
    HTTP partition, and `run` refuses without a pinned ceiling."""
    _a, capsule, keys, counts, _s = load_plan("HEAD")
    assert len(keys) == counts["HTTP_CAPTURE"]
    assert sorted(set(keys)) == keys, "plan keys must be unique"
    # the plan is a SUBSET of nothing else: no REUSE or PREDECESSOR
    # key can appear in it
    assert not (set(keys) & set(capsule["reuse_or_bridge"]))
    assert not (set(keys) & set(capsule["predecessor"]))
    # and every planned key is one the capsule would authorize
    for k in (keys[0], keys[len(keys) // 2], keys[-1]):
        assert DISP.may_fire(capsule, *k.split("/")) is True
    # a REUSE key is NOT in the plan and would be refused
    rk = sorted(capsule["reuse_or_bridge"])[0]
    assert rk not in set(keys)
    try:
        DISP.may_fire(capsule, *rk.split("/"))
        raise AssertionError("a REUSE key must never be firable")
    except DISP.DispositionRefusal:
        pass
    # ---- codex 0110Z P0-1(4): the DESCENDANT-CHANGES-CAPSULE doctor
    # A descendant changes the capsule at the same path while the
    # manifest retains the OLD pin. The candidate plan may report its
    # count; it must NOT report current_pin, and run() must refuse
    # BEFORE any opener. This is the exact shape that would otherwise
    # appear after Step 2.
    real = capsule_pin_status("HEAD", candidate_sha=_s)
    # (a) a candidate whose bytes are NOT the pinned bytes
    drift = capsule_pin_status("HEAD", candidate_sha="0" * 64)
    assert drift["current_pin"] is False, drift
    assert drift["runnable"] is False, drift
    assert "refuse every key" in (drift["reason"] or ""), drift
    # ...while the PIN itself still verifies, so the refusal is about
    # DIVERGENCE and not about a broken manifest
    assert drift["slot_bound"] is True, drift
    assert drift["pin_count"] == 1, drift
    assert drift["pin_recomputes"] is True, drift
    # (b) with no candidate supplied the pin still verifies on its own
    bare = capsule_pin_status("HEAD")
    assert bare["pin_recomputes"] is True and bare["runnable"] is True
    # (c) THREE separate refusal paths, each asserted on ITS OWN
    # distinctive needle.
    #
    # codex 0151Z P0-2 said this doctor was AMBIENT: it ran only
    # under `if not real["runnable"]`, so re-pinning the capsule
    # would have silenced it. Constructing the divergence fixed that
    # -- and immediately exposed a second defect in my own repair:
    # run() now refuses on the ALLOWLIST first, so a doctor asserting
    # only "REFUSING" passed on a refusal it did not construct. That
    # is the accidental-needle defect, in a doctor written minutes
    # after I argued the same point to cayley. Each path below names
    # a string only that path can produce.
    import w2_no_network_grassmann as _NN
    import w2_accrual_instrument_cayley as _ACC
    _real_status = capsule_pin_status
    _real_allow = _ACC.runtime_allowlist_check
    _real_fetch = CAP.http_fetch
    _g = globals()

    def _probe_run(needle, *, allow=None, status=None):
        """Run run('HEAD') under injected conditions; require a typed
        refusal carrying `needle`, zero network attempts, and NO
        opener."""
        reached = []

        def _raiser(*a, **k):
            reached.append(1)
            raise AssertionError("OPENER REACHED")
        if allow is not None:
            _ACC.runtime_allowlist_check = allow
        if status is not None:
            _g["capsule_pin_status"] = status
        try:
            with _NN.no_network() as net:
                CAP.http_fetch = _raiser
                try:
                    run("HEAD")
                    raise AssertionError(
                        f"run() accepted a condition it must refuse "
                        f"({needle})")
                except SystemExit as exc:
                    assert needle in str(exc), \
                        f"expected {needle!r}, got {str(exc)[:160]!r}"
                assert net.attempts == 0, net.attempts
        finally:
            CAP.http_fetch = _real_fetch
            _ACC.runtime_allowlist_check = _real_allow
            _g["capsule_pin_status"] = _real_status
        assert not reached, "an opener was reached before refusal"

    _me_path = "monitoring/src/" + os.path.basename(
        R_FILE if (R_FILE := __file__) else "")

    def _allow_without_me(repo, commit):
        return {"pins_checked": 3,
                "pins": [("accrual_impl", "monitoring/src/other.py")]}

    def _allow_raises(repo, commit):
        raise _ACC.InstrumentRefusal(
            "RUNTIME_ALLOWLIST_VIOLATION: [('accrual_impl', "
            f"'{_me_path}', 'aa11!=bb22')]")

    def _allow_ok(repo, commit):
        return {"pins_checked": 9,
                "pins": [("accrual_impl", _me_path)]}

    def _forced_divergent(commitish, candidate_sha=None):
        st = _real_status(commitish, candidate_sha)
        st.update(current_pin=False, runnable=False,
                  reason="CONSTRUCTED_CAPSULE_DIVERGENCE")
        return st
    # (c1) codex P0-1(4): the runner ABSENT from the checked BOUND pins
    _probe_run("is not among the BOUND pins", allow=_allow_without_me)
    # (c2) codex P0-1(4): the runner's DISK bytes diverge from its pin
    _probe_run("the runtime allowlist refused", allow=_allow_raises)
    # (c3) codex P0-2: a CONSTRUCTED divergent capsule, reached only
    # because the allowlist is stubbed to PASS -- so this asserts the
    # capsule path and cannot pass on the allowlist refusal
    _probe_run("CONSTRUCTED_CAPSULE_DIVERGENCE",
               allow=_allow_ok, status=_forced_divergent)
    print("  P0-1/P0-2 doctors: runner-unbound, runner-disk-divergent "
          "and constructed-capsule-divergence each refused before any "
          "opener, sentinel attempts 0")
    # live status is corroboration ONLY; it never gates the doctors
    print(f"  live corroboration: current_pin={real['current_pin']} "
          f"runnable={real['runnable']}")
    print(f"w2_capture_run_v4 selftest: ALL PASS (plan={len(keys)}, "
          "no network)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if mode == "plan":
        plan(sys.argv[2] if len(sys.argv) > 2 else "HEAD")
    elif mode == "--selftest":
        _selftest()
    elif mode == "run":
        if len(sys.argv) < 3:
            raise SystemExit("usage: run <manifest-commit>")
        run(sys.argv[2])
    else:
        raise SystemExit("usage: plan [commitish] | run <manifest> "
                         "| --selftest")


if __name__ == "__main__":
    main()
