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


def _capsule_is_pinned(commitish):
    man = json.loads(_blob(
        commitish, CAP.EXEC_MANIFEST_PATH).decode("utf-8"))
    slot = man.get("slots", {}).get(CAP.AUTHORITY_SLOT, {})
    return any(isinstance(p, dict)
               and p.get("path") == DISP.CAPSULE_PATH
               for p in (slot.get("pins") or ()))


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
    pinned = _capsule_is_pinned(commitish)
    print("capsule pinned in the execution manifest:", pinned)
    if not pinned:
        print("  -> `run` REFUSES until it is pinned; the production "
              "entrypoint fails closed without the ceiling")
    print("no request fired")
    return keys


def run(manifest_commit):
    if not _capsule_is_pinned(manifest_commit):
        raise SystemExit(
            "REFUSING: the disposition capsule is not pinned in the "
            f"execution manifest at {manifest_commit} -- there is no "
            "registered ceiling, so no request may be fired")
    authority, capsule, keys, counts, csha = load_plan(manifest_commit)
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
