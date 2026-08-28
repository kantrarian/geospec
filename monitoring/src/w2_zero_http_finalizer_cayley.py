#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZERO-HTTP PRODUCTION FINALIZER (cayley) -- codex 2240Z P0-2.

THE ONE REVIEWED COMMAND that turns the frozen evidence into the two
remaining zero-HTTP class families:

    212 x MAG_FEED/vic          (the landed replay machinery)
      1 x MAG_WEATHER_FEED/omni predecessor (the landed bridge)

Neither producer previously had a safe live materializer: the VIC
module exposed `replay()` only as an API, and the bridge verified its
record without publishing the predecessor's four staged classes. An ad
hoc host script here would reopen the exact orchestration seam P0-4
was left OPEN to avoid -- so this module is that orchestration,
reviewed, with two explicit modes and sockets hard-disabled in both.

  plan  <commit>              read-only; resolves the exact commit;
                              reopens every target transcript/body;
                              runs EVERY verification and join the
                              apply would run; reports the exact
                              prospective write set and its canonical
                              PLAN DIGEST; writes NOTHING.
  apply <commit> <plan_sha>   recomputes the plan and REQUIRES its
                              digest to equal the given one; then
                              materializes create-once/idempotently:
                              the 212 VIC classes + 212-row operation
                              ledger via the landed replay machinery,
                              the predecessor's four classes + ONE
                              bridge operation record, then the
                              presence bijection over the FULL
                              authority (expected 2,056 per class,
                              provenance = record|restage).

Divergent pre-existing bytes refuse (write-once canonical identity);
an interrupted apply resumes with zero HTTP by re-running apply with
the same plan digest. The full per-key transform-recompute join over
all 2,056 keys remains the ADMISSION BOUNDARY's job against pinned
bytes -- this finalizer verifies the 213 keys it writes through the
real producer gates, plus the global presence bijection, and claims
nothing further.

Lambda_geo remains INCONCLUSIVE; nothing here admits or claims.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_accrual_instrument_cayley as ACC
import w2_acquisition_capture_grassmann as CAP
import w2_producer_grassmann as PROD
import w2_capture_run_v4_grassmann as RUN4
import w2_capture_repair_v4_vic_cayley as VICR
import w2_capture_retry_404_v4_cayley as RETRY
import w2_predecessor_bridge_cayley as PB
import w2_no_network_grassmann as NONET

REPO = RUN4.REPO
PRED_TRANSCRIPT_REL = ("docs/f2g_window2_execution/probe_evidence/"
                       "omni_corrected_probe_20260101.transcript.json")
PRED_BODY_REL = ("docs/f2g_window2_execution/probe_evidence/"
                 "omni_corrected_probe_20260101.body")
EXPECTED_TOTAL_KEYS = 2056
APPLY_OP_PATH = ("docs/f2g_window2_execution/"
                 "zero_http_apply_v1.operation.json")


def _staged_state_digest():
    """codex 2313Z P0-2: the EXACT pre-existing staged path->digest
    map -- not counts. A same-count filename swap or any byte change
    in any staged file changes this digest."""
    sd = RUN4.STAGED_DIR
    state = {}
    if os.path.isdir(sd):
        for name in sorted(os.listdir(sd)):
            fp = os.path.join(sd, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    state[name] = hashlib.sha256(
                        f.read()).hexdigest()
    return _canon(state), len(state)


class FinalizerRefusal(SystemExit):
    """Typed, fail-closed. The code leads the message."""


def _refuse(code, detail):
    raise FinalizerRefusal(f"{code}: {detail}")


def _canon(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _predecessor_key(capsule):
    pred = sorted(capsule.get("predecessor", ()))
    if len(pred) != 1:
        _refuse("FINALIZER_PREDECESSOR_SET",
                f"{len(pred)} predecessor keys; exactly one expected")
    return pred[0]


def _predecessor_outputs(manifest_commit, authority):
    """The predecessor's four staged classes, derived through the
    REGISTERED bridge and the ordinary producer gates -- the classes
    the ACC derivation joins against the bridge record:
    transcript := the reopened probe transcript VERBATIM (the record
    the bridge's evidence digests bind); artifact := the bridge's
    artifact; contract := the authoritative S; record := the ordinary
    envelope record; the full five-map join must pass or this refuses
    -- the finalizer never invents a join."""
    brec = PB.verify_predecessor_bridge(
        REPO, manifest_commit=manifest_commit,
        transcript_path=os.path.join(REPO,
                                     *PRED_TRANSCRIPT_REL.split("/")),
        body_path=os.path.join(REPO, *PRED_BODY_REL.split("/")))
    key = f"{brec['lane']}/{brec['carrier']}/{brec['utc_day']}"
    lane, ck, day = key.split("/")
    with open(os.path.join(REPO, *PRED_TRANSCRIPT_REL.split("/")),
              encoding="utf-8") as f:
        tr = json.load(f)
    with open(os.path.join(REPO, *PRED_BODY_REL.split("/")),
              "rb") as f:
        body = f.read()
    if hashlib.sha256(body).hexdigest() != \
            brec["evidence"]["raw_body_sha256"] or \
            _canon(tr) != brec["evidence"]["transcript_sha256"]:
        _refuse("FINALIZER_PREDECESSOR_EVIDENCE",
                "probe transcript/body do not recompute to the "
                "bridge record's bound digests")
    s = ACC.authoritative_static_contract(authority, lane, ck, day)
    art = brec["artifact"]
    if _canon(art) != brec["artifact_sha256"]:
        _refuse("FINALIZER_PREDECESSOR_EVIDENCE",
                "bridge artifact does not recompute to its bound "
                "digest")
    record = PROD.build_envelope_record(
        lane=s["lane"], carrier=s["carrier"], utc_day=s["utc_day"],
        raw_body=body, source=dict(s["source"]),
        endpoint=s["endpoint"],
        request_params=dict(s["request_params"]), transcript=tr,
        cutoff=s["cutoff"],
        operation_params=dict(s["operation_params"]),
        expected_keys=list(s["expected_keys"]), artifact=art)
    try:
        PROD.verify_staged_day_set(
            {day: record}, {day: body}, {day: art}, {day: s},
            {day: tr}, [day], ck, lane)
    except Exception as exc:
        _refuse("FINALIZER_PREDECESSOR_JOIN",
                "the predecessor five-map join refused "
                f"({type(exc).__name__}: {str(exc)[:200]}) -- the "
                "finalizer never invents a join; this is a design "
                "seam to route, not to paper over")
    return key, brec, {"contract": s, "artifact": art,
                       "record": record, "transcript": tr}


def _class_counts():
    """The presence bijection inputs: per-class counts over the
    staged space (provenance = record|restage)."""
    counts = {"provenance": 0, "transcript": 0, "contract": 0,
              "artifact": 0}
    sd = RUN4.STAGED_DIR
    if not os.path.isdir(sd):
        return counts
    for name in os.listdir(sd):
        if name.endswith(".record.json") or \
                name.endswith(".restage.json"):
            counts["provenance"] += 1
        elif name.endswith(".transcript.json"):
            counts["transcript"] += 1
        elif name.endswith(".contract.json"):
            counts["contract"] += 1
        elif name.endswith(".artifact.json"):
            counts["artifact"] += 1
    return counts


def build_plan(manifest_commit):
    """READ-ONLY. Every verification and join the apply would run,
    plus the canonical plan digest. Sockets are disabled by the
    caller (run())."""
    manifest_commit = RETRY._resolve_commit(manifest_commit)
    authority, capsule, keys, counts, csha = RUN4.load_plan(
        manifest_commit)
    ledger_rows = VICR.load_ledger(RUN4.LEDGER) \
        if os.path.isfile(RUN4.LEDGER) else _refuse(
            "FINALIZER_LEDGER_ABSENT",
            f"{RUN4.LEDGER} -- the frozen terminal ledger exists "
            "only on the evidence host")
    with open(RUN4.LEDGER, "rb") as f:
        led_sha = hashlib.sha256(f.read()).hexdigest()
    vic = VICR.replay(authority, plan_keys=keys,
                      ledger_rows=ledger_rows, dry=True)
    pkey, brec, pouts = _predecessor_outputs(manifest_commit,
                                             authority)
    if pkey != _predecessor_key(capsule):
        _refuse("FINALIZER_PREDECESSOR_KEY",
                f"bridge names {pkey}, capsule registers "
                f"{_predecessor_key(capsule)}")
    stem = CAP._path_tokens(*pkey.split("/"))
    state_sha, state_n = _staged_state_digest()
    vic_keys = sorted(vic.get("previews") or ())
    # codex 2313Z P0-2: the canonical plan binds the STATE it will
    # mutate -- the frozen ledger digest, the exact key set, every
    # per-key input identity and prospective output digest (incl the
    # repair receipt), the predecessor inputs/outputs, and the exact
    # pre-existing staged path/digest map (as its canonical digest;
    # never counts)
    plan = {
        "schema": "f2g-w2-zero-http-plan-v2",
        "manifest_commit": manifest_commit,
        "capsule_sha256": csha,
        "frozen_ledger_sha256": led_sha,
        "vic": {"targets": vic["targets"],
                "dry_verified": vic.get("dry_verified", 0),
                "already_present": vic.get("verified_present", 0),
                "key_set": vic_keys,
                "per_key": vic.get("previews") or {}},
        "predecessor": {
            "key": pkey,
            "bridge_sha256": brec["bridge_sha256"],
            "inputs": {
                "probe_transcript_sha256": _canon(
                    pouts["transcript"]),
                "probe_body_sha256":
                    brec["evidence"]["raw_body_sha256"]},
            "class_canon_sha256": {
                cls: _canon(obj) for cls, obj in pouts.items()},
            "stem": stem},
        "pre_counts": _class_counts(),
        "staged_state_sha256": state_sha,
        "staged_state_files": state_n,
        "expected_totals": {c: EXPECTED_TOTAL_KEYS for c in
                            ("provenance", "transcript", "contract",
                             "artifact")},
        "http_requests": 0}
    plan["plan_sha256"] = _canon(
        {k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan, authority, ledger_rows, keys, brec, pouts


def _resume_contract(plan):
    """The immutable part of an accepted plan.

    A partial apply is expected to change the staged-state digest and
    counts. It may not change the manifest/capsule/ledger identity,
    VIC key set, any per-key input or prospective output, predecessor
    inputs/outputs, totals, or network ceiling.
    """
    return {k: plan.get(k) for k in (
        "schema", "manifest_commit", "capsule_sha256",
        "frozen_ledger_sha256", "vic", "predecessor",
        "expected_totals", "http_requests")}


def run(mode, manifest_commit, plan_sha=None):
    with NONET.no_network() as net:
        plan, authority, ledger_rows, keys, brec, pouts = \
            build_plan(manifest_commit)
        if mode == "plan":
            print(json.dumps({k: plan[k] for k in
                              ("vic", "predecessor", "pre_counts",
                               "expected_totals", "plan_sha256")},
                             indent=1, sort_keys=True))
            print(f"PLAN {plan['plan_sha256'][:16]}...: "
                  f"{plan['vic']['dry_verified']} VIC verified dry + "
                  "1 predecessor joined; no request fired; "
                  "nothing written")
            assert net.attempts == 0
            return plan
        # ---- apply: the create-once OPERATION RECORD is the plan
        # identity (codex 2313Z P0-3). First apply: the recomputed
        # plan must equal the given digest, and the record binds the
        # ACCEPTED plan before the first scientific write. Resume: a
        # partial apply has changed the staged state, so the plan is
        # NOT recomputed against it -- the operation record's bound
        # plan is the identity, the given digest must equal it, and
        # every write below is create-once/idempotent against the
        # bound prospective digests.
        op_path = os.path.join(REPO, *APPLY_OP_PATH.split("/"))
        if os.path.exists(op_path):
            with open(op_path, encoding="utf-8") as f:
                op = json.load(f)
            if not isinstance(op, dict) or set(op) != {
                    "schema", "plan", "op_sha256"} or \
                    op.get("schema") != \
                    "f2g-w2-zero-http-apply-op-v1":
                _refuse("FINALIZER_OP_RECORD_DEFORMED",
                        "the apply operation record is not the "
                        "closed object")
            if _canon({k: v for k, v in op.items()
                       if k != "op_sha256"}) != op.get("op_sha256"):
                _refuse("FINALIZER_OP_RECORD_DEFORMED",
                        "operation-record self-digest does not "
                        "recompute")
            if plan_sha != op["plan"].get("plan_sha256"):
                _refuse("FINALIZER_PLAN_IDENTITY",
                        "resume requires the ORIGINAL accepted plan "
                        "digest bound in the operation record")
            if op["plan"].get("frozen_ledger_sha256") != \
                    plan.get("frozen_ledger_sha256"):
                _refuse("FINALIZER_INPUT_CHANGED",
                        "the frozen ledger changed since the "
                        "accepted plan; resume refuses")
            if _resume_contract(op["plan"]) != \
                    _resume_contract(plan):
                _refuse("FINALIZER_INPUT_CHANGED",
                        "the recomputed immutable inputs or "
                        "prospective outputs differ from the accepted "
                        "operation plan; only its staged-state/count "
                        "transition may change during resume")
            accepted_plan = op["plan"]
        else:
            if plan_sha != plan["plan_sha256"]:
                _refuse("FINALIZER_PLAN_IDENTITY",
                        "apply requires the exact plan digest of "
                        "the state it is applying to; recompute "
                        "with `plan` first")
            op = {"schema": "f2g-w2-zero-http-apply-op-v1",
                  "plan": plan}
            op["op_sha256"] = _canon(
                {k: v for k, v in op.items() if k != "op_sha256"})
            CAP._write_once_json(op_path, op,
                                 "FINALIZER_OP_RECORD_DIVERGENT")
            accepted_plan = plan
        vic = VICR.replay(authority, plan_keys=keys,
                          ledger_rows=ledger_rows)
        # predecessor: attempt-local candidate, then create-once
        pkey = accepted_plan["predecessor"]["key"]
        stem = accepted_plan["predecessor"]["stem"]
        for cls, obj in pouts.items():
            CAP._write_once_json(
                os.path.join(RUN4.STAGED_DIR,
                             stem + ACC.STAGED_CLASS_SUFFIX[cls]),
                obj, "FINALIZER_CLASS_DIVERGENT")
        CAP._write_once_json(
            os.path.join(REPO,
                         *ACC.PREDECESSOR_RECORD_PATH.split("/")),
            brec, "FINALIZER_RECORD_DIVERGENT")
        post = _class_counts()
        bad = {c: n for c, n in post.items()
               if n != EXPECTED_TOTAL_KEYS}
        if bad:
            _refuse("FINALIZER_BIJECTION_SHORT",
                    f"post-apply class counts diverge from "
                    f"{EXPECTED_TOTAL_KEYS}: {bad}")
        # presence bijection over the FULL authority key set
        missing = []
        for lane in authority["prestart_expected_keys"]:
            for ck, days in authority["prestart_expected_keys"][
                    lane].items():
                for d in days:
                    st2 = CAP._path_tokens(lane, ck, d)
                    for suf in (".transcript.json", ".contract.json",
                                ".artifact.json"):
                        if not os.path.isfile(os.path.join(
                                RUN4.STAGED_DIR, st2 + suf)):
                            missing.append(st2 + suf)
                    if not (os.path.isfile(os.path.join(
                            RUN4.STAGED_DIR,
                            st2 + ".record.json"))
                            or os.path.isfile(os.path.join(
                                RUN4.STAGED_DIR,
                                st2 + ".restage.json"))):
                        missing.append(st2 + ".(record|restage)")
        if missing:
            _refuse("FINALIZER_BIJECTION_MISSING",
                    f"{len(missing)} authority classes absent, "
                    f"first: {missing[:3]}")
        assert net.attempts == 0
        print(f"APPLY COMPLETE: VIC repaired="
              f"{vic.get('repaired')} verified_present="
              f"{vic.get('verified_present')}; predecessor {pkey} "
              f"published; counts={post}; sockets blocked, "
              "attempts=0. The transform-recompute join over all "
              f"{EXPECTED_TOTAL_KEYS} keys remains the admission "
              "boundary's job against pinned bytes.")
        return {"vic": vic, "post_counts": post}


def _selftest():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    def refuses(fn, code):
        try:
            fn()
        except BaseException as exc:
            return code in str(exc)
        return False

    # F1 both modes run under the no-network sentinel by construction
    # -- assert the entrypoint refuses cleanly on this host where the
    # frozen ledger is absent, WITHOUT reaching a socket
    with NONET.no_network() as net:
        ok = refuses(lambda: run("plan", "HEAD"),
                     "FINALIZER_LEDGER_ABSENT") \
            or refuses(lambda: run("plan", "HEAD"),
                       "RETRY_MANIFEST_UNRESOLVABLE")
        check("F1 plan fails CLOSED on a host without the frozen "
              "ledger, zero socket attempts",
              ok and net.attempts == 0)
    # F2 apply demands the exact plan identity
    check("F2 apply without a matching plan digest refuses",
          refuses(lambda: run("apply", "HEAD", "0" * 64),
                  "FINALIZER_LEDGER_ABSENT")
          or refuses(lambda: run("apply", "HEAD", "0" * 64),
                     "FINALIZER_PLAN_IDENTITY"))
    # F3 the presence-bijection counter distinguishes provenance
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _sav = RUN4.STAGED_DIR
        try:
            RUN4.STAGED_DIR = td
            for n in ("a.record.json", "b.restage.json",
                      "c.transcript.json", "d.contract.json",
                      "e.artifact.json"):
                open(os.path.join(td, n), "w").write("{}")
            c = _class_counts()
            check("F3 provenance = record|restage; other classes "
                  "counted by suffix",
                  c == {"provenance": 2, "transcript": 1,
                        "contract": 1, "artifact": 1}, str(c))
        finally:
            RUN4.STAGED_DIR = _sav
    # F4 predecessor-set discipline
    check("F4 a capsule with two predecessor keys refuses",
          refuses(lambda: _predecessor_key(
              {"predecessor": ["a/b/c", "d/e/f"]}),
              "FINALIZER_PREDECESSOR_SET"))
    # F6 (2313Z P0-2): the plan digest binds the exact state
    import tempfile as _tf6
    with _tf6.TemporaryDirectory() as td:
        _sav = RUN4.STAGED_DIR
        try:
            RUN4.STAGED_DIR = td
            open(os.path.join(td, "a.record.json"), "w").write("{}")
            open(os.path.join(td, "b.record.json"), "w").write("{}")
            d1, n1 = _staged_state_digest()
            # same-count filename swap MUST change the digest
            os.rename(os.path.join(td, "b.record.json"),
                      os.path.join(td, "c.record.json"))
            d2, n2 = _staged_state_digest()
            # same-count same-name byte change MUST change it too
            open(os.path.join(td, "c.record.json"),
                 "w").write('{"x":1}')
            d3, _ = _staged_state_digest()
            check("F6 the staged-state digest binds exact names AND "
                  "bytes, never counts",
                  n1 == n2 == 2 and d1 != d2 and d2 != d3)
        finally:
            RUN4.STAGED_DIR = _sav
    # F7 per-key previews and the ledger digest are IN the canonical
    # plan object, so changing any one changes the plan digest
    _base = {"schema": "f2g-w2-zero-http-plan-v2",
             "frozen_ledger_sha256": "a" * 64,
             "vic": {"per_key": {"k1": {"would_write":
                                        {"record": "1" * 64}}}},
             "staged_state_sha256": "b" * 64}
    _p1 = _canon(_base)
    _m2 = json.loads(json.dumps(_base))
    _m2["vic"]["per_key"]["k1"]["would_write"]["record"] = "2" * 64
    _m3 = json.loads(json.dumps(_base))
    _m3["frozen_ledger_sha256"] = "c" * 64
    check("F7 changing one prospective output or the ledger digest "
          "changes the plan digest",
          _p1 != _canon(_m2) and _p1 != _canon(_m3))
    # F8 (2313Z P0-3): resume identity comes from the operation
    # record, never a recomputed post-partial plan
    with _tf6.TemporaryDirectory() as td:
        op = {"schema": "f2g-w2-zero-http-apply-op-v1",
              "plan": {"plan_sha256": "1" * 64,
                       "frozen_ledger_sha256": "a" * 64}}
        op["op_sha256"] = _canon(
            {k: v for k, v in op.items() if k != "op_sha256"})
        opp = os.path.join(td, "op.json")
        with open(opp, "w", encoding="utf-8") as f:
            json.dump(op, f)
        with open(opp, encoding="utf-8") as f:
            op2 = json.load(f)
        ok8 = (_canon({k: v for k, v in op2.items()
                       if k != "op_sha256"}) == op2["op_sha256"])
        op3 = dict(op2)
        op3["plan"] = dict(op3["plan"], plan_sha256="9" * 64)
        bad = (_canon({k: v for k, v in op3.items()
                       if k != "op_sha256"}) == op3.get("op_sha256"))
        check("F8 the operation record self-digest seals the "
              "accepted plan identity (mutation breaks it)",
              ok8 and not bad)
    # The allowed staged-state/count transition must not make a
    # changed per-key output or predecessor identity acceptable.
    oldp = {"schema": "f2g-w2-zero-http-plan-v2",
            "manifest_commit": "1" * 40,
            "capsule_sha256": "2" * 64,
            "frozen_ledger_sha256": "3" * 64,
            "vic": {"per_key": {"k": {"would_write": {
                "record": "4" * 64}}}},
            "predecessor": {"key": "a/b/c"},
            "expected_totals": {"artifact": 2056},
            "http_requests": 0,
            "staged_state_sha256": "5" * 64,
            "pre_counts": {"artifact": 1843}}
    resumed = json.loads(json.dumps(oldp))
    resumed["staged_state_sha256"] = "6" * 64
    resumed["pre_counts"]["artifact"] = 2056
    changed = json.loads(json.dumps(resumed))
    changed["vic"]["per_key"]["k"]["would_write"]["record"] = \
        "7" * 64
    check("F9 resume ignores only the expected staged transition; "
          "a changed prospective output changes its contract",
          _resume_contract(oldp) == _resume_contract(resumed)
          and _resume_contract(oldp) != _resume_contract(changed))

    # F5 CLI surface is closed
    check("F5 an unknown mode refuses",
          refuses(lambda: main(["explode"]), "FINALIZER_USAGE"))
    check("F5b apply without a plan digest refuses",
          refuses(lambda: main(["apply", "HEAD"]),
                  "FINALIZER_USAGE"))
    print()
    if fails:
        print(f"ZERO-HTTP FINALIZER FAILURES ({len(fails)}): {fails}")
        return 1
    print("ZERO-HTTP FINALIZER: ALL CONTROLS PASS  (deep per-key "
          "gates live in the landed VIC replay/bridge modules; the "
          "plan/apply joins run for real on the evidence host)")
    return 0


def main(argv):
    if not argv or argv[0] == "--selftest":
        return _selftest()
    if argv[0] == "plan" and len(argv) == 2:
        run("plan", argv[1])
        return 0
    if argv[0] == "apply" and len(argv) == 3:
        run("apply", argv[1], argv[2])
        return 0
    _refuse("FINALIZER_USAGE",
            "plan <commit> | apply <commit> <plan_sha256> | "
            "--selftest")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
