#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAILY-PATH IMMUTABLE REVISION MODEL -- red-KAT lock (cayley), built to
the CORRECTED contract v1 (grassmann 1432Z C-7..C-11 as corrected by codex
1433Z) and RECUT for codex's four 1755Z findings (F1 legacy capsule
re-derived from git; F2 LF blob authority for the CSV prefix; F3 semantic
inputs capsule; F4 revision<->journal identity).

Every check runs the REAL module (`ensemble_revisions_cayley`) and the REAL
runner functions (`run_ensemble_daily.check_persistence`) over temporary
stores built here with a scripted `git`; the checkout carries CRLF so the
LF-blob authority is exercised on every publish; nothing public is read or
written and no network is touched. Each RED control is one change from the
live positive and refuses for its OWN measured reason.

  R-0  live positive (CRLF checkout): cutover capsule validated against
       git -> publish -> journal -> latest -> csv = LF blob + rows -> staging
       clean -> revision block closed -> inputs_sha256 recomputes
  R-1  duplicate default run REFUSES; rescore needs a prior; empty reason
  R-2  rescore: supersedes == exact current run; C-8 prefix; C-10 latest
  R-3  journal tamper table (six mutants, each its own code)
  R-4  transaction partners: orphan / dirty .txn / dangling line
  R-5  persistence over the public view (revision|hole|legacy) with the
       REAL check_persistence == local replay; four keys
  R-6  source_index snapshot / JOURNAL_MOVED / C-9 replay
  R-7  inputs capsule recompute + tamper
  R-8  B6/C-11 UTC key (AST + mutant) + run-id format/uniqueness
  R-9  corpus-bar contracts intact
  R-10 anti-vacuity (REVISION_EXISTS guard removed on a copy)
  R-11 legacy surfaces: capsule create-once; mutated committed CSV blob
       refuses
  CODEX-1..6  the 1505Z reproductions (ported, refusing typed)
  CODEX-7  F1: forged/incomplete/reordered/substituted/false-parseability
           capsule -> LEGACY_CAPSULE_NOT_DERIVED_FROM_CUTOVER; the
           re-derived capsule passes (positive twin)
  CODEX-8  F2: LF checkout and CRLF checkout of the same store derive
           byte-identical LF CSV; a mutated committed blob refuses
  CODEX-9  F3: empty list / null field / missing-extra-duplicate kind /
           wrong length-day-keyset-hash-path / omitted calibration /
           input-pin mismatch refuse; the full capsule passes
  CODEX-10 F4: reason / schema / fired_utc / source_index / persistence
           mismatch (re-sealed) refuse; the untouched revision passes
"""
import ast
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ensemble_revisions_cayley as REV  # noqa: E402

FAILS = []


def _ok(name, msg):
    print(f"    [PASS] {name}: {msg}")


def _fail(name, msg):
    FAILS.append(name)
    print(f"    [FAIL] {name}: {msg}")


def _check(name, cond, msg):
    (_ok if cond else _fail)(name, msg)


def _refuses(fn, needle):
    try:
        fn()
        return False, "ACCEPTED"
    except REV.RevisionRefusal as e:
        return (needle in str(e)), str(e)[:200]


def _p(repo, rel):
    return os.path.join(repo, rel.replace("/", os.sep))


def _read(p):
    with io.open(p, "rb") as f:
        return f.read()


def _shim_obspy_if_absent():
    """FIXTURE-ONLY import shim (announced, never silent): the runner's
    import chain reaches obspy via the acquisition stack; this bar uses only
    pure runner functions."""
    try:
        import obspy  # noqa: F401
        return False
    except ImportError:
        pass
    import importlib.abc
    import importlib.machinery
    import types

    class _Placeholder:
        def __init__(self, *a, **k):
            raise RuntimeError("FIXTURE obspy shim: acquisition code must not "
                               "execute inside test_daily_revision_model")

    class _L(importlib.abc.Loader):
        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__fixture_only__ = "import-only shim; test_daily_revision_model"
            m.__getattr__ = lambda name: _Placeholder
            return m

        def exec_module(self, module):
            return None

    class _F(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "obspy" or fullname.startswith("obspy."):
                return importlib.machinery.ModuleSpec(fullname, _L(), is_package=True)
            return None
    sys.meta_path.insert(0, _F())
    print("    NOTE: obspy absent on this host -> import-only FIXTURE shim "
          "installed for obspy.* (no acquisition code runs in this bar)")
    return True


LEGACY_CSV = ("date,region,tier,risk,confidence,methods,agreement\n"
              "2026-08-30,alpha,1,0.5000,0.50,1,single_method\n"
              "2026-08-30,beta,0,0.0100,0.50,1,single_method\n"
              "2026-08-31,alpha,0,0.0200,0.50,1,single_method\n").encode()
CAL_REL = REV.CALIBRATION_DIR_REL + "/x.json"
CAL_RAW = b'{"region": "x", "valid_through": "2026-09-09"}\n'
FIRED = datetime(2026, 9, 3, 6, 15, 1, 123456, tzinfo=timezone.utc)


def _rec(date, tiers):
    return {"date": date, "timestamp": "2026-09-02T00:00:00+00:00",
            "regions": {r: {"tier": t, "combined_risk": 0.1 * (t + 1),
                            "confidence": 0.5, "methods_available": 1,
                            "agreement": "single_method",
                            "tier_name": "WATCH" if t else "NORMAL"}
                        for r, t in tiers.items()},
            "summary": {"total_regions": len(tiers)}}


LEGACY_0831 = _rec("2026-08-31", {"alpha": 0})
LEGACY_0831_RAW = REV.record_bytes(LEGACY_0831)
FAKE_GIT = REV.make_fake_git(LEGACY_CSV, LEGACY_0831_RAW)


def _mkstore(eol=b"\r\n"):
    """A temp store whose CHECKOUT csv carries `eol` (CRLF by default, the
    Windows runner case) while git authority is the LF blob."""
    repo = tempfile.mkdtemp(prefix="rev-model-bar-")
    os.makedirs(_p(repo, "docs"))
    os.makedirs(_p(repo, "monitoring/dashboard"))
    with io.open(_p(repo, REV.CSV_REL), "wb") as f:
        f.write(LEGACY_CSV.replace(b"\n", eol))
    cap = REV.build_legacy_baseline(repo, git=FAKE_GIT)
    REV.write_legacy_baseline(repo, cap, git=FAKE_GIT)
    return repo, REV.load_legacy_baseline(repo, git=FAKE_GIT)


EXPECT = {"calibration_paths": [CAL_REL]}


def _inputs(repo, cap, pins, day, tag="a"):
    ents = [REV.input_entry("code", p, None, None, raw_bytes=(tag + p).encode())
            for p in REV.CODE_PATHS]
    ents.append(REV.input_entry("calibration_capsule", CAL_REL, None,
                                ["region", "valid_through"], raw_bytes=CAL_RAW))
    for pin in pins:
        if pin["kind"] != "hole":
            ents.append(REV.pin_input_entry(repo, cap, pin, git=FAKE_GIT))
    ents.append(REV.scored_day_entry(day))
    return {"schema": REV.INPUTS_SCHEMA, "entries": ents}


def _pub(repo, cap, day, tiers, fired, reason=None, snap=None, pins=None, tag="a"):
    snap = REV.journal_bytes(repo) if snap is None else snap
    if pins is None:
        pins = [v[2] for v in REV.prior_days_view(repo, snap, cap, day, 3, git=FAKE_GIT)]
    e = REV.publish_revision(repo, _rec(day, tiers), _inputs(repo, cap, pins, day, tag),
                             snap, pins, fired, rescore_reason=reason,
                             expect_inputs=EXPECT, git=FAKE_GIT)
    return e, snap, pins


def _exp(pins, day):
    return dict(EXPECT, pins=pins, scored_day=day)


def main():
    print("DAILY REVISION MODEL red-KATs (cayley, corrected v1 + F1-F4 recut) -- temp stores, scripted git, CRLF checkout, no network")
    repo, cap = _mkstore()
    try:
        # ---- R-0 live positive (CRLF checkout)
        snap0 = REV.journal_bytes(repo)
        view = REV.prior_days_view(repo, snap0, cap, "2026-09-02", 3, git=FAKE_GIT)
        pins = [v[2] for v in view]
        e1, _s, _p1 = _pub(repo, cap, "2026-09-02", {"alpha": 1, "beta": 0}, FIRED, pins=pins, snap=snap0)
        j1 = REV.journal_bytes(repo)
        rev1_raw = _read(_p(repo, e1["path"]))
        rev1 = json.loads(rev1_raw.decode("utf-8"))
        csv1 = _read(_p(repo, REV.CSV_REL))
        txn_clean = (not os.path.isdir(_p(repo, REV.TXN_DIR_REL))
                     or not os.listdir(_p(repo, REV.TXN_DIR_REL)))
        ok = (len(REV.parse_journal(j1)) == 1 and REV.parse_journal(j1)[0] == e1
              and j1 == REV.canonical_bytes(e1)
              and hashlib.sha256(rev1_raw).hexdigest() == e1["sha256"]
              and _read(_p(repo, REV.LATEST_REL)) == rev1_raw
              and csv1 == LEGACY_CSV + (b"2026-09-02,alpha,1,0.2000,0.50,1,single_method\n"
                                        b"2026-09-02,beta,0,0.1000,0.50,1,single_method\n")
              and b"\r" not in csv1 and txn_clean
              and set(rev1["revision"]) == REV.REVISION_FIELDS
              and rev1["revision"]["source_index"] == {"entry_count": 0,
                                                       "prefix_sha256": hashlib.sha256(b"").hexdigest()}
              and rev1["revision"]["inputs_sha256"] == REV.inputs_sha256(
                  rev1["revision"]["inputs"], expect=_exp(pins, "2026-09-02"))
              and cap["legacy_csv"]["git_blob"] == "c" * 40)
        _check("R-0 LIVE-POSITIVE", ok,
               f"{e1['run_id']}: capsule validated vs git; CRLF checkout, LF derived csv = LF blob + 2 rows; "
               "journal 1 canonical line; latest byte-equal; staging clean; inputs_sha256 recomputes")

        # ---- R-1
        before = (REV.journal_bytes(repo), _read(_p(repo, REV.CSV_REL)),
                  sorted(os.listdir(_p(repo, "docs/ensemble/2026-09-02"))))
        snap1 = REV.journal_bytes(repo)
        hit, msg = _refuses(lambda: _pub(repo, cap, "2026-09-02", {"alpha": 2, "beta": 0}, FIRED,
                                         snap=snap1, pins=pins), "REVISION_EXISTS")
        after = (REV.journal_bytes(repo), _read(_p(repo, REV.CSV_REL)),
                 sorted(os.listdir(_p(repo, "docs/ensemble/2026-09-02"))))
        _check("R-1 DUPLICATE-REFUSES", hit and before == after,
               f"{msg[:60]}... ; store byte-unchanged={before == after}")
        hit, msg = _refuses(lambda: _pub(repo, cap, "2026-09-03", {"alpha": 0}, FIRED, reason="x",
                                         snap=snap1), "RESCORE_WITHOUT_PRIOR")
        _check("R-1b RESCORE-NEEDS-PRIOR", hit, msg[:80])
        hit, msg = _refuses(lambda: _pub(repo, cap, "2026-09-02", {"alpha": 2}, FIRED, reason="  ",
                                         snap=snap1, pins=pins), "RESCORE_REASON_EMPTY")
        _check("R-1c REASON-REQUIRED", hit, msg[:80])

        # ---- R-2 rescore
        e2, _s, _p2 = _pub(repo, cap, "2026-09-02", {"alpha": 2, "beta": 0}, FIRED.replace(hour=7),
                           reason="input fix: THD baseline", snap=snap1, pins=pins, tag="b")
        j2 = REV.journal_bytes(repo)
        csv2 = _read(_p(repo, REV.CSV_REL))
        ok = (e2["supersedes"] == e1["run_id"] and e2["reason"] == "input fix: THD baseline"
              and REV.journal_prefix_ok(j1, j2) and len(REV.parse_journal(j2)) == 2
              and _read(_p(repo, e1["path"])) == rev1_raw
              and csv2 == LEGACY_CSV + (b"2026-09-02,alpha,2,0.3000,0.50,1,single_method\n"
                                        b"2026-09-02,beta,0,0.1000,0.50,1,single_method\n")
              and REV.current_map(REV.parse_journal(j2))["2026-09-02"]["run_id"] == e2["run_id"]
              and _read(_p(repo, REV.LATEST_REL)) == _read(_p(repo, e2["path"])))
        _check("R-2 RESCORE-ONE-RULE", ok,
               "r2 supersedes the exact current run; journal byte-prefix (C-8); only 09-02 rows rewritten; "
               "legacy prefix and r1 byte-identical; latest -> r2 (C-10)")

        # ---- R-3 journal tamper table
        lines = j2.split(b"\n")[:-1]
        e2b = json.loads(lines[1])
        stale = dict(e2b, supersedes="20260101T000000000000Z-deadbeef")
        dup = REV.canonical_bytes(json.loads(lines[0]))
        table = [
            ("truncated", j2[:-1], "JOURNAL_TRUNCATED"),
            ("blank line", j2 + b"\n", "JOURNAL_BLANK_LINE"),
            ("non-canonical", lines[0] + b"\n"
             + json.dumps(e2b, sort_keys=True, separators=(", ", ":")).encode() + b"\n",
             "JOURNAL_NONCANONICAL"),
            ("multi-line object", lines[0] + b"\n" + json.dumps(e2b, indent=1).encode() + b"\n",
             "JOURNAL_UNPARSABLE"),
            ("duplicate run id", j2 + dup, "JOURNAL_DUPLICATE_RUN_ID"),
            ("stale supersedes (fork)", lines[0] + b"\n" + REV.canonical_bytes(stale),
             "JOURNAL_STALE_SUPERSEDES"),
            ("reason-less rescore", lines[0] + b"\n" + REV.canonical_bytes(dict(e2b, reason=None)),
             "JOURNAL_RESCORE_WITHOUT_REASON"),
        ]
        res = []
        for label, raw, needle in table:
            hit, msg = _refuses(lambda raw=raw: REV.parse_journal(raw), needle)
            res.append((label, hit, msg.split(":")[0]))
        _check("R-3 JOURNAL-TAMPER-TABLE", all(h for _l, h, _m in res),
               "; ".join(f"{l}->{m}" for l, _h, m in res))

        # ---- R-4 transaction partners
        orphan = _p(repo, f"{REV.REV_DIR_REL}/2026-09-02/20260101T000000000000Z-0badf00d.json")
        with io.open(orphan, "wb") as f:
            f.write(b"{}\n")
        h1, m1 = _refuses(lambda: REV.check_store_clean(repo), "REVISION_ORPHAN")
        os.remove(orphan)
        os.makedirs(_p(repo, f"{REV.TXN_DIR_REL}/zzz"))
        with io.open(_p(repo, f"{REV.TXN_DIR_REL}/zzz/revision.json"), "wb") as f:
            f.write(b"{}")
        h2, m2 = _refuses(lambda: REV.check_store_clean(repo), "REVISION_TXN_DIRTY")
        shutil.rmtree(_p(repo, REV.TXN_DIR_REL))
        os.rename(_p(repo, e2["path"]), _p(repo, e2["path"]) + ".moved")
        h3, m3 = _refuses(lambda: REV.check_store_clean(repo), "REVISION_DANGLING_JOURNAL_LINE")
        os.rename(_p(repo, e2["path"]) + ".moved", _p(repo, e2["path"]))
        h4 = len(REV.check_store_clean(repo)) == 2
        _check("R-4 TRANSACTION-PARTNERS", h1 and h2 and h3 and h4,
               "orphan / dirty .txn / dangling line each refuse the next run with a RECOVERY instruction; "
               "the clean store passes")

        # ---- R-5 persistence over the public view (real runner)
        _shim_obspy_if_absent()
        import run_ensemble_daily as RED  # noqa
        from ensemble import EnsembleResult  # noqa
        tgt = datetime(2026, 9, 3)

        def result(region, tier):
            return EnsembleResult(region=region, date=tgt, combined_risk=0.3, tier=tier,
                                  tier_name="WATCH" if tier else "NORMAL", components={},
                                  confidence=0.5, agreement="single_method", methods_available=1)
        current = {"alpha": result("alpha", 1), "beta": result("beta", 1)}
        snap = REV.journal_bytes(repo)
        view = REV.prior_days_view(repo, snap, cap, "2026-09-03", 3, git=FAKE_GIT)
        kinds = [v[2]["kind"] for v in view]
        pmap = {k + 1: r for k, (_d, r, _pin) in enumerate(view)}
        pub = RED.check_persistence(current, Path(repo), tgt, 2, loader=lambda k: pmap.get(k))
        ld = tempfile.mkdtemp(prefix="rev-model-local-")
        for d, tiers in (("2026-09-02", {"alpha": 2, "beta": 0}), ("2026-08-31", {"alpha": 0})):
            with io.open(os.path.join(ld, f"ensemble_{d}.json"), "w", encoding="utf-8") as f:
                json.dump(_rec(d, tiers), f)
        loc = RED.check_persistence(current, Path(ld), tgt, 2)
        shutil.rmtree(ld, ignore_errors=True)
        ok = (kinds == ["revision", "hole", "legacy"]
              and view[0][2]["run_id"] == e2["run_id"] and view[0][2]["sha256"] == e2["sha256"]
              and view[2][2]["legacy"]["git_blob"] == "b" * 40
              and pub["alpha"]["tier_history"] == [0, None, 2, 1] == loc["alpha"]["tier_history"]
              and pub["alpha"]["is_confirmed"] and pub["alpha"]["consecutive_days"] == 2
              and pub["beta"]["tier_history"] == [None, None, 0, 1] and not pub["beta"]["is_confirmed"]
              and set(pub["alpha"]) == {"current_tier", "consecutive_days", "is_confirmed", "tier_history"})
        _check("R-5 PUBLIC-PERSISTENCE-UNION", ok,
               f"prior kinds {kinds}; real check_persistence alpha={pub['alpha']['tier_history']} == local replay; four keys")

        # ---- R-6 source_index / JOURNAL_MOVED / C-9
        pins3 = [v[2] for v in view]
        hit, msg = _refuses(lambda: _pub(repo, cap, "2026-09-03", {"alpha": 1, "beta": 1}, FIRED.replace(day=4),
                                         snap=j1, pins=pins3, tag="c"), "JOURNAL_MOVED")
        e3, _s, _p3 = _pub(repo, cap, "2026-09-03", {"alpha": 1, "beta": 1}, FIRED.replace(day=4),
                           snap=snap, pins=pins3, tag="c")
        r3, _ = REV.reopen_revision(repo, e3)
        si = r3["revision"]["source_index"]
        replay_prefix = REV.journal_bytes(repo)[:len(snap)]
        replay = REV.prior_days_view(repo, replay_prefix, cap, "2026-09-03", 3, git=FAKE_GIT)
        ok = (hit and si == {"entry_count": 2, "prefix_sha256": hashlib.sha256(snap).hexdigest()}
              and hashlib.sha256(replay_prefix).hexdigest() == si["prefix_sha256"]
              and [v[2] for v in replay] == r3["revision"]["persistence_inputs"])
        _check("R-6 SOURCE-INDEX-SNAPSHOT", ok,
               f"stale snapshot -> {msg[:30]}...; recorded source_index == the prefix consumed; C-9 replay reproduces the pins")

        # ---- R-7 inputs capsule
        capsule = r3["revision"]["inputs"]
        exp3 = _exp(pins3, "2026-09-03")
        d0 = REV.inputs_sha256(capsule, expect=exp3)
        tampered = json.loads(json.dumps(capsule))
        tampered["entries"][0]["sha256"] = ("0" if tampered["entries"][0]["sha256"][0] != "0" else "1") \
            + tampered["entries"][0]["sha256"][1:]
        d1 = REV.inputs_sha256(tampered, expect=exp3)
        opened = json.loads(json.dumps(capsule))
        opened["entries"][0]["extra"] = 1
        h_open, m_open = _refuses(lambda: REV.inputs_sha256(opened), "INPUTS_CAPSULE_SCHEMA")
        _check("R-7 INPUTS-CAPSULE", d0 == r3["revision"]["inputs_sha256"] and d1 != d0 and h_open,
               "inputs_sha256 recomputes; one digit changed -> different digest; an open entry shape refuses")

        # ---- R-8 B6 / C-11 / run-id
        src = io.open(os.path.join(HERE, "run_ensemble_daily.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        main_fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
        nows = [n for n in ast.walk(main_fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "now"]
        naive = [n for n in nows if not n.args]
        utc = [n for n in nows if n.args and isinstance(n.args[0], ast.Attribute)
               and n.args[0].attr == "utc"]
        mut = src.replace("today_utc = datetime.now(timezone.utc).date()", "today_utc = datetime.now().date()")
        mmain = [n for n in ast.parse(mut).body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
        mnaive = [n for n in ast.walk(mmain) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "now" and not n.args]
        ids = {REV.run_id_for(FIRED) for _ in range(5)}
        ok = (not naive and len(utc) >= 1 and mut != src and len(mnaive) == 1
              and r3["revision"]["scored_day_utc"] == r3["date"] == "2026-09-03"
              and all(REV._RUN_ID_RE.match(i) for i in ids) and len(ids) == 5)
        _check("R-8 B6-C11-RUN-ID", ok,
               f"main(): {len(utc)} UTC now, {len(naive)} naive; mutant detected; scored_day_utc == date; 5 distinct ids at one instant")

        # ---- R-9 corpus-bar contracts
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        cp = [a.arg for a in fns["check_persistence"].args.args]
        sr = [a.arg for a in fns["save_results"].args.args]
        anchor = "is_confirmed = consecutive >= required_consecutive if current_tier >= 1 else False"
        _check("R-9 CORPUS-BAR-CONTRACTS",
               cp[:4] == ["current_results", "output_dir", "target_date", "required_consecutive"]
               and sr == ["results", "output_dir", "target_date", "persistence", "events_data"]
               and anchor in src,
               f"check_persistence{tuple(cp)}, save_results{tuple(sr)}, mutation anchor present")

        # ---- R-10 anti-vacuity
        msrc = io.open(os.path.join(HERE, "ensemble_revisions_cayley.py"), encoding="utf-8").read()
        guard = "    if cur is not None and rescore_reason is None:\n        raise RevisionRefusal(\n"
        assert guard in msrc, "guard text not found -- re-anchor the bar"
        mdir = tempfile.mkdtemp(prefix="rev-model-mut-")
        mpath = os.path.join(mdir, "ensemble_revisions_mutant.py")
        io.open(mpath, "w", encoding="utf-8", newline="\n").write(
            msrc.replace(guard, "    if False:\n        raise RevisionRefusal(\n", 1))
        spec = importlib.util.spec_from_file_location("ensemble_revisions_mutant", mpath)
        MUT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MUT)
        repo2 = tempfile.mkdtemp(prefix="rev-model-bar2-")
        try:
            os.makedirs(_p(repo2, "docs")); os.makedirs(_p(repo2, "monitoring/dashboard"))
            with io.open(_p(repo2, REV.CSV_REL), "wb") as f:
                f.write(LEGACY_CSV)
            fg2 = MUT.make_fake_git(LEGACY_CSV, LEGACY_0831_RAW)
            cap2 = MUT.build_legacy_baseline(repo2, git=fg2); MUT.write_legacy_baseline(repo2, cap2, git=fg2)
            cap2 = MUT.load_legacy_baseline(repo2, git=fg2)
            s0 = MUT.journal_bytes(repo2)
            v0 = [v[2] for v in MUT.prior_days_view(repo2, s0, cap2, "2026-09-02", 3, git=fg2)]

            def inp(tag):
                ents = [MUT.input_entry("code", p, None, None, raw_bytes=(tag + p).encode()) for p in MUT.CODE_PATHS]
                ents.append(MUT.input_entry("calibration_capsule", CAL_REL, None, ["region", "valid_through"], raw_bytes=CAL_RAW))
                for pin in v0:
                    if pin["kind"] != "hole":
                        ents.append(MUT.pin_input_entry(repo2, cap2, pin, git=fg2))
                ents.append(MUT.scored_day_entry("2026-09-02"))
                return {"schema": MUT.INPUTS_SCHEMA, "entries": ents}
            MUT.publish_revision(repo2, _rec("2026-09-02", {"alpha": 1}), inp("a"), s0, v0, FIRED,
                                 expect_inputs=EXPECT, git=fg2)
            s1 = MUT.journal_bytes(repo2)
            try:
                MUT.publish_revision(repo2, _rec("2026-09-02", {"alpha": 2}), inp("a"), s1, v0,
                                     FIRED.replace(hour=8), expect_inputs=EXPECT, git=fg2)
                outcome = "ACCEPTED"
            except MUT.RevisionRefusal as e:
                outcome = str(e)
            _check("R-10 ANTI-VACUITY", "REVISION_EXISTS" not in outcome,
                   f"guard removed on a COPY -> outcome: {outcome[:80]}")
        finally:
            shutil.rmtree(repo2, ignore_errors=True)
            shutil.rmtree(mdir, ignore_errors=True)

        # ---- R-11 legacy surfaces
        h_once, m_once = _refuses(lambda: REV.write_legacy_baseline(repo, cap, git=FAKE_GIT), "REVISION_PATH_EXISTS")
        fg_bad = REV.make_fake_git(LEGACY_CSV.replace(b"0.5000", b"0.5001"), LEGACY_0831_RAW)
        h_blob, m_blob = _refuses(lambda: REV.derive_csv_bytes(repo, cap, REV.parse_journal(REV.journal_bytes(repo)), git=fg_bad),
                                  "CSV_LEGACY_BLOB_MISMATCH")
        _check("R-11 LEGACY-SURFACES", h_once and h_blob,
               "capsule is create-once; a mutated committed CSV blob refuses the derivation")
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    _codex_partners_1505()
    _codex_partners_1755()

    if FAILS:
        print(f"DAILY REVISION MODEL: {len(FAILS)} FAIL -> {FAILS}")
        return 1
    print("DAILY REVISION MODEL (corrected v1 + F1-F4): ALL PASS (temp stores; scripted git; CRLF checkout; nothing scored; nothing public touched)")
    return 0


def _codex_partners_1505():
    """codex 1505Z reproductions, ported to the recut API."""
    # CODEX-1 index rewrite / forged earlier entry
    repo, cap = _mkstore()
    try:
        e1, _s, _p1 = _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="one")
        prefix = REV.journal_bytes(repo)
        e2, _s, _p2 = _pub(repo, cap, "2026-09-02", {"alpha": 1}, FIRED.replace(day=4), tag="two")
        later = REV.journal_bytes(repo)
        lines = later.split(b"\n")[:-1]
        forged_first = json.loads(lines[0]); forged_first["appended_utc"] = "2099-01-01T00:00:00Z"
        forged_journal = REV.canonical_bytes(forged_first) + lines[1] + b"\n"
        r2, _raw = REV.reopen_revision(repo, e2)
        rec1_path = _p(repo, e1["path"])
        rec1 = json.loads(_read(rec1_path).decode("utf-8"))
        rec1["revision"]["inputs"]["entries"][0]["identity"] = "forged/x.py"
        with io.open(rec1_path, "wb") as f:
            f.write(REV.record_bytes(rec1))
        h_reopen, m_reopen = _refuses(lambda: REV.reopen_revision(repo, e1), "REVISION_DIGEST_MISMATCH")
        ok = (later.startswith(prefix) and not REV.journal_prefix_ok(prefix, forged_journal)
              and r2["revision"]["source_index"]["prefix_sha256"] == hashlib.sha256(prefix).hexdigest()
              and h_reopen)
        _check("CODEX-1 INDEX-REWRITE", ok, "prefix comparator + source_index + digest reopen all refuse the rewrite")
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-2 legacy drift -> the checkout is irrelevant now (blob authority); a mutated BLOB refuses
    repo, cap = _mkstore()
    try:
        _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="one")
        csvp = _p(repo, REV.CSV_REL)
        with io.open(csvp, "wb") as f:
            f.write(_read(csvp).replace(b"0.5000", b"0.9999", 1))
        e2, _s, _p2 = _pub(repo, cap, "2026-09-02", {"alpha": 1}, FIRED.replace(day=4), tag="two")
        csv_after = _read(csvp)
        _check("CODEX-2 LEGACY-DRIFT", b"0.9999" not in csv_after and csv_after.startswith(LEGACY_CSV)
               and os.path.exists(_p(repo, REV.LEGACY_REL)),
               "a checkout edit of a pre-model row is OVERWRITTEN by the committed LF blob on the next publish "
               "(authority is git, never the checkout); the capsule exists")
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-3 stale prior identities (TOCTOU)
    repo, cap = _mkstore()
    try:
        e1, _s, _p1 = _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="v1")
        snap = REV.journal_bytes(repo)
        stale_pins = [v[2] for v in REV.prior_days_view(repo, snap, cap, "2026-09-02", 3, git=FAKE_GIT)]
        e2, _s, _p2 = _pub(repo, cap, "2026-09-01", {"alpha": 2}, FIRED.replace(hour=7),
                           reason="later correction", tag="v2")
        h, m = _refuses(lambda: _pub(repo, cap, "2026-09-02", {"alpha": 1}, FIRED.replace(day=4),
                                     snap=snap, pins=stale_pins, tag="t"), "JOURNAL_MOVED")
        _check("CODEX-3 STALE-PRIOR", h and stale_pins[0]["run_id"] == e1["run_id"],
               f"captured r1, rescore to r2, publish with the stale snapshot -> {m.split(':')[0]}")
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-4 empty / untyped inputs
    repo, cap = _mkstore()
    try:
        snap = REV.journal_bytes(repo)
        pins = [v[2] for v in REV.prior_days_view(repo, snap, cap, "2026-09-01", 3, git=FAKE_GIT)]
        h, m = _refuses(lambda: REV.publish_revision(repo, _rec("2026-09-01", {"alpha": 1}), {}, snap, pins, FIRED,
                                                     expect_inputs=EXPECT, git=FAKE_GIT), "INPUTS_CAPSULE_SCHEMA")
        _check("CODEX-4 EMPTY-INPUTS", h and not os.path.isdir(_p(repo, "docs/ensemble/2026-09-01")),
               f"inputs={{}} -> {m.split(':')[0]}; nothing written")
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-5 planted crash after revision creation -> typed recovery
    repo, cap = _mkstore()
    try:
        real = REV._write_atomic

        def crash_journal(path, data):
            if path.endswith("index.ndjson") and os.sep + ".txn" + os.sep not in path:
                raise RuntimeError("planted crash before the journal publish")
            return real(path, data)
        REV._write_atomic = crash_journal
        try:
            _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="v1")
            crashed = False
        except RuntimeError:
            crashed = True
        finally:
            REV._write_atomic = real
        revs = os.listdir(_p(repo, "docs/ensemble/2026-09-01"))
        h, m = _refuses(lambda: _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED.replace(hour=8), tag="v1"),
                        "REVISION_TXN_DIRTY")
        _check("CODEX-5 CRASH-RECOVERY", crashed and len(revs) == 1 and h and "RECOVERY" in m,
               f"crash after create-once: retry refuses {m.split(':')[0]} with a RECOVERY instruction")
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-6 revision fields present
    repo, cap = _mkstore()
    try:
        e, _s, _p1 = _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="v1")
        r, _raw = REV.reopen_revision(repo, e)
        missing = sorted({"scored_day_utc", "source_index"} - set(r["revision"]))
        _check("CODEX-6 REVISION-FIELDS", not missing, f"missing={missing}")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def _codex_partners_1755():
    """codex 1755Z four findings, ported to the recut API with positive twins."""
    # CODEX-7 F1: legacy capsule re-derived from git
    repo = tempfile.mkdtemp(prefix="rev-model-f1-")
    try:
        os.makedirs(_p(repo, "docs")); os.makedirs(_p(repo, "monitoring/dashboard"))
        with io.open(_p(repo, REV.CSV_REL), "wb") as f:
            f.write(LEGACY_CSV)
        good = REV.build_legacy_baseline(repo, git=FAKE_GIT)
        pos = REV.validate_legacy_baseline(repo, good, git=FAKE_GIT) is True
        forged = {"schema": REV.LEGACY_SCHEMA, "records": [],
                  "legacy_csv": {"row_count": 1, "prefix_sha256": REV.sha256_bytes(LEGACY_CSV)}}
        rec0 = good["records"][0]
        cases = [
            ("codex's incomplete capsule", forged, "LEGACY_CAPSULE_NOT_DERIVED_FROM_CUTOVER"),
            ("records deleted", dict(good, records=[]), "record vector"),
            ("record substituted", dict(good, records=[dict(rec0, git_blob="d" * 40)]), "record vector"),
            ("record reordered/duplicated", dict(good, records=[rec0, dict(rec0, git_blob="e" * 40)]), "record vector"),
            ("wrong commit", dict(good, records=[dict(rec0, commit="c2")]), "record vector"),
            ("wrong path", dict(good, record_path="docs/other.json"), "record_path"),
            ("false parseability/date", dict(good, records=[dict(rec0, parseable=False, date=None)]), "record vector"),
            ("csv row_count altered", dict(good, legacy_csv=dict(good["legacy_csv"], row_count=9)), "recompute"),
            ("csv blob altered", dict(good, legacy_csv=dict(good["legacy_csv"], git_blob="d" * 40)), "git_blob"),
            ("cutover commit != HEAD", dict(good, cutover_commit="e" * 40), "HEAD"),
        ]
        res = []
        for label, bad, needle in cases:
            h, m = _refuses(lambda bad=bad: REV.validate_legacy_baseline(repo, bad, git=FAKE_GIT), needle)
            res.append((label, h))
        # committed capsule whose add-commit parent is not the cutover commit
        fg_parent = REV.make_fake_git(LEGACY_CSV, LEGACY_0831_RAW, capsule_add="f" * 40 + " " + "e" * 40 + "\n")
        h_par, m_par = _refuses(lambda: REV.validate_legacy_baseline(repo, good, git=fg_parent), "capsule-add commit's parent")
        _check("CODEX-7 F1-LEGACY-CAPSULE-RE-DERIVED", pos and all(h for _l, h in res) and h_par,
               "re-derived capsule passes; " + "; ".join(f"{l}->{'refused' if h else 'ACCEPTED'}" for l, h in res)
               + "; wrong add-commit parent->refused")
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-8 F2: LF and CRLF checkouts derive identical LF bytes; blob mutation refuses
    outs = {}
    for label, eol in (("LF", b"\n"), ("CRLF", b"\r\n")):
        repo, cap = _mkstore(eol=eol)
        try:
            e1, _s, _p1 = _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="one")
            outs[label] = _read(_p(repo, REV.CSV_REL))
            if label == "CRLF":
                fg_bad = REV.make_fake_git(LEGACY_CSV.replace(b"0.0100", b"0.0101"), LEGACY_0831_RAW)
                h_mut, m_mut = _refuses(lambda: REV.derive_csv_bytes(repo, cap, REV.parse_journal(REV.journal_bytes(repo)), git=fg_bad),
                                        "CSV_LEGACY_BLOB_MISMATCH")
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    _check("CODEX-8 F2-LF-BLOB-AUTHORITY", outs["LF"] == outs["CRLF"] and b"\r" not in outs["CRLF"]
           and outs["LF"].startswith(LEGACY_CSV) and h_mut,
           f"LF and CRLF checkouts derive byte-identical LF csv ({len(outs['LF'])} B); mutated committed blob refuses")
    # CODEX-9 F3: semantic inputs
    repo, cap = _mkstore()
    try:
        snap = REV.journal_bytes(repo)
        pins = [v[2] for v in REV.prior_days_view(repo, snap, cap, "2026-09-01", 3, git=FAKE_GIT)]
        full = _inputs(repo, cap, pins, "2026-09-01")
        exp = _exp(pins, "2026-09-01")
        # the reopened lengths of the pinned bytes (what the publisher passes)
        exp["pin_byte_lengths"] = {e["identity"]: e["byte_length"] for e in full["entries"]
                                   if e["kind"] in ("prior_revision", "legacy_record")}
        pos = REV.validate_inputs_capsule(full, expect=exp) is True
        E = full["entries"]

        def swap(kind, **kw):
            return {"schema": REV.INPUTS_SCHEMA,
                    "entries": [dict(e, **kw) if e["kind"] == kind else e for e in E]}
        cases = [
            ("empty list", {"schema": REV.INPUTS_SCHEMA, "entries": []}, "empty"),
            ("null field (codex's all-None calibration)", {"schema": REV.INPUTS_SCHEMA, "entries": E + [
                {"kind": "calibration_capsule", "identity": None, "data_day": None, "keyset": None,
                 "byte_length": None, "sha256": None}]}, "INPUTS_CAPSULE_SCHEMA"),
            ("missing kind (no scored_day)", {"schema": REV.INPUTS_SCHEMA, "entries": [e for e in E if e["kind"] != "scored_day"]}, "scored_day"),
            ("extra kind (second scored_day)", {"schema": REV.INPUTS_SCHEMA, "entries": E + [REV.scored_day_entry("2026-09-01")]}, "scored_day"),
            ("duplicate code identity", {"schema": REV.INPUTS_SCHEMA, "entries": E + [E[0]]}, "duplicate"),
            ("wrong length", swap("legacy_record", byte_length=1), "one-to-one"),
            ("wrong day", swap("legacy_record", data_day="2026-01-01"), "one-to-one"),
            ("wrong keyset", swap("calibration_capsule", keyset=[]), "keyset"),
            ("wrong hash", swap("legacy_record", sha256="0" * 64), "one-to-one"),
            ("wrong path", swap("calibration_capsule", identity="docs/x.json"), "calibration"),
            ("omitted calibration", {"schema": REV.INPUTS_SCHEMA, "entries": [e for e in E if e["kind"] != "calibration_capsule"]}, "calibration set"),
            ("input/persistence mismatch", {"schema": REV.INPUTS_SCHEMA, "entries": [e for e in E if e["kind"] != "legacy_record"]}, "one-to-one"),
        ]
        res = []
        for label, bad, needle in cases:
            h, m = _refuses(lambda bad=bad: REV.validate_inputs_capsule(bad, expect=exp), needle)
            res.append((label, h))
        _check("CODEX-9 F3-INPUTS-SEMANTICS", pos and all(h for _l, h in res),
               "full capsule passes; " + "; ".join(f"{l}->{'refused' if h else 'ACCEPTED'}" for l, h in res))
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    # CODEX-10 F4: revision <-> journal identity
    repo, cap = _mkstore()
    try:
        e1, _s, pins = _pub(repo, cap, "2026-09-01", {"alpha": 1}, FIRED, tag="v1")
        e2, _s, _p2 = _pub(repo, cap, "2026-09-01", {"alpha": 2}, FIRED.replace(hour=7), reason="fix", tag="v2", pins=pins)
        r2, raw2 = REV.reopen_revision(repo, e2)
        pos = REV.validate_revision_against_entry(r2, e2, expect_inputs=EXPECT) is True

        def resealed(mut):
            r = json.loads(raw2.decode("utf-8")); mut(r); d = REV.record_bytes(r)
            return r, dict(e2, sha256=REV.sha256_bytes(d))
        cases = [
            ("codex's reason mismatch", lambda r: r["revision"].__setitem__("reason", "revision-only reason"), "reason"),
            ("schema", lambda r: r["revision"].__setitem__("schema", "x"), "schema"),
            ("fired_utc non-canonical", lambda r: r["revision"].__setitem__("fired_utc", "2026-09-03T07:15:01Z"), "fired_utc"),
            ("fired_utc != run-id prefix", lambda r: r["revision"].__setitem__("fired_utc", "2026-09-03T07:15:02.123456Z"), "run_id time prefix"),
            ("source_index open", lambda r: r["revision"].__setitem__("source_index", {"entry_count": 1}), "source_index"),
            ("persistence semantic", lambda r: r["revision"]["persistence_inputs"].__setitem__(
                0, {"date": "2026-08-31", "kind": "hole", "run_id": "x", "sha256": None, "legacy": None}), "hole"),
            ("supersedes", lambda r: r["revision"].__setitem__("supersedes", None), "supersedes"),
        ]
        res = []
        for label, mut, needle in cases:
            r, en = resealed(mut)
            h, m = _refuses(lambda r=r, en=en: REV.validate_revision_against_entry(r, en, expect_inputs=EXPECT), needle)
            res.append((label, h))
        # and the same through the store: a re-sealed reason mismatch must refuse on reopen
        r, en = resealed(lambda r: r["revision"].__setitem__("reason", "revision-only reason"))
        with io.open(_p(repo, e2["path"]), "wb") as f:
            f.write(REV.record_bytes(r))
        h_store, m_store = _refuses(lambda: REV.reopen_revision(repo, en), "reason")
        _check("CODEX-10 F4-REVISION-JOURNAL-IDENTITY", pos and all(h for _l, h in res) and h_store,
               "untouched revision passes; " + "; ".join(f"{l}->{'refused' if h else 'ACCEPTED'}" for l, h in res)
               + "; re-sealed reason mismatch refuses on reopen")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
