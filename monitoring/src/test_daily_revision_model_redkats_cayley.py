#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAILY-PATH IMMUTABLE REVISION MODEL -- red-KAT lock (cayley), built to
the CORRECTED contract v1 (grassmann 1432Z C-7..C-11 as corrected by codex
1433Z, five corrections).

Every check runs the REAL module (`ensemble_revisions_cayley`) and the REAL
runner functions (`run_ensemble_daily.check_persistence`) over temporary
stores built here with a scripted `git` (the legacy capsule derivation is
exercised on synthetic committed history); nothing public is read or
written and no network is touched. Each RED control is one change from the
live positive and refuses for its OWN measured reason.

  R-0  live positive: cutover capsule -> publish -> journal (1 canonical
       line) -> latest byte-equal -> csv = bound legacy prefix + rows ->
       staging clean -> revision block closed set, source_index of the
       snapshot, inputs_sha256 recomputes
  R-1  duplicate default run REFUSES (store byte-unchanged); rescore needs
       a prior; empty reason refuses
  R-2  rescore: supersedes == the EXACT current run; the journal is a byte
       prefix of the new one (C-8); only that date's csv rows change; the
       legacy prefix and the first revision are byte-identical
  R-3  journal tamper table: truncated / blank / non-canonical / duplicate
       run id / stale supersedes (fork) / reason-less rescore -> refuse
  R-4  transaction partners (correction 4): orphan revision, dangling
       journal line, dirty .txn each make the NEXT run refuse, typed
  R-5  persistence over the public view: closed union revision|legacy|hole;
       the REAL check_persistence reproduces the local replay where days
       exist; a hole only where neither source has the date; per-region
       dict keeps exactly the four keys the corpus bar compares
  R-6  source_index snapshot (correction 3): JOURNAL_MOVED when the journal
       changed between resolve and publish; C-9 replay from the recorded
       prefix reproduces the pins the revision carries
  R-7  inputs capsule (correction 5): inputs_sha256 recomputes from the
       stored capsule; one entry digit changed -> different digest; an
       open shape refuses
  R-8  B6/C-11: main() derives the day key from the UTC clock (AST, mutant
       detected); scored_day_utc == date; run ids match
       YYYYMMDDTHHMMSSffffffZ-hex8 and two publishes at the SAME instant
       get distinct ids (correction 2)
  R-9  corpus-bar contracts intact (signatures + mutation anchor)
  R-10 anti-vacuity: the REVISION_EXISTS guard removed on a COPY -> the
       duplicate is no longer refused as REVISION_EXISTS
  R-11 legacy surfaces: capsule is create-once; a changed legacy csv prefix
       refuses the derivation
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
        return (needle in str(e)), str(e)[:160]


def _p(repo, rel):
    return os.path.join(repo, rel.replace("/", os.sep))


def _read(p):
    with io.open(p, "rb") as f:
        return f.read()


def _shim_obspy_if_absent():
    """FIXTURE-ONLY import shim: the runner's import chain reaches obspy via
    the acquisition stack; this bar uses only pure runner functions. On a
    host without obspy an inert `obspy.*` placeholder namespace is served,
    announced, never silent; nothing in it computes."""
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


def fake_git(_repo, *a):
    """Scripted committed history: ONE committed ensemble_latest.json blob
    (scored day 2026-08-31) and the legacy data.csv at HEAD."""
    if a[0] == "log":
        return b"c1\n"
    if a[0] == "rev-parse" and a[1] == "c1:docs/ensemble_latest.json":
        return b"b" * 40 + b"\n"
    if a[0] == "rev-parse" and a[1] == "HEAD":
        return b"h" * 40 + b"\n"
    if a[0] == "cat-file":
        return LEGACY_0831_RAW
    if a[0] == "show":
        return LEGACY_CSV
    raise AssertionError(a)


def _mkstore():
    repo = tempfile.mkdtemp(prefix="rev-model-bar-")
    os.makedirs(_p(repo, "docs"))
    os.makedirs(_p(repo, "monitoring/dashboard"))
    with io.open(_p(repo, REV.CSV_REL), "wb") as f:
        f.write(LEGACY_CSV)
    cap = REV.build_legacy_baseline(repo, git=fake_git)
    REV.write_legacy_baseline(repo, cap)
    return repo, REV.load_legacy_baseline(repo)


def _inputs(tag):
    return {"schema": REV.INPUTS_SCHEMA, "entries": [
        REV.input_entry("code", "monitoring/src/x.py", None, None,
                        raw_bytes=tag.encode()),
        REV.input_entry("scored_day", "2026-09-02", "2026-09-02", None)]}


FIRED = datetime(2026, 9, 3, 6, 15, 1, 123456, tzinfo=timezone.utc)


def main():
    print("DAILY REVISION MODEL red-KATs (cayley, corrected v1) -- temp stores, scripted git, no network")
    repo, cap = _mkstore()
    try:
        # ---- R-0 live positive
        snap0 = REV.journal_bytes(repo)
        view = REV.prior_days_view(repo, snap0, cap, "2026-09-02", 3, git=fake_git)
        pins = [v[2] for v in view]
        inputs1 = _inputs("a")
        e1 = REV.publish_revision(repo, _rec("2026-09-02", {"alpha": 1, "beta": 0}),
                                  inputs1, snap0, pins, FIRED)
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
              and txn_clean
              and set(rev1["revision"]) == REV.REVISION_FIELDS
              and rev1["revision"]["source_index"] == {"entry_count": 0,
                                                       "prefix_sha256": hashlib.sha256(b"").hexdigest()}
              and rev1["revision"]["inputs_sha256"] == REV.inputs_sha256(rev1["revision"]["inputs"]))
        _check("R-0 LIVE-POSITIVE", ok,
               f"{e1['run_id']}: 1 canonical journal line, latest byte-equal, csv = legacy prefix + 2 rows, "
               "staging clean, closed revision block, source_index of the empty snapshot, inputs_sha256 recomputes")

        # ---- R-1 duplicate / rescore-without-prior / empty reason
        before = (REV.journal_bytes(repo), _read(_p(repo, REV.CSV_REL)),
                  sorted(os.listdir(_p(repo, "docs/ensemble/2026-09-02"))))
        snap1 = REV.journal_bytes(repo)
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-02", {"alpha": 2, "beta": 0}), inputs1, snap1, pins, FIRED),
            "REVISION_EXISTS")
        after = (REV.journal_bytes(repo), _read(_p(repo, REV.CSV_REL)),
                 sorted(os.listdir(_p(repo, "docs/ensemble/2026-09-02"))))
        _check("R-1 DUPLICATE-REFUSES", hit and before == after,
               f"{msg[:60]}... ; store byte-unchanged={before == after}")
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-03", {"alpha": 0}), inputs1, snap1, pins, FIRED,
            rescore_reason="x"), "RESCORE_WITHOUT_PRIOR")
        _check("R-1b RESCORE-NEEDS-PRIOR", hit, msg[:80])
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-02", {"alpha": 2}), inputs1, snap1, pins, FIRED,
            rescore_reason="  "), "RESCORE_REASON_EMPTY")
        _check("R-1c REASON-REQUIRED", hit, msg[:80])

        # ---- R-2 rescore
        inputs2 = _inputs("b")
        e2 = REV.publish_revision(repo, _rec("2026-09-02", {"alpha": 2, "beta": 0}),
                                  inputs2, snap1, pins, FIRED.replace(hour=7),
                                  rescore_reason="input fix: THD baseline")
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
               "r2 supersedes the exact current run; journal is a byte prefix (C-8); only 09-02 rows "
               "rewritten; legacy prefix and r1 byte-identical; latest -> r2 (C-10)")

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
               "orphan revision / dirty .txn / dangling journal line each refuse the next run with a "
               "typed RECOVERY instruction; the clean store passes")

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
        view = REV.prior_days_view(repo, snap, cap, "2026-09-03", 3, git=fake_git)
        kinds = [v[2]["kind"] for v in view]
        pmap = {k + 1: r for k, (_d, r, _pin) in enumerate(view)}
        pub = RED.check_persistence(current, Path(repo), tgt, 2, loader=lambda k: pmap.get(k))
        ld = tempfile.mkdtemp(prefix="rev-model-local-")
        for d, tiers in (("2026-09-02", {"alpha": 2, "beta": 0}), ("2026-08-31", {"alpha": 0})):
            with io.open(os.path.join(ld, f"ensemble_{d}.json"), "w", encoding="utf-8") as f:
                json.dump(_rec(d, tiers), f)
        loc = RED.check_persistence(current, Path(ld), tgt, 2)
        shutil.rmtree(ld, ignore_errors=True)
        # days back 1..3 from 09-03: 09-02 = journaled revision r2, 09-01 = HOLE
        # (neither source has it), 08-31 = legacy committed blob
        ok = (kinds == ["revision", "hole", "legacy"]
              and view[0][2]["run_id"] == e2["run_id"] and view[0][2]["sha256"] == e2["sha256"]
              and view[2][2]["legacy"]["git_blob"] == "b" * 40
              and pub["alpha"]["tier_history"] == [0, None, 2, 1] == loc["alpha"]["tier_history"]
              and pub["alpha"]["is_confirmed"] and pub["alpha"]["consecutive_days"] == 2
              and pub["beta"]["tier_history"] == [None, None, 0, 1] and not pub["beta"]["is_confirmed"]
              and set(pub["alpha"]) == {"current_tier", "consecutive_days", "is_confirmed", "tier_history"})
        _check("R-5 PUBLIC-PERSISTENCE-UNION", ok,
               f"prior kinds {kinds} (09-02 revision r2 bound by sha; 09-01 hole in both sources; "
               f"08-31 legacy blob); real check_persistence alpha={pub['alpha']['tier_history']} == "
               "local replay; four keys")

        # ---- R-6 source_index snapshot / JOURNAL_MOVED / C-9 replay
        stale_snap = j1                     # resolved before the rescore landed
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-03", {"alpha": 1, "beta": 1}), _inputs("c"), stale_snap,
            [v[2] for v in view], FIRED.replace(day=4)), "JOURNAL_MOVED")
        e3 = REV.publish_revision(repo, _rec("2026-09-03", {"alpha": 1, "beta": 1}), _inputs("c"),
                                  snap, [v[2] for v in view], FIRED.replace(day=4))
        r3, _ = REV.reopen_revision(repo, e3)
        si = r3["revision"]["source_index"]
        replay_prefix = REV.journal_bytes(repo)[:len(snap)]
        replay = REV.prior_days_view(repo, replay_prefix, cap, "2026-09-03", 3, git=fake_git)
        ok = (hit and si == {"entry_count": 2, "prefix_sha256": hashlib.sha256(snap).hexdigest()}
              and hashlib.sha256(replay_prefix).hexdigest() == si["prefix_sha256"]
              and [v[2] for v in replay] == r3["revision"]["persistence_inputs"])
        _check("R-6 SOURCE-INDEX-SNAPSHOT", ok,
               f"stale snapshot -> {msg[:40]}...; recorded source_index {{entry_count 2, prefix sha}} "
               "== the prefix actually consumed; C-9 replay from that prefix reproduces the pins")

        # ---- R-7 inputs capsule
        capsule = r3["revision"]["inputs"]
        d0 = REV.inputs_sha256(capsule)
        tampered = json.loads(json.dumps(capsule))
        tampered["entries"][0]["sha256"] = ("0" if tampered["entries"][0]["sha256"][0] != "0" else "1") \
            + tampered["entries"][0]["sha256"][1:]
        d1 = REV.inputs_sha256(tampered)
        opened = json.loads(json.dumps(capsule))
        opened["entries"][0]["extra"] = 1
        h_open, m_open = _refuses(lambda: REV.inputs_sha256(opened), "INPUTS_CAPSULE_SCHEMA")
        _check("R-7 INPUTS-CAPSULE", d0 == r3["revision"]["inputs_sha256"] and d1 != d0 and h_open,
               "inputs_sha256 recomputes from the stored capsule; one digit changed -> different digest; "
               "an open entry shape refuses")

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
               f"main(): {len(utc)} UTC now, {len(naive)} naive; mutant detected; scored_day_utc == date; "
               "5 ids at the same instant are 5 distinct YYYYMMDDTHHMMSSffffffZ-hex8 ids")

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
        repo2, cap2 = _mkstore()
        try:
            s0 = MUT.journal_bytes(repo2)
            v0 = MUT.prior_days_view(repo2, s0, cap2, "2026-09-02", 3, git=fake_git)
            MUT.publish_revision(repo2, _rec("2026-09-02", {"alpha": 1}), _inputs("a"), s0,
                                 [v[2] for v in v0], FIRED)
            s1 = MUT.journal_bytes(repo2)
            try:
                MUT.publish_revision(repo2, _rec("2026-09-02", {"alpha": 2}), _inputs("a"), s1,
                                     [v[2] for v in v0], FIRED.replace(hour=8))
                outcome = "ACCEPTED"
            except MUT.RevisionRefusal as e:
                outcome = str(e)
            _check("R-10 ANTI-VACUITY", "REVISION_EXISTS" not in outcome,
                   f"guard removed on a COPY -> outcome: {outcome[:80]}")
        finally:
            shutil.rmtree(repo2, ignore_errors=True)
            shutil.rmtree(mdir, ignore_errors=True)

        # ---- R-11 legacy surfaces
        h_once, m_once = _refuses(lambda: REV.write_legacy_baseline(repo, cap), "REVISION_PATH_EXISTS")
        csv_now = _read(_p(repo, REV.CSV_REL))
        with io.open(_p(repo, REV.CSV_REL), "wb") as f:
            f.write(csv_now.replace(b"0.5000", b"0.5001", 1))
        h_pref, m_pref = _refuses(lambda: REV.derive_csv_bytes(repo, cap, REV.parse_journal(REV.journal_bytes(repo))),
                                  "CSV_LEGACY_PREFIX_CHANGED")
        with io.open(_p(repo, REV.CSV_REL), "wb") as f:
            f.write(csv_now)
        _check("R-11 LEGACY-SURFACES", h_once and h_pref,
               "capsule is create-once; a changed legacy csv prefix refuses the derivation")
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    if FAILS:
        print(f"DAILY REVISION MODEL: {len(FAILS)} FAIL -> {FAILS}")
        return 1
    print("DAILY REVISION MODEL (corrected v1): ALL PASS (temp stores; scripted git; nothing scored; nothing public touched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
