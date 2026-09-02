#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAILY-PATH IMMUTABLE REVISION MODEL -- red-KAT lock (cayley).

asylum 2026-09-02: "1. use immutable revision 2. fix B5 and land B6".
codex's model (2026-09-01 2303Z): create-once per-day revisions, an
append-only index, a derived current pointer, duplicate-date REFUSAL,
owner --rescore <reason> appending a NEW revision, data.csv DERIVED by
one writer, persistence bound to the exact prior revisions consumed.

Every check below runs the REAL module (`ensemble_revisions_cayley`) and
the REAL runner functions (`run_ensemble_daily.check_persistence`) over
a temporary store built here; nothing public is read or written and no
network is touched. Each RED control is one change away from the live
positive and refuses for its OWN measured reason.

  R-0  live positive: publish -> index -> current -> latest -> csv
  R-1  duplicate default run REFUSES (REVISION_EXISTS), writes nothing
  R-2  --rescore appends r02 naming r01; ONLY that date's csv rows are
       rewritten; frozen pre-model rows are byte-identical and ordered
  R-3  index is append-only: a reordered / edited / duplicated index
       REFUSES on validation; a tampered committed revision REFUSES on
       reopen (digest)
  R-4  persistence over PUBLIC revisions: identical verdicts to the
       local-dir replay when the days exist; a missing public day is a
       HOLE (None in tier_history), never a confirmation
  R-5  B6: the production scored-day key is derived from the UTC clock
       (AST of main); the naive-local form is absent
  R-6  contracts the landed corpus bar depends on are intact: the
       check_persistence / save_results positional signatures and the
       bar's exact mutation-anchor source line
  R-7  anti-vacuity: the R-1 refusal disappears when the guard is
       removed from a COPY of the module (the control fires because of
       the guard, not because of the fixture)
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
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ensemble_revisions_cayley as REV  # noqa: E402

FAILS = []


def _shim_obspy_if_absent():
    """FIXTURE-ONLY import shim. The runner's import chain reaches obspy
    through the acquisition stack (ensemble -> fault_correlation ->
    seismic_data). This bar exercises only pure functions of the runner
    (check_persistence, the AST of main, signatures), so on a host without
    obspy it registers an import-only dummy carrying the two names
    seismic_data binds at import. It is installed ONLY when the real
    package is absent and is announced, never silent; it computes
    nothing."""
    try:
        import obspy  # noqa: F401
        return False
    except ImportError:
        pass
    import importlib.abc
    import importlib.machinery
    import types

    class _Placeholder:
        """Inert stand-in for any name imported from the shimmed package."""
        def __init__(self, *a, **k):
            raise RuntimeError("FIXTURE obspy shim: acquisition code must not "
                               "execute inside test_daily_revision_model")

    class _ShimLoader(importlib.abc.Loader):
        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []          # every submodule import resolves here too
            m.__fixture_only__ = ("import-only shim; "
                                  "test_daily_revision_model_redkats_cayley")
            m.__getattr__ = lambda name: _Placeholder
            return m

        def exec_module(self, module):
            return None

    class _ShimFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "obspy" or fullname.startswith("obspy."):
                return importlib.machinery.ModuleSpec(
                    fullname, _ShimLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _ShimFinder())
    print("    NOTE: obspy absent on this host -> import-only FIXTURE shim "
          "installed for obspy.* (no acquisition code runs in this bar)")
    return True


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


def _mkrepo():
    repo = tempfile.mkdtemp(prefix="rev-model-bar-")
    os.makedirs(_p(repo, "docs"))
    os.makedirs(_p(repo, "monitoring/dashboard"))
    return repo


FROZEN = ("date,region,tier,risk,confidence,methods,agreement\n"
          "2026-08-30,alpha,1,0.5000,0.50,1,single_method\n"
          "2026-08-30,beta,0,0.0100,0.50,1,single_method\n"
          "2026-08-31,alpha,0,0.0200,0.50,1,single_method\n")


def _rec(date, tiers):
    return {"date": date,
            "timestamp": "2026-09-02T00:00:00+00:00",
            "regions": {r: {"tier": t, "combined_risk": 0.1 * (t + 1),
                            "confidence": 0.5, "methods_available": 1,
                            "agreement": "single_method",
                            "tier_name": "WATCH" if t else "NORMAL"}
                        for r, t in tiers.items()},
            "summary": {"total_regions": len(tiers)}}


def _csv(repo):
    with io.open(_p(repo, REV.CSV_REL), "r", newline="", encoding="utf-8") as f:
        return f.read()


def main():
    print("DAILY REVISION MODEL red-KATs (cayley) -- temp store, no network")
    repo = _mkrepo()
    try:
        with io.open(_p(repo, REV.CSV_REL), "wb") as f:
            f.write(FROZEN.encode())
        inputs1 = {"schema": "x", "code": "a" * 64}

        # ---- R-0 live positive
        e1 = REV.publish_revision(repo, _rec("2026-09-01", {"alpha": 1, "beta": 0}),
                                  inputs1, created_utc="2026-09-02T07:00:00Z")
        idx = REV.load_index(repo)
        cur = json.load(io.open(_p(repo, REV.CURRENT_REL), encoding="utf-8"))
        latest = io.open(_p(repo, REV.LATEST_REL), "rb").read()
        rev_bytes = io.open(_p(repo, e1["path"]), "rb").read()
        csv1 = _csv(repo)
        ok = (e1["run_id"] == "r01" and e1["supersedes"] is None
              and len(idx["revisions"]) == 1
              and cur["current"]["2026-09-01"]["run_id"] == "r01"
              and cur["latest_date"] == "2026-09-01"
              and latest == rev_bytes
              and hashlib.sha256(rev_bytes).hexdigest() == e1["sha256"]
              and csv1.startswith(FROZEN)
              and csv1[len(FROZEN):] == ("2026-09-01,alpha,1,0.2000,0.50,1,single_method\n"
                                         "2026-09-01,beta,0,0.1000,0.50,1,single_method\n"))
        _check("R-0 LIVE-POSITIVE", ok,
               f"r01 published; index 1; current/latest derived; csv = frozen + 2 rows ({e1['sha256'][:12]})")
        rec1 = json.loads(rev_bytes.decode("utf-8"))
        _check("R-0b RECORD-CARRIES-REVISION",
               rec1.get("revision", {}).get("run_id") == "r01"
               and rec1["revision"]["inputs"] == inputs1
               and rec1["revision"]["supersedes_inputs"] is None,
               "revision block with inputs and null supersedes_inputs")

        # ---- R-1 duplicate default run refuses, writes nothing
        before = (idx, csv1, sorted(os.listdir(_p(repo, "docs/ensemble/2026-09-01"))))
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-01", {"alpha": 2, "beta": 0}), inputs1), "REVISION_EXISTS")
        after = (REV.load_index(repo), _csv(repo),
                 sorted(os.listdir(_p(repo, "docs/ensemble/2026-09-01"))))
        _check("R-1 DUPLICATE-REFUSES", hit and before == after,
               f"{msg[:70]}... ; store byte-unchanged={before == after}")
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-03", {"alpha": 0}), inputs1, rescore_reason="x"),
            "RESCORE_WITHOUT_PRIOR")
        _check("R-1b RESCORE-NEEDS-PRIOR", hit, msg[:90])
        hit, msg = _refuses(lambda: REV.publish_revision(
            repo, _rec("2026-09-01", {"alpha": 2}), inputs1, rescore_reason="   "),
            "RESCORE_REASON_EMPTY")
        _check("R-1c REASON-REQUIRED", hit, msg[:90])

        # ---- R-2 rescore appends; only that date's rows rewritten
        inputs2 = {"schema": "x", "code": "b" * 64}
        e2 = REV.publish_revision(repo, _rec("2026-09-01", {"alpha": 2, "beta": 0}),
                                  inputs2, rescore_reason="input fix: THD baseline",
                                  created_utc="2026-09-02T08:00:00Z")
        idx2 = REV.load_index(repo)
        csv2 = _csv(repo)
        rec2 = json.load(io.open(_p(repo, e2["path"]), encoding="utf-8"))
        ok = (e2["run_id"] == "r02" and e2["supersedes"] == "r01"
              and [e["run_id"] for e in idx2["revisions"]] == ["r01", "r02"]
              and idx2["revisions"][0] == e1
              and os.path.exists(_p(repo, e1["path"]))
              and io.open(_p(repo, e1["path"]), "rb").read() == rev_bytes
              and rec2["revision"]["supersedes_inputs"] == inputs1
              and csv2.startswith(FROZEN)
              and csv2[len(FROZEN):] == ("2026-09-01,alpha,2,0.3000,0.50,1,single_method\n"
                                         "2026-09-01,beta,0,0.1000,0.50,1,single_method\n")
              and json.load(io.open(_p(repo, REV.CURRENT_REL), encoding="utf-8"))
              ["current"]["2026-09-01"]["run_id"] == "r02")
        _check("R-2 RESCORE-APPENDS-ONE-RULE", ok,
               "r02 supersedes r01 (r01 bytes intact); csv rows for 09-01 rewritten from r02; "
               "frozen rows byte-identical; current -> r02")

        # ---- R-3 append-only / digest locks
        good = REV.load_index(repo)
        bad_order = {"schema": REV.INDEX_SCHEMA,
                     "revisions": [good["revisions"][1], good["revisions"][0]]}
        hit1, m1 = _refuses(lambda: REV.validate_index(bad_order), "REVISION_INDEX_ORDER")
        edited = json.loads(json.dumps(good))
        edited["revisions"][0]["sha256"] = "0" * 64
        # an edited FIRST entry is caught when its revision is reopened
        hit2, m2 = _refuses(lambda: REV.reopen_revision(repo, edited["revisions"][0]),
                            "REVISION_DIGEST_MISMATCH")
        dup = json.loads(json.dumps(good))
        dup["revisions"].append(dict(good["revisions"][1]))
        hit3, m3 = _refuses(lambda: REV.validate_index(dup), "REVISION_INDEX_DUPLICATE")
        nosup = json.loads(json.dumps(good))
        nosup["revisions"][1]["supersedes"] = None
        hit4, m4 = _refuses(lambda: REV.validate_index(nosup), "REVISION_INDEX_SUPERSEDES")
        with io.open(_p(repo, e2["path"]), "ab") as f:
            f.write(b"\n")
        hit5, m5 = _refuses(lambda: REV.reopen_revision(repo, e2), "REVISION_DIGEST_MISMATCH")
        with io.open(_p(repo, e2["path"]), "wb") as f:
            f.write(REV.record_bytes(rec2))
        _check("R-3 APPEND-ONLY-LOCKS", hit1 and hit2 and hit3 and hit4 and hit5,
               "reordered / edited-digest / duplicated / supersedes-cleared index refuse; "
               "tampered committed revision refuses on reopen")

        # ---- R-4 persistence: public loader == local replay; holes explicit
        _shim_obspy_if_absent()
        import run_ensemble_daily as RED  # noqa  (real runner; import only)
        from ensemble import EnsembleResult, MethodResult  # noqa
        tgt = datetime(2026, 9, 2)

        def result(region, tier):
            return EnsembleResult(region=region, date=tgt, combined_risk=0.3,
                                  tier=tier, tier_name="WATCH" if tier else "NORMAL",
                                  components={}, confidence=0.5,
                                  agreement="single_method", methods_available=1)
        current = {"alpha": result("alpha", 1), "beta": result("beta", 1)}
        # local replay dir: 09-01 (alpha WATCH), 08-31 alpha NORMAL; no 08-30
        ld = tempfile.mkdtemp(prefix="rev-model-local-")
        for d, tiers in (("2026-09-01", {"alpha": 2, "beta": 0}),
                         ("2026-08-31", {"alpha": 0})):
            with io.open(os.path.join(ld, f"ensemble_{d}.json"), "w", encoding="utf-8") as f:
                json.dump(_rec(d, tiers), f)
        # public store: 09-01 is r02 (alpha 2); 08-31 / 08-30 are holes
        idxp = REV.load_index(repo)
        prior = REV.prior_days(repo, idxp, "2026-09-02", 3)
        pmap = {k + 1: r for k, (_d, r, _i) in enumerate(prior)}
        pub = RED.check_persistence(current, Path(repo), tgt, 2,
                                    loader=lambda k: pmap.get(k))
        # local replay with the SAME facts except 08-31 present as NORMAL:
        # alpha: [None(08-30), 0(08-31), 2(09-01), 1] -> consecutive 2, confirmed
        loc = RED.check_persistence(current, Path(ld), tgt, 2)
        ok = (pub["alpha"]["tier_history"] == [None, None, 2, 1]
              and pub["alpha"]["consecutive_days"] == 2 and pub["alpha"]["is_confirmed"]
              and pub["beta"]["tier_history"] == [None, None, 0, 1]
              and pub["beta"]["consecutive_days"] == 1 and not pub["beta"]["is_confirmed"]
              and loc["alpha"]["tier_history"] == [None, 0, 2, 1]
              and loc["alpha"]["is_confirmed"] == pub["alpha"]["is_confirmed"]
              and prior[0][2]["run_id"] == "r02" and prior[1] == ("2026-08-31", None, None))
        _check("R-4 PUBLIC-PERSISTENCE-BINDS-AND-HOLES", ok,
               f"public tier_history alpha={pub['alpha']['tier_history']} (08-31/08-30 holes), "
               f"prior[0] binds r02 sha {prior[0][2]['sha256'][:12]}; verdicts equal the local "
               "replay where days exist")
        four = {"current_tier", "consecutive_days", "is_confirmed", "tier_history"}
        _check("R-4b PER-REGION-KEYS", set(pub["alpha"]) == four,
               f"per-region keys {sorted(pub['alpha'])} == the four the corpus bar compares")
        shutil.rmtree(ld, ignore_errors=True)

        # ---- R-5 B6: UTC-derived scored day in main()
        src = io.open(os.path.join(HERE, "run_ensemble_daily.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        main_fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
        naive_now = [n for n in ast.walk(main_fn)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "now" and not n.args]
        utc_now = [n for n in ast.walk(main_fn)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "now" and n.args
                   and isinstance(n.args[0], ast.Attribute) and n.args[0].attr == "utc"]
        _check("R-5 B6-UTC-DAY-KEY", not naive_now and len(utc_now) >= 1,
               f"main(): {len(utc_now)} datetime.now(timezone.utc) call(s), {len(naive_now)} naive now() call(s)")
        # the mutation control: the pre-B6 form must be DETECTED by this same scan
        mut = src.replace("today_utc = datetime.now(timezone.utc).date()",
                          "today_utc = datetime.now().date()")
        mtree = ast.parse(mut)
        mmain = [n for n in mtree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
        mnaive = [n for n in ast.walk(mmain)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "now" and not n.args]
        _check("R-5b B6-MUTANT-DETECTED", mut != src and len(mnaive) == 1,
               "naive datetime.now() reintroduced -> the scan goes RED")

        # ---- R-6 corpus-bar contracts intact
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        cp = [a.arg for a in fns["check_persistence"].args.args]
        sr = [a.arg for a in fns["save_results"].args.args]
        anchor = ("is_confirmed = consecutive >= required_consecutive "
                  "if current_tier >= 1 else False")
        _check("R-6 CORPUS-BAR-CONTRACTS",
               cp[:4] == ["current_results", "output_dir", "target_date", "required_consecutive"]
               and sr == ["results", "output_dir", "target_date", "persistence", "events_data"]
               and anchor in src,
               f"check_persistence{tuple(cp)}, save_results{tuple(sr)}, mutation anchor present")

        # ---- R-7 anti-vacuity: R-1 fires because of the guard
        msrc = io.open(os.path.join(HERE, "ensemble_revisions_cayley.py"), encoding="utf-8").read()
        guard = "    if existing and rescore_reason is None:\n        raise RevisionRefusal(\n"
        assert guard in msrc, "guard text not found -- bar needs re-anchoring"
        msrc2 = msrc.replace(guard, "    if False:\n        raise RevisionRefusal(\n", 1)
        mdir = tempfile.mkdtemp(prefix="rev-model-mut-")
        mpath = os.path.join(mdir, "ensemble_revisions_mutant.py")
        io.open(mpath, "w", encoding="utf-8", newline="\n").write(msrc2)
        spec = importlib.util.spec_from_file_location("ensemble_revisions_mutant", mpath)
        MUT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MUT)
        repo2 = _mkrepo()
        try:
            with io.open(_p(repo2, REV.CSV_REL), "wb") as f:
                f.write(FROZEN.encode())
            MUT.publish_revision(repo2, _rec("2026-09-01", {"alpha": 1}), inputs1,
                                 created_utc="2026-09-02T07:00:00Z")
            try:
                MUT.publish_revision(repo2, _rec("2026-09-01", {"alpha": 2}), inputs1,
                                     created_utc="2026-09-02T07:01:00Z")
                outcome = "ACCEPTED"
            except MUT.RevisionRefusal as e:
                outcome = str(e)
            # with the guard gone the duplicate must NOT be refused as
            # REVISION_EXISTS any more (it is accepted, or caught only
            # downstream by the index validator as a reason-less seq 2) --
            # so the R-1 control measures the guard, not the fixture
            _check("R-7 ANTI-VACUITY", "REVISION_EXISTS" not in outcome,
                   f"guard removed on a COPY -> outcome: {outcome[:90]}")
        finally:
            shutil.rmtree(repo2, ignore_errors=True)
            shutil.rmtree(mdir, ignore_errors=True)
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    if FAILS:
        print(f"DAILY REVISION MODEL: {len(FAILS)} FAIL -> {FAILS}")
        return 1
    print("DAILY REVISION MODEL: ALL PASS (temp stores; nothing scored; nothing public touched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
