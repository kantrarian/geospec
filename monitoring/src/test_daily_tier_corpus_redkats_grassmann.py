#!/usr/bin/env python3
"""
DAILY-TIER-CORPUS red-KAT bar -- grassmann, 2026-09-01.

Locks the daily monitor's tier / renormalization / single-method-cap /
persistence / summary / append-only-history layer against the program's
own committed output: every `docs/ensemble_latest.json` blob in the
ancestry of a pinned commit (the current-output layer, one blob per
"Daily monitoring" commit) plus `docs/data.csv` (the append-only
history) at that commit.

The bar drives the REAL production functions -- `GeoSpecEnsemble.compute_risk`,
`check_persistence`, `save_results` -- with only the three DATA-producing
component computations stubbed from each committed record. It never
re-implements the combine logic as an oracle: if the logic in
`ensemble.py` / `run_ensemble_daily.py` changes, the committed corpus stops
reproducing and the bar names the exact region-days and fields that flip.

LOCKS (typed, PASS/FAIL, exit 1 on any FAIL):
  C-0 BINDING      --commit resolves; the worktree copies of the two audited
                   modules are byte-identical (LF) to the blobs at that commit
                   (else CORPUS_WORKTREE_DIVERGENT); corpus = every
                   docs/ensemble_latest.json blob in `git rev-list <commit>`.
  C-1 COMBINE      real compute_risk (components stubbed from the record, the
                   record's own freeze flags honored) reproduces combined_risk,
                   tier, tier_name, confidence, agreement, methods_available,
                   effective_weights, notes, coverage and every component's
                   post-freeze notes. Divergences are classified mechanically;
                   only the pre-era classes pinned in LEDGER (with a max date and
                   an exact count) are tolerated. Anything else FAILS.
  C-2a PERSISTENCE (exact) real check_persistence over a temp dir holding the
                   prior days' records AS THEY WERE COMMITTED BEFORE that run.
                   Where the public history lacks a prior day, or holds a
                   different version than the runner saw locally, the row is
                   typed (UNCOMMITTED_PRIOR_DAY / LOCAL_HISTORY_NE_COMMITTED)
                   and its count pinned in LEDGER -- those are findings about
                   the public history, not about the counting rule.
  C-2b PERSISTENCE (self-consistent) real check_persistence over the record's
                   OWN tier_history must reproduce consecutive_days /
                   is_confirmed / tier_history for EVERY row. No tolerance.
  C-3 SUMMARY      real save_results into an empty temp dir reproduces the
                   record's summary block (on the record's keys) and every
                   region/component key present in the record round-trips.
  C-4 DATA.CSV     rows dated on/after the append-only era must equal the
                   EARLIEST committed record for their (date, region) with the
                   writer's own f-strings (the writer skips regions already
                   present for a date, so a re-score never rewrites a row).
                   Pre-era rows and rows with no committed record are typed
                   and their counts pinned.
  C-5 APPEND-ONLY  from APPEND_ONLY_SINCE forward, every older docs/data.csv
                   content is a line-prefix of every newer one. The pre-era
                   rewrites (there were 22) are pinned as an exact list.
  C-6 LEDGER       every pinned class count and max date matches exactly. A
                   new divergence OR a vanished one fails.

NOT LOCKED: fetch, envelope, correlation, THD, Lambda_geo computation,
capsule admission, dashboard HTML, scored-day selection. Authorizes nothing.

--selftest plants source-level mutations into TEMP COPIES of the audited
modules (never the tree) and proves each lock goes RED for exactly the
region-days an independent scan of the corpus predicts, plus a no-op
mutation that must stay CLEAN and a data.csv edit named by line.

Usage (from monitoring/src):
  python test_daily_tier_corpus_redkats_grassmann.py --repo <root> [--commit <rev>] [--selftest]
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RECORD_PATH = "docs/ensemble_latest.json"
CSV_PATH = "docs/data.csv"
AUDITED = ("monitoring/src/ensemble.py", "monitoring/src/run_ensemble_daily.py")
FROZEN_NOTE = "FROZEN (incident 2026-07-31): excluded from tier pending fix"
FLOAT_TOL = 1e-12
CSV_HEADER = ["date", "region", "tier", "risk", "confidence", "methods", "agreement"]

# The last docs/data.csv history rewrite on public master: 2026-06-10
# "fix: restore 5-week data hole (May 2-Jun 7)". From this content forward the
# history is append-only; every earlier rewrite is pinned below.
APPEND_ONLY_SINCE = "81f704570e9d5381ff79424dd0f477330582cd2f"
APPEND_ONLY_SINCE_DATE = "2026-06-10"

# Frozen ledger: (lock, class) -> (exact count, max scored date). Measured at
# public 94968394; every entry was inspected before it was pinned.
LEDGER: Dict[Tuple[str, str], Tuple[int, str]] = {
    # C-1: January-2026 records predate the DEGRADED tier, the three-component
    # record shape and the current combine formula.
    ("C-1", "PRE_ERA_NO_DEGRADED_TIER"): (9, "2026-01-10"),
    ("C-1", "PRE_ERA_LAMBDA_ONLY_RECORD"): (6, "2026-01-10"),
    # Jan-2026 combine multiplied an all_elevated row by exactly 1.1 (measured ratio); gone since.
    ("C-1", "PRE_ERA_ALL_ELEVATED_BOOST_1P1"): (9, "2026-01-21"),
    # C-2a: the runner reads prior days from its LOCAL results dir; the public
    # history is missing some of those days or holds a different version.
    ("C-2a", "UNCOMMITTED_PRIOR_DAY"): (179, "2026-08-10"),
    ("C-2a", "LOCAL_HISTORY_NE_COMMITTED"): (242, "2026-06-08"),
    # C-4: rows before the append-only era / rows predating the record corpus.
    ("C-4", "NO_COMMITTED_RECORD"): (1304, "2026-08-07"),
    ("C-4", "PRE_ERA_ROW_MISMATCH"): (182, "2026-05-01"),
    ("C-4", "PRE_ERA_LATER_VERSION"): (45, "2026-05-01"),
    # append-only era rows that match NO committed record for their (date, region): a disagreement
    # between docs/data.csv and the current-output layer. FINDING, pinned; growth fails.
    ("C-4", "CSV_RECORD_DISAGREEMENT"): (39, "2026-08-08"),
}
PRE_ERA_MAX_DATE = "2026-01-21"   # last scored day on which a C-1 pre-era class may occur

# The 22 pre-era docs/data.csv rewrites, (older commit, newer commit), pinned.
PRE_ERA_CSV_REWRITES: Tuple[Tuple[str, str], ...] = (
    ("4583f78be1", "7f39bc5d95"),
    ("534ed57fc3", "f092bfcc5c"),
    ("8ee0aa4963", "a698091f44"),
    ("a698091f44", "717eca7c50"),
    ("717eca7c50", "946a0afaa4"),
    ("946a0afaa4", "ab19b4dc1d"),
    ("ab19b4dc1d", "30f2a5c21a"),
    ("d75e374ddd", "68462d0980"),
    ("68462d0980", "1c5dd55d45"),
    ("1c5dd55d45", "225742d8cd"),
    ("225742d8cd", "798571c61e"),
    ("798571c61e", "e99cfc4015"),
    ("e99cfc4015", "7c967ec8f1"),
    ("7c967ec8f1", "4e580be9e2"),
    ("4e580be9e2", "992b968a4d"),
    ("992b968a4d", "2f55cd595e"),
    ("c8118ca242", "3fc2f5177a"),
    ("cb0fd37a5a", "92dd964c60"),
    ("92dd964c60", "4c046b7343"),
    ("c7723b274c", "dba4489a71"),
    ("f1d783a11a", "d9494aa198"),
    ("d9494aa198", "81f704570e"),
)

FAILS: List[str] = []
PASSES: List[str] = []


def _ok(name: str, detail: str = ""):
    PASSES.append(name)
    print(f"    [PASS] {name}" + (f" -- {detail}" if detail else ""))


def _fail(name: str, detail: str):
    FAILS.append(f"{name} {detail}")
    print(f"    [FAIL] {name} -- {detail}")


def _note(name: str, detail: str):
    print(f"    [NOTE] {name} -- {detail}")


# --------------------------------------------------------------------------- git
def _git(repo: str, args: List[str], binary: bool = False):
    r = subprocess.run(["git", "-C", repo] + args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def _blob(repo: str, rev: str, path: str) -> bytes:
    return _git(repo, ["show", f"{rev}:{path}"], binary=True)


def _lf_sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


# --------------------------------------------------------------------------- corpus
class Corpus:
    """Every committed record in the ancestry of `commit`, newest first."""

    def __init__(self, repo: str, commit: str):
        self.repo = repo
        self.commit = commit
        self.commits: List[str] = _git(repo, ["rev-list", commit, "--", RECORD_PATH]).split()
        self.records: List[Tuple[int, str, dict]] = []   # (index newest=0, commit, record)
        self.unparseable: List[str] = []
        for i, c in enumerate(self.commits):
            raw = _blob(repo, c, RECORD_PATH)
            try:
                d = json.loads(raw)
            except Exception:
                self.unparseable.append(c)
                continue
            if not isinstance(d, dict) or "regions" not in d or "date" not in d:
                self.unparseable.append(c)
                continue
            self.records.append((i, c, d))
        self.by_date: Dict[str, List[Tuple[int, str, dict]]] = {}
        for i, c, d in self.records:
            self.by_date.setdefault(d["date"], []).append((i, c, d))

    def version_before(self, date: str, index: int) -> Optional[dict]:
        """Most recent committed record for `date` OLDER than commit `index` (None if none)."""
        cands = [(i, d) for i, _, d in self.by_date.get(date, []) if i > index]
        return min(cands, key=lambda t: t[0])[1] if cands else None

    def versions(self, date: str) -> List[dict]:
        return [d for _, _, d in self.by_date.get(date, [])]

    def earliest(self, date: str) -> Optional[dict]:
        cands = self.by_date.get(date, [])
        return max(cands, key=lambda t: t[0])[2] if cands else None


# --------------------------------------------------------------------------- module loading
def _load_module_from_source(name: str, source: str, tmpdir: str):
    """Load `source` as module `name` from a temp file without disturbing sys.modules."""
    # unique file per load + no bytecode: mutants can be byte-length-identical to the original and
    # written within one mtime tick, which would let the pyc cache serve a stale mutant
    _load_module_from_source.n = getattr(_load_module_from_source, "n", 0) + 1
    sys.dont_write_bytecode = True
    path = os.path.join(tmpdir, f"{name}_mutant_{_load_module_from_source.n}.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)
    return mod


def _strip_frozen_note(notes: str) -> str:
    if notes == FROZEN_NOTE:
        return ""
    suffix = " | " + FROZEN_NOTE
    return notes[: -len(suffix)] if notes.endswith(suffix) else notes


def _method_result(ens_mod, cdict: dict, keep_frozen: bool = False):
    b = cdict.get("baseline") or {}
    return ens_mod.MethodResult(
        name=cdict["name"],
        available=bool(cdict["available"]),
        raw_value=cdict.get("raw_value"),
        raw_secondary=cdict.get("raw_secondary"),
        risk_score=float(cdict.get("risk_score", 0.0)),
        is_elevated=bool(cdict.get("is_elevated", False)),
        is_critical=bool(cdict.get("is_critical", False)),
        notes=(cdict.get("notes", "") or "") if keep_frozen else _strip_frozen_note(cdict.get("notes", "") or ""),
        baseline_mean=float(b.get("mean", 0.0)),
        baseline_std=float(b.get("std", 0.0)),
        baseline_n=int(b.get("n_samples", 0)),
        baseline_window=b.get("window", ""),
        baseline_quality=b.get("quality", "unknown"),
        z_score=float(cdict.get("z_score", 0.0)),
        sample_rate_hz=float(cdict.get("sample_rate_hz", 0.0)),
        frozen=bool(cdict.get("frozen", False)) if keep_frozen else False,
    )


def _feq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= FLOAT_TOL
        except (TypeError, ValueError):
            return False
    return a == b


def _diff_fields(got: dict, exp: dict, keys) -> List[str]:
    """Field diffs restricted to keys PRESENT in the expected (committed) record."""
    out = []
    for k in keys:
        if k not in exp:
            continue
        g, e = got.get(k, "<absent>"), exp[k]
        if isinstance(e, dict) and isinstance(g, dict):
            gs = {str(a): b for a, b in g.items()}
            es = {str(a): b for a, b in e.items()}
            out += [f"{k}.{d}" for d in _diff_fields(gs, es, tuple(es.keys()))]
        elif not _feq(g, e):
            out.append(f"{k}: got {g!r} expected {e!r}")
    return out


Key = Tuple[str, str, str]   # (scored date, region, commit[:10])


def _region_result(ens_mod, region: str, rv: dict):
    cov = rv.get("coverage", {})
    comps = {k: _method_result(ens_mod, v, keep_frozen=True) for k, v in rv.get("components", {}).items()}
    return ens_mod.EnsembleResult(
        region=region, date=datetime.fromisoformat(rv["date"]), combined_risk=rv["combined_risk"],
        tier=rv["tier"], tier_name=rv["tier_name"], components=comps, confidence=rv["confidence"],
        agreement=rv["agreement"], methods_available=rv["methods_available"],
        notes=rv.get("notes", ""), segments_defined=int(cov.get("segments_defined", 0)),
        segments_working=int(cov.get("segments_working", 0)),
        segment_names=list(cov.get("segment_names", [])),
        effective_weights=dict(rv.get("effective_weights", {})))


# --------------------------------------------------------------------------- C-1
def lock_c1_combine(corpus: Corpus, ens_mod) -> Dict[Key, List[str]]:
    divergent: Dict[Key, List[str]] = {}
    ensembles: Dict[str, object] = {}
    saved_frozen = ens_mod.component_frozen
    try:
        for _, c, rec in corpus.records:
            for region, rv in rec["regions"].items():
                key = (rec["date"], region, c[:10])
                comps = rv.get("components", {})
                mr = {k: _method_result(ens_mod, v) for k, v in comps.items()}
                if set(mr) != {"lambda_geo", "fault_correlation", "seismic_thd"}:
                    divergent[key] = [f"components: {sorted(mr)}"]
                    continue
                frozen_flags = {k: bool(v.get("frozen", False)) for k, v in comps.items()}
                ens_mod.component_frozen = lambda r, cn, _f=frozen_flags: _f.get(cn, False)
                ens = ensembles.get(region)
                if ens is None:
                    ens = ens_mod.GeoSpecEnsemble(region)
                    ensembles[region] = ens
                cov = rv.get("coverage", {})
                ens.compute_lambda_geo_risk = lambda d, _m=mr: _m["lambda_geo"]
                ens.compute_fault_correlation_risk = lambda d, _m=mr, _c=cov: (
                    _m["fault_correlation"], int(_c.get("segments_defined", 0)),
                    int(_c.get("segments_working", 0)), list(_c.get("segment_names", [])))
                ens.compute_thd_risk = lambda d, _m=mr, **kw: _m["seismic_thd"]
                got = ens.compute_risk(datetime.fromisoformat(rv["date"])).to_dict()
                diffs = _diff_fields(got, rv, ("combined_risk", "tier", "tier_name", "confidence",
                                               "agreement", "methods_available", "notes",
                                               "effective_weights", "coverage"))
                for cn, v in comps.items():
                    diffs += [f"components.{cn}.{d}" for d in
                              _diff_fields(got["components"][cn], v,
                                           ("available", "risk_score", "is_elevated",
                                            "is_critical", "frozen", "notes"))]
                if diffs:
                    divergent[key] = diffs
    finally:
        ens_mod.component_frozen = saved_frozen
    return divergent


def classify_c1(key: Key, diffs: List[str], rv: dict) -> str:
    date = key[0]
    if date > PRE_ERA_MAX_DATE:
        return "UNEXPLAINED"
    if diffs[0].startswith("components: ") and "effective_weights" not in rv:
        return "PRE_ERA_LAMBDA_ONLY_RECORD"
    fields = {d.split(":")[0] for d in diffs}
    if fields <= {"tier", "tier_name", "notes"} and rv.get("methods_available", 1) == 0 \
            and any("got -1" in d for d in diffs):
        return "PRE_ERA_NO_DEGRADED_TIER"
    if fields <= {"combined_risk", "tier", "tier_name"} and rv.get("agreement") == "all_elevated":
        m = re.search(r"combined_risk: got ([0-9.eE+-]+) expected ([0-9.eE+-]+)", diffs[0])
        if m and abs(float(m.group(2)) / float(m.group(1)) - 1.1) <= 1e-9:
            return "PRE_ERA_ALL_ELEVATED_BOOST_1P1"
    return "UNEXPLAINED"


# --------------------------------------------------------------------------- C-2
def _persistence_replay(red_mod, ens_mod, td: str, rec: dict, prior: Dict[str, Optional[dict]]):
    for f in os.listdir(td):
        os.remove(os.path.join(td, f))
    for dstr, prev in prior.items():
        if prev is not None:
            with open(os.path.join(td, f"ensemble_{dstr}.json"), "w") as f:
                json.dump(prev, f)
    target = datetime.fromisoformat(next(iter(rec["regions"].values()))["date"])
    current = {r: _region_result(ens_mod, r, rv) for r, rv in rec["regions"].items()}
    return red_mod.check_persistence(current, Path(td), target)


def lock_c2a_persistence_exact(corpus: Corpus, ens_mod, red_mod) -> Dict[Key, Tuple[str, List[str]]]:
    """Returns {key: (class, diffs)}; class in EXACT-mismatch classes or UNEXPLAINED."""
    out: Dict[Key, Tuple[str, List[str]]] = {}
    with tempfile.TemporaryDirectory(prefix="corpus-persist-") as td:
        for idx, c, rec in corpus.records:
            if not any("persistence" in rv for rv in rec["regions"].values()):
                continue
            target = datetime.fromisoformat(next(iter(rec["regions"].values()))["date"])
            prior = {}
            for back in range(1, 5):
                dstr = (target - timedelta(days=back)).strftime("%Y-%m-%d")
                prior[dstr] = corpus.version_before(dstr, idx)
            got = _persistence_replay(red_mod, ens_mod, td, rec, prior)
            for region, rv in rec["regions"].items():
                if "persistence" not in rv:
                    continue
                exp = rv["persistence"]
                diffs = _diff_fields(got[region], exp,
                                     ("current_tier", "consecutive_days", "is_confirmed", "tier_history"))
                if not diffs:
                    continue
                gh, eh = got[region]["tier_history"], exp["tier_history"]
                if len(gh) == len(eh) and all(g == e or g is None for g, e in zip(gh, eh)):
                    cls = "UNCOMMITTED_PRIOR_DAY"
                elif len(gh) == len(eh) and gh[-1] == eh[-1]:
                    cls = "LOCAL_HISTORY_NE_COMMITTED"
                else:
                    cls = "UNEXPLAINED"
                out[(rec["date"], region, c[:10])] = (cls, diffs)
    return out


def lock_c2b_persistence_self(corpus: Corpus, ens_mod, red_mod) -> Dict[Key, List[str]]:
    """The record's own tier_history must reproduce its own consecutive/confirmed under the real rule."""
    divergent: Dict[Key, List[str]] = {}
    with tempfile.TemporaryDirectory(prefix="corpus-persist-self-") as td:
        for idx, c, rec in corpus.records:
            rows = {r: rv for r, rv in rec["regions"].items() if "persistence" in rv}
            if not rows:
                continue
            target = datetime.fromisoformat(next(iter(rec["regions"].values()))["date"])
            # synthesize the prior-day files from each region's own tier_history[:-1] (oldest first)
            prior: Dict[str, dict] = {}
            for region, rv in rows.items():
                hist = rv["persistence"]["tier_history"]
                n_prior = len(hist) - 1
                for k, t in enumerate(hist[:-1]):
                    back = n_prior - k
                    dstr = (target - timedelta(days=back)).strftime("%Y-%m-%d")
                    if t is None:
                        continue
                    prior.setdefault(dstr, {"regions": {}})["regions"][region] = {"tier": t}
            got = _persistence_replay(red_mod, ens_mod, td, rec, prior)
            for region, rv in rows.items():
                diffs = _diff_fields(got[region], rv["persistence"],
                                     ("current_tier", "consecutive_days", "is_confirmed", "tier_history"))
                if diffs:
                    divergent[(rec["date"], region, c[:10])] = diffs
    return divergent


# --------------------------------------------------------------------------- C-3
def lock_c3_summary(corpus: Corpus, ens_mod, red_mod) -> Dict[Key, List[str]]:
    divergent: Dict[Key, List[str]] = {}
    with tempfile.TemporaryDirectory(prefix="corpus-summary-") as td:
        for _, c, rec in corpus.records:
            for f in os.listdir(td):
                os.remove(os.path.join(td, f))
            results = {r: _region_result(ens_mod, r, rv) for r, rv in rec["regions"].items()}
            persistence = {r: rv["persistence"] for r, rv in rec["regions"].items() if "persistence" in rv}
            target = datetime.fromisoformat(next(iter(rec["regions"].values()))["date"])
            out = red_mod.save_results(results, Path(td), target, persistence=persistence or None,
                                       events_data=rec.get("earthquake_events"))
            with open(out) as f:
                written = json.load(f)
            diffs = _diff_fields({"summary": written.get("summary", {})}, {"summary": rec.get("summary", {})},
                                 ("summary",))
            if diffs:
                divergent[(rec["date"], "<summary>", c[:10])] = diffs
            for region, rv in rec["regions"].items():
                wr = written["regions"][region]
                d2 = _diff_fields(wr, rv, ("combined_risk", "tier", "tier_name", "confidence",
                                           "agreement", "methods_available", "notes",
                                           "effective_weights", "coverage", "persistence"))
                for cn, cv in rv.get("components", {}).items():
                    d2 += [f"components.{cn}.{d}" for d in
                           _diff_fields(wr["components"][cn], cv, tuple(cv.keys()))]
                if d2:
                    divergent[(rec["date"], region, c[:10])] = d2
    return divergent


# --------------------------------------------------------------------------- C-4
def _csv_row_for(rv: dict, date: str, region: str) -> List[str]:
    return [date, region, str(rv["tier"]), f"{rv['combined_risk']:.4f}",
            f"{rv['confidence']:.2f}", str(rv["methods_available"]), rv["agreement"] or ""]


def lock_c4_data_csv(corpus: Corpus, csv_text: str) -> Tuple[List[str], Dict[str, Tuple[int, str]], int]:
    """Returns (hard problems, {class: (count, max date)}, n rows)."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    problems: List[str] = []
    classes: Dict[str, List[str]] = {"NO_COMMITTED_RECORD": [], "PRE_ERA_ROW_MISMATCH": [],
                                     "PRE_ERA_LATER_VERSION": [], "CSV_RECORD_DISAGREEMENT": []}
    lines_by_class: Dict[str, List[str]] = {k: [] for k in classes}
    lock_c4_data_csv.last_lines = lines_by_class
    if not rows or rows[0] != CSV_HEADER:
        return [f"header: {rows[:1]!r}"], {k: (0, "") for k in classes}, 0
    seen = set()
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != 7:
            problems.append(f"line {n}: {len(row)} fields")
            continue
        date, region = row[0], row[1]
        if (date, region) in seen:
            problems.append(f"line {n}: duplicate row for {date} {region}")
        seen.add((date, region))
        earliest = corpus.earliest(date)
        if earliest is None or region not in earliest["regions"]:
            classes["NO_COMMITTED_RECORD"].append(date)
            continue
        exp = _csv_row_for(earliest["regions"][region], date, region)
        if row == exp:
            continue
        alts = [_csv_row_for(v["regions"][region], date, region)
                for v in corpus.versions(date) if region in v["regions"]]
        if date >= APPEND_ONLY_SINCE_DATE:
            # append-only era: the row must be the earliest committed record; a row matching NO
            # committed version is a public-history / current-output disagreement (pinned finding)
            cls = "PRE_ERA_LATER_VERSION" if row in alts else "CSV_RECORD_DISAGREEMENT"
        else:
            cls = "PRE_ERA_LATER_VERSION" if row in alts else "PRE_ERA_ROW_MISMATCH"
        classes[cls].append(date)
        lines_by_class[cls].append(f"line {n}: {date} {region}: row {row[2:]} vs earliest {exp[2:]}")
    summary = {k: (len(v), max(v) if v else "") for k, v in classes.items()}
    return problems, summary, len(rows) - 1


# --------------------------------------------------------------------------- C-5
def _append_only_violations(seq: List[Tuple[str, bytes]]) -> List[Tuple[str, str, str]]:
    """(older label, newer label, description) for every non-prefix step in seq (oldest first)."""
    out = []
    prev_lines, prev_label = None, None
    for label, content in seq:
        lines = content.replace(b"\r\n", b"\n").split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]
        if prev_lines is not None:
            if len(lines) < len(prev_lines):
                out.append((prev_label, label, f"history SHRANK {len(prev_lines)} -> {len(lines)}"))
            else:
                for k, (a, b) in enumerate(zip(prev_lines, lines)):
                    if a != b:
                        out.append((prev_label, label, f"line {k + 1} EDITED: "
                                    f"{a.decode('utf-8', 'replace')!r} -> {b.decode('utf-8', 'replace')!r}"))
                        break
        prev_lines, prev_label = lines, label
    return out


def lock_c5_append_only(repo: str, commit: str):
    commits = _git(repo, ["rev-list", "--reverse", commit, "--", CSV_PATH]).split()
    seq = [(c, _blob(repo, c, CSV_PATH)) for c in commits]
    viol = _append_only_violations(seq)
    if APPEND_ONLY_SINCE not in commits:
        raise SystemExit(f"CORPUS_APPEND_ONLY_ANCHOR_ABSENT: {APPEND_ONLY_SINCE[:12]} not in {CSV_PATH} history")
    anchor = commits.index(APPEND_ONLY_SINCE)
    pre = [(a[:10], b[:10]) for a, b, _ in viol if commits.index(b) <= anchor]
    post = [(a[:10], b[:10], d) for a, b, d in viol if commits.index(b) > anchor]
    return pre, post, len(commits) - anchor


# --------------------------------------------------------------------------- bar
def run_bar(repo: str, commit: str, ens_mod=None, red_mod=None, quiet: bool = False,
            csv_override: Optional[str] = None) -> dict:
    full = _git(repo, ["rev-parse", f"{commit}^{{commit}}"])
    for rel in AUDITED:
        blob_sha = _lf_sha(_blob(repo, full, rel))
        with open(os.path.join(repo, rel.replace("/", os.sep)), "rb") as f:
            live_sha = _lf_sha(f.read())
        if blob_sha != live_sha and ens_mod is None:
            raise SystemExit(f"CORPUS_WORKTREE_DIVERGENT: {rel} worktree {live_sha[:12]} != "
                             f"blob {blob_sha[:12]} at {full[:12]}")
    if ens_mod is None:
        import ensemble as ens_mod  # noqa
    if red_mod is None:
        import run_ensemble_daily as red_mod  # noqa
    corpus = Corpus(repo, full)
    n_rd = sum(len(r["regions"]) for _, _, r in corpus.records)
    if not quiet:
        print(f"  corpus @ {full[:12]}: {len(corpus.commits)} commits, {len(corpus.records)} records, "
              f"{n_rd} region-days, {len(corpus.by_date)} dates, "
              f"{sum(1 for v in corpus.by_date.values() if len(v) > 1)} dates with >1 version, "
              f"{len(corpus.unparseable)} unparseable")
    c1 = lock_c1_combine(corpus, ens_mod)
    c2a = lock_c2a_persistence_exact(corpus, ens_mod, red_mod)
    c2b = lock_c2b_persistence_self(corpus, ens_mod, red_mod)
    c3 = lock_c3_summary(corpus, ens_mod, red_mod)
    csv_text = csv_override if csv_override is not None else _blob(repo, full, CSV_PATH).decode("utf-8")
    c4, c4cls, n_rows = lock_c4_data_csv(corpus, csv_text)
    c4lines = lock_c4_data_csv.last_lines
    c5pre, c5post, n_post = lock_c5_append_only(repo, full)
    return {"commit": full, "corpus": corpus, "n_region_days": n_rd, "c1": c1, "c2a": c2a, "c2b": c2b,
            "c3": c3, "c4": c4, "c4cls": c4cls, "c4lines": c4lines, "n_rows": n_rows,
            "c5pre": c5pre, "c5post": c5post, "n_post": n_post}


def _by_class(items: Dict[Key, str]) -> Dict[str, Tuple[int, str]]:
    out: Dict[str, List[str]] = {}
    for (d, _, _), cls in items.items():
        out.setdefault(cls, []).append(d)
    return {k: (len(v), max(v)) for k, v in out.items()}


def report(res: dict) -> None:
    corpus = res["corpus"]
    _ok("C-0 BINDING", f"{res['commit'][:12]}: audited modules byte-bound; "
                       f"{len(corpus.records)} records / {res['n_region_days']} region-days")
    if corpus.unparseable:
        _note("C-0", f"{len(corpus.unparseable)} UNPARSEABLE_RECORD blob(s) "
                     f"{[c[:10] for c in corpus.unparseable]} excluded from replay (typed)")
    measured: Dict[Tuple[str, str], Tuple[int, str]] = {}

    # C-1
    c1cls = {}
    for key, diffs in res["c1"].items():
        rv = next(rec for _, c, rec in corpus.records if c.startswith(key[2]))["regions"][key[1]]
        c1cls[key] = classify_c1(key, diffs, rv)
    unexplained = [(k, res["c1"][k][0]) for k, cls in c1cls.items() if cls == "UNEXPLAINED"]
    for cls, cnt in _by_class(c1cls).items():
        measured[("C-1", cls)] = cnt
    if unexplained:
        _fail("C-1 COMBINE", f"{len(unexplained)} UNEXPLAINED divergent region-day(s): "
                             + "; ".join(f"{k[0]} {k[1]} @{k[2]}: {d}" for k, d in unexplained[:5]))
    else:
        _ok("C-1 COMBINE", f"{res['n_region_days'] - len(res['c1'])} region-days reproduce; "
                           f"{len(res['c1'])} pre-era ledgered")

    # C-2a
    c2acls = {k: v[0] for k, v in res["c2a"].items()}
    unexplained = [(k, res["c2a"][k][1][0]) for k, cls in c2acls.items() if cls == "UNEXPLAINED"]
    for cls, cnt in _by_class(c2acls).items():
        measured[("C-2a", cls)] = cnt
    n_pers = sum(1 for _, _, r in corpus.records for rv in r["regions"].values() if "persistence" in rv)
    if unexplained:
        _fail("C-2a PERSISTENCE-EXACT", f"{len(unexplained)} UNEXPLAINED: "
                                       + "; ".join(f"{k[0]} {k[1]} @{k[2]}: {d}" for k, d in unexplained[:5]))
    else:
        _ok("C-2a PERSISTENCE-EXACT", f"{n_pers - len(res['c2a'])} of {n_pers} rows reproduce from the "
                                      f"committed history; {len(res['c2a'])} typed (public history incomplete)")

    # C-2b
    if res["c2b"]:
        k, d = next(iter(res["c2b"].items()))
        _fail("C-2b PERSISTENCE-SELF", f"{len(res['c2b'])} row(s) violate the counting rule on their own "
                                      f"history: {k[0]} {k[1]} @{k[2]}: {d[0]}")
    else:
        _ok("C-2b PERSISTENCE-SELF", f"{n_pers} rows: own tier_history -> consecutive_days/is_confirmed "
                                     f"reproduce under the real rule")

    # C-3
    if res["c3"]:
        k, d = next(iter(res["c3"].items()))
        _fail("C-3 SUMMARY", f"{len(res['c3'])} record(s): {k[0]} {k[1]} @{k[2]}: {d[0]}")
    else:
        _ok("C-3 SUMMARY", f"{len(corpus.records)} records: summary + region/component round-trip")

    # C-4
    for cls, cnt in res["c4cls"].items():
        measured[("C-4", cls)] = cnt
    if res["c4"]:
        _fail("C-4 DATA.CSV", f"{len(res['c4'])} problem(s): {res['c4'][:4]}")
    else:
        n_typed = sum(c for c, _ in res["c4cls"].values())
        dis = res["c4cls"].get("CSV_RECORD_DISAGREEMENT", (0, ""))
        _ok("C-4 DATA.CSV", f"{res['n_rows'] - n_typed} rows == earliest committed record (writer format); "
                            f"{n_typed} typed incl. {dis[0]} CSV_RECORD_DISAGREEMENT (pinned finding)")

    # C-5
    if res["c5post"]:
        _fail("C-5 APPEND-ONLY", f"{len(res['c5post'])} rewrite(s) AFTER {APPEND_ONLY_SINCE[:10]}: "
                                 f"{res['c5post'][:3]}")
    elif tuple(res["c5pre"]) != PRE_ERA_CSV_REWRITES:
        _fail("C-5 APPEND-ONLY", f"pre-era rewrite list changed: {len(res['c5pre'])} vs "
                                 f"{len(PRE_ERA_CSV_REWRITES)} pinned; first diff "
                                 f"{next((p for p in res['c5pre'] if p not in PRE_ERA_CSV_REWRITES), None)}")
    else:
        _ok("C-5 APPEND-ONLY", f"{res['n_post']} data.csv commits since {APPEND_ONLY_SINCE[:10]} "
                               f"({APPEND_ONLY_SINCE_DATE}), every older content a prefix; "
                               f"{len(res['c5pre'])} pre-era rewrites pinned")

    # C-6
    problems = []
    for k, (cnt, mx) in LEDGER.items():
        m = measured.get(k, (0, ""))
        if m != (cnt, mx):
            problems.append(f"{k}: measured {m} pinned {(cnt, mx)}")
    for k, m in measured.items():
        if k not in LEDGER and m[0]:
            problems.append(f"{k}: measured {m} NOT PINNED")
    if problems:
        _fail("C-6 LEDGER", "; ".join(problems[:6]))
    else:
        _ok("C-6 LEDGER", f"{len(LEDGER)} pinned classes match exactly (count + max date)")


# --------------------------------------------------------------------------- selftest
def _mutate(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"SELFTEST_MUTATION_ANCHOR: {old!r} occurs {src.count(old)} times")
    return src.replace(old, new)


def selftest(repo: str, commit: str) -> int:
    import ensemble as real_ens
    import run_ensemble_daily as real_red
    with open(os.path.join(repo, "monitoring", "src", "ensemble.py"), encoding="utf-8") as f:
        ens_src = f.read()
    with open(os.path.join(repo, "monitoring", "src", "run_ensemble_daily.py"), encoding="utf-8") as f:
        red_src = f.read()
    base = run_bar(repo, commit, real_ens, real_red, quiet=True)
    corpus = base["corpus"]
    fails = 0

    def check(name, got: set, exp: set):
        nonlocal fails
        if got == exp:
            _ok(name, f"exactly the {len(exp)} predicted key(s)" if exp else "0 keys, as predicted")
        else:
            fails += 1
            _fail(name, f"got {len(got)} exp {len(exp)}; got-exp {sorted(got - exp)[:3]} "
                        f"exp-got {sorted(exp - got)[:3]}")

    def rows(pred):
        return {(rec["date"], r, c[:10]) for _, c, rec in corpus.records
                for r, rv in rec["regions"].items() if pred(rv)}

    with tempfile.TemporaryDirectory(prefix="corpus-selftest-") as td:
        # M-A: WATCH floor 0.25 -> 0.30. RED exactly where 0.25 <= risk < 0.30 and the tier is real.
        m = _mutate(_mutate(ens_src, "'min_risk': 0.0, 'max_risk': 0.25", "'min_risk': 0.0, 'max_risk': 0.30"),
                    "'min_risk': 0.25, 'max_risk': 0.50", "'min_risk': 0.30, 'max_risk': 0.50")
        got = set(lock_c1_combine(corpus, _load_module_from_source("ensemble", m, td)))
        exp = rows(lambda rv: 0.25 <= rv["combined_risk"] < 0.30 and rv["tier"] == 1) | set(base["c1"])
        check("M-A WATCH floor 0.25->0.30 names exactly the band", got, exp)

        # M-B: MIN_METHODS_FOR_OPERATIONAL 1 -> 2. RED exactly on every single-method row.
        m = _mutate(ens_src, "MIN_METHODS_FOR_OPERATIONAL = 1", "MIN_METHODS_FOR_OPERATIONAL = 2")
        got = set(lock_c1_combine(corpus, _load_module_from_source("ensemble", m, td)))
        # single-method rows flip to DEGRADED; zero-method rows keep tier -1 but their DEGRADED note
        # text ("need >=N") changes, so they flip too
        exp = rows(lambda rv: rv.get("methods_available", 0) <= 1) | set(base["c1"])
        check("M-B MIN_METHODS 1->2 names exactly the <=1-method rows", got, exp)

        # M-C: confirmation rule '>=' -> '>'. RED (self-consistency) exactly on confirmed rows with 2 days.
        m = _mutate(red_src, "is_confirmed = consecutive >= required_consecutive if current_tier >= 1 else False",
                    "is_confirmed = consecutive > required_consecutive if current_tier >= 1 else False")
        got = set(lock_c2b_persistence_self(corpus, real_ens, _load_module_from_source("run_ensemble_daily", m, td)))
        exp = rows(lambda rv: "persistence" in rv and rv["persistence"]["is_confirmed"]
                   and rv["persistence"]["consecutive_days"] == 2) | set(base["c2b"])
        check("M-C confirmation >= -> > names exactly the 2-day confirmed rows", got, exp)

        # M-D: no-op mutation must stay CLEAN (same divergence set as the real module).
        got = set(lock_c1_combine(corpus, _load_module_from_source("ensemble", ens_src + "\n# no-op\n", td)))
        check("M-D no-op mutation stays clean", got, set(base["c1"]))

        # M-E: one post-era data.csv row's tier flipped in memory -> C-4 names exactly that line.
        csv_text = _blob(repo, base["commit"], CSV_PATH).decode("utf-8")
        lines = csv_text.split("\n")
        k = next(i for i in range(len(lines) - 1, 0, -1) if lines[i].startswith(APPEND_ONLY_SINCE_DATE[:7]))
        cells = lines[k].split(",")
        cells[2] = "3" if cells[2] != "3" else "0"
        lines[k] = ",".join(cells)
        lock_c4_data_csv(corpus, "\n".join(lines))
        named = {p.split(":")[0] for p in lock_c4_data_csv.last_lines["CSV_RECORD_DISAGREEMENT"]}
        base_named = {p.split(":")[0] for p in base["c4lines"]["CSV_RECORD_DISAGREEMENT"]}
        check("M-E data.csv row edit -> CSV_RECORD_DISAGREEMENT names exactly that line", named,
              base_named | {f"line {k + 1}"})

        # M-F: the real append-only comparator on a synthetic history with one edited line + one shrink.
        seq = [("c1", b"h\na\n"), ("c2", b"h\na\nb\n"), ("c3", b"h\nX\nb\n"), ("c4", b"h\nX\n")]
        v = [(a, b, d.split(":")[0]) for a, b, d in _append_only_violations(seq)]
        if v == [("c2", "c3", "line 2 EDITED"), ("c3", "c4", "history SHRANK 3 -> 2")]:
            _ok("M-F append-only comparator names the edited line and the shrink", "2 of 3 steps")
        else:
            fails += 1
            _fail("M-F append-only comparator", f"{v}")

    print()
    print("DAILY-TIER-CORPUS SELFTEST: " + ("ALL PASS" if not fails else f"{fails} FAIL"))
    return 1 if fails else 0


# --------------------------------------------------------------------------- main
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-ledger", action="store_true", help="print measured classes (for pinning)")
    a = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)
    repo = os.path.abspath(a.repo)
    try:
        _git(repo, ["rev-parse", f"{a.commit}^{{commit}}"])
    except RuntimeError as e:
        print(f"CORPUS_REVISION_UNRESOLVABLE: {e}")
        return 2
    print(f"DAILY-TIER-CORPUS red-KAT bar (grassmann) -- repo {repo} commit {a.commit}")
    if a.selftest:
        return selftest(repo, a.commit)
    res = run_bar(repo, a.commit)
    if a.dump_ledger:
        corpus = res["corpus"]
        c1cls = {}
        for key, diffs in res["c1"].items():
            rv = next(rec for _, c, rec in corpus.records if c.startswith(key[2]))["regions"][key[1]]
            c1cls[key] = classify_c1(key, diffs, rv)
        print("C-1", _by_class(c1cls))
        for k, cls in c1cls.items():
            if cls == "UNEXPLAINED":
                print("   ", k, res["c1"][k][:2])
        print("C-2a", _by_class({k: v[0] for k, v in res["c2a"].items()}))
        for k, (cls, d) in res["c2a"].items():
            if cls == "UNEXPLAINED":
                print("   ", k, d[:2])
        print("C-2b", len(res["c2b"]), list(res["c2b"].items())[:3])
        print("C-3", len(res["c3"]), list(res["c3"].items())[:3])
        print("C-4", res["c4cls"], res["c4"][:3])
        for ln in res["c4lines"]["CSV_RECORD_DISAGREEMENT"]:
            print("   ", ln[:150])
        print("C-5 pre", res["c5pre"])
        print("C-5 post", res["c5post"])
        return 0
    report(res)
    print()
    if FAILS:
        print(f"DAILY-TIER-CORPUS RED-KAT FAILURES ({len(FAILS)}): {[f.split(' ')[0] for f in FAILS]}")
        return 1
    print("ALL DAILY-TIER-CORPUS RED-KATs PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
