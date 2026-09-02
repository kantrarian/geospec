#!/usr/bin/env python3
"""
DAILY-TIER-CORPUS red-KAT bar -- grassmann, 2026-09-02 (REV 3: codex 0003Z repairs 1-2 on top of 2303Z).

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

TWO COMMITS. `--commit` (target_commit) is the production history under
audit. `--bar-commit` (bar_commit) is the commit the EXECUTING bar and its
sidecar ledger are bound to; it defaults to the worktree's exact HEAD and is
overridable only explicitly. Before the ledger is loaded or any lock is
reported, SOURCE BINDING (exit 3, typed) requires, in this order:
  CORPUS_BAR_NOT_IN_TREE    the executing module's __file__ is not (samefile)
                            <repo>/monitoring/src/<this file>;
  CORPUS_BAR_NOT_COMMITTED  bar_commit has no blob at that path (or none for
                            the sidecar);
  CORPUS_BAR_DIVERGENT      the executing bar's LF bytes != its blob at
                            bar_commit;
  CORPUS_LEDGER_DIVERGENT   the live sidecar's LF bytes != its blob at
                            bar_commit (the independent LEDGER_SHA256 pin in
                            C-6 is retained as defense in depth).
Both commits and all four full blob digests (bar / ledger at bar_commit, and
the live files) are printed on every run. A byte-modified external copy of
this bar, an edited in-tree bar, or an edited sidecar therefore cannot emit a
commit-labelled verdict (selftest M-K proves each refusal, and the untouched
committed positive proceeds).

LOCKS (typed, PASS/FAIL, exit 1 on any FAIL):
  C-0 BINDING      --commit resolves; the worktree copies of the two audited
                   modules are byte-identical (LF) to the blobs at that commit
                   (else CORPUS_WORKTREE_DIVERGENT); the source binding above
                   is reported here with its digests; corpus = every
                   docs/ensemble_latest.json blob in `git rev-list <commit>`.
                   Blobs that are not a record are a FROZEN EXCLUSION SET
                   (UNPARSEABLE_EXCLUSIONS, full commit sha -> blob digest):
                   any added, removed or changed exclusion FAILS; the set and
                   the coverage denominator are printed every run.
  C-1 COMBINE      real compute_risk (components stubbed from the record, the
                   record's own freeze flags honored) reproduces combined_risk,
                   tier, tier_name, confidence, agreement, methods_available,
                   effective_weights, notes, coverage and every component's
                   post-freeze notes. Divergences are classified mechanically
                   (pre-era classes only, each bounded by PRE_ERA_MAX_DATE);
                   an UNEXPLAINED divergence FAILS outright.
  C-2a PERSISTENCE (exact) real check_persistence over a temp dir holding the
                   prior days' records AS THEY WERE COMMITTED BEFORE that run.
                   Where the public history lacks a prior day, or holds a
                   different version than the runner saw locally, the row is
                   typed (UNCOMMITTED_PRIOR_DAY / LOCAL_HISTORY_NE_COMMITTED):
                   findings about the public history, not the counting rule.
  C-2b PERSISTENCE (self-consistent) real check_persistence over the record's
                   OWN tier_history must reproduce consecutive_days /
                   is_confirmed / tier_history for EVERY row. No tolerance.
  C-3 SUMMARY      real save_results into an empty temp dir reproduces the
                   record's summary block (on the record's keys) and every
                   region/component key present in the record round-trips.
  C-4 DATA.CSV     rows dated on/after the append-only era must equal the
                   EARLIEST committed record for their (date, region) with the
                   writer's own f-strings. Rows that match NO committed version
                   are CSV_RECORD_DISAGREEMENT (a finding). Pre-era rows and
                   rows with no committed record are typed.
  C-5 APPEND-ONLY  from APPEND_ONLY_SINCE forward, every older docs/data.csv
                   content is a line-prefix of every newer one. The pre-era
                   rewrites (there were 22) are pinned as an exact list.
  C-6 LEDGER       EXACT-SET lock. Every typed exception of C-0/C-1/C-2a/C-4 is
                   an identity: (lock, class, date, region, FULL source commit
                   or CSV line) + the FULL 64-hex sha256 of its classified
                   content, hashed as canonical compact sorted-key JSON of the
                   STRUCTURED value ({"diffs": [...]} for C-1/C-2a,
                   {"actual": [cells], "expected": [cells]} for C-4, the raw
                   blob for C-0) -- never a newline- or '|'-joined string, so
                   values containing those characters cannot share a preimage
                   (selftest M-J proves the old joined framing collides and the
                   canonical framing separates). The full set lives in the
                   sidecar LEDGER_FILE whose sha256 is pinned here
                   (LEDGER_SHA256). The measured set must EQUAL the frozen set;
                   any added, removed, moved, reclassified or content-changed
                   exception FAILS and the set difference is printed. Count and
                   max-date per class are still printed, as information only.

NOT LOCKED: fetch, envelope, correlation, THD, Lambda_geo computation,
capsule admission, dashboard HTML, scored-day selection. Authorizes nothing.

--selftest plants source-level mutations into TEMP COPIES of the audited
modules (never the tree) and proves each lock goes RED for exactly the
region-days an independent scan of the corpus predicts; a no-op mutation
must stay CLEAN; a same-count/same-max-date swap of two CSV exceptions
(codex's line-3962/3970 substitution) must go RED under the exact-set
ledger while the old count+max summary would have accepted it; making a
parseable blob unparseable must go RED, and a swap that preserves the
exclusion count must go RED; a same-key content change of one exception
must change its full identity (M-I); the framing KAT (M-J) must show the
old joined preimage colliding and the canonical one separating; and the
source-binding KATs (M-K, run as real child processes in a throwaway
detached worktree at bar_commit) must show an external byte-modified bar,
an edited in-tree bar and an edited sidecar each REFUSED with the typed
code before any lock line, while the untouched committed positive binds.

Usage (from monitoring/src):
  python test_daily_tier_corpus_redkats_grassmann.py --repo <root> [--commit <rev>]
      [--bar-commit <rev>] [--selftest] [--bind-only] [--dump-ledger] [--write-ledger <path>]
  --write-ledger is the only mode that runs UNBOUND (it authors the sidecar and
  emits no verdict); it says so loudly.
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
BAR_FILE = "test_daily_tier_corpus_redkats_grassmann.py"
BAR_PATH = "monitoring/src/" + BAR_FILE
BOUND: Optional[dict] = None   # set by bind_sources(); report() refuses to run without it
FROZEN_NOTE = "FROZEN (incident 2026-07-31): excluded from tier pending fix"
FLOAT_TOL = 1e-12
CSV_HEADER = ["date", "region", "tier", "risk", "confidence", "methods", "agreement"]

# The last docs/data.csv history rewrite on public master: 2026-06-10
# "fix: restore 5-week data hole (May 2-Jun 7)". From this content forward the
# history is append-only; every earlier rewrite is pinned below.
APPEND_ONLY_SINCE = "81f704570e9d5381ff79424dd0f477330582cd2f"
APPEND_ONLY_SINCE_DATE = "2026-06-10"
PRE_ERA_MAX_DATE = "2026-01-21"   # last scored day on which a C-1 pre-era class may occur

# Frozen exclusion set: commits whose docs/ensemble_latest.json blob is not a
# record. Both are ZERO-BYTE files committed by "Daily monitoring" runs
# (2026-03-22 and 2026-06-07) -- the publish-stall mechanism cayley measured
# 2026-09-01T22:53Z. Full commit sha -> sha256 of the blob (empty file).
UNPARSEABLE_EXCLUSIONS: Dict[str, str] = {
    "83fee4ec07b8290152d02d328d65c708f4a20381": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Daily monitoring 2026-03-22, 0 B
    "d48c4786008f36ada5feebf771dc4277a97c20eb": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Daily monitoring 2026-06-07, 0 B
}

# Exact exception ledger: sidecar JSON (sorted list of identity dicts), sha256 pinned.
LEDGER_FILE = "daily_tier_corpus_ledger_grassmann.json"
LEDGER_PATH = "monitoring/src/" + LEDGER_FILE
LEDGER_SHA256 = "5de4948a29d435ee8f882f460a01cd8485a031b72fde873e54527add917804d3"   # 2,017 identities @ public 94968394 (REV 3 full-digest canonical framing)

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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(obj) -> bytes:
    """Canonical compact sorted-key UTF-8 JSON: length-framed by construction, so structured values
    containing newlines or '|' cannot share a preimage."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest_json(obj) -> str:
    """FULL 64-hex sha256 of the canonical structured value (codex 0003Z #2)."""
    return _sha(_canon(obj))


# --------------------------------------------------------------------------- git
def _git(repo: str, args: List[str], binary: bool = False):
    r = subprocess.run(["git", "-C", repo] + args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def _blob(repo: str, rev: str, path: str) -> bytes:
    return _git(repo, ["show", f"{rev}:{path}"], binary=True)


def _lf_sha(data: bytes) -> str:
    return _sha(data.replace(b"\r\n", b"\n"))


# --------------------------------------------------------------------------- corpus
class Corpus:
    """Every committed record in the ancestry of `commit`, newest first."""

    def __init__(self, repo: str, commit: str, blob_reader=None):
        self.repo = repo
        self.commit = commit
        read = blob_reader or (lambda c: _blob(repo, c, RECORD_PATH))
        self.commits: List[str] = _git(repo, ["rev-list", commit, "--", RECORD_PATH]).split()
        self.records: List[Tuple[int, str, dict]] = []   # (index newest=0, FULL commit, record)
        self.unparseable: Dict[str, str] = {}             # full commit -> blob sha256
        for i, c in enumerate(self.commits):
            raw = read(c)
            try:
                d = json.loads(raw)
            except Exception:
                self.unparseable[c] = _sha(raw)
                continue
            if not isinstance(d, dict) or "regions" not in d or "date" not in d:
                self.unparseable[c] = _sha(raw)
                continue
            self.records.append((i, c, d))
        self.by_date: Dict[str, List[Tuple[int, str, dict]]] = {}
        for i, c, d in self.records:
            self.by_date.setdefault(d["date"], []).append((i, c, d))
        self.by_commit: Dict[str, dict] = {c: d for _, c, d in self.records}

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
    """Field diffs restricted to keys PRESENT in the expected (committed) record; recursive on dicts."""
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


Key = Tuple[str, str, str]   # (scored date, region, FULL source commit)


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
                key = (rec["date"], region, c)
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
    """Returns {key: (class, diffs)}; class is a typed public-history class or UNEXPLAINED."""
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
                out[(rec["date"], region, c)] = (cls, diffs)
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
            prior: Dict[str, dict] = {}
            for region, rv in rows.items():
                hist = rv["persistence"]["tier_history"]
                n_prior = len(hist) - 1
                for k, t in enumerate(hist[:-1]):
                    dstr = (target - timedelta(days=n_prior - k)).strftime("%Y-%m-%d")
                    if t is None:
                        continue
                    prior.setdefault(dstr, {"regions": {}})["regions"][region] = {"tier": t}
            got = _persistence_replay(red_mod, ens_mod, td, rec, prior)
            for region, rv in rows.items():
                diffs = _diff_fields(got[region], rv["persistence"],
                                     ("current_tier", "consecutive_days", "is_confirmed", "tier_history"))
                if diffs:
                    divergent[(rec["date"], region, c)] = diffs
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
                divergent[(rec["date"], "<summary>", c)] = diffs
            for region, rv in rec["regions"].items():
                wr = written["regions"][region]
                d2 = _diff_fields(wr, rv, ("combined_risk", "tier", "tier_name", "confidence",
                                           "agreement", "methods_available", "notes",
                                           "effective_weights", "coverage", "persistence"))
                for cn, cv in rv.get("components", {}).items():
                    d2 += [f"components.{cn}.{d}" for d in
                           _diff_fields(wr["components"][cn], cv, tuple(cv.keys()))]
                if d2:
                    divergent[(rec["date"], region, c)] = d2
    return divergent


# --------------------------------------------------------------------------- C-4
def _csv_row_for(rv: dict, date: str, region: str) -> List[str]:
    return [date, region, str(rv["tier"]), f"{rv['combined_risk']:.4f}",
            f"{rv['confidence']:.2f}", str(rv["methods_available"]), rv["agreement"] or ""]


def lock_c4_data_csv(corpus: Corpus, csv_text: str) -> Tuple[List[str], List[dict], int]:
    """Returns (hard problems, typed entries [{cls, line, date, region, actual, expected}], n rows)."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    problems: List[str] = []
    entries: List[dict] = []
    if not rows or rows[0] != CSV_HEADER:
        return [f"header: {rows[:1]!r}"], entries, 0
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
            entries.append({"cls": "NO_COMMITTED_RECORD", "line": n, "date": date, "region": region,
                            "actual": list(row), "expected": []})
            continue
        exp = _csv_row_for(earliest["regions"][region], date, region)
        if row == exp:
            continue
        alts = [_csv_row_for(v["regions"][region], date, region)
                for v in corpus.versions(date) if region in v["regions"]]
        if date >= APPEND_ONLY_SINCE_DATE:
            # append-only era: the row must be the earliest committed record; a row matching NO
            # committed version is a public-history / current-output disagreement (a finding)
            cls = "PRE_ERA_LATER_VERSION" if row in alts else "CSV_RECORD_DISAGREEMENT"
        else:
            cls = "PRE_ERA_LATER_VERSION" if row in alts else "PRE_ERA_ROW_MISMATCH"
        entries.append({"cls": cls, "line": n, "date": date, "region": region,
                        "actual": list(row), "expected": list(exp)})
    return problems, entries, len(rows) - 1


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


# --------------------------------------------------------------------------- exception set (C-6)
def exception_set(res: dict) -> List[dict]:
    """Every typed exception as an identity dict. Sorted, canonical; this is what the ledger freezes."""
    corpus: Corpus = res["corpus"]
    out: List[dict] = []
    for c, dg in corpus.unparseable.items():
        out.append({"lock": "C-0", "cls": "UNPARSEABLE_RECORD", "commit": c, "digest": dg})
    for (d, r, c), diffs in res["c1"].items():
        cls = classify_c1((d, r, c), diffs, corpus.by_commit[c]["regions"][r])
        out.append({"lock": "C-1", "cls": cls, "date": d, "region": r, "commit": c,
                    "digest": _digest_json({"diffs": list(diffs)})})
    for (d, r, c), (cls, diffs) in res["c2a"].items():
        out.append({"lock": "C-2a", "cls": cls, "date": d, "region": r, "commit": c,
                    "digest": _digest_json({"diffs": list(diffs)})})
    for e in res["c4entries"]:
        out.append({"lock": "C-4", "cls": e["cls"], "date": e["date"], "region": e["region"],
                    "line": e["line"], "digest": _digest_json({"actual": list(e["actual"]),
                                                                "expected": list(e["expected"])})})
    return sorted(out, key=lambda e: json.dumps(e, sort_keys=True))


def _identity(e: dict) -> str:
    return json.dumps(e, sort_keys=True)


def load_ledger() -> Tuple[List[dict], str]:
    path = os.path.join(HERE, LEDGER_FILE)
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.replace(b"\r\n", b"\n")), _lf_sha(raw)


def bind_sources(repo: str, bar_commit_rev: str) -> dict:
    """codex 0003Z #1: bind the EXECUTING bar and its live sidecar to their blobs at bar_commit.
    Refuses (SystemExit, exit 3, typed) BEFORE the ledger is loaded or any lock is reported.
    Order: in-tree (samefile) -> committed at bar_commit -> bar bytes -> ledger bytes."""
    global BOUND
    bar_commit = _git(repo, ["rev-parse", f"{bar_commit_rev}^{{commit}}"])
    expected_bar = os.path.join(repo, BAR_PATH.replace("/", os.sep))
    executing = os.path.abspath(__file__)
    try:
        in_tree = os.path.samefile(executing, expected_bar)
    except OSError:
        in_tree = False
    if not in_tree:
        raise SystemExit(f"CORPUS_BAR_NOT_IN_TREE: executing {executing} is not {expected_bar}")
    blobs = {}
    for rel in (BAR_PATH, LEDGER_PATH):
        try:
            blobs[rel] = _lf_sha(_blob(repo, bar_commit, rel))
        except RuntimeError as e:
            raise SystemExit(f"CORPUS_BAR_NOT_COMMITTED: {rel} has no blob at bar_commit {bar_commit[:12]}: {e}")
    with open(executing, "rb") as f:
        live_bar = _lf_sha(f.read())
    if live_bar != blobs[BAR_PATH]:
        raise SystemExit(f"CORPUS_BAR_DIVERGENT: executing bar {live_bar} != blob {blobs[BAR_PATH]} "
                         f"at bar_commit {bar_commit[:12]}")
    ledger_live_path = os.path.join(HERE, LEDGER_FILE)
    try:
        with open(ledger_live_path, "rb") as f:
            live_ledger = _lf_sha(f.read())
    except FileNotFoundError:
        raise SystemExit(f"CORPUS_LEDGER_DIVERGENT: live sidecar {ledger_live_path} missing; blob at "
                         f"bar_commit {bar_commit[:12]} = {blobs[LEDGER_PATH]}")
    if live_ledger != blobs[LEDGER_PATH]:
        raise SystemExit(f"CORPUS_LEDGER_DIVERGENT: live sidecar {live_ledger} != blob {blobs[LEDGER_PATH]} "
                         f"at bar_commit {bar_commit[:12]}")
    BOUND = {"bar_commit": bar_commit, "bar_blob": blobs[BAR_PATH], "bar_live": live_bar,
             "ledger_blob": blobs[LEDGER_PATH], "ledger_live": live_ledger, "bar_file": executing}
    return BOUND


def print_binding(bound: dict, target_commit: str) -> None:
    print(f"  SOURCE BINDING: bar_commit {bound['bar_commit']} / target_commit {target_commit}")
    print(f"    bar    {BAR_PATH}: blob {bound['bar_blob']}")
    print(f"           live == blob: {bound['bar_live']} ({bound['bar_file']})")
    print(f"    ledger {LEDGER_PATH}: blob {bound['ledger_blob']}")
    print(f"           live == blob: {bound['ledger_live']}")


def class_summary(entries: List[dict]) -> Dict[Tuple[str, str], Tuple[int, str]]:
    by: Dict[Tuple[str, str], List[str]] = {}
    for e in entries:
        by.setdefault((e["lock"], e["cls"]), []).append(e.get("date", ""))
    return {k: (len(v), max(v)) for k, v in by.items()}


# --------------------------------------------------------------------------- bar
def run_bar(repo: str, commit: str, ens_mod=None, red_mod=None, quiet: bool = False,
            csv_override: Optional[str] = None, blob_reader=None) -> dict:
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
    corpus = Corpus(repo, full, blob_reader=blob_reader)
    n_rd = sum(len(r["regions"]) for _, _, r in corpus.records)
    if not quiet:
        print(f"  corpus @ {full[:12]}: {len(corpus.commits)} commits, {len(corpus.records)} records, "
              f"{n_rd} region-days, {len(corpus.by_date)} dates, "
              f"{sum(1 for v in corpus.by_date.values() if len(v) > 1)} dates with >1 version, "
              f"{len(corpus.unparseable)} unparseable (frozen exclusion set)")
    c1 = lock_c1_combine(corpus, ens_mod)
    c2a = lock_c2a_persistence_exact(corpus, ens_mod, red_mod)
    c2b = lock_c2b_persistence_self(corpus, ens_mod, red_mod)
    c3 = lock_c3_summary(corpus, ens_mod, red_mod)
    csv_text = csv_override if csv_override is not None else _blob(repo, full, CSV_PATH).decode("utf-8")
    c4, c4entries, n_rows = lock_c4_data_csv(corpus, csv_text)
    c5pre, c5post, n_post = lock_c5_append_only(repo, full)
    res = {"commit": full, "corpus": corpus, "n_region_days": n_rd, "c1": c1, "c2a": c2a, "c2b": c2b,
           "c3": c3, "c4": c4, "c4entries": c4entries, "n_rows": n_rows,
           "c5pre": c5pre, "c5post": c5post, "n_post": n_post}
    res["exceptions"] = exception_set(res)
    return res


def report(res: dict) -> None:
    if BOUND is None:
        raise SystemExit("CORPUS_UNBOUND_REPORT: report() called before bind_sources()")
    corpus: Corpus = res["corpus"]
    exc = res["exceptions"]
    summ = class_summary(exc)

    # C-0: binding + frozen exclusion set
    measured_excl = dict(corpus.unparseable)
    excl_diff = []
    for c, dg in measured_excl.items():
        if c not in UNPARSEABLE_EXCLUSIONS:
            excl_diff.append(f"ADDED {c[:12]} blob {dg[:12]}")
        elif UNPARSEABLE_EXCLUSIONS[c] != dg:
            excl_diff.append(f"CHANGED {c[:12]} blob {dg[:12]} != pinned {UNPARSEABLE_EXCLUSIONS[c][:12]}")
    for c in UNPARSEABLE_EXCLUSIONS:
        if c not in measured_excl:
            excl_diff.append(f"REMOVED {c[:12]} (now parseable or absent)")
    cov = f"{len(corpus.records)} of {len(corpus.commits)} commits replayed / {res['n_region_days']} region-days"
    if excl_diff:
        _fail("C-0 BINDING", f"frozen exclusion set changed: {excl_diff[:4]}")
    else:
        _ok("C-0 BINDING", f"target {res['commit'][:12]}: audited modules byte-bound; bar+sidecar bound to "
                           f"bar_commit {BOUND['bar_commit'][:12]} (bar blob {BOUND['bar_blob'][:12]}, "
                           f"ledger blob {BOUND['ledger_blob'][:12]}); {cov}; "
                           f"exclusion set = {len(UNPARSEABLE_EXCLUSIONS)} pinned zero-byte blob(s) "
                           f"{[c[:10] for c in UNPARSEABLE_EXCLUSIONS]}, exact")

    # C-1
    unexplained = [e for e in exc if e["lock"] == "C-1" and e["cls"] == "UNEXPLAINED"]
    if unexplained:
        e = unexplained[0]
        _fail("C-1 COMBINE", f"{len(unexplained)} UNEXPLAINED divergent region-day(s): "
                             f"{e['date']} {e['region']} @{e['commit'][:10]}: "
                             f"{res['c1'][(e['date'], e['region'], e['commit'])][0]}")
    else:
        _ok("C-1 COMBINE", f"{res['n_region_days'] - len(res['c1'])} region-days reproduce; "
                           f"{len(res['c1'])} pre-era typed")

    # C-2a
    n_pers = sum(1 for _, _, r in corpus.records for rv in r["regions"].values() if "persistence" in rv)
    unexplained = [e for e in exc if e["lock"] == "C-2a" and e["cls"] == "UNEXPLAINED"]
    if unexplained:
        e = unexplained[0]
        _fail("C-2a PERSISTENCE-EXACT", f"{len(unexplained)} UNEXPLAINED: {e['date']} {e['region']} "
                                       f"@{e['commit'][:10]}: {res['c2a'][(e['date'], e['region'], e['commit'])][1][0]}")
    else:
        _ok("C-2a PERSISTENCE-EXACT", f"{n_pers - len(res['c2a'])} of {n_pers} rows reproduce from the "
                                      f"committed history; {len(res['c2a'])} typed (public history incomplete)")

    # C-2b
    if res["c2b"]:
        k, d = next(iter(res["c2b"].items()))
        _fail("C-2b PERSISTENCE-SELF", f"{len(res['c2b'])} row(s) violate the counting rule on their own "
                                      f"history: {k[0]} {k[1]} @{k[2][:10]}: {d[0]}")
    else:
        _ok("C-2b PERSISTENCE-SELF", f"{n_pers} rows: own tier_history -> consecutive_days/is_confirmed "
                                     f"reproduce under the real rule")

    # C-3
    if res["c3"]:
        k, d = next(iter(res["c3"].items()))
        _fail("C-3 SUMMARY", f"{len(res['c3'])} record(s): {k[0]} {k[1]} @{k[2][:10]}: {d[0]}")
    else:
        _ok("C-3 SUMMARY", f"{len(corpus.records)} records: summary + region/component round-trip")

    # C-4
    if res["c4"]:
        _fail("C-4 DATA.CSV", f"{len(res['c4'])} structural problem(s): {res['c4'][:4]}")
    else:
        n_typed = len(res["c4entries"])
        dis = summ.get(("C-4", "CSV_RECORD_DISAGREEMENT"), (0, ""))
        _ok("C-4 DATA.CSV", f"{res['n_rows'] - n_typed} rows == earliest committed record (writer format); "
                            f"{n_typed} typed incl. {dis[0]} CSV_RECORD_DISAGREEMENT (finding, exact-set pinned)")

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

    # C-6: exact-set ledger
    for (lock, cls), (n, mx) in sorted(summ.items()):
        print(f"           {lock} {cls}: {n} (max scored date {mx or '-'})")
    try:
        frozen, sha = load_ledger()
    except FileNotFoundError:
        _fail("C-6 LEDGER", f"sidecar {LEDGER_FILE} missing")
        return
    if sha != LEDGER_SHA256:
        _fail("C-6 LEDGER", f"sidecar sha256 {sha[:16]} != pinned {LEDGER_SHA256[:16]}")
        return
    if sha != BOUND["ledger_blob"]:
        _fail("C-6 LEDGER", f"sidecar sha256 {sha[:16]} != blob {BOUND['ledger_blob'][:16]} at bar_commit")
        return
    bad_digest = [e for e in frozen if not re.fullmatch(r"[0-9a-f]{64}", str(e.get("digest", "")))]
    if bad_digest:
        _fail("C-6 LEDGER", f"{len(bad_digest)} frozen identities without a full 64-hex digest: {bad_digest[:2]}")
        return
    fro = {_identity(e) for e in frozen}
    mea = {_identity(e) for e in exc}
    added, removed = sorted(mea - fro), sorted(fro - mea)
    if added or removed:
        _fail("C-6 LEDGER", f"exception set differs from frozen: +{len(added)} -{len(removed)}; "
                            f"added {added[:3]}; removed {removed[:3]}")
    else:
        _ok("C-6 LEDGER", f"{len(fro)} frozen exception identities == measured set exactly "
                          f"(sidecar {LEDGER_FILE} sha {sha[:12]} == pin == blob at bar_commit "
                          f"{BOUND['bar_commit'][:12]}; full 64-hex canonical-JSON content digests)")


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
        return {(rec["date"], r, c) for _, c, rec in corpus.records
                for r, rv in rec["regions"].items() if pred(rv)}

    with tempfile.TemporaryDirectory(prefix="corpus-selftest-") as td:
        # M-A: WATCH floor 0.25 -> 0.30. RED exactly where 0.25 <= risk < 0.30 and the tier is real.
        m = _mutate(_mutate(ens_src, "'min_risk': 0.0, 'max_risk': 0.25", "'min_risk': 0.0, 'max_risk': 0.30"),
                    "'min_risk': 0.25, 'max_risk': 0.50", "'min_risk': 0.30, 'max_risk': 0.50")
        got = set(lock_c1_combine(corpus, _load_module_from_source("ensemble", m, td)))
        exp = rows(lambda rv: 0.25 <= rv["combined_risk"] < 0.30 and rv["tier"] == 1) | set(base["c1"])
        check("M-A WATCH floor 0.25->0.30 names exactly the band", got, exp)

        # M-B: MIN_METHODS_FOR_OPERATIONAL 1 -> 2. RED exactly on every <=1-method row.
        m = _mutate(ens_src, "MIN_METHODS_FOR_OPERATIONAL = 1", "MIN_METHODS_FOR_OPERATIONAL = 2")
        got = set(lock_c1_combine(corpus, _load_module_from_source("ensemble", m, td)))
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

        # M-E: one append-era data.csv row's tier flipped in memory -> exactly one new C-4 identity.
        csv_text = _blob(repo, base["commit"], CSV_PATH).decode("utf-8")
        lines = csv_text.split("\n")
        k = next(i for i in range(len(lines) - 1, 0, -1) if lines[i].startswith(APPEND_ONLY_SINCE_DATE[:7]))
        cells = lines[k].split(",")
        cells[2] = "3" if cells[2] != "3" else "0"
        lines[k] = ",".join(cells)
        _, ents, _ = lock_c4_data_csv(corpus, "\n".join(lines))
        got = {(e["cls"], e["line"]) for e in ents}
        exp = {(e["cls"], e["line"]) for e in base["c4entries"]} | {("CSV_RECORD_DISAGREEMENT", k + 1)}
        check("M-E data.csv row edit -> one new CSV_RECORD_DISAGREEMENT identity at that line", got, exp)

        # M-F: the real append-only comparator on a synthetic history with one edited line + one shrink.
        seq = [("c1", b"h\na\n"), ("c2", b"h\na\nb\n"), ("c3", b"h\nX\nb\n"), ("c4", b"h\nX\n")]
        v = [(a, b, d.split(":")[0]) for a, b, d in _append_only_violations(seq)]
        if v == [("c2", "c3", "line 2 EDITED"), ("c3", "c4", "history SHRANK 3 -> 2")]:
            _ok("M-F append-only comparator names the edited line and the shrink", "2 of 3 steps")
        else:
            fails += 1
            _fail("M-F append-only comparator", f"{v}")

        # M-G (codex 2303Z #1): same-count / same-max-date SWAP of two CSV exceptions. Repair one
        # existing CSV_RECORD_DISAGREEMENT row to its earliest committed record and break one
        # previously-matching row on the SAME date. The (count, max date) summary is unchanged --
        # the old ledger would have accepted it -- but the exact identity set must differ.
        dis = [e for e in base["c4entries"] if e["cls"] == "CSV_RECORD_DISAGREEMENT"]
        target = dis[-1]
        lines = csv_text.split("\n")
        lines[target["line"] - 1] = ",".join(target["expected"])             # repair
        victim = next(i for i in range(1, len(lines)) if lines[i].startswith(target["date"] + ",")
                      and (i + 1) not in {e["line"] for e in dis})
        vc = lines[victim].split(","); vc[3] = "0.9999"; lines[victim] = ",".join(vc)   # break
        _, ents, _ = lock_c4_data_csv(corpus, "\n".join(lines))
        swapped = {"corpus": corpus, "c1": base["c1"], "c2a": base["c2a"], "c4entries": ents}
        swapped_exc = exception_set(swapped)
        same_summary = class_summary(swapped_exc) == class_summary(base["exceptions"])
        set_differs = {_identity(e) for e in swapped_exc} != {_identity(e) for e in base["exceptions"]}
        if same_summary and set_differs:
            _ok("M-G same-count/same-max swap of CSV exceptions goes RED under the exact set",
                f"summary unchanged ({len(dis)}, {target['date']}), identity set differs")
        else:
            fails += 1
            _fail("M-G swap KAT", f"same_summary={same_summary} set_differs={set_differs}")

        # M-H (codex 2303Z #2): (a) a parseable record blob made empty -> exclusion set gains one
        # identity; (b) a swap that empties one parseable blob AND makes one excluded blob parseable
        # preserves the exclusion COUNT but must still differ from the frozen set.
        victim_c = corpus.records[5][1]
        excluded_c = next(iter(corpus.unparseable))
        good = json.dumps(corpus.records[5][2]).encode("utf-8")
        real_read = lambda c: _blob(repo, c, RECORD_PATH)
        ca = Corpus(repo, base["commit"], blob_reader=lambda c: b"" if c == victim_c else real_read(c))
        cb = Corpus(repo, base["commit"], blob_reader=lambda c: b"" if c == victim_c else
                    (good if c == excluded_c else real_read(c)))
        frozen_ids = {json.dumps({"lock": "C-0", "cls": "UNPARSEABLE_RECORD", "commit": c, "digest": d},
                                 sort_keys=True) for c, d in UNPARSEABLE_EXCLUSIONS.items()}
        ids_a = {json.dumps({"lock": "C-0", "cls": "UNPARSEABLE_RECORD", "commit": c, "digest": d},
                            sort_keys=True) for c, d in ca.unparseable.items()}
        ids_b = {json.dumps({"lock": "C-0", "cls": "UNPARSEABLE_RECORD", "commit": c, "digest": d},
                            sort_keys=True) for c, d in cb.unparseable.items()}
        ok_a = len(ca.unparseable) == len(UNPARSEABLE_EXCLUSIONS) + 1 and ids_a != frozen_ids
        ok_b = len(cb.unparseable) == len(UNPARSEABLE_EXCLUSIONS) and ids_b != frozen_ids
        if ok_a and ok_b:
            _ok("M-H unparseable-blob KATs: added exclusion RED; count-preserving swap RED",
                f"{victim_c[:10]} emptied; {excluded_c[:10]} made parseable")
        else:
            fails += 1
            _fail("M-H unparseable-blob KATs", f"added={ok_a} swap={ok_b}")

        # M-I (codex 0003Z #2): SAME-KEY content change. One C-2a exception keeps (lock, cls, date,
        # region, commit) and has one character of its classified content changed. Every non-digest
        # key of the measured set must be unchanged, yet the identity set must differ by exactly one
        # identity each way (RED under set equality).
        k2, (cls2, diffs2) = next(iter(base["c2a"].items()))
        alt_c2a = dict(base["c2a"])
        alt_c2a[k2] = (cls2, [diffs2[0] + "X"] + list(diffs2[1:]))
        alt = exception_set({"corpus": corpus, "c1": base["c1"], "c2a": alt_c2a, "c4entries": base["c4entries"]})
        strip = lambda e: {k: v for k, v in e.items() if k != "digest"}
        same_keys = sorted(_identity(strip(e)) for e in alt) == sorted(_identity(strip(e)) for e in base["exceptions"])
        base_ids, alt_ids = {_identity(e) for e in base["exceptions"]}, {_identity(e) for e in alt}
        full_hex = all(re.fullmatch(r"[0-9a-f]{64}", e["digest"]) for e in alt)
        if same_keys and len(alt_ids - base_ids) == 1 and len(base_ids - alt_ids) == 1 and full_hex:
            _ok("M-I same-key content change flips exactly one full 64-hex identity",
                f"{k2[0]} {k2[1]} @{k2[2][:10]} ({cls2})")
        else:
            fails += 1
            _fail("M-I same-key content change", f"same_keys={same_keys} +{len(alt_ids - base_ids)} "
                                                 f"-{len(base_ids - alt_ids)} full_hex={full_hex}")

        # M-J (codex 0003Z #2): FRAMING. Values containing newline / '|' / ',' : REV 2's joined-text
        # preimages COLLIDE (proved here), the canonical structured JSON digests SEPARATE.
        old_c1 = ("\n".join(["x\ny", "z"]), "\n".join(["x", "y\nz"]))
        old_c4 = ("a|b" + "|" + "c", "a" + "|" + "b|c")
        old_cells = (",".join(["a,b", "c"]), ",".join(["a", "b,c"]))
        new_c1 = (_digest_json({"diffs": ["x\ny", "z"]}), _digest_json({"diffs": ["x", "y\nz"]}))
        new_c4 = (_digest_json({"actual": ["a|b"], "expected": ["c"]}),
                  _digest_json({"actual": ["a"], "expected": ["b|c"]}))
        new_cells = (_digest_json({"actual": ["a,b", "c"], "expected": []}),
                     _digest_json({"actual": ["a", "b,c"], "expected": []}))
        collide = old_c1[0] == old_c1[1] and old_c4[0] == old_c4[1] and old_cells[0] == old_cells[1]
        separate = new_c1[0] != new_c1[1] and new_c4[0] != new_c4[1] and new_cells[0] != new_cells[1]
        if collide and separate:
            _ok("M-J framing: joined-text preimages collide (newline, '|', ','); canonical JSON separates",
                "3 of 3 pairs each way")
        else:
            fails += 1
            _fail("M-J framing", f"old_collide={collide} new_separate={separate}")

        # M-K (codex 0003Z #1): SOURCE BINDING, as REAL child processes. (a) an external byte-modified
        # copy of this bar (with a matching sidecar beside it) run against the pristine tree; then in a
        # throwaway DETACHED worktree at bar_commit: (d) the untouched committed positive BINDS,
        # (b) an edited in-tree bar and (c) an edited sidecar are each REFUSED with the typed code,
        # exit 3, before any lock line.
        bar_commit = BOUND["bar_commit"]

        def child(bar_path: str, repo_arg: str) -> Tuple[int, str]:
            r = subprocess.run([sys.executable, bar_path, "--repo", repo_arg, "--commit", base["commit"],
                                "--bind-only"], capture_output=True, text=True, cwd=os.path.dirname(bar_path))
            return r.returncode, r.stdout + r.stderr

        def refused(code: int, out: str, typed: str) -> bool:
            return code == 3 and typed in out and "[PASS]" not in out and "[FAIL]" not in out

        ext = os.path.join(td, "external_copy")
        os.makedirs(ext)
        with open(__file__, "rb") as f:
            bar_bytes = f.read()
        ext_bar = os.path.join(ext, BAR_FILE)
        with open(ext_bar, "wb") as f:
            f.write(bar_bytes + b"\n# externally modified\n")
        with open(os.path.join(HERE, LEDGER_FILE), "rb") as f:
            ledger_bytes = f.read()
        with open(os.path.join(ext, LEDGER_FILE), "wb") as f:
            f.write(ledger_bytes)
        code_a, out_a = child(ext_bar, repo)
        ok_ka = refused(code_a, out_a, "CORPUS_BAR_NOT_IN_TREE")

        wt = os.path.join(td, "wt_bind")
        _git(repo, ["worktree", "add", "--detach", wt, bar_commit])
        try:
            wt_bar = os.path.join(wt, BAR_PATH.replace("/", os.sep))
            wt_ledger = os.path.join(wt, LEDGER_PATH.replace("/", os.sep))
            code_d, out_d = child(wt_bar, wt)
            ok_kd = code_d == 0 and "SOURCE BINDING" in out_d and "BIND-ONLY" in out_d \
                and f"bar_commit {bar_commit}" in out_d and "[PASS]" not in out_d
            with open(wt_bar, "ab") as f:
                f.write(b"\n# edited in tree\n")
            code_b, out_b = child(wt_bar, wt)
            ok_kb = refused(code_b, out_b, "CORPUS_BAR_DIVERGENT")
            _git(wt, ["checkout", "--", BAR_PATH])
            with open(wt_ledger, "ab") as f:
                f.write(b"\n")
            code_c, out_c = child(wt_bar, wt)
            ok_kc = refused(code_c, out_c, "CORPUS_LEDGER_DIVERGENT")
        finally:
            _git(repo, ["worktree", "remove", "--force", wt])
        if ok_ka and ok_kb and ok_kc and ok_kd:
            _ok("M-K source binding: external copy / edited in-tree bar / edited sidecar REFUSED (exit 3, "
                "typed, no lock line); untouched committed positive BINDS", f"bar_commit {bar_commit[:12]}")
        else:
            fails += 1
            _fail("M-K source binding", f"external={ok_ka}({code_a}) in-tree={ok_kb}({code_b}) "
                                        f"sidecar={ok_kc}({code_c}) positive={ok_kd}({code_d}); "
                                        f"tail: {out_d.strip().splitlines()[-1:] if out_d.strip() else out_d!r}")

    print()
    print("DAILY-TIER-CORPUS SELFTEST: " + ("ALL PASS" if not fails else f"{fails} FAIL"))
    return 1 if fails else 0


# --------------------------------------------------------------------------- main
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", default="HEAD", help="target_commit: the production history under audit")
    ap.add_argument("--bar-commit", default="HEAD",
                    help="bar_commit: the commit the EXECUTING bar + sidecar must be byte-bound to "
                         "(default: the worktree's exact HEAD; override only explicitly)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bind-only", action="store_true", help="perform source binding, print it, emit no verdict")
    ap.add_argument("--dump-ledger", action="store_true", help="print the measured class summary + exclusions")
    ap.add_argument("--write-ledger", default=None, help="write the measured exception set to this path (UNBOUND)")
    a = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)
    repo = os.path.abspath(a.repo)
    try:
        target = _git(repo, ["rev-parse", f"{a.commit}^{{commit}}"])
    except RuntimeError as e:
        print(f"CORPUS_REVISION_UNRESOLVABLE: {e}")
        return 2
    print(f"DAILY-TIER-CORPUS red-KAT bar (grassmann, REV 3) -- repo {repo} commit {a.commit}")
    if a.write_ledger:
        print("  UNBOUND: --write-ledger authors the sidecar from the measured set; it emits NO verdict")
    else:
        try:
            bound = bind_sources(repo, a.bar_commit)
        except RuntimeError as e:
            print(f"CORPUS_REVISION_UNRESOLVABLE: bar_commit {a.bar_commit}: {e}")
            return 2
        except SystemExit as e:
            print(str(e))
            print("SOURCE BINDING REFUSED -- no corpus verdict")
            return 3
        print_binding(bound, target)
        if a.bind_only:
            print("BIND-ONLY: sources bound; no corpus verdict emitted")
            return 0
    if a.selftest:
        return selftest(repo, a.commit)
    res = run_bar(repo, a.commit)
    if a.write_ledger:
        with open(a.write_ledger, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res["exceptions"], f, indent=0, sort_keys=True)
            f.write("\n")
        with open(a.write_ledger, "rb") as f:
            print(f"wrote {a.write_ledger}: {len(res['exceptions'])} identities, sha256 {_lf_sha(f.read())}")
        return 0
    if a.dump_ledger:
        for k, v in sorted(class_summary(res["exceptions"]).items()):
            print(k, v)
        print("exclusions", {c: d for c, d in res["corpus"].unparseable.items()})
        for e in res["exceptions"]:
            if e["cls"] == "UNEXPLAINED":
                print("   UNEXPLAINED", e)
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
