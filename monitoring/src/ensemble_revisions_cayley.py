#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMMUTABLE REVISION STORE for the daily ensemble record (cayley) --
corrected contract v1, recut for codex's four 1755Z findings.

Owner: asylum 2026-09-02 "use immutable revision" / "land B6".
Contract: grassmann 1432Z (C-7..C-11 layout) as CORRECTED by codex 1433Z
(five corrections) and RECUT per codex 1755Z (four findings):

  F1 the cutover capsule is RE-DERIVED from git at its declared
     cutover_commit and never self-attested: validate_legacy_baseline()
  F2 the legacy CSV prefix is the exact GIT BLOB (LF), reopened through
     git, never checkout-translated bytes; derived files are written LF
  F3 the inputs capsule has per-kind non-null/type rules and mandatory
     cardinalities, cross-checked against the persistence pins, the
     scored day, the registered code paths and the committed calibration
     set: validate_inputs_capsule(cap, expect=...)
  F4 a revision is identity-linked to its journal line, not only
     digest-linked: validate_revision_against_entry(record, entry)

LAYOUT
  docs/ensemble/<YYYY-MM-DD>/<run_id>.json   one immutable REVISION per
                                            run, CREATE-ONCE (O_EXCL)
  docs/ensemble/index.ndjson                APPEND-ONLY journal: one
                                            canonical JSON object per
                                            line; a newer journal carries
                                            every older one as an exact
                                            byte prefix (C-8)
  docs/ensemble/legacy_baseline_v1.json     the CUTOVER CAPSULE: created
                                            once by the owner-run
                                            --cutover, never amended,
                                            re-derivable from git
  docs/ensemble_latest.json                 DERIVED: byte copy of the
                                            current revision of the max
                                            date (C-10)
  docs/data.csv                             DERIVED by ONE writer: the
                                            bound legacy GIT BLOB + the
                                            CURRENT revision's rows for
                                            every journaled date, LF

RULES
  * run_id = fired_utc as YYYYMMDDTHHMMSSffffffZ + "-" + 8 hex of uuid4;
    there is NO `current` field anywhere -- current is the LAST valid
    journal event for the date; a re-score's `supersedes` must equal
    that exact run.
  * every revision records `source_index = {entry_count, prefix_sha256}`
    of the exact journal bytes persistence resolved against;
    `persistence_inputs` entries are the closed union revision | legacy |
    hole, each resolved against that prefix plus the legacy capsule.
  * a run stages all surfaces under docs/ensemble/.txn/<run_id>/,
    validates them, then publishes; the operator commits everything in
    ONE commit. A dirty transaction, an orphan revision or a dangling
    journal line makes the NEXT run REFUSE with a typed recovery
    instruction.
  * B6 / C-11: `scored_day_utc == date`, derived from the UTC clock.

Nothing here scores anything; nothing here rewrites history.
"""
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timedelta, timezone

REVISION_SCHEMA = "geospec-ensemble-revision-v1"
JOURNAL_ENTRY_SCHEMA = "geospec-ensemble-journal-entry-v1"
LEGACY_SCHEMA = "geospec-ensemble-legacy-baseline-v1"
INPUTS_SCHEMA = "geospec-ensemble-inputs-capsule-v1"
REV_DIR_REL = "docs/ensemble"
JOURNAL_REL = REV_DIR_REL + "/index.ndjson"
LEGACY_REL = REV_DIR_REL + "/legacy_baseline_v1.json"
TXN_DIR_REL = REV_DIR_REL + "/.txn"
LATEST_REL = "docs/ensemble_latest.json"
CSV_REL = "docs/data.csv"
DASHBOARD_CSV_REL = "monitoring/dashboard/data.csv"
CSV_HEADER = ["date", "region", "tier", "risk", "confidence", "methods",
              "agreement"]
CODE_PATHS = ("monitoring/src/run_ensemble_daily.py",
              "monitoring/src/ensemble.py",
              "monitoring/src/ensemble_revisions_cayley.py")
CALIBRATION_DIR_REL = "monitoring/data/calibration"
RESOLUTION_RULE = ("for a legacy date, the record consumed is the LAST "
                   "committed parseable blob of that date in first-parent "
                   "order (the record as last published); the frozen CSV "
                   "prefix is the exact committed blob, copied byte-for-byte, "
                   "never regenerated and never read from a checkout")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_ID_RE = re.compile(r"^(\d{8}T\d{12}Z)-[0-9a-f]{8}$")
_FIRED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
JOURNAL_FIELDS = {"schema", "date", "run_id", "path", "sha256", "supersedes",
                  "reason", "appended_utc"}
REVISION_FIELDS = {"schema", "date", "run_id", "fired_utc", "scored_day_utc",
                   "supersedes", "reason", "inputs", "inputs_sha256",
                   "source_index", "persistence_inputs"}
LEGACY_FIELDS = {"schema", "created_utc", "cutover_commit", "record_path",
                 "records", "legacy_csv", "resolution_rule"}
LEGACY_RECORD_FIELDS = {"date", "record_path", "commit", "git_blob", "sha256",
                        "byte_length", "parseable"}
LEGACY_CSV_FIELDS = {"path", "header", "row_count", "prefix_sha256",
                     "csv_row_sha256_by_date", "git_blob", "byte_length"}
INPUT_ENTRY_FIELDS = {"kind", "identity", "data_day", "keyset", "byte_length",
                      "sha256"}
INPUT_KINDS = {"code", "calibration_capsule", "prior_revision",
               "legacy_record", "scored_day"}
PERSISTENCE_KINDS = {"revision", "legacy", "hole"}
PIN_FIELDS = {"date", "kind", "run_id", "sha256", "legacy"}
PIN_LEGACY_FIELDS = {"capsule", "capsule_sha256", "record_path", "git_blob"}


class RevisionRefusal(ValueError):
    """Typed refusal; the code leads the message."""


# ------------------------------------------------------------ helpers --
def utc_now():
    return datetime.now(timezone.utc)


def iso_z(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fired_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run_id_for(fired):
    """microsecond instant + 8 hex of uuid4 (codex correction 2)."""
    return fired.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]


def canonical_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _p(repo, rel):
    return os.path.join(repo, rel.replace("/", os.sep))


def _read(path):
    with io.open(path, "rb") as f:
        return f.read()


def _write_atomic(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _create_once(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RevisionRefusal(
            f"REVISION_PATH_EXISTS: {path} already exists -- revisions are "
            "create-once")
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def record_bytes(record):
    """Committed form: sorted keys, 2-space indent, LF, trailing newline."""
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _git(repo, *a):
    return subprocess.check_output(["git", "-C", repo] + list(a))


# ------------------------------------------------------------ journal --
def journal_bytes(repo):
    p = _p(repo, JOURNAL_REL)
    return _read(p) if os.path.exists(p) else b""


def parse_journal(raw):
    """Parse and validate the NDJSON journal (C-8)."""
    if raw == b"":
        return []
    if not raw.endswith(b"\n"):
        raise RevisionRefusal("JOURNAL_TRUNCATED: index.ndjson does not end "
                              "with a newline -- an unterminated line is a "
                              "partial transaction")
    entries = []
    seen_ids = set()
    current = {}
    for i, line in enumerate(raw.split(b"\n")[:-1], 1):
        if line.strip() == b"":
            raise RevisionRefusal(f"JOURNAL_BLANK_LINE: index.ndjson line {i}")
        try:
            e = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise RevisionRefusal(f"JOURNAL_UNPARSABLE: index.ndjson line {i}")
        if canonical_bytes(e) != line + b"\n":
            raise RevisionRefusal(
                f"JOURNAL_NONCANONICAL: index.ndjson line {i} is not the "
                "canonical encoding of its object")
        _validate_journal_entry(e, i)
        if e["run_id"] in seen_ids:
            raise RevisionRefusal(
                f"JOURNAL_DUPLICATE_RUN_ID: {e['run_id']} at line {i}")
        seen_ids.add(e["run_id"])
        cur = current.get(e["date"])
        if cur is None:
            if e["supersedes"] is not None:
                raise RevisionRefusal(
                    f"JOURNAL_SUPERSEDES_WITHOUT_PRIOR: line {i} {e['date']} "
                    f"names {e['supersedes']} but the date has no prior event")
        elif e["supersedes"] != cur["run_id"]:
            raise RevisionRefusal(
                f"JOURNAL_STALE_SUPERSEDES: line {i} {e['date']} names "
                f"{e['supersedes']!r}; the current run at that point was "
                f"{cur['run_id']}")
        current[e["date"]] = e
        entries.append(e)
    return entries


def _validate_journal_entry(e, i):
    if not isinstance(e, dict) or set(e) != JOURNAL_FIELDS:
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} field set not closed")
    if e["schema"] != JOURNAL_ENTRY_SCHEMA:
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} schema")
    if not (isinstance(e["date"], str) and _DATE_RE.match(e["date"])):
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} date")
    if not (isinstance(e["run_id"], str) and _RUN_ID_RE.match(e["run_id"])):
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} run_id "
                              f"{e['run_id']!r} is not YYYYMMDDTHHMMSSffffffZ-hex8")
    if e["path"] != f"{REV_DIR_REL}/{e['date']}/{e['run_id']}.json":
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} path is not the "
                              "registered layout")
    if not (isinstance(e["sha256"], str) and _HEX64_RE.match(e["sha256"])):
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} sha256")
    if e["supersedes"] is not None and not (
            isinstance(e["supersedes"], str) and _RUN_ID_RE.match(e["supersedes"])):
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} supersedes")
    if e["supersedes"] is None and e["reason"] is not None:
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} reason without "
                              "supersedes")
    if e["supersedes"] is not None and not (
            isinstance(e["reason"], str) and e["reason"].strip()):
        raise RevisionRefusal(f"JOURNAL_RESCORE_WITHOUT_REASON: line {i}")
    if not isinstance(e["appended_utc"], str):
        raise RevisionRefusal(f"JOURNAL_SCHEMA: line {i} appended_utc")


def current_map(entries):
    cur = {}
    for e in entries:
        cur[e["date"]] = e
    return cur


def journal_prefix_ok(older, newer):
    """C-8 comparator: every older journal is an exact byte prefix of
    every newer one."""
    return newer[:len(older)] == older


def source_index_of(raw):
    return {"entry_count": len(parse_journal(raw)),
            "prefix_sha256": sha256_bytes(raw)}


# ------------------------------------------------------- legacy capsule --
def _derive_records(repo, cutover_commit, record_rel, git):
    """F1: the exact first-parent record vector of `record_rel` reachable
    from `cutover_commit`, one entry per DISTINCT blob, newest first."""
    log = git(repo, "log", "--first-parent", "--format=%H", cutover_commit,
              "--", record_rel).decode().split()
    seen = set()
    records = []
    for c in log:
        try:
            blob = git(repo, "rev-parse", f"{c}:{record_rel}").decode().strip()
        except subprocess.CalledProcessError:
            continue
        if blob in seen:
            continue
        seen.add(blob)
        raw = git(repo, "cat-file", "blob", blob)
        try:
            rec = json.loads(raw.decode("utf-8"))
            d = rec.get("date") if isinstance(rec, dict) else None
        except (ValueError, UnicodeDecodeError):
            d = None
        ok = isinstance(d, str) and bool(_DATE_RE.match(d))
        records.append({"date": d if ok else None,
                        "record_path": record_rel, "commit": c,
                        "git_blob": blob, "sha256": sha256_bytes(raw),
                        "byte_length": len(raw), "parseable": ok})
    return records


def _csv_meta(csv_raw):
    rows = list(csv.reader(io.StringIO(csv_raw.decode("utf-8"), newline="")))
    header, body = (rows[0], rows[1:]) if rows else (list(CSV_HEADER), [])
    per_date = {}
    for r in body:
        if r:
            per_date.setdefault(r[0], []).append(",".join(r))
    return header, len(body), {d: sha256_bytes(("\n".join(v) + "\n").encode())
                               for d, v in sorted(per_date.items())}


def build_legacy_baseline(repo, record_rel=LATEST_REL, csv_rel=CSV_REL,
                          git=_git):
    """Derive the cutover capsule from COMMITTED bytes at HEAD: every
    committed pre-cutover record (F1) and the legacy CSV as the exact GIT
    BLOB (F2). Pure derivation; no choice is made here."""
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    records = _derive_records(repo, head, record_rel, git)
    csv_blob = git(repo, "rev-parse", f"{head}:{csv_rel}").decode().strip()
    csv_raw = git(repo, "cat-file", "blob", csv_blob)
    header, n, per_date = _csv_meta(csv_raw)
    return {"schema": LEGACY_SCHEMA,
            "created_utc": iso_z(utc_now()),
            "cutover_commit": head,
            "record_path": record_rel,
            "records": records,
            "legacy_csv": {"path": csv_rel, "header": header,
                           "row_count": n,
                           "prefix_sha256": sha256_bytes(csv_raw),
                           "csv_row_sha256_by_date": per_date,
                           "git_blob": csv_blob,
                           "byte_length": len(csv_raw)},
            "resolution_rule": RESOLUTION_RULE}


def validate_legacy_baseline(repo, cap, git=_git):
    """F1 (shared with REV 6): the capsule must be exactly what git says at
    its declared cutover_commit -- closed schema, fixed paths/rule, the exact
    first-parent record vector with every blob reachable at commit:path and
    matching bytes/date/length/parseability, the CSV blob/header/row-count/
    per-date digests, and the capsule-add commit's parent (or HEAD, before
    the cutover commit exists) equal to cutover_commit. Refuses typed
    LEGACY_CAPSULE_NOT_DERIVED_FROM_CUTOVER: <what>."""
    def refuse(what):
        raise RevisionRefusal(f"LEGACY_CAPSULE_NOT_DERIVED_FROM_CUTOVER: {what}")
    if not isinstance(cap, dict) or set(cap) - {"_sha256"} != LEGACY_FIELDS:
        refuse("capsule field set not closed")
    if cap["schema"] != LEGACY_SCHEMA:
        refuse("schema")
    if not isinstance(cap["created_utc"], str):
        refuse("created_utc")
    cc = cap["cutover_commit"]
    if not (isinstance(cc, str) and _HEX40_RE.match(cc)):
        refuse("cutover_commit is not a 40-hex commit")
    if cap["record_path"] != LATEST_REL:
        refuse(f"record_path {cap['record_path']!r} is not {LATEST_REL}")
    if cap["resolution_rule"] != RESOLUTION_RULE:
        refuse("resolution_rule is not the registered rule")
    lc = cap["legacy_csv"]
    if not isinstance(lc, dict) or set(lc) != LEGACY_CSV_FIELDS:
        refuse("legacy_csv field set not closed")
    if lc["path"] != CSV_REL:
        refuse(f"legacy_csv.path {lc['path']!r} is not {CSV_REL}")
    if not isinstance(cap["records"], list):
        refuse("records is not a list")
    for i, r in enumerate(cap["records"]):
        if not isinstance(r, dict) or set(r) != LEGACY_RECORD_FIELDS:
            refuse(f"record {i} field set not closed")
    # --- authority: the declared cutover commit must be reachable and, if
    # the capsule is committed, be the parent of the commit that added it
    try:
        git(repo, "cat-file", "-e", f"{cc}^{{commit}}")
    except subprocess.CalledProcessError:
        refuse(f"cutover_commit {cc[:12]} is not a reachable commit")
    try:
        add = git(repo, "log", "--first-parent", "--diff-filter=A",
                  "--format=%H %P", "--", LEGACY_REL).decode().split("\n")
        add = [ln for ln in add if ln.strip()]
    except subprocess.CalledProcessError:
        add = []
    if add:
        parts = add[-1].split()          # the ORIGINAL add (oldest)
        parent = parts[1] if len(parts) > 1 else ""
        if parent != cc:
            refuse(f"the capsule-add commit's parent {parent[:12]} is not "
                   f"the declared cutover_commit {cc[:12]}")
    else:
        head = git(repo, "rev-parse", "HEAD").decode().strip()
        if head != cc:
            refuse(f"uncommitted capsule declares cutover {cc[:12]} but HEAD "
                   f"is {head[:12]}")
    # --- exact record vector re-derived from git
    expected = _derive_records(repo, cc, cap["record_path"], git)
    if cap["records"] != expected:
        n_e, n_c = len(expected), len(cap["records"])
        first = next((i for i, (a, b) in enumerate(zip(expected, cap["records"]))
                      if a != b), min(n_e, n_c))
        refuse(f"record vector diverges from first-parent history at "
               f"{cc[:12]} (expected {n_e} records, capsule has {n_c}; first "
               f"divergence at index {first})")
    # --- CSV authority
    try:
        blob = git(repo, "rev-parse", f"{cc}:{CSV_REL}").decode().strip()
    except subprocess.CalledProcessError:
        refuse(f"{CSV_REL} is not committed at {cc[:12]}")
    if lc["git_blob"] != blob:
        refuse(f"legacy_csv.git_blob {str(lc['git_blob'])[:12]} is not the "
               f"blob at {cc[:12]} ({blob[:12]})")
    csv_raw = git(repo, "cat-file", "blob", blob)
    if lc["prefix_sha256"] != sha256_bytes(csv_raw) or \
            lc["byte_length"] != len(csv_raw):
        refuse("legacy_csv digest/length do not match the committed blob")
    header, n, per_date = _csv_meta(csv_raw)
    if lc["header"] != header or lc["row_count"] != n or \
            lc["csv_row_sha256_by_date"] != per_date:
        refuse("legacy_csv header/row_count/per-date digests do not recompute "
               "from the committed blob")
    if header != CSV_HEADER:
        refuse("committed legacy CSV header is not the registered header")
    return True


def write_legacy_baseline(repo, capsule, git=_git):
    validate_legacy_baseline(repo, capsule, git=git)
    path = _p(repo, LEGACY_REL)
    _create_once(path, record_bytes(capsule))
    return path


def load_legacy_baseline(repo, git=_git):
    """The capsule, VALIDATED against git (F1), or None if absent."""
    path = _p(repo, LEGACY_REL)
    if not os.path.exists(path):
        return None
    raw = _read(path)
    try:
        cap = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise RevisionRefusal("LEGACY_CAPSULE_UNREADABLE")
    validate_legacy_baseline(repo, cap, git=git)
    cap["_sha256"] = sha256_bytes(raw)
    return cap


def legacy_record_for(cap, date_str):
    """The legacy record consumed for a pre-cutover date: the LAST committed
    parseable blob of that date in first-parent order (records are listed
    newest-first, so the first match)."""
    if cap is None:
        return None
    for r in cap["records"]:
        if r["parseable"] and r["date"] == date_str:
            return r
    return None


def reopen_legacy_record(repo, cap, rec, git=_git):
    raw = git(repo, "cat-file", "blob", rec["git_blob"])
    if sha256_bytes(raw) != rec["sha256"] or len(raw) != rec["byte_length"]:
        raise RevisionRefusal(
            f"LEGACY_RECORD_DIGEST_MISMATCH: {rec['git_blob'][:12]}")
    return json.loads(raw.decode("utf-8"))


# ------------------------------------------------------ inputs capsule --
def input_entry(kind, identity, data_day, keyset, raw_bytes=None,
                sha256=None, byte_length=None):
    if kind not in INPUT_KINDS:
        raise RevisionRefusal(f"INPUTS_CAPSULE_SCHEMA: kind {kind!r}")
    if raw_bytes is not None:
        sha256, byte_length = sha256_bytes(raw_bytes), len(raw_bytes)
    return {"kind": kind, "identity": identity, "data_day": data_day,
            "keyset": sorted(keyset) if keyset is not None else None,
            "byte_length": byte_length, "sha256": sha256}


def scored_day_preimage(date_str):
    """F3: the canonical preimage of the scored-day input, so the entry
    has a real length and digest."""
    return canonical_bytes({"kind": "scored_day", "date": date_str})


def scored_day_entry(date_str):
    return input_entry("scored_day", date_str, date_str, None,
                       raw_bytes=scored_day_preimage(date_str))


def _refuse_inputs(what):
    raise RevisionRefusal(f"INPUTS_CAPSULE_SCHEMA: {what}")


def validate_inputs_capsule(cap, expect=None):
    """F3: closed shape AND per-kind semantics AND cardinalities.

    Per kind (identity / data_day / keyset / byte_length / sha256):
      code                 repo-relative .py path / None / None / int>0 / hex64
      calibration_capsule  repo-relative .json path / None / non-empty list / int>0 / hex64
      prior_revision       run_id / date / None / int>0 / hex64
      legacy_record        40-hex git blob / date / None / int>0 / hex64
      scored_day           date == data_day / date / None / len(preimage) / sha256(preimage)
    Mandatory cardinalities: exactly the three CODE_PATHS in order, at
    least one entry (an empty capsule is meaningless), exactly one
    scored_day, no duplicate identities within a kind.
    `expect` (production and REV 6 cross-check) may carry
      calibration_paths: the complete committed calibration set (exact set)
      pins: the persistence_inputs (one prior_revision / legacy_record per
            non-hole pin with matching identity, day and sha256)
      scored_day: the date being published
    """
    if not isinstance(cap, dict) or set(cap) != {"schema", "entries"}:
        _refuse_inputs("not the closed shape")
    if cap["schema"] != INPUTS_SCHEMA or not isinstance(cap["entries"], list):
        _refuse_inputs("schema/entries")
    ents = cap["entries"]
    if not ents:
        _refuse_inputs("empty entries -- an inputs capsule must name inputs")
    by_kind = {k: [] for k in INPUT_KINDS}
    for i, e in enumerate(ents):
        if not isinstance(e, dict) or set(e) != INPUT_ENTRY_FIELDS:
            _refuse_inputs(f"entry {i} fields")
        k = e["kind"]
        if k not in INPUT_KINDS:
            _refuse_inputs(f"entry {i} kind")
        ident, day, keys, n, h = (e["identity"], e["data_day"], e["keyset"],
                                  e["byte_length"], e["sha256"])
        if not (isinstance(h, str) and _HEX64_RE.match(h)):
            _refuse_inputs(f"entry {i} ({k}) sha256 must be hex64")
        if not (type(n) is int and n > 0):
            _refuse_inputs(f"entry {i} ({k}) byte_length must be a positive int")
        if k == "code":
            if not (isinstance(ident, str) and ident.endswith(".py")
                    and "/" in ident):
                _refuse_inputs(f"entry {i} code identity must be a repo path")
            if day is not None or keys is not None:
                _refuse_inputs(f"entry {i} code carries a day/keyset")
        elif k == "calibration_capsule":
            if not (isinstance(ident, str) and ident.endswith(".json")
                    and ident.startswith(CALIBRATION_DIR_REL + "/")):
                _refuse_inputs(f"entry {i} calibration identity must be a "
                               f"{CALIBRATION_DIR_REL}/*.json path")
            if day is not None:
                _refuse_inputs(f"entry {i} calibration carries a day")
            if not (isinstance(keys, list) and keys
                    and all(isinstance(x, str) for x in keys)
                    and keys == sorted(keys)):
                _refuse_inputs(f"entry {i} calibration keyset must be a "
                               "non-empty sorted list")
        elif k == "prior_revision":
            if not (isinstance(ident, str) and _RUN_ID_RE.match(ident)):
                _refuse_inputs(f"entry {i} prior_revision identity must be a "
                               "run_id")
            if not (isinstance(day, str) and _DATE_RE.match(day)):
                _refuse_inputs(f"entry {i} prior_revision data_day")
            if keys is not None:
                _refuse_inputs(f"entry {i} prior_revision carries a keyset")
        elif k == "legacy_record":
            if not (isinstance(ident, str) and _HEX40_RE.match(ident)):
                _refuse_inputs(f"entry {i} legacy_record identity must be a "
                               "40-hex git blob")
            if not (isinstance(day, str) and _DATE_RE.match(day)):
                _refuse_inputs(f"entry {i} legacy_record data_day")
            if keys is not None:
                _refuse_inputs(f"entry {i} legacy_record carries a keyset")
        else:  # scored_day
            if not (isinstance(ident, str) and _DATE_RE.match(ident)
                    and day == ident):
                _refuse_inputs(f"entry {i} scored_day identity must equal its "
                               "data_day and be a date")
            if keys is not None:
                _refuse_inputs(f"entry {i} scored_day carries a keyset")
            pre = scored_day_preimage(ident)
            if n != len(pre) or h != sha256_bytes(pre):
                _refuse_inputs(f"entry {i} scored_day length/digest are not "
                               "those of the canonical preimage")
        by_kind[k].append(e)
    for k, lst in by_kind.items():
        idents = [e["identity"] for e in lst]
        if len(set(idents)) != len(idents):
            _refuse_inputs(f"duplicate {k} identity")
    if [e["identity"] for e in by_kind["code"]] != list(CODE_PATHS):
        _refuse_inputs(f"code entries must be exactly {list(CODE_PATHS)} in "
                       f"order, got {[e['identity'] for e in by_kind['code']]}")
    if len(by_kind["scored_day"]) != 1:
        _refuse_inputs(f"exactly one scored_day entry, got "
                       f"{len(by_kind['scored_day'])}")
    if expect:
        if "scored_day" in expect and \
                by_kind["scored_day"][0]["identity"] != expect["scored_day"]:
            _refuse_inputs("scored_day entry is not the day being published")
        if "calibration_paths" in expect:
            want = sorted(expect["calibration_paths"])
            got = sorted(e["identity"] for e in by_kind["calibration_capsule"])
            if got != want:
                _refuse_inputs(f"calibration set {got} != committed set {want}")
        if "pins" in expect:
            want = {}
            for pin in expect["pins"]:
                if pin["kind"] == "revision":
                    want[("prior_revision", pin["run_id"])] = (pin["date"],
                                                               pin["sha256"])
                elif pin["kind"] == "legacy":
                    want[("legacy_record", pin["legacy"]["git_blob"])] = (
                        pin["date"], pin["sha256"])
            got = {(e["kind"], e["identity"]): (e["data_day"], e["sha256"])
                   for k in ("prior_revision", "legacy_record")
                   for e in by_kind[k]}
            if got != want:
                _refuse_inputs("prior_revision/legacy_record entries do not "
                               "match the persistence pins one-to-one "
                               f"(got {sorted(got)}, want {sorted(want)})")
        if "pin_byte_lengths" in expect:
            # the REAL length of every pinned byte string (reopened by the
            # publisher / the bar), so a wrong length refuses one-to-one too
            for k in ("prior_revision", "legacy_record"):
                for e in by_kind[k]:
                    want_len = expect["pin_byte_lengths"].get(e["identity"])
                    if want_len is not None and e["byte_length"] != want_len:
                        _refuse_inputs(f"{k} {e['identity'][:12]} byte_length "
                                       f"{e['byte_length']} != reopened "
                                       f"{want_len} (one-to-one)")
    return True


def inputs_sha256(cap, expect=None):
    validate_inputs_capsule(cap, expect=expect)
    return sha256_bytes(canonical_bytes(cap))


# ---------------------------------------------------- revision identity --
def _refuse_rev(what):
    raise RevisionRefusal(f"REVISION_IDENTITY: {what}")


def validate_persistence_inputs(pins):
    if not isinstance(pins, list):
        _refuse_rev("persistence_inputs is not a list")
    for i, pe in enumerate(pins):
        if not isinstance(pe, dict) or set(pe) != PIN_FIELDS:
            _refuse_rev(f"persistence entry {i} field set not closed")
        if not (isinstance(pe["date"], str) and _DATE_RE.match(pe["date"])):
            _refuse_rev(f"persistence entry {i} date")
        k = pe["kind"]
        if k == "revision":
            if not (isinstance(pe["run_id"], str) and _RUN_ID_RE.match(pe["run_id"])
                    and isinstance(pe["sha256"], str) and _HEX64_RE.match(pe["sha256"])
                    and pe["legacy"] is None):
                _refuse_rev(f"persistence entry {i} revision shape")
        elif k == "legacy":
            lg = pe["legacy"]
            if not (pe["run_id"] is None
                    and isinstance(pe["sha256"], str) and _HEX64_RE.match(pe["sha256"])
                    and isinstance(lg, dict) and set(lg) == PIN_LEGACY_FIELDS
                    and lg["capsule"] == LEGACY_REL
                    and isinstance(lg["capsule_sha256"], str)
                    and _HEX64_RE.match(lg["capsule_sha256"])
                    and lg["record_path"] == LATEST_REL
                    and isinstance(lg["git_blob"], str)
                    and _HEX40_RE.match(lg["git_blob"])):
                _refuse_rev(f"persistence entry {i} legacy shape")
        elif k == "hole":
            if not (pe["run_id"] is None and pe["sha256"] is None
                    and pe["legacy"] is None):
                _refuse_rev(f"persistence entry {i} hole must be all-null")
        else:
            _refuse_rev(f"persistence entry {i} kind {k!r}")
    return True


def validate_revision_against_entry(record, entry, expect_inputs=None):
    """F4 (shared with REV 6): the revision block must be identity-linked
    to its journal line -- schema, date, run_id, supersedes AND reason
    equal; canonical aware-UTC fired_utc whose compact form is the run-id
    time prefix; closed source_index; per-kind persistence entries; scored
    day == date; inputs capsule valid with inputs_sha256 recomputing."""
    if not isinstance(record, dict) or not isinstance(record.get("revision"),
                                                      dict):
        _refuse_rev("record carries no revision block")
    rv = record["revision"]
    if set(rv) != REVISION_FIELDS:
        _refuse_rev("revision field set not closed")
    if rv["schema"] != REVISION_SCHEMA:
        _refuse_rev(f"schema {rv['schema']!r}")
    for k in ("date", "run_id", "supersedes", "reason"):
        if rv[k] != entry[k]:
            _refuse_rev(f"revision.{k} {rv[k]!r} != journal {entry[k]!r}")
    if record.get("date") != rv["date"]:
        _refuse_rev("record date != revision date")
    if rv["scored_day_utc"] != rv["date"]:
        _refuse_rev("scored_day_utc != date")
    if not (isinstance(rv["fired_utc"], str) and _FIRED_RE.match(rv["fired_utc"])):
        _refuse_rev("fired_utc is not canonical YYYY-MM-DDTHH:MM:SS.ffffffZ")
    compact = rv["fired_utc"].replace("-", "").replace(":", "").replace(".", "")
    m = _RUN_ID_RE.match(rv["run_id"])
    if not m or m.group(1) != compact:
        _refuse_rev("run_id time prefix != fired_utc")
    si = rv["source_index"]
    if not (isinstance(si, dict) and set(si) == {"entry_count", "prefix_sha256"}
            and type(si["entry_count"]) is int and si["entry_count"] >= 0
            and isinstance(si["prefix_sha256"], str)
            and _HEX64_RE.match(si["prefix_sha256"])):
        _refuse_rev("source_index not closed/typed")
    validate_persistence_inputs(rv["persistence_inputs"])
    exp = dict(expect_inputs or {})
    exp.setdefault("pins", rv["persistence_inputs"])
    exp.setdefault("scored_day", rv["date"])
    if rv["inputs_sha256"] != inputs_sha256(rv["inputs"], expect=exp):
        _refuse_rev("inputs_sha256 does not recompute")
    return True


# ----------------------------------------------------------- revisions --
def reopen_revision(repo, entry):
    path = _p(repo, entry["path"])
    if not os.path.exists(path):
        raise RevisionRefusal(
            f"REVISION_MISSING: {entry['path']} is journaled but absent")
    raw = _read(path)
    if sha256_bytes(raw) != entry["sha256"]:
        raise RevisionRefusal(
            f"REVISION_DIGEST_MISMATCH: {entry['path']} does not hash to its "
            "journal line")
    try:
        rec = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise RevisionRefusal(f"REVISION_UNPARSABLE: {entry['path']}")
    validate_revision_against_entry(rec, entry)
    return rec, raw


def prior_days_view(repo, journal_raw, cap, target_date_str, days,
                    git=_git):
    """For days_back = 1..days, resolve the prior day against EXACTLY the
    given journal bytes plus the (validated) legacy capsule.
    Returns [(date, record|None, persistence_input_entry)]."""
    entries = parse_journal(journal_raw)
    cur = current_map(entries)
    y, m, d = (int(x) for x in target_date_str.split("-"))
    t0 = datetime(y, m, d)
    out = []
    for k in range(1, days + 1):
        ds = (t0 - timedelta(days=k)).strftime("%Y-%m-%d")
        e = cur.get(ds)
        if e is not None:
            rec, _raw = reopen_revision(repo, e)
            out.append((ds, rec, {"date": ds, "kind": "revision",
                                  "run_id": e["run_id"], "sha256": e["sha256"],
                                  "legacy": None}))
            continue
        lr = legacy_record_for(cap, ds)
        if lr is not None:
            rec = reopen_legacy_record(repo, cap, lr, git=git)
            out.append((ds, rec, {"date": ds, "kind": "legacy", "run_id": None,
                                  "sha256": lr["sha256"],
                                  "legacy": {"capsule": LEGACY_REL,
                                             "capsule_sha256": cap["_sha256"],
                                             "record_path": lr["record_path"],
                                             "git_blob": lr["git_blob"]}}))
            continue
        out.append((ds, None, {"date": ds, "kind": "hole", "run_id": None,
                               "sha256": None, "legacy": None}))
    return out


def pin_input_entry(repo, cap, pin, git=_git):
    """F3: the inputs entry for a non-hole persistence pin, with the real
    byte length of the named bytes."""
    if pin["kind"] == "revision":
        # any JOURNALED revision (a superseded one is still journaled; the
        # pin's currency is adjudicated by C-9 against the snapshot)
        e = next((x for x in parse_journal(journal_bytes(repo))
                  if x["run_id"] == pin["run_id"]), None)
        if e is None:
            raise RevisionRefusal(f"PIN_NOT_JOURNALED: {pin['run_id']}")
        raw = _read(_p(repo, e["path"]))
        return input_entry("prior_revision", pin["run_id"], pin["date"], None,
                           raw_bytes=raw)
    if pin["kind"] == "legacy":
        rec = next((r for r in cap["records"]
                    if r["git_blob"] == pin["legacy"]["git_blob"]), None)
        if rec is None:
            raise RevisionRefusal("PIN_NOT_IN_CAPSULE")
        return input_entry("legacy_record", rec["git_blob"], pin["date"], None,
                           sha256=rec["sha256"], byte_length=rec["byte_length"])
    raise RevisionRefusal("PIN_IS_A_HOLE")


# --------------------------------------------------------- transaction --
def check_store_clean(repo):
    """Refuse to start a run over a dirty transaction, an orphan revision,
    or a dangling journal line. Returns the parsed journal when clean."""
    txn = _p(repo, TXN_DIR_REL)
    if os.path.isdir(txn) and os.listdir(txn):
        raise RevisionRefusal(
            f"REVISION_TXN_DIRTY: {TXN_DIR_REL} holds an unfinished "
            f"transaction {sorted(os.listdir(txn))}. RECOVERY: inspect it; "
            "if its journal candidate is a byte-prefix extension of the "
            "committed journal and its revision hashes to its line, publish "
            "it by moving the staged surfaces into place and commit them "
            "together; otherwise delete the staging directory. Never start "
            "a new run around it.")
    raw = journal_bytes(repo)
    entries = parse_journal(raw)
    journaled = {e["path"] for e in entries}
    for e in entries:
        if not os.path.exists(_p(repo, e["path"])):
            raise RevisionRefusal(
                f"REVISION_DANGLING_JOURNAL_LINE: {e['path']} is journaled "
                "but absent. RECOVERY: restore the committed revision file; "
                "the journal is never edited.")
    base = _p(repo, REV_DIR_REL)
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if not _DATE_RE.match(d):
                continue
            for fn in sorted(os.listdir(os.path.join(base, d))):
                rel = f"{REV_DIR_REL}/{d}/{fn}"
                if fn.endswith(".json") and rel not in journaled:
                    raise RevisionRefusal(
                        f"REVISION_ORPHAN: {rel} exists with no journal line. "
                        "RECOVERY: if it is the revision of an interrupted "
                        "publish, append its journal line and regenerate the "
                        "derived surfaces in one commit; otherwise remove it. "
                        "Never start a new run around it.")
    return entries


def _csv_rows_for_record(rec):
    rows = []
    for region, r in rec["regions"].items():
        rows.append([rec["date"], region, str(int(r["tier"])),
                     f"{float(r['combined_risk']):.4f}",
                     f"{float(r['confidence']):.2f}",
                     str(int(r["methods_available"])),
                     r.get("agreement") or ""])
    return rows


def derive_csv_bytes(repo, cap, entries, git=_git):
    """F2: the legacy prefix is the exact committed GIT BLOB (LF) named by
    the capsule, reopened through git -- never checkout-translated bytes --
    plus the CURRENT revisions' rows (LF) for every journaled date."""
    if cap is None:
        raise RevisionRefusal("LEGACY_CAPSULE_ABSENT: the cutover capsule "
                              "must exist before the first revision")
    lc = cap["legacy_csv"]
    prefix = git(repo, "cat-file", "blob", lc["git_blob"])
    if sha256_bytes(prefix) != lc["prefix_sha256"] or \
            len(prefix) != lc["byte_length"]:
        raise RevisionRefusal(
            "CSV_LEGACY_BLOB_MISMATCH: the committed legacy CSV blob does not "
            "hash to the capsule")
    if b"\r" in prefix:
        raise RevisionRefusal("CSV_LEGACY_BLOB_NOT_LF: the committed legacy "
                              "CSV blob carries CR bytes")
    cur = current_map(entries)
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    for d in sorted(cur):
        rec, _raw = reopen_revision(repo, cur[d])
        for r in _csv_rows_for_record(rec):
            w.writerow(r)
    return prefix + out.getvalue().encode("utf-8")


def publish_revision(repo, record, inputs_capsule, journal_snapshot,
                     persistence_inputs, fired, rescore_reason=None,
                     expect_inputs=None, git=_git):
    """Stage, validate and publish ONE revision + its journal line + the
    derived surfaces. `journal_snapshot` must be the exact journal bytes
    the caller resolved persistence against. Returns the journal entry."""
    if not isinstance(record, dict) or not isinstance(record.get("date"), str) \
            or not _DATE_RE.match(record["date"]):
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: date")
    if not isinstance(record.get("regions"), dict) or not record["regions"]:
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: regions")
    date_str = record["date"]
    if rescore_reason is not None and not (isinstance(rescore_reason, str)
                                           and rescore_reason.strip()):
        raise RevisionRefusal("RESCORE_REASON_EMPTY")
    validate_persistence_inputs(persistence_inputs)
    check_store_clean(repo)
    cap = load_legacy_baseline(repo, git=git)
    if cap is None:
        raise RevisionRefusal("LEGACY_CAPSULE_ABSENT: run the cutover first")
    exp = dict(expect_inputs or {})
    exp["pins"] = persistence_inputs
    exp["scored_day"] = date_str
    exp["pin_byte_lengths"] = {
        pe["identity"]: pe["byte_length"]
        for pe in (pin_input_entry(repo, cap, pin, git=git)
                   for pin in persistence_inputs if pin["kind"] != "hole")}
    inputs_digest = inputs_sha256(inputs_capsule, expect=exp)   # F3, before any write
    live = journal_bytes(repo)
    if live != journal_snapshot:
        raise RevisionRefusal(
            "JOURNAL_MOVED: index.ndjson changed after persistence was "
            "resolved; the run must be repeated from the current journal")
    entries = parse_journal(live)
    cur = current_map(entries).get(date_str)
    if cur is not None and rescore_reason is None:
        raise RevisionRefusal(
            f"REVISION_EXISTS: {date_str} already has current revision "
            f"{cur['run_id']}; a second run is a RE-SCORE and needs an "
            "explicit --rescore <reason>; nothing was written")
    if cur is None and rescore_reason is not None:
        raise RevisionRefusal(
            f"RESCORE_WITHOUT_PRIOR: {date_str} has no revision to supersede")
    run_id = run_id_for(fired)
    rel = f"{REV_DIR_REL}/{date_str}/{run_id}.json"
    rec = dict(record)
    rec["revision"] = {
        "schema": REVISION_SCHEMA, "date": date_str, "run_id": run_id,
        "fired_utc": fired_iso(fired),
        "scored_day_utc": date_str,
        "supersedes": cur["run_id"] if cur else None,
        "reason": rescore_reason,
        "inputs": inputs_capsule,
        "inputs_sha256": inputs_digest,
        "source_index": source_index_of(journal_snapshot),
        "persistence_inputs": persistence_inputs}
    data = record_bytes(rec)
    entry = {"schema": JOURNAL_ENTRY_SCHEMA, "date": date_str, "run_id": run_id,
             "path": rel, "sha256": sha256_bytes(data),
             "supersedes": cur["run_id"] if cur else None,
             "reason": rescore_reason, "appended_utc": iso_z(utc_now())}
    validate_revision_against_entry(rec, entry, expect_inputs=exp)   # F4
    new_journal = live + canonical_bytes(entry)
    parse_journal(new_journal)
    assert journal_prefix_ok(live, new_journal)
    # ---- stage
    txn = _p(repo, f"{TXN_DIR_REL}/{run_id}")
    os.makedirs(txn)
    try:
        _write_atomic(os.path.join(txn, "revision.json"), data)
        _write_atomic(os.path.join(txn, "index.ndjson"), new_journal)
        _create_once(_p(repo, rel), data)              # the irreversible step
        new_entries = entries + [entry]
        csv_bytes = derive_csv_bytes(repo, cap, new_entries, git=git)
        _write_atomic(os.path.join(txn, "data.csv"), csv_bytes)
        latest_date = max(current_map(new_entries))
        latest_entry = current_map(new_entries)[latest_date]
        _rec, latest_raw = reopen_revision(repo, latest_entry)
        _write_atomic(os.path.join(txn, "ensemble_latest.json"), latest_raw)
        # ---- publish
        _write_atomic(_p(repo, LATEST_REL), latest_raw)
        _write_atomic(_p(repo, CSV_REL), csv_bytes)
        dash = _p(repo, DASHBOARD_CSV_REL)
        if os.path.isdir(os.path.dirname(dash)):
            _write_atomic(dash, csv_bytes)
        _write_atomic(_p(repo, JOURNAL_REL), new_journal)
    finally:
        if os.path.exists(_p(repo, JOURNAL_REL)) and \
                journal_bytes(repo) == new_journal:
            for fn in os.listdir(txn):
                os.remove(os.path.join(txn, fn))
            os.rmdir(txn)
    return entry


# ------------------------------------------------------------ selftest --
def make_fake_git(csv_raw, record_blob_raw, *, csv_blob="c" * 40,
                  rec_blob="b" * 40, head="1" * 40, capsule_add=None):
    """A scripted git over ONE committed record (scored day from the blob)
    and ONE committed legacy CSV blob at HEAD, for temp-store fixtures."""
    def fake_git(_repo, *a):
        a = list(a)
        if a[:1] == ["log"]:
            if "--diff-filter=A" in a:
                return (capsule_add or "").encode()
            return b"c1\n"                           # first-parent history
        if a[:1] == ["rev-parse"]:
            if a[1] == "HEAD":
                return head.encode() + b"\n"
            if a[1].endswith(":" + LATEST_REL):
                return rec_blob.encode() + b"\n"
            if a[1].endswith(":" + CSV_REL):
                return csv_blob.encode() + b"\n"
        if a[:2] == ["cat-file", "-e"]:
            return b""
        if a[:2] == ["cat-file", "blob"]:
            if a[2] == rec_blob:
                return record_blob_raw
            if a[2] == csv_blob:
                return csv_raw
        raise AssertionError(f"fake git: {a}")
    return fake_git


def _selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ens-rev3-selftest-")
    try:
        repo = tmp
        os.makedirs(_p(repo, "docs"))
        os.makedirs(_p(repo, "monitoring/dashboard"))
        legacy_csv = ("date,region,tier,risk,confidence,methods,agreement\n"
                      "2026-08-30,a,1,0.5000,0.50,1,single_method\n"
                      "2026-08-31,a,0,0.0200,0.50,1,single_method\n").encode()
        # the checkout carries CRLF (Windows); authority is the LF blob
        _write_atomic(_p(repo, CSV_REL), legacy_csv.replace(b"\n", b"\r\n"))
        legacy_rec = {"date": "2026-08-31", "regions": {
            "a": {"tier": 0, "combined_risk": 0.02, "confidence": 0.5,
                  "methods_available": 1, "agreement": "single_method"}}}
        blob_raw = record_bytes(legacy_rec)
        fg = make_fake_git(legacy_csv, blob_raw)
        cap = build_legacy_baseline(repo, git=fg)
        assert cap["records"][0]["date"] == "2026-08-31"
        assert cap["legacy_csv"]["row_count"] == 2 and cap["legacy_csv"]["git_blob"] == "c" * 40
        write_legacy_baseline(repo, cap, git=fg)
        for bad, needle in (
                (dict(cap, records=[]), "record vector diverges"),
                (dict(cap, records=list(reversed(cap["records"] + [dict(cap["records"][0], git_blob="d" * 40)]))), "record vector"),
                (dict(cap, cutover_commit="e" * 40), "HEAD"),
                (dict(cap, legacy_csv=dict(cap["legacy_csv"], row_count=3)), "recompute"),
                (dict(cap, legacy_csv=dict(cap["legacy_csv"], git_blob="d" * 40)), "git_blob"),
                (dict(cap, resolution_rule="whatever"), "resolution_rule"),
                (dict(cap, records=[dict(cap["records"][0], parseable=False, date=None)]), "record vector")):
            try:
                validate_legacy_baseline(repo, bad, git=fg)
                raise AssertionError(f"forged capsule accepted: {needle}")
            except RevisionRefusal as x:
                assert "LEGACY_CAPSULE_NOT_DERIVED_FROM_CUTOVER" in str(x) and needle in str(x), (needle, x)
        cap = load_legacy_baseline(repo, git=fg)
        snap = journal_bytes(repo)
        view = prior_days_view(repo, snap, cap, "2026-09-02", 3, git=fg)
        assert [v[2]["kind"] for v in view] == ["hole", "legacy", "hole"]
        pins = [v[2] for v in view]
        cal_raw = b'{"region": "x", "valid_through": "2026-09-09"}\n'
        def inputs(tag, pins, day):
            ents = [input_entry("code", p, None, None, raw_bytes=(tag + p).encode())
                    for p in CODE_PATHS]
            ents.append(input_entry("calibration_capsule",
                                    CALIBRATION_DIR_REL + "/x.json", None,
                                    ["region", "valid_through"], raw_bytes=cal_raw))
            for pin in pins:
                if pin["kind"] != "hole":
                    ents.append(pin_input_entry(repo, cap, pin, git=fg))
            ents.append(scored_day_entry(day))
            return {"schema": INPUTS_SCHEMA, "entries": ents}
        exp = {"calibration_paths": [CALIBRATION_DIR_REL + "/x.json"]}
        rec = {"date": "2026-09-02", "regions": {
            "a": {"tier": 1, "combined_risk": 0.3, "confidence": 0.5,
                  "methods_available": 1, "agreement": "single_method"}},
            "summary": {}}
        fired = datetime(2026, 9, 3, 6, 15, 1, 123456, tzinfo=timezone.utc)
        cap1 = inputs("a", pins, "2026-09-02")
        # F3 negatives, each one change from cap1
        bad_caps = [
            ({"schema": INPUTS_SCHEMA, "entries": []}, "empty"),
            ({"schema": INPUTS_SCHEMA, "entries": [dict(cap1["entries"][0], sha256=None)]}, "sha256"),
            ({"schema": INPUTS_SCHEMA, "entries": cap1["entries"][1:]}, "code entries"),
            ({"schema": INPUTS_SCHEMA, "entries": cap1["entries"] + [cap1["entries"][-1]]}, "scored_day"),
            ({"schema": INPUTS_SCHEMA, "entries": [dict(e, byte_length=None) if e["kind"] == "legacy_record" else e for e in cap1["entries"]]}, "byte_length"),
            ({"schema": INPUTS_SCHEMA, "entries": [e for e in cap1["entries"] if e["kind"] != "calibration_capsule"]}, "calibration set"),
            ({"schema": INPUTS_SCHEMA, "entries": [e for e in cap1["entries"] if e["kind"] != "legacy_record"]}, "one-to-one"),
            ({"schema": INPUTS_SCHEMA, "entries": [dict(e, data_day="2026-01-01") if e["kind"] == "scored_day" else e for e in cap1["entries"]]}, "scored_day"),
        ]
        for bc, needle in bad_caps:
            try:
                validate_inputs_capsule(bc, expect=dict(exp, pins=pins, scored_day="2026-09-02"))
                raise AssertionError(f"bad inputs accepted: {needle}")
            except RevisionRefusal as x:
                assert "INPUTS_CAPSULE_SCHEMA" in str(x) and needle in str(x), (needle, x)
        e1 = publish_revision(repo, rec, cap1, snap, pins, fired, expect_inputs=exp, git=fg)
        assert _RUN_ID_RE.match(e1["run_id"]) and e1["supersedes"] is None
        j1 = journal_bytes(repo)
        csv1 = _read(_p(repo, CSV_REL))
        # F2: derived from the LF BLOB, not the CRLF checkout; written LF
        assert csv1 == legacy_csv + b"2026-09-02,a,1,0.3000,0.50,1,single_method\n"
        assert b"\r" not in csv1
        r1, raw1 = reopen_revision(repo, e1)
        assert r1["revision"]["inputs_sha256"] == inputs_sha256(cap1, expect=dict(exp, pins=pins, scored_day="2026-09-02"))
        # F4 negatives on a re-sealed copy of the revision
        def resealed(mut):
            r = json.loads(raw1.decode("utf-8")); mut(r); d = record_bytes(r)
            return r, dict(e1, sha256=sha256_bytes(d))
        for mut, needle in (
                (lambda r: r["revision"].__setitem__("reason", "revision-only reason"), "reason"),
                (lambda r: r["revision"].__setitem__("schema", "x"), "schema"),
                (lambda r: r["revision"].__setitem__("fired_utc", "2026-09-03T06:15:01Z"), "fired_utc"),
                (lambda r: r["revision"].__setitem__("fired_utc", "2026-09-03T06:15:02.123456Z"), "run_id time prefix"),
                (lambda r: r["revision"].__setitem__("source_index", {"entry_count": -1, "prefix_sha256": "0" * 64}), "source_index"),
                (lambda r: r["revision"]["persistence_inputs"].__setitem__(0, {"date": "2026-09-01", "kind": "hole", "run_id": "x", "sha256": None, "legacy": None}), "hole"),
                (lambda r: r["revision"].__setitem__("scored_day_utc", "1999-01-01"), "scored_day_utc"),
                (lambda r: r["revision"]["inputs"]["entries"].__setitem__(0, dict(r["revision"]["inputs"]["entries"][0], sha256="0" * 64)), "inputs_sha256")):
            r, en = resealed(mut)
            try:
                validate_revision_against_entry(r, en, expect_inputs=exp)
                raise AssertionError(f"identity mismatch accepted: {needle}")
            except RevisionRefusal as x:
                assert needle in str(x), (needle, x)
        # journal <-> revision: a journal line whose reason differs refuses on reopen
        try:
            reopen_revision(repo, dict(e1, reason="other", supersedes=None))
            raise AssertionError("journal/revision reason mismatch accepted")
        except RevisionRefusal as x:
            assert "reason" in str(x)
        # duplicate / moved / rescore as before
        snap2 = journal_bytes(repo)
        try:
            publish_revision(repo, rec, cap1, snap2, pins, fired, expect_inputs=exp, git=fg)
            raise AssertionError("duplicate accepted")
        except RevisionRefusal as x:
            assert "REVISION_EXISTS" in str(x)
        assert journal_bytes(repo) == j1
        view2 = prior_days_view(repo, snap2, cap, "2026-09-03", 3, git=fg)
        pins2 = [v[2] for v in view2]
        assert [p["kind"] for p in pins2] == ["revision", "hole", "legacy"]
        cap2 = inputs("b", pins2, "2026-09-03")
        assert any(e["kind"] == "prior_revision" and e["byte_length"] == len(raw1) for e in cap2["entries"])
        try:
            publish_revision(repo, dict(rec, date="2026-09-03"), cap2, snap, pins2, fired, expect_inputs=exp, git=fg)
            raise AssertionError("moved journal accepted")
        except RevisionRefusal as x:
            assert "JOURNAL_MOVED" in str(x)
        rec2 = dict(rec, regions={"a": dict(rec["regions"]["a"], tier=2)})
        e2 = publish_revision(repo, rec2, inputs("c", pins, "2026-09-02"), snap2, pins,
                              fired + timedelta(hours=1), rescore_reason="fix",
                              expect_inputs=exp, git=fg)
        assert e2["supersedes"] == e1["run_id"] and e2["reason"] == "fix"
        j2 = journal_bytes(repo)
        assert journal_prefix_ok(j1, j2) and len(parse_journal(j2)) == 2
        assert _read(_p(repo, CSV_REL)) == legacy_csv + b"2026-09-02,a,2,0.3000,0.50,1,single_method\n"
        r2, _ = reopen_revision(repo, e2)
        assert r2["revision"]["reason"] == "fix"
        # store cleanliness partners
        orphan = _p(repo, f"{REV_DIR_REL}/2026-09-02/20260101T000000000000Z-0badf00d.json")
        _write_atomic(orphan, b"{}\n")
        try:
            check_store_clean(repo); raise AssertionError("orphan accepted")
        except RevisionRefusal as x:
            assert "REVISION_ORPHAN" in str(x)
        os.remove(orphan)
        # F2 partner: a mutated committed CSV blob refuses the derivation
        fg_bad = make_fake_git(legacy_csv.replace(b"0.5000", b"0.5001"), blob_raw)
        try:
            derive_csv_bytes(repo, cap, parse_journal(j2), git=fg_bad)
            raise AssertionError("mutated blob accepted")
        except RevisionRefusal as x:
            assert "CSV_LEGACY_BLOB_MISMATCH" in str(x)
        print("ensemble_revisions (corrected v1, recut F1-F4) selftest: ALL PASS "
              "(temp store; scripted git; CRLF checkout carrier; nothing scored; "
              "nothing public touched)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--selftest"]:
        raise SystemExit(_selftest())
    raise SystemExit("usage: ensemble_revisions_cayley.py --selftest (the "
                     "runner imports this module; the cutover capsule is "
                     "written by run_ensemble_daily.py --cutover)")
