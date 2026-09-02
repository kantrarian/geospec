#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMMUTABLE REVISION STORE for the daily ensemble record (cayley) --
corrected contract v1.

Owner: asylum 2026-09-02 "use immutable revision" / "land B6".
Contract: grassmann 1432Z (C-7..C-11 layout) as CORRECTED by codex 1433Z
(five corrections). Everything below is built to the corrected text:

LAYOUT
  docs/ensemble/<YYYY-MM-DD>/<run_id>.json   one immutable REVISION per
                                            run, CREATE-ONCE (O_EXCL)
  docs/ensemble/index.ndjson                APPEND-ONLY journal: one
                                            canonical JSON object per
                                            line; a newer journal must
                                            carry every older one as an
                                            exact byte prefix (C-8)
  docs/ensemble/legacy_baseline_v1.json     the CUTOVER CAPSULE (codex
                                            correction 1): created once,
                                            never amended; enumerates
                                            every committed pre-cutover
                                            (date, record_path, git_blob,
                                            sha256, csv_row_sha256) and
                                            binds the frozen legacy
                                            data.csv prefix
  docs/ensemble_latest.json                 DERIVED: byte copy of the
                                            current revision of the max
                                            date (C-10)
  docs/data.csv                             DERIVED by ONE writer: the
                                            bound legacy prefix bytes +
                                            the CURRENT revision's rows
                                            for every journaled date

RULES
  * run_id = fired_utc as YYYYMMDDTHHMMSSffffffZ + "-" + 8 hex of uuid4
    (codex correction 2: seconds alone are not uniqueness); there is NO
    `current` field anywhere -- current is the LAST valid journal event
    for the date; a re-score's `supersedes` must equal that exact run.
  * every revision records `source_index = {entry_count, prefix_sha256}`,
    the exact journal prefix visible when persistence ran (correction 3);
    `persistence_inputs` entries are the closed union revision | legacy |
    hole, each resolved against that prefix plus the legacy capsule; a
    hole is valid only when neither source has the date.
  * a run stages all surfaces under docs/ensemble/.txn/<run_id>/,
    validates them, then publishes; the operator commits everything in
    ONE commit. A dirty/incomplete transaction, an orphan revision (file
    with no journal line) or a dangling journal line (no file) makes the
    NEXT run REFUSE with a typed recovery instruction; it never starts a
    new run around it (correction 4).
  * `inputs` is a closed ORDERED capsule of {kind, path|identity,
    data_day, keyset, byte_length, sha256} entries; `inputs_sha256` is
    sha256 over the contract's canonical JSON encoding of that capsule
    and is recomputed by the bar from the named bytes (correction 5).
  * B6 / C-11: `scored_day_utc == date`, derived from the UTC clock.

Nothing here scores anything; nothing here rewrites history: the
committed pre-cutover records and the frozen CSV prefix stay as they are.
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
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}\d{6}Z-[0-9a-f]{8}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
JOURNAL_FIELDS = {"schema", "date", "run_id", "path", "sha256", "supersedes",
                  "reason", "appended_utc"}
REVISION_FIELDS = {"schema", "date", "run_id", "fired_utc", "scored_day_utc",
                   "supersedes", "reason", "inputs", "inputs_sha256",
                   "source_index", "persistence_inputs"}
INPUT_ENTRY_FIELDS = {"kind", "identity", "data_day", "keyset", "byte_length",
                      "sha256"}
INPUT_KINDS = {"code", "calibration_capsule", "prior_revision",
               "legacy_record", "scored_day"}
PERSISTENCE_KINDS = {"revision", "legacy", "hole"}


class RevisionRefusal(ValueError):
    """Typed refusal; the code leads the message."""


# ------------------------------------------------------------ helpers --
def utc_now():
    return datetime.now(timezone.utc)


def iso_z(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id_for(fired):
    """codex correction 2: microsecond instant + 8 hex of uuid4."""
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
    """The committed form of a record / capsule: sorted keys, 2-space
    indent, LF, trailing newline."""
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


# ------------------------------------------------------------ journal --
def journal_bytes(repo):
    p = _p(repo, JOURNAL_REL)
    return _read(p) if os.path.exists(p) else b""


def parse_journal(raw):
    """Parse and validate the NDJSON journal (C-8). Returns the entries
    in order. Refuses blank/truncated lines, non-canonical lines,
    duplicate run ids, stale supersedes, forks in per-date lineage."""
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
        else:
            if e["supersedes"] != cur["run_id"]:
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
def _git(repo, *a):
    return subprocess.check_output(["git", "-C", repo] + list(a))


def build_legacy_baseline(repo, record_rel=LATEST_REL, csv_rel=CSV_REL,
                          git=_git):
    """codex correction 1: enumerate every COMMITTED pre-cutover record
    reachable from HEAD -- (date, record_path, git_blob, sha256) for each
    distinct blob of `record_rel`, in first-parent history order -- and
    bind the frozen legacy CSV prefix (bytes as committed at HEAD) with a
    per-date csv_row_sha256. Pure derivation from committed bytes; makes
    no choice about which blob "wins" -- per-date resolution is a stated
    rule at read time (see legacy_record_for)."""
    log = git(repo, "log", "--first-parent", "--format=%H",
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
            d = rec.get("date")
        except (ValueError, UnicodeDecodeError):
            d = None
        records.append({"date": d if isinstance(d, str) and _DATE_RE.match(d)
                        else None,
                        "record_path": record_rel, "commit": c,
                        "git_blob": blob, "sha256": sha256_bytes(raw),
                        "byte_length": len(raw),
                        "parseable": d is not None})
    csv_raw = git(repo, "show", f"HEAD:{csv_rel}")
    rows = list(csv.reader(io.StringIO(csv_raw.decode("utf-8"), newline="")))
    header, body = (rows[0], rows[1:]) if rows else (CSV_HEADER, [])
    per_date = {}
    for r in body:
        if r:
            per_date.setdefault(r[0], []).append(",".join(r))
    csv_rows = {d: sha256_bytes(("\n".join(v) + "\n").encode())
                for d, v in sorted(per_date.items())}
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    return {"schema": LEGACY_SCHEMA,
            "created_utc": iso_z(utc_now()),
            "cutover_commit": head,
            "record_path": record_rel,
            "records": records,
            "legacy_csv": {"path": csv_rel, "header": header,
                           "row_count": len(body),
                           "prefix_sha256": sha256_bytes(csv_raw),
                           "csv_row_sha256_by_date": csv_rows},
            "resolution_rule": ("for a legacy date, the record consumed is "
                                "the LAST committed parseable blob of that "
                                "date in first-parent order (the record as "
                                "last published); the frozen CSV prefix is "
                                "copied byte-for-byte, never regenerated")}


def write_legacy_baseline(repo, capsule):
    path = _p(repo, LEGACY_REL)
    _create_once(path, record_bytes(capsule))
    return path


def load_legacy_baseline(repo):
    path = _p(repo, LEGACY_REL)
    if not os.path.exists(path):
        return None
    raw = _read(path)
    try:
        cap = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise RevisionRefusal("LEGACY_CAPSULE_UNREADABLE")
    if not isinstance(cap, dict) or cap.get("schema") != LEGACY_SCHEMA:
        raise RevisionRefusal("LEGACY_CAPSULE_SCHEMA")
    cap["_sha256"] = sha256_bytes(raw)
    return cap


def legacy_record_for(cap, date_str):
    """The legacy record consumed for a pre-cutover date: the LAST
    committed parseable blob of that date in first-parent order
    (records are listed newest-first, so the first match)."""
    if cap is None:
        return None
    for r in cap["records"]:
        if r["parseable"] and r["date"] == date_str:
            return r
    return None


def reopen_legacy_record(repo, cap, rec, git=_git):
    raw = git(repo, "cat-file", "blob", rec["git_blob"])
    if sha256_bytes(raw) != rec["sha256"]:
        raise RevisionRefusal(
            f"LEGACY_RECORD_DIGEST_MISMATCH: {rec['git_blob'][:12]}")
    return json.loads(raw.decode("utf-8"))


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
        return json.loads(raw.decode("utf-8")), raw
    except (ValueError, UnicodeDecodeError):
        raise RevisionRefusal(f"REVISION_UNPARSABLE: {entry['path']}")


def prior_days_view(repo, journal_raw, cap, target_date_str, days,
                    git=_git):
    """For days_back = 1..days, resolve the prior day against EXACTLY the
    given journal bytes (the snapshot the run will record) plus the legacy
    capsule. Returns [(date, record|None, persistence_input_entry)]."""
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


# ------------------------------------------------------ inputs capsule --
def input_entry(kind, identity, data_day, keyset, raw_bytes=None,
                sha256=None, byte_length=None):
    if kind not in INPUT_KINDS:
        raise RevisionRefusal(f"INPUTS_KIND: {kind!r}")
    if raw_bytes is not None:
        sha256, byte_length = sha256_bytes(raw_bytes), len(raw_bytes)
    return {"kind": kind, "identity": identity, "data_day": data_day,
            "keyset": sorted(keyset) if keyset is not None else None,
            "byte_length": byte_length, "sha256": sha256}


def validate_inputs_capsule(cap):
    if not isinstance(cap, dict) or set(cap) != {"schema", "entries"}:
        raise RevisionRefusal("INPUTS_CAPSULE_SCHEMA: not the closed shape")
    if cap["schema"] != INPUTS_SCHEMA or not isinstance(cap["entries"], list):
        raise RevisionRefusal("INPUTS_CAPSULE_SCHEMA")
    for i, e in enumerate(cap["entries"]):
        if not isinstance(e, dict) or set(e) != INPUT_ENTRY_FIELDS:
            raise RevisionRefusal(f"INPUTS_CAPSULE_SCHEMA: entry {i} fields")
        if e["kind"] not in INPUT_KINDS:
            raise RevisionRefusal(f"INPUTS_CAPSULE_SCHEMA: entry {i} kind")
        if e["sha256"] is not None and not (isinstance(e["sha256"], str)
                                            and _HEX64_RE.match(e["sha256"])):
            raise RevisionRefusal(f"INPUTS_CAPSULE_SCHEMA: entry {i} sha256")
    return True


def inputs_sha256(cap):
    validate_inputs_capsule(cap)
    return sha256_bytes(canonical_bytes(cap))


# --------------------------------------------------------- transaction --
def check_store_clean(repo):
    """codex correction 4: refuse to start a run over a dirty transaction,
    an orphan revision, or a dangling journal line. Returns the parsed
    journal entries when clean."""
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


def derive_csv_bytes(repo, cap, entries):
    """Legacy prefix (bound bytes, copied) + CURRENT revisions' rows for
    every journaled date, dates sorted."""
    if cap is None:
        raise RevisionRefusal("LEGACY_CAPSULE_ABSENT: the cutover capsule "
                              "must exist before the first revision")
    prefix = None
    p = _p(repo, CSV_REL)
    if os.path.exists(p):
        raw = _read(p)
        # the committed legacy prefix is the leading bytes of the current
        # file up to its bound length; verify by digest
        hdr_and_rows = raw.split(b"\n")
        # reconstruct the legacy prefix from the capsule's row count
        n = cap["legacy_csv"]["row_count"]
        prefix = b"\n".join(hdr_and_rows[:n + 1]) + b"\n"
        if sha256_bytes(prefix) != cap["legacy_csv"]["prefix_sha256"]:
            raise RevisionRefusal(
                "CSV_LEGACY_PREFIX_CHANGED: docs/data.csv no longer starts "
                "with the bound legacy prefix")
    else:
        raise RevisionRefusal("CSV_ABSENT: docs/data.csv missing")
    cur = current_map(entries)
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    for d in sorted(cur):
        rec, _raw = reopen_revision(repo, cur[d])
        for r in _csv_rows_for_record(rec):
            w.writerow(r)
    return prefix + out.getvalue().encode("utf-8")


def publish_revision(repo, record, inputs_capsule, journal_snapshot,
                     persistence_inputs, fired, rescore_reason=None):
    """Stage, validate and publish ONE revision + its journal line + the
    derived surfaces. `journal_snapshot` must be the exact journal bytes
    the caller resolved persistence against (recorded as source_index).
    Returns the journal entry."""
    if not isinstance(record, dict) or not isinstance(record.get("date"), str) \
            or not _DATE_RE.match(record["date"]):
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: date")
    if not isinstance(record.get("regions"), dict) or not record["regions"]:
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: regions")
    date_str = record["date"]
    if rescore_reason is not None and not (isinstance(rescore_reason, str)
                                           and rescore_reason.strip()):
        raise RevisionRefusal("RESCORE_REASON_EMPTY")
    for pe in persistence_inputs:
        if set(pe) != {"date", "kind", "run_id", "sha256", "legacy"} or \
                pe["kind"] not in PERSISTENCE_KINDS:
            raise RevisionRefusal("PERSISTENCE_INPUTS_SCHEMA")
    check_store_clean(repo)
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
    cap = load_legacy_baseline(repo)
    if cap is None:
        raise RevisionRefusal("LEGACY_CAPSULE_ABSENT: run the cutover first")
    run_id = run_id_for(fired)
    rel = f"{REV_DIR_REL}/{date_str}/{run_id}.json"
    rec = dict(record)
    rec["revision"] = {
        "schema": REVISION_SCHEMA, "date": date_str, "run_id": run_id,
        "fired_utc": fired.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "scored_day_utc": date_str,
        "supersedes": cur["run_id"] if cur else None,
        "reason": rescore_reason,
        "inputs": inputs_capsule,
        "inputs_sha256": inputs_sha256(inputs_capsule),
        "source_index": source_index_of(journal_snapshot),
        "persistence_inputs": persistence_inputs}
    data = record_bytes(rec)
    entry = {"schema": JOURNAL_ENTRY_SCHEMA, "date": date_str, "run_id": run_id,
             "path": rel, "sha256": sha256_bytes(data),
             "supersedes": cur["run_id"] if cur else None,
             "reason": rescore_reason, "appended_utc": iso_z(utc_now())}
    new_journal = live + canonical_bytes(entry)
    parse_journal(new_journal)                      # validates the append
    assert journal_prefix_ok(live, new_journal)
    # ---- stage
    txn = _p(repo, f"{TXN_DIR_REL}/{run_id}")
    os.makedirs(txn)
    try:
        _write_atomic(os.path.join(txn, "revision.json"), data)
        _write_atomic(os.path.join(txn, "index.ndjson"), new_journal)
        # the CSV derivation needs the revision reopenable: stage a copy at
        # its final path FIRST via create-once (the irreversible step),
        # then derive
        _create_once(_p(repo, rel), data)
        new_entries = entries + [entry]
        csv_bytes = derive_csv_bytes(repo, cap, new_entries)
        _write_atomic(os.path.join(txn, "data.csv"), csv_bytes)
        latest_date = max(current_map(new_entries))
        latest_entry = current_map(new_entries)[latest_date]
        _rec, latest_raw = reopen_revision(repo, latest_entry)
        _write_atomic(os.path.join(txn, "ensemble_latest.json"), latest_raw)
        # ---- publish (journal last among the derived, latest/csv before)
        _write_atomic(_p(repo, LATEST_REL), latest_raw)
        _write_atomic(_p(repo, CSV_REL), csv_bytes)
        dash = _p(repo, DASHBOARD_CSV_REL)
        if os.path.isdir(os.path.dirname(dash)):
            _write_atomic(dash, csv_bytes)
        _write_atomic(_p(repo, JOURNAL_REL), new_journal)
    finally:
        # a completed publish leaves no staging; an exception leaves it for
        # the typed recovery refusal on the next run
        if os.path.exists(_p(repo, JOURNAL_REL)) and \
                journal_bytes(repo) == new_journal:
            for fn in os.listdir(txn):
                os.remove(os.path.join(txn, fn))
            os.rmdir(txn)
    return entry


# ------------------------------------------------------------ selftest --
def _selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ens-rev2-selftest-")
    try:
        repo = tmp
        os.makedirs(_p(repo, "docs"))
        os.makedirs(_p(repo, "monitoring/dashboard"))
        legacy_csv = ("date,region,tier,risk,confidence,methods,agreement\n"
                      "2026-08-30,a,1,0.5000,0.50,1,single_method\n"
                      "2026-08-31,a,0,0.0200,0.50,1,single_method\n").encode()
        _write_atomic(_p(repo, CSV_REL), legacy_csv)
        legacy_rec_0831 = {"date": "2026-08-31", "regions": {
            "a": {"tier": 0, "combined_risk": 0.02, "confidence": 0.5,
                  "methods_available": 1, "agreement": "single_method"}}}
        blob_raw = record_bytes(legacy_rec_0831)

        def fake_git(_repo, *a):
            if a[0] == "log":
                return b"c1\n"
            if a[0] == "rev-parse" and a[1] == "c1:docs/ensemble_latest.json":
                return b"b" * 40 + b"\n"
            if a[0] == "rev-parse" and a[1] == "HEAD":
                return b"h" * 40 + b"\n"
            if a[0] == "cat-file":
                return blob_raw
            if a[0] == "show":
                return legacy_csv
            raise AssertionError(a)
        cap = build_legacy_baseline(repo, git=fake_git)
        assert cap["records"][0]["date"] == "2026-08-31"
        assert cap["legacy_csv"]["row_count"] == 2
        write_legacy_baseline(repo, cap)
        try:
            write_legacy_baseline(repo, cap)
            raise AssertionError("capsule rewritten")
        except RevisionRefusal as x:
            assert "REVISION_PATH_EXISTS" in str(x)
        cap = load_legacy_baseline(repo)
        # persistence view over the empty journal: 09-01 hole, 08-31 legacy
        snap = journal_bytes(repo)
        view = prior_days_view(repo, snap, cap, "2026-09-02", 3, git=fake_git)
        assert view[0][2]["kind"] == "hole"
        assert view[1][2]["kind"] == "legacy" and view[1][1] == legacy_rec_0831
        assert view[2][2]["kind"] == "hole"
        pins = [v[2] for v in view]
        rec = {"date": "2026-09-02", "regions": {
            "a": {"tier": 1, "combined_risk": 0.3, "confidence": 0.5,
                  "methods_available": 1, "agreement": "single_method"}},
            "summary": {}}
        inputs = {"schema": INPUTS_SCHEMA, "entries": [
            input_entry("code", "monitoring/src/x.py", None, None,
                        raw_bytes=b"code"),
            input_entry("scored_day", "2026-09-02", "2026-09-02", None)]}
        fired = datetime(2026, 9, 3, 6, 15, 1, 123456, tzinfo=timezone.utc)
        e1 = publish_revision(repo, rec, inputs, snap, pins, fired)
        assert _RUN_ID_RE.match(e1["run_id"]) and e1["supersedes"] is None
        j1 = journal_bytes(repo)
        assert parse_journal(j1)[0] == e1 and journal_prefix_ok(snap, j1)
        assert not os.path.exists(_p(repo, TXN_DIR_REL)) or \
            not os.listdir(_p(repo, TXN_DIR_REL))
        csv1 = _read(_p(repo, CSV_REL))
        assert csv1.startswith(legacy_csv) and \
            csv1[len(legacy_csv):] == b"2026-09-02,a,1,0.3000,0.50,1,single_method\n"
        assert _read(_p(repo, LATEST_REL)) == _read(_p(repo, e1["path"]))
        r1, _ = reopen_revision(repo, e1)
        assert r1["revision"]["source_index"] == {"entry_count": 0,
                                                  "prefix_sha256": sha256_bytes(b"")}
        assert r1["revision"]["inputs_sha256"] == inputs_sha256(inputs)
        assert r1["revision"]["scored_day_utc"] == "2026-09-02"
        # duplicate default run refuses, store unchanged
        snap2 = journal_bytes(repo)
        try:
            publish_revision(repo, rec, inputs, snap2, pins, fired)
            raise AssertionError("duplicate accepted")
        except RevisionRefusal as x:
            assert "REVISION_EXISTS" in str(x)
        assert journal_bytes(repo) == j1
        # journal moved between resolve and publish refuses
        try:
            publish_revision(repo, dict(rec, date="2026-09-03"), inputs, snap,
                             pins, fired)
            raise AssertionError("moved journal accepted")
        except RevisionRefusal as x:
            assert "JOURNAL_MOVED" in str(x)
        # rescore: supersedes the exact current run; csv row rewritten
        rec2 = dict(rec, regions={"a": dict(rec["regions"]["a"], tier=2)})
        e2 = publish_revision(repo, rec2, inputs, snap2, pins,
                              fired + timedelta(hours=1), rescore_reason="fix")
        assert e2["supersedes"] == e1["run_id"] and e2["reason"] == "fix"
        j2 = journal_bytes(repo)
        assert journal_prefix_ok(j1, j2) and len(parse_journal(j2)) == 2
        csv2 = _read(_p(repo, CSV_REL))
        assert csv2 == legacy_csv + b"2026-09-02,a,2,0.3000,0.50,1,single_method\n"
        assert os.path.exists(_p(repo, e1["path"]))
        # stale supersedes / forks / blank / truncated refuse
        bad = j2.replace(e1["run_id"].encode(), b"20260101T000000000000Z-deadbeef", 1)
        for raw, needle in ((j2[:-1], "JOURNAL_TRUNCATED"),
                            (j2 + b"\n", "JOURNAL_BLANK_LINE"),
                            (j2.replace(b'"reason":"fix"', b'"reason":null'),
                             "JOURNAL_RESCORE_WITHOUT_REASON"),):
            try:
                parse_journal(raw)
                raise AssertionError(needle)
            except RevisionRefusal as x:
                assert needle in str(x), (needle, x)
        lines = j2.split(b"\n")[:-1]
        e2b = json.loads(lines[1])
        e2b["supersedes"] = "20260101T000000000000Z-deadbeef"
        try:
            parse_journal(lines[0] + b"\n" + canonical_bytes(e2b))
            raise AssertionError("stale supersedes accepted")
        except RevisionRefusal as x:
            assert "JOURNAL_STALE_SUPERSEDES" in str(x)
        # orphan revision / dangling line / dirty txn refuse the next run
        orphan = _p(repo, f"{REV_DIR_REL}/2026-09-02/20260101T000000000000Z-0badf00d.json")
        _write_atomic(orphan, b"{}\n")
        try:
            check_store_clean(repo)
            raise AssertionError("orphan accepted")
        except RevisionRefusal as x:
            assert "REVISION_ORPHAN" in str(x)
        os.remove(orphan)
        os.makedirs(_p(repo, f"{TXN_DIR_REL}/zzz"))
        _write_atomic(_p(repo, f"{TXN_DIR_REL}/zzz/revision.json"), b"{}")
        try:
            check_store_clean(repo)
            raise AssertionError("dirty txn accepted")
        except RevisionRefusal as x:
            assert "REVISION_TXN_DIRTY" in str(x)
        shutil.rmtree(_p(repo, TXN_DIR_REL))
        os.rename(_p(repo, e2["path"]), _p(repo, e2["path"]) + ".moved")
        try:
            check_store_clean(repo)
            raise AssertionError("dangling accepted")
        except RevisionRefusal as x:
            assert "REVISION_DANGLING_JOURNAL_LINE" in str(x)
        os.rename(_p(repo, e2["path"]) + ".moved", _p(repo, e2["path"]))
        assert len(check_store_clean(repo)) == 2
        # legacy prefix tamper refuses the derivation
        _write_atomic(_p(repo, CSV_REL), csv2.replace(b"0.5000", b"0.5001"))
        try:
            derive_csv_bytes(repo, cap, parse_journal(j2))
            raise AssertionError("prefix tamper accepted")
        except RevisionRefusal as x:
            assert "CSV_LEGACY_PREFIX_CHANGED" in str(x)
        print("ensemble_revisions (corrected v1) selftest: ALL PASS (temp store; "
              "fake git; nothing scored; nothing public touched)")
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
