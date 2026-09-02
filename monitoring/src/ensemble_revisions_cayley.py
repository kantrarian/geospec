#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMMUTABLE REVISION STORE for the daily ensemble record (cayley).

Owner decision 2026-09-02 (asylum: "use immutable revision"), codex's
model (2026-09-01 2303Z, corpus-bar review, disposition item 2):

  * every scored day is published CREATE-ONCE as
        docs/ensemble/<date>/<run_id>.json
    where run_id = r01, r02, ... in issue order; a file that exists is
    never rewritten (the open is exclusive-create);
  * docs/ensemble/index.json is APPEND-ONLY: one entry per revision,
    in issue order, carrying path + sha256 + supersedes + reason; an
    existing entry never changes (the bar re-derives and compares);
  * docs/ensemble/current.json is DERIVED from the index: for each date
    the LAST revision is current;
  * a second run for a date that already has a revision REFUSES typed
    unless the operator passes --rescore <reason>; then a NEW revision
    is appended with `supersedes` naming the previous run_id, the
    reason, and the input identities of both;
  * docs/data.csv is DERIVED from the index by ONE writer: rows for
    dates that have revisions come from the CURRENT revision only, so
    the record and the CSV can never disagree for a modelled date; rows
    for dates that predate the model are FROZEN (kept byte-for-byte,
    never regenerated, never reordered) -- they are the pinned history
    of the old two-writer behaviour (grassmann's ledger, codex item 4);
  * docs/ensemble_latest.json is DERIVED: the byte copy of the current
    revision of the latest date (the dashboard keeps reading it);
  * persistence reads PRIOR DAYS from the public revisions only and
    binds the exact (date, run_id, sha256) it consumed; a day with no
    public revision is an explicit hole, never a local-only
    confirmation.

Nothing here scores anything, and nothing here touches history before
the model: the frozen CSV rows and the committed ensemble_latest.json
history stay as they are.
"""
import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone

INDEX_SCHEMA = "geospec-ensemble-revision-index-v1"
CURRENT_SCHEMA = "geospec-ensemble-current-v1"
REVISION_SCHEMA = "geospec-ensemble-revision-v1"
REV_DIR_REL = "docs/ensemble"
INDEX_REL = REV_DIR_REL + "/index.json"
CURRENT_REL = REV_DIR_REL + "/current.json"
LATEST_REL = "docs/ensemble_latest.json"
CSV_REL = "docs/data.csv"
DASHBOARD_CSV_REL = "monitoring/dashboard/data.csv"
CSV_HEADER = ["date", "region", "tier", "risk", "confidence", "methods",
              "agreement"]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_ID_RE = re.compile(r"^r\d{2,}$")
INDEX_ENTRY_FIELDS = {"date", "run_id", "seq", "path", "sha256",
                      "supersedes", "rescore_reason", "created_utc",
                      "inputs"}


class RevisionRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _p(repo, rel):
    return os.path.join(repo, rel.replace("/", os.sep))


def _read_json(path):
    with io.open(path, "rb") as f:
        raw = f.read()
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (ValueError, UnicodeDecodeError) as e:
        raise RevisionRefusal(f"REVISION_STORE_UNREADABLE: {path}: {e}")


def _write_bytes_atomic(path, data):
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
            "create-once; a new revision gets a new run_id")
    with os.fdopen(fd, "wb") as f:
        f.write(data)


# ---------------------------------------------------------------- index --
def load_index(repo):
    """The append-only index, or the empty index if the store does not
    exist yet. Shape and ordering are validated on every load."""
    path = _p(repo, INDEX_REL)
    if not os.path.exists(path):
        return {"schema": INDEX_SCHEMA, "revisions": []}
    idx, _raw = _read_json(path)
    validate_index(idx)
    return idx


def validate_index(idx):
    if not isinstance(idx, dict) or idx.get("schema") != INDEX_SCHEMA:
        raise RevisionRefusal("REVISION_INDEX_SCHEMA: not the registered "
                              "index schema")
    revs = idx.get("revisions")
    if not isinstance(revs, list):
        raise RevisionRefusal("REVISION_INDEX_SCHEMA: revisions is not a list")
    seen = set()
    per_date = {}
    for i, e in enumerate(revs):
        if not isinstance(e, dict) or set(e) != INDEX_ENTRY_FIELDS:
            raise RevisionRefusal(
                f"REVISION_INDEX_SCHEMA: entry {i} field set not closed")
        if not (isinstance(e["date"], str) and _DATE_RE.match(e["date"])):
            raise RevisionRefusal(f"REVISION_INDEX_SCHEMA: entry {i} date")
        if not (isinstance(e["run_id"], str) and _RUN_ID_RE.match(e["run_id"])):
            raise RevisionRefusal(f"REVISION_INDEX_SCHEMA: entry {i} run_id")
        key = (e["date"], e["run_id"])
        if key in seen:
            raise RevisionRefusal(
                f"REVISION_INDEX_DUPLICATE: {e['date']} {e['run_id']}")
        seen.add(key)
        n = per_date.get(e["date"], 0) + 1
        per_date[e["date"]] = n
        if e["seq"] != n or e["run_id"] != f"r{n:02d}":
            raise RevisionRefusal(
                f"REVISION_INDEX_ORDER: {e['date']} entry {i} is seq "
                f"{e['seq']} / {e['run_id']} but is the {n}th revision of "
                "that date in issue order")
        exp_sup = f"r{n - 1:02d}" if n > 1 else None
        if e["supersedes"] != exp_sup:
            raise RevisionRefusal(
                f"REVISION_INDEX_SUPERSEDES: {e['date']} {e['run_id']} names "
                f"{e['supersedes']!r}, expected {exp_sup!r}")
        if n > 1 and not (isinstance(e["rescore_reason"], str)
                          and e["rescore_reason"].strip()):
            raise RevisionRefusal(
                f"REVISION_INDEX_REASON: {e['date']} {e['run_id']} is a "
                "re-score without a reason")
        if n == 1 and e["rescore_reason"] is not None:
            raise RevisionRefusal(
                f"REVISION_INDEX_REASON: {e['date']} r01 carries a re-score "
                "reason but supersedes nothing")
        if e["path"] != f"{REV_DIR_REL}/{e['date']}/{e['run_id']}.json":
            raise RevisionRefusal(
                f"REVISION_INDEX_PATH: {e['date']} {e['run_id']} path "
                f"{e['path']!r} is not the registered layout")
        if not (isinstance(e["sha256"], str) and len(e["sha256"]) == 64):
            raise RevisionRefusal(
                f"REVISION_INDEX_SCHEMA: {e['date']} {e['run_id']} sha256")
    return True


def revisions_for(idx, date_str):
    return [e for e in idx["revisions"] if e["date"] == date_str]


def current_map(idx):
    """date -> the LAST (current) revision entry, derived from the index."""
    cur = {}
    for e in idx["revisions"]:
        cur[e["date"]] = e
    return cur


def derive_current(idx):
    cur = current_map(idx)
    return {"schema": CURRENT_SCHEMA,
            "latest_date": max(cur) if cur else None,
            "current": {d: {"run_id": e["run_id"], "path": e["path"],
                            "sha256": e["sha256"]}
                        for d, e in sorted(cur.items())}}


def reopen_revision(repo, entry):
    """Reopen a revision named by an index entry and REQUIRE its digest."""
    path = _p(repo, entry["path"])
    if not os.path.exists(path):
        raise RevisionRefusal(
            f"REVISION_MISSING: {entry['path']} is in the index but absent")
    rec, raw = _read_json(path)
    if sha256_bytes(raw) != entry["sha256"]:
        raise RevisionRefusal(
            f"REVISION_DIGEST_MISMATCH: {entry['path']} does not hash to the "
            "index entry")
    return rec, raw


def prior_revision(repo, idx, date_str):
    """The CURRENT public revision for a date, reopened and digest-checked,
    or (None, None) when the date has no public revision (a hole)."""
    e = current_map(idx).get(date_str)
    if e is None:
        return None, None
    rec, _raw = reopen_revision(repo, e)
    return rec, {"date": e["date"], "run_id": e["run_id"],
                 "sha256": e["sha256"]}


# -------------------------------------------------------------- publish --
def record_bytes(record):
    """The committed form of a record: sorted keys, 2-space indent, LF,
    trailing newline (readable, deterministic, same shape the dashboard
    already parses)."""
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def publish_revision(repo, record, inputs, rescore_reason=None,
                     created_utc=None):
    """Publish `record` (a dict carrying at least "date" and "regions")
    as the next revision of its date.

    Default mode: the date must have NO revision yet, else
    REVISION_EXISTS (typed refusal; nothing written).
    Re-score mode (rescore_reason is a non-empty string): the date MUST
    already have a revision; a new one is appended naming the previous
    run_id in `supersedes`.

    `inputs` is the caller's closed dict of input identities (code
    blobs, calibration capsules, prior revisions consumed, ...); it is
    stored verbatim in the record and in the index entry.
    Returns the index entry. The record dict is NOT mutated; a copy
    carrying the `revision` block is what gets written.
    """
    if not isinstance(record, dict) or not isinstance(record.get("date"),
                                                      str):
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: record carries no "
                              "string date")
    date_str = record["date"]
    if not _DATE_RE.match(date_str):
        raise RevisionRefusal(f"REVISION_RECORD_SCHEMA: date {date_str!r}")
    if not isinstance(record.get("regions"), dict) or not record["regions"]:
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: record carries no "
                              "regions")
    if not isinstance(inputs, dict):
        raise RevisionRefusal("REVISION_RECORD_SCHEMA: inputs must be a dict")
    if rescore_reason is not None and not (isinstance(rescore_reason, str)
                                           and rescore_reason.strip()):
        raise RevisionRefusal("RESCORE_REASON_EMPTY: --rescore needs a "
                              "non-empty reason")
    idx = load_index(repo)
    existing = revisions_for(idx, date_str)
    if existing and rescore_reason is None:
        raise RevisionRefusal(
            f"REVISION_EXISTS: {date_str} already has revision "
            f"{existing[-1]['run_id']} ({existing[-1]['path']}); a second "
            "run of a scored day is a RE-SCORE and needs an explicit "
            "--rescore <reason>; nothing was written")
    if not existing and rescore_reason is not None:
        raise RevisionRefusal(
            f"RESCORE_WITHOUT_PRIOR: {date_str} has no revision to "
            "supersede; run it without --rescore")
    seq = len(existing) + 1
    run_id = f"r{seq:02d}"
    rel = f"{REV_DIR_REL}/{date_str}/{run_id}.json"
    supersedes = existing[-1]["run_id"] if existing else None
    created = created_utc or utc_now_iso()
    rec = dict(record)
    rec["revision"] = {
        "schema": REVISION_SCHEMA, "date": date_str, "run_id": run_id,
        "seq": seq, "supersedes": supersedes,
        "rescore_reason": rescore_reason, "created_utc": created,
        "inputs": inputs,
        "supersedes_inputs": existing[-1]["inputs"] if existing else None}
    data = record_bytes(rec)
    entry = {"date": date_str, "run_id": run_id, "seq": seq, "path": rel,
             "sha256": sha256_bytes(data), "supersedes": supersedes,
             "rescore_reason": rescore_reason, "created_utc": created,
             "inputs": inputs}
    # order: the create-once record first (the irreversible step), then
    # the index append, then the derived surfaces
    _create_once(_p(repo, rel), data)
    new_idx = {"schema": INDEX_SCHEMA,
               "revisions": list(idx["revisions"]) + [entry]}
    validate_index(new_idx)
    _write_bytes_atomic(_p(repo, INDEX_REL), record_bytes(new_idx))
    write_derived_surfaces(repo, new_idx)
    return entry


def write_derived_surfaces(repo, idx=None):
    """current.json, ensemble_latest.json, docs/data.csv and the
    dashboard copy -- all derived from the index, one writer."""
    idx = idx or load_index(repo)
    cur = derive_current(idx)
    _write_bytes_atomic(_p(repo, CURRENT_REL), record_bytes(cur))
    if cur["latest_date"] is not None:
        e = current_map(idx)[cur["latest_date"]]
        _rec, raw = reopen_revision(repo, e)
        _write_bytes_atomic(_p(repo, LATEST_REL), raw)
    write_data_csv(repo, idx)
    return cur


# ------------------------------------------------------------ data.csv --
def _csv_rows_for_record(rec):
    """The dashboard rows a record implies (date,region,tier,risk,
    confidence,methods,agreement), in the record's region order."""
    rows = []
    for region, r in rec["regions"].items():
        rows.append([rec["date"], region, str(int(r["tier"])),
                     f"{float(r['combined_risk']):.4f}",
                     f"{float(r['confidence']):.2f}",
                     str(int(r["methods_available"])),
                     r.get("agreement") or ""])
    return rows


def _read_csv_rows(path):
    if not os.path.exists(path):
        return None, []
    with io.open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    return rows[0], rows[1:]


def derive_csv_rows(existing_header, existing_rows, idx):
    """FROZEN rows (dates with no revision) kept byte-for-byte in their
    order, then for every modelled date (sorted) the CURRENT revision's
    rows. Modelled dates' OLD rows are the ones replaced -- that is the
    single conflict rule."""
    header = existing_header or CSV_HEADER
    if header != CSV_HEADER:
        raise RevisionRefusal(
            f"CSV_HEADER_MISMATCH: {header!r} is not the registered "
            "dashboard header")
    modelled = set(current_map(idx))
    frozen = [r for r in existing_rows
              if not (len(r) >= 1 and r[0] in modelled)]
    return header, frozen, sorted(modelled)


def write_data_csv(repo, idx=None):
    idx = idx or load_index(repo)
    path = _p(repo, CSV_REL)
    header, old_rows = _read_csv_rows(path)
    header, frozen, modelled_dates = derive_csv_rows(header, old_rows, idx)
    cur = current_map(idx)
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(header)
    for r in frozen:
        w.writerow(r)
    for d in modelled_dates:
        rec, _raw = reopen_revision(repo, cur[d])
        for r in _csv_rows_for_record(rec):
            w.writerow(r)
    data = out.getvalue().encode("utf-8")
    # never-shrink on the FROZEN part: every non-modelled old row is kept
    kept = len(frozen)
    old_non_modelled = sum(1 for r in old_rows
                           if not (len(r) >= 1 and r[0] in set(cur)))
    if kept != old_non_modelled:
        raise RevisionRefusal("CSV_FROZEN_ROWS_CHANGED: the frozen history "
                              "would not be preserved")
    _write_bytes_atomic(path, data)
    dash = _p(repo, DASHBOARD_CSV_REL)
    if os.path.exists(os.path.dirname(dash)):
        _write_bytes_atomic(dash, data)
    return path


# ---------------------------------------------------------- persistence --
def prior_days(repo, idx, target_date_str, days):
    """For days_back = 1..days: (date, record-or-None, identity-or-None).
    Only PUBLIC revisions count; a missing day is a hole."""
    y, m, d = (int(x) for x in target_date_str.split("-"))
    t0 = datetime(y, m, d, tzinfo=timezone.utc)
    out = []
    for k in range(1, days + 1):
        ds = (t0 - _timedelta_days(k)).strftime("%Y-%m-%d")
        rec, ident = prior_revision(repo, idx, ds)
        out.append((ds, rec, ident))
    return out


def _timedelta_days(k):
    from datetime import timedelta
    return timedelta(days=k)


# ------------------------------------------------------------ selftest --
def _selftest():
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="ens-rev-selftest-")
    try:
        repo = tmp
        os.makedirs(_p(repo, "docs"), exist_ok=True)
        os.makedirs(_p(repo, "monitoring/dashboard"), exist_ok=True)
        # frozen pre-model history: two dates, one of them the day we
        # will later re-score (must NOT be touched: it is pre-model)
        old = ("date,region,tier,risk,confidence,methods,agreement\n"
               "2026-08-30,a,1,0.5000,0.50,1,single_method\n"
               "2026-08-30,b,0,0.0100,0.50,1,single_method\n"
               "2026-08-31,a,0,0.0200,0.50,1,single_method\n")
        with io.open(_p(repo, CSV_REL), "wb") as f:
            f.write(old.encode())

        def rec(date, tiers):
            return {"date": date, "regions": {
                r: {"tier": t, "combined_risk": 0.1 * (t + 1),
                    "confidence": 0.5, "methods_available": 1,
                    "agreement": "single_method"} for r, t in tiers.items()},
                    "summary": {}}
        inputs = {"code": "c" * 64}
        e1 = publish_revision(repo, rec("2026-09-01", {"a": 1, "b": 0}),
                              inputs, created_utc="2026-09-02T00:00:00Z")
        assert e1["run_id"] == "r01" and e1["supersedes"] is None
        assert os.path.exists(_p(repo, e1["path"]))
        # duplicate default run refuses, writes nothing
        try:
            publish_revision(repo, rec("2026-09-01", {"a": 2, "b": 0}),
                             inputs)
            raise AssertionError("duplicate accepted")
        except RevisionRefusal as x:
            assert "REVISION_EXISTS" in str(x), x
        assert len(load_index(repo)["revisions"]) == 1
        # rescore without prior refuses
        try:
            publish_revision(repo, rec("2026-09-03", {"a": 0}), inputs,
                             rescore_reason="x")
            raise AssertionError("rescore without prior accepted")
        except RevisionRefusal as x:
            assert "RESCORE_WITHOUT_PRIOR" in str(x), x
        # empty reason refuses
        try:
            publish_revision(repo, rec("2026-09-01", {"a": 2}), inputs,
                             rescore_reason="  ")
            raise AssertionError("empty reason accepted")
        except RevisionRefusal as x:
            assert "RESCORE_REASON_EMPTY" in str(x), x
        # csv derived: frozen rows intact, modelled date appended
        h, rows = _read_csv_rows(_p(repo, CSV_REL))
        assert h == CSV_HEADER
        assert rows[:3] == [r.split(",") for r in old.splitlines()[1:]], rows
        assert rows[3:] == [["2026-09-01", "a", "1", "0.2000", "0.50", "1",
                             "single_method"],
                            ["2026-09-01", "b", "0", "0.1000", "0.50", "1",
                             "single_method"]], rows[3:]
        # latest derived == byte copy of the revision
        with io.open(_p(repo, LATEST_REL), "rb") as f:
            latest = f.read()
        with io.open(_p(repo, e1["path"]), "rb") as f:
            assert latest == f.read()
        # rescore: appends r02 naming r01, rewrites ONLY that date's rows
        e2 = publish_revision(repo, rec("2026-09-01", {"a": 2, "b": 0}),
                              {"code": "d" * 64}, rescore_reason="input fix",
                              created_utc="2026-09-02T01:00:00Z")
        assert e2["run_id"] == "r02" and e2["supersedes"] == "r01"
        idx = load_index(repo)
        assert [e["run_id"] for e in idx["revisions"]] == ["r01", "r02"]
        assert idx["revisions"][0] == e1, "append-only violated"
        r2, _ = reopen_revision(repo, e2)
        assert r2["revision"]["supersedes_inputs"] == inputs
        h, rows = _read_csv_rows(_p(repo, CSV_REL))
        assert rows[:3] == [r.split(",") for r in old.splitlines()[1:]]
        assert rows[3][2] == "2" and len(rows) == 5, rows
        cur = derive_current(idx)
        assert cur["current"]["2026-09-01"]["run_id"] == "r02"
        assert cur["latest_date"] == "2026-09-01"
        # persistence sees the CURRENT revision, and a hole for 08-31
        pd = prior_days(repo, idx, "2026-09-02", 2)
        assert pd[0][0] == "2026-09-01" and pd[0][2]["run_id"] == "r02"
        assert pd[1] == ("2026-08-31", None, None), pd[1]
        # tampered committed revision refuses on reopen
        with io.open(_p(repo, e2["path"]), "ab") as f:
            f.write(b" ")
        try:
            reopen_revision(repo, e2)
            raise AssertionError("tamper accepted")
        except RevisionRefusal as x:
            assert "REVISION_DIGEST_MISMATCH" in str(x), x
        # a doctored index (reordered) refuses on load
        bad = {"schema": INDEX_SCHEMA,
               "revisions": [idx["revisions"][1], idx["revisions"][0]]}
        try:
            validate_index(bad)
            raise AssertionError("reordered index accepted")
        except RevisionRefusal as x:
            assert "REVISION_INDEX_ORDER" in str(x), x
        print("ensemble_revisions selftest: ALL PASS (temp store; nothing "
              "scored; nothing public touched)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--selftest"]:
        raise SystemExit(_selftest())
    raise SystemExit("usage: ensemble_revisions_cayley.py --selftest "
                     "(the runner imports this module; there is no "
                     "production CLI)")
