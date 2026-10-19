#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAW-FIRST HISTORY for the public dashboard (cayley 2026-10-19; codex bbe31a10, owner "agreed - proceed").

docs/data.csv carries only the combined score. This writer reads every RETAINED daily report AT A COMMIT, by
identity (git objects, never the working tree), and writes the measured values beside the recorded scores:

  legacy dates   docs/ensemble/legacy_baseline_v1.json: each record names the LAST committed parseable blob of its
                 date in first-parent order (the record as last published); it is opened by blob id and checked
                 against the recorded sha256 and length. Earlier distinct blobs of the same date in that order are
                 kept as SUPERSEDED revisions, never merged into the series.
  revision dates docs/ensemble/index.ndjson: the date's last journal entry is current, opened by path at the commit
                 and checked against the journal sha256; earlier entries of the date are SUPERSEDED revisions.
  docs/data.csv  the published combined score, kept beside each day (and the only record for CSV-only dates).

Output (LF, sorted keys, deterministic): <out>/index.json and <out>/<region>.json. Nothing here scores,
recalibrates or rewrites a published value; scores are shown as recorded under the baseline in force that day.

Per method (seismic_thd, lambda_geo, fault_correlation) and day:
  recorded      the report's own fields, verbatim (available, raw_value, raw_secondary, risk_score, frozen, notes)
  raw.value     raw_value ONLY when available is true and it is a finite number; otherwise null
  raw.state     MEASURED | AVAILABLE_NONNUMERIC | UNAVAILABLE_ZERO_PLACEHOLDER (available not true with raw_value 0: a
                legacy placeholder, never a measurement) | UNCONFIRMED_NONZERO (available not true with a finite nonzero
                raw_value: kept as raw.unconfirmed_value, shown flagged, never a measurement or admitted -- as in codex's
                viewer 5d8095ca) | UNAVAILABLE | NOT_IN_REPORT | NO_REPORT
  score.value   the recorded risk_score only when the method was available and it is a finite number
  score.state   RECORDED | WITHHELD (null, with any recorded score_basis) | NONNUMERIC | METHOD_UNAVAILABLE
                | NOT_IN_REPORT | NO_REPORT
  admission     with a method set: its state (VALID, NOT_QUALIFIED, UNAVAILABLE, ...), exclusion code and reason,
                inclusion and effective weight; without one: PRE_QUALIFICATION_RECORD and the recorded effective weight
  calibration   baseline, z_score, calibration, score_basis, sample_rate_hz -- only the keys the report has (an absent
                key stays absent: unknown; no formula is applied to a record that did not carry it)
  station       for THD, the recorded notes' "sta=" text (basis NOTES_TEXT); otherwise absent
  warnings      UNCONFIRMED_MEASUREMENT_STATUS, FROZEN, CALIBRATION_<STATUS> (status other than calibrated), BASELINE_QUALITY_<QUALITY> (other than
                good), ZERO_SAMPLE_BASELINE, NOT_ADMITTED_<STATE> (a measured value its method set excluded), and one
                DERIVED here: REPEATS_PREVIOUS_MEASURED_VALUE (exactly equal, at full precision, to the method's previous
                measured day in this series: a likely carried-forward value; the report's own frozen flag may be false)
Per day: combined.value is the recorded combined_risk only when at least one method carried it; otherwise null with
NO_QUALIFIED_METHOD (a method set with nothing included) or NO_METHOD_AVAILABLE. The CSV row is kept as published.

Usage: python monitoring/src/raw_history_cayley.py --repo <repo> --rev <commit> --out <new empty dir>
       python monitoring/src/raw_history_cayley.py ... --replace-own   (only a directory this writer produced)
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys

SCHEMA = "geospec-raw-history-v1"
REGION_SCHEMA = "geospec-raw-history-region-v1"
CAPSULE_REL = "docs/ensemble/legacy_baseline_v1.json"
JOURNAL_REL = "docs/ensemble/index.ndjson"
CSV_REL = "docs/data.csv"
METHODS = ("seismic_thd", "lambda_geo", "fault_correlation")
# What docs/JSON_SCHEMA.md says each raw field means. Documentation, not a retained record: reports carry no unit.
METHOD_DOCS = {
    "seismic_thd": {"label": "THD", "raw": "THD value (dimensionless)", "secondary": "documented as signal-to-noise ratio"},
    "lambda_geo": {"label": "LG", "raw": "baseline ratio (1.0 = normal)", "secondary": "not documented"},
    "fault_correlation": {"label": "FC", "raw": "L2/L1 eigenvalue ratio", "secondary": "documented as participation ratio"},
}
RECORDED_KEYS = ("available", "raw_value", "raw_secondary", "risk_score", "frozen", "notes")
CALIBRATION_KEYS = ("baseline", "z_score", "calibration", "score_basis", "sample_rate_hz")
COMBINED_KEYS = ("combined_risk", "tier", "tier_name", "methods_available", "agreement", "confidence")
_STATION_RE = re.compile(r"\bsta=([A-Za-z0-9]+\.[A-Za-z0-9]+)")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REGION_RE = re.compile(r"^[a-z0-9_]+$")


class Refused(Exception):
    pass


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo] + list(args), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise Refused("git %s: %s" % (" ".join(args), e.stderr.decode("utf-8", "replace").strip()))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _show(repo, rev, path):
    return _git(repo, "show", rev + ":" + path)


def method_entry(name, comp, region_report):
    """One method on one day, from the report's own fields only."""
    if not isinstance(comp, dict):
        return {"raw": {"value": None, "state": "NOT_IN_REPORT"}, "score": {"value": None, "state": "NOT_IN_REPORT"}}
    available, raw, risk = comp.get("available"), comp.get("raw_value"), comp.get("risk_score")
    if available is True:
        raw_state = "MEASURED" if _finite(raw) else "AVAILABLE_NONNUMERIC"
    elif _finite(raw):
        raw_state = "UNAVAILABLE_ZERO_PLACEHOLDER" if raw == 0 else "UNCONFIRMED_NONZERO"
    else:
        raw_state = "UNAVAILABLE"
    measured = raw_state == "MEASURED"
    secondary = comp.get("raw_secondary")
    if available is not True:
        score_state = "METHOD_UNAVAILABLE"
    elif _finite(risk):
        score_state = "RECORDED"
    elif risk is None:
        score_state = "WITHHELD"
    else:
        score_state = "NONNUMERIC"
    entry = {"recorded": {k: comp[k] for k in RECORDED_KEYS if k in comp},
             "raw": {"value": raw if measured else None, "state": raw_state,
                     "secondary": secondary if measured and _finite(secondary) else None},
             "score": {"value": risk if score_state == "RECORDED" else None, "state": score_state}}
    if raw_state == "UNCONFIRMED_NONZERO":
        entry["raw"]["unconfirmed_value"] = raw
    calibration = {k: comp[k] for k in CALIBRATION_KEYS if k in comp}
    if calibration:
        entry["calibration"] = calibration
    method_set = region_report.get("method_set")
    if isinstance(method_set, dict):
        excluded = (method_set.get("excluded") or {}).get(name) or {}
        admission = {"basis": "METHOD_SET", "state": (method_set.get("states") or {}).get(name),
                     "included": name in (method_set.get("included") or []),
                     "weight": (method_set.get("effective_weights") or {}).get(name),
                     "code": excluded.get("code"), "reason": excluded.get("reason")}
    else:
        weights = region_report.get("effective_weights") or {}
        admission = {"basis": "PRE_QUALIFICATION_RECORD", "state": None, "included": name in weights,
                     "weight": weights.get(name), "code": None, "reason": None}
    entry["admission"] = admission
    notes = comp.get("notes")
    if name == "seismic_thd" and isinstance(notes, str):
        match = _STATION_RE.search(notes)
        if match:
            entry["station"] = {"value": match.group(1), "basis": "NOTES_TEXT"}
    warnings = ["UNCONFIRMED_MEASUREMENT_STATUS"] if raw_state == "UNCONFIRMED_NONZERO" else []
    if comp.get("frozen") is True:
        warnings.append("FROZEN")
    status = (calibration.get("calibration") or {}).get("status") if isinstance(calibration.get("calibration"), dict) else None
    if isinstance(status, str) and status != "calibrated":
        warnings.append("CALIBRATION_" + status.upper())
    baseline = calibration.get("baseline") if isinstance(calibration.get("baseline"), dict) else {}
    quality = baseline.get("quality")
    if isinstance(quality, str) and quality != "good":
        warnings.append("BASELINE_QUALITY_" + quality.upper())
    if baseline.get("n_samples") == 0:
        warnings.append("ZERO_SAMPLE_BASELINE")
    if measured and admission["basis"] == "METHOD_SET" and not admission["included"] and admission["state"]:
        warnings.append("NOT_ADMITTED_" + str(admission["state"]).upper())
    entry["warnings"] = warnings
    return entry


def combined_entry(region_report):
    recorded = {k: region_report[k] for k in COMBINED_KEYS if k in region_report}
    methods_available, risk = region_report.get("methods_available"), region_report.get("combined_risk")
    method_set = region_report.get("method_set")
    if isinstance(methods_available, int) and not isinstance(methods_available, bool) and methods_available > 0 and _finite(risk):
        state, value = "SCORED", risk
    else:
        value = None
        state = ("NO_QUALIFIED_METHOD" if isinstance(method_set, dict) and not method_set.get("included")
                 else "NO_METHOD_AVAILABLE")
    out = {"recorded": recorded, "value": value, "state": state,
           "effective_weights": region_report.get("effective_weights")}
    if isinstance(method_set, dict):
        out["method_set"] = {k: method_set.get(k) for k in ("label", "rule_version", "contract_version",
                                                           "comparability_key", "risk_basis")}
    return out


def _csv_shape(recorded):
    """The report's combined fields as the CSV writer formats them (risk to 4 decimals)."""
    risk = recorded.get("combined_risk")
    return ("%.4f" % risk if _finite(risk) else None, str(recorded.get("tier")),
            str(recorded.get("methods_available")), recorded.get("agreement"))


def day_from_report(report, region, source):
    regions = report.get("regions")
    region_report = regions.get(region) if isinstance(regions, dict) else None
    if not isinstance(region_report, dict):
        return {"source": source, "combined": {"recorded": {}, "value": None, "state": "REGION_NOT_IN_REPORT"},
                "methods": {m: {"raw": {"value": None, "state": "NOT_IN_REPORT"},
                                "score": {"value": None, "state": "NOT_IN_REPORT"}} for m in METHODS}}
    components = region_report.get("components") or {}
    return {"source": source, "combined": combined_entry(region_report),
            "methods": {m: method_entry(m, components.get(m), region_report) for m in METHODS}}


def _parse_csv(data):
    text = data.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows or list(rows[0].keys())[:7] != ["date", "region", "tier", "risk", "confidence", "methods", "agreement"]:
        raise Refused("DATA_CSV_HEADER_UNEXPECTED")
    return rows


def load_reports(repo, rev):
    """{date: {"current": (report, source), "superseded": [(report, source), ...]}} from the commit, by identity."""
    capsule = json.loads(_show(repo, rev, CAPSULE_REL))
    reports = {}
    legacy_dates = set()
    undated = []
    for record in capsule["records"]:
        date = record["date"]
        if date is None and not record.get("parseable"):
            # An unparseable publication names no date (an empty blob); it is listed, never attributed to a day.
            undated.append({k: record.get(k) for k in ("record_path", "commit", "git_blob", "sha256", "byte_length")})
            continue
        if not isinstance(date, str) or not _DATE_RE.match(date):
            raise Refused("CAPSULE_DATE " + date)
        legacy_dates.add(date)
        if not record.get("parseable"):
            reports[date] = {"current": None, "superseded": [],
                             "unparseable": {"kind": "LEGACY_UNPARSEABLE", "commit": record.get("commit"),
                                             "git_blob": record.get("git_blob")}}
            continue
        data = _git(repo, "cat-file", "blob", record["git_blob"])
        if _sha(data) != record["sha256"] or len(data) != record["byte_length"]:
            raise Refused("LEGACY_BLOB_MISMATCH " + date)
        source = {"kind": "LEGACY_LAST_COMMITTED_BLOB", "record_path": record["record_path"], "commit": record["commit"],
                  "git_blob": record["git_blob"], "sha256": record["sha256"]}
        reports[date] = {"current": (json.loads(data), source), "superseded": []}
    # Earlier same-date legacy blobs, in the capsule's own resolution order (first parent, oldest first).
    paths = sorted({r["record_path"] for r in capsule["records"]})
    for path in paths:
        log = _git(repo, "log", "--first-parent", "--reverse", "--format=%H", capsule["cutover_commit"], "--", path).decode().split()
        seen = {}
        for commit in log:
            try:
                blob = _git(repo, "rev-parse", commit + ":" + path).decode().strip()
            except Refused:
                continue
            data = _git(repo, "cat-file", "blob", blob)
            try:
                date = str(json.loads(data).get("date", ""))[:10]
            except (ValueError, AttributeError):
                continue
            if date not in reports or reports[date].get("current") is None:
                continue
            seen.setdefault(date, [])
            if blob not in [b for b, _, _ in seen[date]]:
                seen[date].append((blob, commit, data))
        for date, blobs in seen.items():
            current = reports[date]["current"][1]
            if blobs[-1][0] != current["git_blob"] and current["git_blob"] not in [b for b, _, _ in blobs]:
                raise Refused("LEGACY_RESOLUTION_DISAGREES " + date)
            for blob, commit, data in blobs:
                if blob == current["git_blob"]:
                    continue
                reports[date]["superseded"].append((json.loads(data), {
                    "kind": "LEGACY_EARLIER_COMMITTED_BLOB", "record_path": path, "commit": commit,
                    "git_blob": blob, "sha256": _sha(data)}))
    journal_bytes = _show(repo, rev, JOURNAL_REL)
    by_date = {}
    for line in journal_bytes.decode("utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            by_date.setdefault(entry["date"], []).append(entry)
    for date, entries in by_date.items():
        if date in legacy_dates:
            raise Refused("JOURNAL_DATE_ALSO_LEGACY " + date)
        loaded = []
        for entry in entries:
            data = _show(repo, rev, entry["path"])
            if _sha(data) != entry["sha256"]:
                raise Refused("REVISION_SHA256_MISMATCH " + entry["path"])
            loaded.append((json.loads(data), {"kind": "REVISION", "path": entry["path"], "run_id": entry["run_id"],
                                              "sha256": entry["sha256"], "supersedes": entry.get("supersedes"),
                                              "reason": entry.get("reason")}))
        current = loaded[-1]
        current[1]["kind"] = "REVISION_CURRENT"
        superseded = []
        for report, source in loaded[:-1]:
            source["kind"] = "REVISION_SUPERSEDED"
            superseded.append((report, source))
        reports[date] = {"current": current, "superseded": superseded}
    identity = {"capsule_sha256": _sha(_show(repo, rev, CAPSULE_REL)), "journal_sha256": _sha(journal_bytes),
                "undated_unparseable_capsule_records": undated}
    return reports, identity


def build(repo, rev):
    commit = _git(repo, "rev-parse", rev + "^{commit}").decode().strip()
    reports, identity = load_reports(repo, commit)
    csv_bytes = _show(repo, commit, CSV_REL)
    csv_rows = _parse_csv(csv_bytes)
    csv_by = {}
    for row in csv_rows:
        csv_by[(row["date"], row["region"])] = row
    regions = sorted({r["region"] for r in csv_rows} |
                     {g for v in reports.values() if v.get("current") for g in (v["current"][0].get("regions") or {})})
    for region in regions:
        if not _REGION_RE.match(region):
            raise Refused("REGION_ID " + region)
    dates = sorted({d for d, _ in csv_by} | set(reports))
    files, preservation = {}, {"compared": 0, "equal": 0, "mismatches": []}
    for region in regions:
        days = []
        for date in dates:
            row = csv_by.get((date, region))
            entry = reports.get(date)
            if entry and entry.get("current"):
                report, source = entry["current"]
                day = day_from_report(report, region, source)
                day["superseded"] = [day_from_report(r, region, s) for r, s in entry["superseded"]]
            else:
                if row is None:
                    continue
                source = (entry or {}).get("unparseable") or {"kind": "NO_REPORT"}
                methods_n = int(row["methods"]) if row["methods"].isdigit() else None
                day = {"source": source,
                       "combined": {"recorded": {}, "value": float(row["risk"]) if methods_n else None,
                                    "state": "CSV_ONLY_SCORED" if methods_n else "CSV_ONLY_NO_METHOD"},
                       "methods": {m: {"raw": {"value": None, "state": "NO_REPORT"},
                                       "score": {"value": None, "state": "NO_REPORT"}} for m in METHODS},
                       "superseded": []}
            day["date"] = date
            day["csv"] = {k: row[k] for k in ("risk", "tier", "methods", "agreement")} if row else None
            if row and source.get("kind") in ("LEGACY_LAST_COMMITTED_BLOB", "REVISION_CURRENT"):
                preservation["compared"] += 1
                published = (row["risk"], row["tier"], row["methods"], row["agreement"])
                derived = _csv_shape(day["combined"].get("recorded") or {})
                if published == derived:
                    preservation["equal"] += 1
                else:
                    # The published row is kept as published; the reason it differs is derived, never repaired.
                    matches = [s["source"].get("commit") or s["source"].get("run_id") for s in day["superseded"]
                               if _csv_shape(s["combined"].get("recorded") or {}) == published]
                    reason = ("REGION_NOT_IN_REPORT" if day["combined"]["state"] == "REGION_NOT_IN_REPORT" else
                              "CSV_EQUALS_A_SUPERSEDED_REVISION" if matches else
                              "CSV_EQUALS_NO_RETAINED_REVISION" if day["superseded"] else "NO_OTHER_RETAINED_REVISION")
                    day["csv_disagrees"] = {"reason": reason, "matching_revisions": matches}
                    preservation["mismatches"].append({"date": date, "region": region, "csv": list(published),
                                                       "report": list(derived), "reason": reason,
                                                       "matching_revisions": matches})
            days.append(day)
        for method in METHODS:
            previous = None
            for day in days:
                entry = day["methods"][method]
                if entry["raw"]["state"] != "MEASURED":
                    continue
                value = entry["raw"]["value"]
                if previous is not None and value == previous:
                    entry["warnings"].append("REPEATS_PREVIOUS_MEASURED_VALUE")
                previous = value
        body = {"schema": REGION_SCHEMA, "region": region, "source_commit": commit, "days": days}
        files[region + ".json"] = (json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    reasons = {}
    for item in preservation["mismatches"]:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    preservation["mismatch_reasons"] = dict(sorted(reasons.items()))
    index = {"schema": SCHEMA, "source_commit": commit, "inputs": dict(identity, data_csv_sha256=_sha(csv_bytes)),
             "regions": regions, "dates": {"first": dates[0], "last": dates[-1]},
             "methods": METHOD_DOCS, "documentation": "docs/JSON_SCHEMA.md (meanings only; reports record no unit)",
             "preservation": preservation,
             "files": {name: _sha(data) for name, data in sorted(files.items())}}
    files["index.json"] = (json.dumps(index, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
    return files


def write(files, out, replace_own=False):
    if os.path.exists(out) and os.listdir(out):
        if not replace_own:
            raise Refused("OUT_NOT_EMPTY " + out)
        try:
            with open(os.path.join(out, "index.json"), "rb") as handle:
                prior = json.loads(handle.read())
        except (OSError, ValueError):
            raise Refused("OUT_NOT_PRODUCED_BY_THIS_WRITER " + out)
        own = set(prior.get("files", {})) | {"index.json"}
        if prior.get("schema") != SCHEMA or set(os.listdir(out)) - own:
            raise Refused("OUT_NOT_PRODUCED_BY_THIS_WRITER " + out)
        for name in set(os.listdir(out)) - set(files):
            os.remove(os.path.join(out, name))
    os.makedirs(out, exist_ok=True)
    for name, data in sorted(files.items()):
        tmp = os.path.join(out, "." + name + ".tmp")
        with open(tmp, "xb") as handle:
            handle.write(data)
        os.replace(tmp, os.path.join(out, name))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace-own", action="store_true")
    args = ap.parse_args(argv)
    try:
        files = build(args.repo, args.rev)
        write(files, args.out, args.replace_own)
    except Refused as e:
        print("REFUSED: %s" % e, file=sys.stderr)
        return 2
    index = json.loads(files["index.json"])
    print(json.dumps({"source_commit": index["source_commit"], "regions": len(index["regions"]),
                      "preservation": {k: (len(v) if isinstance(v, list) else v) for k, v in index["preservation"].items()},
                      "bytes": sum(len(v) for v in files.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
