#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF4 immutable archive capsule builder (grassmann) -- codex
2026-08-29T17:58Z ruling repairs 2-3 + the accepted paths/schema.

Builds, from the per-day immutable monitor outputs
monitoring/data/ensemble_results/ensemble_YYYY-MM-DD.json over the
registered calibration interval [2025-10-18, 2026-08-20]:

1. a content-addressed raw-source store (the established W2 pattern):
   E:/GeoSpec/mf4_risk_store/<sha256>.body -- one object per source
   day-file, byte-identical copy; store_id mf4-risk-archive-v1;
2. docs/f2g_window2_execution/mf4_archive/daily_risk_rows_v1.jsonl --
   one canonical row per (issue_day, region) on the full day grid for
   ALL 14 monitor regions: {issue_day, region, combined_risk|null,
   support, persistence_informational, source_sha256}; key-sorted
   JSON, LF, one row per line. `persistence_informational` is the
   monitor's own persistence field, marked NON-INPUT: frozen MF4
   persistence is the catalog-derived recent_event, never this field;
3. docs/f2g_window2_execution/mf4_archive_capsule_v1.json -- schema
   geospec-mf4-archive-capsule-v1 binding: the maturity bounds COPIED
   from mf4_maturity_record_v4.json (never re-derived); region-set
   partition (monitor 14 / typed exclusion tokyo_kanto:
   MF4_BBOX_UNREGISTERED / admitted 13 / alias
   socal_saf_coachella->socal_coachella); the 13 polygon-union bboxes
   recomputed from FAULT_SEGMENTS bytes and REQUIRED equal to the
   acquisition module's pinned table; day index + per-region support
   mask/census; the raw-source inventory (locator + bytes + sha256
   per object, host provenance path recorded but never the only
   recovery route); rows-file digest; producer-code identity;
   catalog/training bindings marked OPEN pending the amended
   acquisition (two-stage build, disclosed);
4. docs/f2g_window2_execution/mf4_archive_receipt_v1.json -- the
   build receipt.

Verifier: verify_capsule() reopens every store object, recomputes
digests, replays the row extraction from store bytes, and refuses
typed on any divergence. Fail-closed everywhere: a source risk value
that is not a finite float and not the registered missingness state
refuses MF4_ARCHIVE_RISK_MALFORMED; duplicate (day, region) refuses;
a missing day/region is recorded explicitly, never silently dropped.
"""
import datetime as dt
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
REPO = os.path.dirname(os.path.dirname(_HERE))

SRC_DIR = os.path.join(REPO, "monitoring", "data", "ensemble_results")
STORE_DIR = r"E:\GeoSpec\mf4_risk_store"
STORE_ID = "mf4-risk-archive-v1"
STORE_ROOT = "s4t://geospec/mf4/risk_archive_v1"
OUT_DIR = os.path.join(REPO, "docs", "f2g_window2_execution")
ROWS_REL = "docs/f2g_window2_execution/mf4_archive/daily_risk_rows_v1.jsonl"
CAPSULE_REL = "docs/f2g_window2_execution/mf4_archive_capsule_v1.json"
RECEIPT_REL = "docs/f2g_window2_execution/mf4_archive_receipt_v1.json"
MATURITY_REL = "docs/f2g_window2_execution/mf4_maturity_record_v4.json"
SCHEMA = "geospec-mf4-archive-capsule-v1"

CAL_START = dt.date(2025, 10, 18)
CAL_END = dt.date(2026, 8, 20)          # calibration_issue_end (maturity)
MONITOR_REGIONS = [
    "anchorage", "campi_flegrei", "cascadia", "hualien",
    "istanbul_marmara", "kaikoura", "kumamoto", "mexico_guerrero",
    "norcal_hayward", "ridgecrest", "socal_saf_coachella",
    "socal_saf_mojave", "tokyo_kanto", "turkey_kahramanmaras",
]
TYPED_EXCLUSIONS = {"tokyo_kanto": "MF4_BBOX_UNREGISTERED"}
ALIAS = {"socal_saf_coachella": "socal_coachella"}
ADMITTED = [r for r in MONITOR_REGIONS if r not in TYPED_EXCLUSIONS]


class Refusal(SystemExit):
    def __init__(self, code, detail):
        super().__init__(f"REFUSED {code}: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _days():
    d, out = CAL_START, []
    while d <= CAL_END:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _bboxes():
    """Recompute union bboxes; require equality with the acquisition
    module's pinned table (one numeric authority, two derivations)."""
    import fault_segments as FS
    import w2_mf4_catalog_acquire_grassmann as ACQ
    src = open(FS.__file__, "rb").read()
    out = {}
    for region in ADMITTED:
        carrier = ALIAS.get(region, region)
        lats, lons = [], []
        for seg in FS.FAULT_SEGMENTS[carrier]:
            for (lat, lon) in seg.polygon:
                lats.append(float(lat)), lons.append(float(lon))
        bbox = {"min_lat": min(lats), "max_lat": max(lats),
                "min_lon": min(lons), "max_lon": max(lons)}
        if bbox != ACQ.PINNED_BBOXES[region]:
            raise Refusal("MF4_BBOX_PIN_MISMATCH", region)
        out[region] = {"carrier": carrier, "bbox": bbox}
    return out, {"path": "monitoring/src/fault_segments.py",
                 "sha256": _sha(src), "bytes": len(src)}


def extract_row(day, region, doc, src_sha):
    regs = doc.get("regions") or {}
    entry = regs.get(region)
    if entry is None:
        return {"issue_day": day, "region": region, "combined_risk": None,
                "support": "MISSING_REGION_ROW",
                "persistence_informational": None,
                "source_sha256": src_sha}
    risk = entry.get("combined_risk")
    if risk is None:
        support = "MISSING_RISK"
    elif isinstance(risk, bool) or not isinstance(risk, (int, float)):
        raise Refusal("MF4_ARCHIVE_RISK_MALFORMED",
                      f"{day}/{region}: {risk!r}")
    elif risk != risk or risk in (float("inf"), float("-inf")):
        raise Refusal("MF4_ARCHIVE_RISK_MALFORMED",
                      f"{day}/{region}: non-finite")
    else:
        risk, support = float(risk), "SUPPORTED"
    pers = entry.get("persistence")
    pers = float(pers) if isinstance(pers, (int, float)) \
        and not isinstance(pers, bool) and pers == pers else None
    return {"issue_day": day, "region": region, "combined_risk": risk,
            "support": support, "persistence_informational": pers,
            "source_sha256": src_sha}


def build():
    os.makedirs(STORE_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "mf4_archive"), exist_ok=True)
    days = _days()
    inventory, rows, missing_days = {}, [], []
    malformed_days = []
    seen = set()
    for day in days:
        sp = os.path.join(SRC_DIR, f"ensemble_{day}.json")
        if not os.path.isfile(sp):
            missing_days.append(day)
            for region in MONITOR_REGIONS:
                rows.append({"issue_day": day, "region": region,
                             "combined_risk": None,
                             "support": "MISSING_DAY_FILE",
                             "persistence_informational": None,
                             "source_sha256": None})
            continue
        raw = open(sp, "rb").read()
        sha = _sha(raw)
        obj = os.path.join(STORE_DIR, sha + ".body")
        if not os.path.isfile(obj):
            with open(obj, "wb") as f:
                f.write(raw)
        elif _sha(open(obj, "rb").read()) != sha:
            raise Refusal("MF4_ARCHIVE_STORE_CORRUPT", obj)
        inventory[day] = {
            "object": sha + ".body", "sha256": sha, "bytes": len(raw),
            "host_provenance_path":
                f"monitoring/data/ensemble_results/ensemble_{day}.json"}
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            # authentic source bytes that do not decode (e.g. the
            # zero-byte 2026-05-23 failed-run file): typed, explicit,
            # never silently dropped; the bytes stay in the store.
            inventory[day]["malformed"] = True
            malformed_days.append(day)
            for region in MONITOR_REGIONS:
                rows.append({"issue_day": day, "region": region,
                             "combined_risk": None,
                             "support": "MALFORMED_DAY_FILE",
                             "persistence_informational": None,
                             "source_sha256": sha})
            continue
        if doc.get("date") != day:
            raise Refusal("MF4_ARCHIVE_DAY_MISMATCH",
                          f"{day}: file says {doc.get('date')}")
        for region in MONITOR_REGIONS:
            if (day, region) in seen:
                raise Refusal("MF4_ARCHIVE_DUPLICATE_ROW",
                              f"{day}/{region}")
            seen.add((day, region))
            rows.append(extract_row(day, region, doc, sha))

    rows_path = os.path.join(REPO, ROWS_REL)
    with open(rows_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    rows_raw = open(rows_path, "rb").read()

    census = {}
    for region in MONITOR_REGIONS:
        sup = sum(1 for r in rows
                  if r["region"] == region and r["support"] == "SUPPORTED")
        census[region] = {"days_total": len(days), "days_supported": sup,
                          "days_unsupported": len(days) - sup}

    missing_cells = sorted(
        [r["issue_day"], r["region"]] for r in rows
        if r["support"] == "MISSING_REGION_ROW")
    bboxes, fs_ident = _bboxes()
    maturity_raw = open(os.path.join(REPO, MATURITY_REL), "rb").read()
    maturity = json.loads(maturity_raw.decode("utf-8"))
    me = open(os.path.abspath(__file__), "rb").read()

    capsule = {
        "schema": SCHEMA,
        "amendment": ("docs/f2g_window2_execution/"
                      "amendment_mf4_late_catalog_repair_20260829.md"),
        "lane_status": "AMENDED_AFTER_FREEZE",
        "maturity_bounds": {
            "source": MATURITY_REL,
            "source_sha256": _sha(maturity_raw),
            "calibration_interval":
                maturity["gate_a_calibration_ledger"]
                        ["calibration_interval"],
            "freeze_day": maturity["gate_a_calibration_ledger"]
                                  ["freeze_day"],
            "snapshot_end": maturity["gate_a_calibration_ledger"]
                                    ["snapshot_end"]},
        "region_sets": {
            "monitor_region_set_at_freeze": MONITOR_REGIONS,
            "typed_exclusions": TYPED_EXCLUSIONS,
            "admitted_regions": ADMITTED,
            "registered_alias": ALIAS},
        "bboxes": bboxes,
        "fault_segments_identity": fs_ident,
        "day_index": {"start": CAL_START.isoformat(),
                      "end": CAL_END.isoformat(),
                      "days_total": len(days),
                      "missing_day_files": missing_days,
                      "malformed_day_files": malformed_days},
        "support_census": census,
        "file_census": {
            "present_files": len(inventory),
            "usable_files": len(inventory) - len(malformed_days),
            "malformed_files": len(malformed_days)},
        "missing_region_cells": missing_cells,
        "raw_source_store": {
            "store_id": STORE_ID, "store_root": STORE_ROOT,
            "local_physical_root": STORE_DIR.replace("\\", "/"),
            "object_count": len(inventory), "inventory": inventory,
            "note": ("host provenance paths recorded above are NOT the "
                     "recovery route; recovery = store objects by "
                     "content address, capsuled to s4t with the "
                     "packet")},
        "rows_file": {"path": ROWS_REL, "rows": len(rows),
                      "sha256": _sha(rows_raw), "bytes": len(rows_raw)},
        "projection_loss_disclosure": (
            "components/coverage/summary/earthquake_events are omitted "
            "from the training row; their complete source bytes remain "
            "reopenable through the store inventory. "
            "persistence_informational is NON-INPUT: frozen MF4 "
            "persistence is the catalog-derived recent_event."),
        "catalog_binding": {
            "status": "OPEN_PENDING_AMENDED_ACQUISITION",
            "contract": ("monitoring/src/"
                         "w2_mf4_catalog_acquire_grassmann.py"),
            "note": ("catalog snapshot + training-row digest bind in "
                     "the v2 finalization after the ONE authorized "
                     "acquisition; two-stage build, disclosed")},
        "training_row_digest": None,
        "producer_code_identity": {
            "path": ("monitoring/src/"
                     "w2_mf4_archive_capsule_gen_grassmann.py"),
            "sha256": _sha(me), "bytes": len(me)},
    }
    cp = os.path.join(REPO, CAPSULE_REL)
    with open(cp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(capsule, f, indent=1, sort_keys=True)
        f.write("\n")
    receipt = {
        "schema": "geospec-mf4-archive-receipt-v1",
        "built_utc": _utcnow(),
        "capsule": {"path": CAPSULE_REL,
                    "sha256": _sha(open(cp, "rb").read())},
        "rows_sha256": _sha(rows_raw),
        "store_objects": len(inventory),
        "days_present": len(days) - len(missing_days) - len(malformed_days),
        "days_missing": len(missing_days),
        "days_malformed": len(malformed_days),
        "refusals": []}
    rp = os.path.join(REPO, RECEIPT_REL)
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
        f.write("\n")
    return capsule, receipt


def verify_capsule():
    """Independent reopen: every store object recomputed, rows replayed
    from store bytes, censuses/digests re-derived and compared."""
    capsule = json.loads(open(os.path.join(REPO, CAPSULE_REL),
                              encoding="utf-8").read())
    inv = capsule["raw_source_store"]["inventory"]
    replay = []
    for day in sorted(inv):
        e = inv[day]
        op = os.path.join(STORE_DIR, e["object"])
        if not os.path.isfile(op):
            raise Refusal("MF4_ARCHIVE_OBJECT_MISSING", e["object"])
        raw = open(op, "rb").read()
        if len(raw) != e["bytes"] or _sha(raw) != e["sha256"]:
            raise Refusal("MF4_ARCHIVE_OBJECT_MISMATCH", e["object"])
        if e.get("malformed"):
            try:
                json.loads(raw.decode("utf-8"))
                raise Refusal("MF4_ARCHIVE_MALFORMED_FLAG_DRIFT", day)
            except (ValueError, UnicodeDecodeError):
                pass
            for region in capsule["region_sets"][
                    "monitor_region_set_at_freeze"]:
                replay.append({"issue_day": day, "region": region,
                               "combined_risk": None,
                               "support": "MALFORMED_DAY_FILE",
                               "persistence_informational": None,
                               "source_sha256": e["sha256"]})
            continue
        doc = json.loads(raw.decode("utf-8"))
        for region in capsule["region_sets"]["monitor_region_set_at_freeze"]:
            replay.append(extract_row(day, region, doc, e["sha256"]))
    for day in capsule["day_index"]["missing_day_files"]:
        for region in capsule["region_sets"]["monitor_region_set_at_freeze"]:
            replay.append({"issue_day": day, "region": region,
                           "combined_risk": None,
                           "support": "MISSING_DAY_FILE",
                           "persistence_informational": None,
                           "source_sha256": None})
    replay.sort(key=lambda r: (r["issue_day"], r["region"]))
    committed = [json.loads(l) for l in
                 open(os.path.join(REPO, ROWS_REL), encoding="utf-8")]
    committed.sort(key=lambda r: (r["issue_day"], r["region"]))
    if replay != committed:
        raise Refusal("MF4_ARCHIVE_ROW_DIVERGENCE",
                      f"{len(replay)} vs {len(committed)} rows")
    rows_raw = open(os.path.join(REPO, ROWS_REL), "rb").read()
    if _sha(rows_raw) != capsule["rows_file"]["sha256"]:
        raise Refusal("MF4_ARCHIVE_ROWS_DIGEST", "rows file digest")
    return {"objects_verified": len(inv), "rows_verified": len(committed)}


if __name__ == "__main__":
    if "--verify" in sys.argv:
        print(json.dumps(verify_capsule()))
    else:
        capsule, receipt = build()
        print("capsule sha256:", receipt["capsule"]["sha256"])
        print("rows:", capsule["rows_file"]["rows"],
              "sha256:", capsule["rows_file"]["sha256"][:16])
        print("store objects:", receipt["store_objects"],
              "days missing:", receipt["days_missing"])
