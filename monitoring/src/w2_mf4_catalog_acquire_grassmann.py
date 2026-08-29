#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF4 late-calibration-catalog acquisition (grassmann) -- the ONE
receipted ComCat snapshot under the codex 2026-08-29T17:58Z pinned
contract and the append-only repair amendment
docs/f2g_window2_execution/amendment_mf4_late_catalog_repair_20260829.md.
This lane is AMENDED_AFTER_FREEZE: the annex's pre-freeze snapshot
prerequisite was missed and is repaired late, disclosed, never
represented as the original untouched preregistration.

Contract (verbatim bounds from the ruling):
- provider: USGS ComCat FDSN event query
  (https://earthquake.usgs.gov/fdsnws/event/1/query)
- one query per each of the 13 registered polygon-union bboxes
  (FAULT_SEGMENTS vertices; alias socal_saf_coachella->socal_coachella;
  tokyo_kanto typed-excluded MF4_BBOX_UNREGISTERED)
- origin-time superset [2025-10-11T00:00:00Z, 2026-08-28T00:00:00Z];
  local admitted filter 2025-10-11T00:00:00Z <= t < 2026-08-28T00:00:00Z
- minmagnitude=4.0, format=geojson, orderby=time-asc, limit=20000;
  count == limit refuses MF4_CATALOG_QUERY_LIMIT
- event fields kept: id, exact UTC origin time (ms), lat, lon, magnitude
- refusals (typed, fail-closed): query limit hit; missing/duplicate IDs
  within a region; the same ID inconsistent across regions; malformed or
  null coordinates/magnitude/time; event outside the registered
  temporal/spatial filter. Later catalog revisions never enter.
- receipts bind: exact URL + params, request/response UTC times, HTTP
  status, content type, raw response bytes sha256 + byte length + event
  count, acquisition code identity (this file's sha256), bbox identity
  (numeric bbox + carrier + fault_segments source sha256), completeness
  policy (ComCat as-is; per-region completeness caveat disclosed).

Modes:
  --plan  print the 13 exact URLs + the recomputed-vs-pinned bbox check;
          ZERO HTTP.
  --fire  perform the 13 queries (the ONE authorized acquisition) and
          write raw bytes + receipts under
          docs/f2g_window2_execution/mf4_catalog_snapshot/.
Firing is discipline-gated on: the registered amendment, codex's
pre-HTTP review pass, and asylum's in-session go.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
REPO = os.path.dirname(os.path.dirname(_HERE))

PROVIDER_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
T_START = "2025-10-11T00:00:00"       # superset start (UTC, inclusive)
T_END = "2026-08-28T00:00:00"         # superset end (UTC, exclusive)
MINMAG = "4.0"
LIMIT = 20000
ALIAS = {"socal_saf_coachella": "socal_coachella"}
TYPED_EXCLUSIONS = {"tokyo_kanto": "MF4_BBOX_UNREGISTERED"}
ADMITTED = [
    "anchorage", "campi_flegrei", "cascadia", "hualien",
    "istanbul_marmara", "kaikoura", "kumamoto", "mexico_guerrero",
    "norcal_hayward", "ridgecrest", "socal_saf_coachella",
    "socal_saf_mojave", "turkey_kahramanmaras",
]
# The registered numeric bboxes, pinned at amendment time. build_bboxes()
# recomputes them from the FAULT_SEGMENTS bytes and REFUSES on any
# mismatch -- the pinned table and the recompute must agree.
PINNED_BBOXES = {
    "anchorage":            {"min_lat": 60.0,  "max_lat": 62.0,  "min_lon": -151.0, "max_lon": -148.0},
    "campi_flegrei":        {"min_lat": 40.8,  "max_lat": 40.85, "min_lon": 14.05,  "max_lon": 14.2},
    "cascadia":             {"min_lat": 45.0,  "max_lat": 51.0,  "min_lon": -128.0, "max_lon": -121.5},
    "hualien":              {"min_lat": 22.0,  "max_lat": 25.5,  "min_lon": 120.0,  "max_lon": 122.5},
    "istanbul_marmara":     {"min_lat": 40.3,  "max_lat": 41.1,  "min_lon": 27.0,   "max_lon": 31.0},
    "kaikoura":             {"min_lat": -43.0, "max_lat": -41.5, "min_lon": 171.5,  "max_lon": 174.5},
    "kumamoto":             {"min_lat": 32.0,  "max_lat": 34.0,  "min_lon": 130.0,  "max_lon": 132.0},
    "mexico_guerrero":      {"min_lat": 15.5,  "max_lat": 18.5,  "min_lon": -101.0, "max_lon": -97.0},
    "norcal_hayward":       {"min_lat": 37.0,  "max_lat": 38.5,  "min_lon": -123.0, "max_lon": -121.2},
    "ridgecrest":           {"min_lat": 35.3,  "max_lat": 36.0,  "min_lon": -117.9, "max_lon": -117.45},
    "socal_saf_coachella":  {"min_lat": 32.8,  "max_lat": 34.0,  "min_lon": -116.8, "max_lon": -115.2},
    "socal_saf_mojave":     {"min_lat": 33.8,  "max_lat": 36.0,  "min_lon": -118.5, "max_lon": -116.0},
    "turkey_kahramanmaras": {"min_lat": 36.0,  "max_lat": 39.0,  "min_lon": 35.0,   "max_lon": 40.0},
}
OUT_DIR = os.path.join(REPO, "docs", "f2g_window2_execution",
                       "mf4_catalog_snapshot")
COMPLETENESS_POLICY = ("ComCat as-is at snapshot time; per-region "
                       "completeness caveat disclosed; no post-snapshot "
                       "revision enters this calibration snapshot")


class Refusal(SystemExit):
    def __init__(self, code, detail):
        super().__init__(f"REFUSED {code}: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def code_identity():
    raw = open(os.path.abspath(__file__), "rb").read()
    return {"path": "monitoring/src/w2_mf4_catalog_acquire_grassmann.py",
            "sha256": _sha(raw), "bytes": len(raw)}


def build_bboxes():
    """Recompute union bboxes from FAULT_SEGMENTS bytes; refuse on any
    divergence from the pinned numeric table."""
    import fault_segments as FS
    src_raw = open(FS.__file__, "rb").read()
    out = {}
    for region in ADMITTED:
        carrier = ALIAS.get(region, region)
        if carrier not in FS.FAULT_SEGMENTS:
            raise Refusal("MF4_BBOX_UNREGISTERED",
                          f"{region}: carrier {carrier} absent")
        lats, lons = [], []
        for seg in FS.FAULT_SEGMENTS[carrier]:
            for (lat, lon) in seg.polygon:
                lats.append(float(lat)), lons.append(float(lon))
        bbox = {"min_lat": min(lats), "max_lat": max(lats),
                "min_lon": min(lons), "max_lon": max(lons)}
        if bbox != PINNED_BBOXES[region]:
            raise Refusal("MF4_BBOX_PIN_MISMATCH",
                          f"{region}: recomputed {bbox} != pinned")
        out[region] = {"carrier": carrier, "bbox": bbox}
    return out, {"path": "monitoring/src/fault_segments.py",
                 "sha256": _sha(src_raw), "bytes": len(src_raw)}


def query_url(bbox):
    params = [
        ("format", "geojson"), ("starttime", T_START), ("endtime", T_END),
        ("minmagnitude", MINMAG),
        ("minlatitude", repr(bbox["min_lat"])),
        ("maxlatitude", repr(bbox["max_lat"])),
        ("minlongitude", repr(bbox["min_lon"])),
        ("maxlongitude", repr(bbox["max_lon"])),
        ("orderby", "time-asc"), ("limit", str(LIMIT)),
    ]
    return PROVIDER_URL + "?" + urllib.parse.urlencode(params), dict(params)


def validate_events(region, bbox, raw):
    doc = json.loads(raw.decode("utf-8"))
    feats = doc.get("features")
    if not isinstance(feats, list):
        raise Refusal("MF4_CATALOG_MALFORMED", f"{region}: no features list")
    if len(feats) >= LIMIT:
        raise Refusal("MF4_CATALOG_QUERY_LIMIT",
                      f"{region}: count {len(feats)} hit limit {LIMIT}")
    t_lo = dt.datetime(2025, 10, 11, tzinfo=dt.timezone.utc)
    t_hi = dt.datetime(2026, 8, 28, tzinfo=dt.timezone.utc)
    events, seen = [], {}
    for f in feats:
        eid = f.get("id")
        p = f.get("properties") or {}
        g = (f.get("geometry") or {}).get("coordinates")
        if not eid:
            raise Refusal("MF4_CATALOG_MISSING_ID", f"{region}: {f!r:.120}")
        mag, tms = p.get("mag"), p.get("time")
        if (not isinstance(g, list) or len(g) < 2
                or not isinstance(g[0], (int, float))
                or not isinstance(g[1], (int, float))
                or not isinstance(mag, (int, float))
                or not isinstance(tms, int)):
            raise Refusal("MF4_CATALOG_MALFORMED",
                          f"{region}: {eid} null/malformed field")
        t = dt.datetime.fromtimestamp(tms / 1000.0, dt.timezone.utc)
        if not (t_lo <= t < t_hi):
            raise Refusal("MF4_CATALOG_TEMPORAL_FILTER",
                          f"{region}: {eid} at {t.isoformat()}")
        lon, lat = float(g[0]), float(g[1])
        if not (bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]):
            raise Refusal("MF4_CATALOG_SPATIAL_FILTER",
                          f"{region}: {eid} at ({lat},{lon})")
        ev = {"id": eid, "time_ms": tms,
              "time_utc": t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
              "lat": lat, "lon": lon, "mag": float(mag)}
        if eid in seen:
            raise Refusal("MF4_CATALOG_DUPLICATE_ID", f"{region}: {eid}")
        seen[eid] = ev
        events.append(ev)
    return events


def cross_region_consistency(per_region):
    """The same event ID appearing in overlapping bboxes must carry
    byte-identical canonical fields everywhere it appears."""
    seen = {}
    for region, evs in per_region.items():
        for ev in evs:
            key = ev["id"]
            canon = json.dumps(ev, sort_keys=True)
            if key in seen and seen[key][1] != canon:
                raise Refusal("MF4_CATALOG_INCONSISTENT_ID",
                              f"{key}: {seen[key][0]} vs {region}")
            seen.setdefault(key, (region, canon))


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--fire", action="store_true")
    args = ap.parse_args()

    bboxes, fs_identity = build_bboxes()
    ident = code_identity()
    print(f"bbox pin check: 13/13 exact (fault_segments "
          f"{fs_identity['sha256'][:16]})")
    if args.plan:
        for region in ADMITTED:
            url, _ = query_url(bboxes[region]["bbox"])
            print(f"{region}: {url}")
        print("PLAN ONLY -- zero HTTP performed")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    receipts, per_region = {}, {}
    for region in ADMITTED:
        url, params = query_url(bboxes[region]["bbox"])
        t_req = _utcnow()
        with urllib.request.urlopen(url, timeout=120) as resp:
            status = resp.status
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read()
        t_resp = _utcnow()
        if status != 200:
            raise Refusal("MF4_CATALOG_HTTP_STATUS", f"{region}: {status}")
        events = validate_events(region, bboxes[region]["bbox"], raw)
        raw_name = f"raw_{region}.geojson"
        with open(os.path.join(OUT_DIR, raw_name), "wb") as f:
            f.write(raw)
        per_region[region] = events
        receipts[region] = {
            "url": url, "params": params,
            "request_utc": t_req, "response_utc": t_resp,
            "http_status": status, "content_type": ctype,
            "raw_file": raw_name, "raw_bytes": len(raw),
            "raw_sha256": _sha(raw), "event_count": len(events),
            "bbox": bboxes[region]["bbox"],
            "carrier": bboxes[region]["carrier"],
        }
        print(f"{region}: {len(events)} events, {len(raw)} B, "
              f"{_sha(raw)[:12]}")
    cross_region_consistency(per_region)

    snap = {"schema": "geospec-mf4-calibration-catalog-snapshot-v1",
            "amendment": ("docs/f2g_window2_execution/"
                          "amendment_mf4_late_catalog_repair_20260829.md"),
            "lane_status": "AMENDED_AFTER_FREEZE",
            "provider": PROVIDER_URL,
            "superset_utc": [T_START + "Z", T_END + "Z"],
            "minmagnitude": float(MINMAG), "limit": LIMIT,
            "completeness_policy": COMPLETENESS_POLICY,
            "typed_exclusions": TYPED_EXCLUSIONS, "alias": ALIAS,
            "fault_segments_identity": fs_identity,
            "acquisition_code_identity": ident,
            "events_by_region": per_region}
    sp = os.path.join(OUT_DIR, "catalog_snapshot_v1.json")
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
        f.write("\n")
    rec = {"schema": "geospec-mf4-catalog-acquisition-receipt-v1",
           "fired_utc": _utcnow(), "region_receipts": receipts,
           "snapshot_file": "catalog_snapshot_v1.json",
           "snapshot_sha256": _sha(open(sp, "rb").read()),
           "acquisition_code_identity": ident,
           "fault_segments_identity": fs_identity}
    rp = os.path.join(OUT_DIR, "acquisition_receipt_v1.json")
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    print("SNAPSHOT WRITTEN:", sp)
    print("  snapshot sha256:", rec["snapshot_sha256"])
    print("RECEIPT WRITTEN:", rp)


if __name__ == "__main__":
    main()
