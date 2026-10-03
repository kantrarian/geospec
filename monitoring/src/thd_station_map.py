"""THD station-to-region map for the public dashboard (cayley 2026-09-27; owner-approved presentation change).

Derives docs/thd_station_map.json from committed inputs only; nothing here is typed in by hand except the declared
official-agency links:

  monitoring/src/run_ensemble_daily.py   REGIONS: configured thd_station/thd_network and fallback_*   (read by AST)
  monitoring/src/event_scorer.py         REGION_BOUNDS: the lat/lon box of each region                 (read by AST)
  monitoring/config/station_metadata/    retained FDSN station responses + sources.json (URL, UTC, bytes, sha256)
  docs/ensemble_latest.json              the station the latest report RECORDED (THD notes 'sta=NET.STA')

Distances are declared, never implied:
  centre_km  great-circle km from the station to the midpoint of the region's REGION_BOUNDS box
  box_km     great-circle km to the lat/lon-clamped point of that box; 0 = inside, not a geodesic minimum
Neither is a distance to a fault trace.

    python monitoring/src/thd_station_map.py            # rebuild docs/thd_station_map.json from retained inputs
    python monitoring/src/thd_station_map.py --fetch    # first refresh the retained FDSN responses, then rebuild
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

SCHEMA = "geospec.thd-station-map.v1"
REPO = Path(__file__).resolve().parents[2]
REGIONS_SRC = "monitoring/src/run_ensemble_daily.py"
BOUNDS_SRC = "monitoring/src/event_scorer.py"
META_DIR = "monitoring/config/station_metadata"
REPORT = "docs/ensemble_latest.json"
OUTPUT = "docs/thd_station_map.json"
EARTH_RADIUS_KM = 6371.0088

# One FDSN station query per data centre; the file name is the retained copy under META_DIR.
FDSN_QUERIES = {
    "earthscope.txt": "https://service.iris.edu/fdsnws/station/1/query?net=IU,AK,G,MX"
                      "&sta=TUC,COR,MAJO,ANTO,SNZO,COLA,TATO,SSL,UNM,TLIG&level=station&format=text",
    "ncedc.txt": "https://service.ncedc.org/fdsnws/station/1/query?net=BK&sta=BKS&level=station&format=text",
    "ingv.txt": "https://webservices.ingv.it/fdsnws/station/1/query?net=IV&sta=CAFE&level=station&format=text",
}

# Declared: the official source a reader should use for each region (not derived from any GeoSpec output).
AGENCIES = {
    "USGS": "https://earthquake.usgs.gov/earthquakes/map/",
    "JMA": "https://www.jma.go.jp/jma/indexe.html",
    "GeoNet": "https://www.geonet.org.nz/",
    "AFAD": "https://deprem.afad.gov.tr/",
    "CWA": "https://scweb.cwa.gov.tw/en-US",
    "INGV": "https://www.ov.ingv.it/",
    "SSN": "http://www.ssn.unam.mx/",
}
REGION_AGENCY = {
    "ridgecrest": "USGS", "socal_saf_mojave": "USGS", "socal_saf_coachella": "USGS", "norcal_hayward": "USGS",
    "cascadia": "USGS", "anchorage": "USGS", "tokyo_kanto": "JMA", "kumamoto": "JMA", "kaikoura": "GeoNet",
    "istanbul_marmara": "AFAD", "turkey_kahramanmaras": "AFAD", "hualien": "CWA", "campi_flegrei": "INGV",
    "mexico_guerrero": "SSN",
}

# A station key is NET.STA, or a three-part key such as HINET.N.KI2H (network 'HINET.N'); the first group may hold
# one dot so the whole key is kept (cayley b1765426 defect: the two-part pattern truncated three-part codes).
STA_RE = re.compile(r"\bsta=([A-Z0-9]+(?:\.[A-Z0-9]+)?)\.([A-Z0-9]+)\b")
ATTEMPT_RE = re.compile(r"\bfrom ([A-Z0-9]+(?:\.[A-Z0-9]+)?)\.([A-Z0-9]+)\b")
N_RE = re.compile(r"\bn=(\d+)\b")


class MapError(Exception):
    """A named reason the map could not be built; nothing is written."""


def _literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise MapError("ASSIGNMENT_ABSENT: %s in %s" % (name, path))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def region_distances(lat, lon, bounds):
    (lat0, lat1), (lon0, lon1) = bounds["lat"], bounds["lon"]
    centre = ((lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0)
    if abs(lon - centre[1]) > 180:
        raise MapError("ANTIMERIDIAN_NOT_HANDLED: station lon %r vs box centre %r" % (lon, centre[1]))
    near = (min(max(lat, lat0), lat1), min(max(lon, lon0), lon1))
    inside = near == (lat, lon)
    return dict(centre_km=round(haversine_km(lat, lon, *centre), 1),
                box_km=0.0 if inside else round(haversine_km(lat, lon, *near), 1))


def parse_fdsn_text(text, source):
    """FDSN station text (level=station) -> {NET.STA: [epoch, ...]}."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 8:
            raise MapError("FDSN_ROW_MALFORMED: %s: %r" % (source, line))
        net, sta, lat, lon, elev, site, start, end = cols[:8]
        out.setdefault("%s.%s" % (net, sta), []).append(dict(
            lat=float(lat), lon=float(lon), elevation_m=float(elev), site=site,
            epoch_start=start, epoch_end=end or None, source=source))
    return out


def _epoch_for(epochs, day):
    """The one epoch in force on `day` (YYYY-MM-DD). Several open epochs must agree on coordinates."""
    live = [e for e in epochs if e["epoch_start"][:10] <= day and (e["epoch_end"] is None or e["epoch_end"][:10] > day)]
    if not live:
        return None
    if len({(e["lat"], e["lon"]) for e in live}) != 1:
        raise MapError("EPOCH_COORDINATES_DISAGREE: %r" % (live,))
    return live[-1]


def recorded_station(region_report):
    """(NET.STA or None, basis, baseline_n or None) from the latest report's THD notes."""
    component = (((region_report or {}).get("components") or {}).get("seismic_thd") or {})
    notes = component.get("notes") or ""
    m = STA_RE.search(notes)
    if m:
        n = N_RE.search(notes)
        raw = component.get("raw_value")
        measured = component.get("available") is True and type(raw) in (int, float) and math.isfinite(raw)
        basis = "LATEST_REPORT_NOTES_STA" if measured else "NOTES_STATION_WITHOUT_AVAILABLE_NUMERIC_VALUE"
        return "%s.%s" % m.groups(), basis, int(n.group(1)) if n else None
    m = ATTEMPT_RE.search(notes)
    if m:
        return "%s.%s" % m.groups(), "LATEST_REPORT_NOTES_ATTEMPTED_NO_VALUE", None
    return None, "NOT_RECORDED", None


def build(repo=REPO):
    repo = Path(repo)
    regions = _literal_assignment(repo / REGIONS_SRC, "REGIONS")
    bounds = _literal_assignment(repo / BOUNDS_SRC, "REGION_BOUNDS")
    sources = json.loads((repo / META_DIR / "sources.json").read_text(encoding="utf-8"))
    stations_all = {}
    for name, meta in sorted(sources["files"].items()):
        raw = (repo / META_DIR / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["sha256"] or len(raw) != meta["bytes"]:
            raise MapError("RETAINED_METADATA_HASH_MISMATCH: %s" % name)
        for key, epochs in parse_fdsn_text(raw.decode("utf-8"), name).items():
            stations_all.setdefault(key, []).extend(epochs)
    report = json.loads((repo / REPORT).read_text(encoding="utf-8"))
    day = str(report["date"])

    stations, rows = {}, []
    for region in sorted(regions):
        cfg = regions[region]
        if region not in bounds:
            raise MapError("REGION_BOUNDS_ABSENT: %s" % region)
        primary = "%s.%s" % (cfg.get("thd_network", "CI"), cfg["thd_station"]) if cfg.get("thd_station") else None
        fallback = ("%s.%s" % (cfg.get("fallback_network", "IU"), cfg["fallback_station"])
                    if cfg.get("fallback_station") else None)
        # second configured fallback (run_ensemble_daily reads fallback2_* the same way; cayley b1765426 defect)
        fallback2 = ("%s.%s" % (cfg.get("fallback2_network", "IU"), cfg["fallback2_station"])
                     if cfg.get("fallback2_station") else None)
        rec, basis, n_days = recorded_station((report.get("regions") or {}).get(region))
        distances = {}
        for key in [k for k in (primary, fallback, fallback2, rec) if k]:
            epochs = stations_all.get(key)
            epoch = _epoch_for(epochs, day) if epochs else None
            if epoch is None:
                distances[key] = dict(status="COORDINATES_NOT_ON_FILE")
                continue
            stations[key] = epoch
            distances[key] = region_distances(epoch["lat"], epoch["lon"], bounds[region])
        agency = REGION_AGENCY.get(region)
        rows.append(dict(region=region, name=cfg.get("name", region), configured_primary=primary,
                         configured_fallback=fallback, configured_fallback2=fallback2,
                         recorded_station=rec, recorded_basis=basis,
                         baseline_n_days=n_days, distances=distances,
                         official_source=dict(name=agency, url=AGENCIES[agency]) if agency else None))
    by_station = {}
    for row in rows:
        if row["recorded_station"] and row["recorded_basis"] == "LATEST_REPORT_NOTES_STA":
            by_station.setdefault(row["recorded_station"], []).append(row["region"])
    for row in rows:
        peers = by_station.get(row["recorded_station"], []) if row["recorded_basis"] == "LATEST_REPORT_NOTES_STA" else []
        row["shares_recorded_station_with"] = [r for r in peers if r != row["region"]]
    return dict(
        schema=SCHEMA, report_date=day,
        inputs=dict(regions=REGIONS_SRC + ":REGIONS", bounds=BOUNDS_SRC + ":REGION_BOUNDS", report=REPORT,
                    station_metadata=sources["files"]),
        distance_basis=dict(
            centre_km="great-circle km from the station to the midpoint of the region's REGION_BOUNDS lat/lon box",
            box_km="great-circle km to the lat/lon-clamped point of that box; 0 = inside, not a geodesic minimum",
            note="neither is a distance to a fault trace"),
        stations=dict(sorted(stations.items())), regions=rows)


def fetch(repo=REPO, clock=lambda: dt.datetime.now(dt.timezone.utc)):
    repo = Path(repo)
    files = {}
    for name, url in sorted(FDSN_QUERIES.items()):
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw, effective = resp.read(), resp.geturl()
        (repo / META_DIR / name).write_bytes(raw)
        files[name] = dict(url=url, url_effective=effective, retrieved_utc=clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
                           bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (repo / META_DIR / "sources.json").write_text(json.dumps(dict(files=files), indent=1, sort_keys=True) + "\n",
                                                  encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true", help="refresh the retained FDSN responses first")
    args = ap.parse_args(argv)
    (REPO / META_DIR).mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch()
    body = build()
    (REPO / OUTPUT).write_text(json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                               encoding="utf-8", newline="\n")
    missing = sorted(k for r in body["regions"] for k, d in r["distances"].items() if d.get("status"))
    print("THD_STATION_MAP report_date=%s regions=%d stations=%d coordinates_not_on_file=%s"
          % (body["report_date"], len(body["regions"]), len(body["stations"]), ",".join(missing) or "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
