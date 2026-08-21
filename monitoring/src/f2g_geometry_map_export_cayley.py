#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo2graph GEOMETRY-ONLY map export (cayley) -- owner-directed 2026-08-21
(asylum quote sha bb94a28b..., 'then go and start the geometry build').

Exports the fault2graph Phase-A network GEOMETRY -- and nothing else -- to
GeoJSON + a provenance manifest + a self-contained Leaflet page under
docs/geo2graph_map/. PRIVATE-REPO artifact; publication is separately
gated (asylum go + codex claims-hygiene pass).

READ DISCIPLINE (enforced in code): only the ALLOWED_INPUTS below are
opened. The builder artifact's snapshots/, deltas/, and
pyg_sidecar_ledger.json contain real-graph measurement values (r) and are
NEVER opened here -- the fresh seal's run discipline forbids reading them
before the sealed-run instrument check. Every byte source is recorded in
the manifest with its sha256.

Layers (each provenance-labeled, claim_status explicit):
  station        Phase-A station registry points (110 pool; 35 selected)
  near_edge      Phase-A k=3 kNN geometric edges w/ true metres (206)
  segment_box    candidate-pool segment_polygons (coarse boxes, NOT traces)
  fault_polygon  legacy fault registry (fault_segments.py) attitude/trace
                 context ONLY for the three carriers; its station list is
                 deliberately NOT used
Typed absence: KO.KHMN has coordinates_available=false -> rendered as a
listed absence, never an invented coordinate.
Usage: exporter.py <repo_root>
"""
import hashlib
import json
import os
import sys
import time

ALLOWED_INPUTS = (
    "data/phase_a_builder_artifact_v1/tables/station.jsonl",
    "data/phase_a_builder_artifact_v1/tables/near.jsonl",
    "data/phase_a_builder_artifact_v1/tables/segment.jsonl",
    "data/phase_a_builder_artifact_v1/tables/member_of.jsonl",
    "data/phase_a_builder_artifact_v1/tables/contains.jsonl",
    "docs/evidence_phase_a_result_anchor.json",
    "docs/f2g_phase_b_shared_calendar_v1.json",
    "monitoring/src/fault_segments.py",
)
CARRIERS = ("istanbul_marmara", "socal_coachella", "turkey_kahramanmaras")
NON_CLAIMS = ("fault2graph network GEOMETRY only -- no measurement data; "
              "method validation status: INCONCLUSIVE; no earthquake "
              "forecast, precursor, or displacement claims; segment boxes "
              "are coarse pool polygons, not fault traces")
OUT_DIR = "docs/geo2graph_map"

_READ = {}


def _read(repo, rel):
    assert rel in ALLOWED_INPUTS, f"READ_DISCIPLINE: {rel} is not allowlisted"
    b = open(os.path.join(repo, rel), "rb").read()
    _READ[rel] = hashlib.sha256(b).hexdigest()
    return b


def _jsonl(repo, rel):
    return [json.loads(l) for l in
            _read(repo, rel).decode("utf-8").splitlines() if l.strip()]


def ring_latlon(poly):
    """[(lat,lon)...] -> closed GeoJSON ring [[lon,lat]...]."""
    r = [[float(lon), float(lat)] for lat, lon in poly]
    if r and r[0] != r[-1]:
        r.append(list(r[0]))
    return r


def main(repo):
    stations = _jsonl(repo, ALLOWED_INPUTS[0])
    near = _jsonl(repo, ALLOWED_INPUTS[1])
    segments = _jsonl(repo, ALLOWED_INPUTS[2])
    member_of = _jsonl(repo, ALLOWED_INPUTS[3])
    _contains = _jsonl(repo, ALLOWED_INPUTS[4])
    anchor = json.loads(_read(repo, ALLOWED_INPUTS[5]))
    calendar = json.loads(_read(repo, ALLOWED_INPUTS[6]))
    _read(repo, ALLOWED_INPUTS[7])  # bytes recorded; imported below
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    import fault_segments as FS

    crs = anchor.get("bar_results", {}).get("carrier_metric_crs", {})
    excluded = anchor.get("geometry_excluded_station_ids", [])
    selected = {(m["carrier_key"], m["station_id"]) for m in member_of}

    feats = []
    absences = []
    coord = {}
    for s in stations:
        props = {"layer": "station", "station_id": s["station_id"],
                 "carrier_key": s["carrier_key"],
                 "segment_name": s["segment_name"],
                 "network": s["network"],
                 "pool_member": s["pool_member"] is True,
                 "registry_selected": (s["carrier_key"],
                                       s["station_id"]) in selected}
        if s.get("coordinates_available") and s["lat"] is not None \
                and s["lon"] is not None:
            coord[(s["carrier_key"], s["station_id"])] = (float(s["lon"]),
                                                          float(s["lat"]))
            feats.append({"type": "Feature", "geometry":
                          {"type": "Point", "coordinates":
                           [float(s["lon"]), float(s["lat"])]},
                          "properties": props})
        else:
            props["typed_absence"] = True
            absences.append({"station_id": s["station_id"],
                             "carrier_key": s["carrier_key"],
                             "segment_name": s["segment_name"],
                             "reason": "coordinates_available=false "
                                       "(typed absence; never invented)"})
            feats.append({"type": "Feature", "geometry": None,
                          "properties": props})
    n_plotted = sum(1 for f in feats if f["geometry"])
    for e in near:
        a = coord[(e["carrier_key"], e["station_a"])]
        b = coord[(e["carrier_key"], e["station_b"])]
        feats.append({"type": "Feature", "geometry":
                      {"type": "LineString", "coordinates": [list(a),
                                                             list(b)]},
                      "properties": {"layer": "near_edge",
                                     "station_a": e["station_a"],
                                     "station_b": e["station_b"],
                                     "distance_m": e["distance_m"],
                                     "carrier_key": e["carrier_key"]}})
    for g in segments:
        feats.append({"type": "Feature", "geometry":
                      {"type": "Polygon",
                       "coordinates": [ring_latlon(g["polyline"])]},
                      "properties": {"layer": "segment_box",
                                     "segment_name": g["segment_name"],
                                     "carrier_key": g["carrier_key"],
                                     "n_members":
                                         len(g["member_stations"])}})
    n_fault = 0
    for ck in CARRIERS:
        for fs in FS.FAULT_SEGMENTS.get(ck, []):
            n_fault += 1
            feats.append({"type": "Feature", "geometry":
                          {"type": "Polygon",
                           "coordinates": [ring_latlon(fs.polygon)]},
                          "properties": {"layer": "fault_polygon",
                                         "name": fs.name,
                                         "region": fs.region,
                                         "strike": fs.strike,
                                         "dip": fs.dip, "rake": fs.rake,
                                         "carrier_key": ck}})
    geo = {"type": "FeatureCollection",
           "name": "geo2graph_geometry_v1",
           "features": feats}
    # ---- KATs ----
    assert len(stations) == 110 and n_plotted == 109
    assert absences == [{"station_id": "KO.KHMN",
                         "carrier_key": "turkey_kahramanmaras",
                         "segment_name": "east_anatolian_central",
                         "reason": absences[0]["reason"]}], absences
    assert excluded == ["KO.KHMN"]
    assert len(near) == 206 and len(segments) == 9
    assert n_fault > 0
    for f in feats:
        g = f["geometry"]
        if not g:
            continue
        pts = [g["coordinates"]] if g["type"] == "Point" else (
            g["coordinates"] if g["type"] == "LineString"
            else g["coordinates"][0])
        for lon, lat in pts:
            assert -125.0 <= lon <= 45.0 and 25.0 <= lat <= 45.0, (f, lon,
                                                                   lat)
    cal_meta = {ck: {"registered_days":
                     len(calendar["carrier_masks"][ck]["registered_days"]),
                     "absent_days":
                     len(calendar["carrier_masks"][ck].get("absent_days",
                                                           []))}
                for ck in CARRIERS if ck in calendar.get("carrier_masks",
                                                         {})}
    self_lf = hashlib.sha256(open(__file__, "rb").read().replace(
        b"\r\n", b"\n")).hexdigest()
    manifest = {
        "schema": "geo2graph-map-layers-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "exporter_lf_sha256": self_lf,
        "non_claims": NON_CLAIMS,
        "read_discipline": "allowlisted r-free sources only; snapshots/, "
                           "deltas/, pyg_sidecar_ledger.json NEVER opened "
                           "(real-graph measurement values, sealed)",
        "inputs": [{"path": p, "sha256": _READ[p]} for p in ALLOWED_INPUTS],
        "carrier_metric_crs": crs,
        "layers": [
            {"id": "station", "claim_status": "geometry",
             "count_plotted": n_plotted, "count_typed_absent":
             len(absences),
             "source": ALLOWED_INPUTS[0]},
            {"id": "near_edge", "claim_status": "geometry",
             "count": len(near), "source": ALLOWED_INPUTS[1],
             "note": "k=3 kNN geometric proximity, true metres in the "
                     "pinned per-carrier CRS; NOT coherence edges"},
            {"id": "segment_box", "claim_status": "geometry",
             "count": len(segments), "source": ALLOWED_INPUTS[2],
             "note": "coarse candidate-pool polygons, not fault traces"},
            {"id": "fault_polygon", "claim_status":
             "legacy-fault-registry", "count": n_fault,
             "source": ALLOWED_INPUTS[7],
             "note": "attitude/trace context only; the legacy station "
                     "list in that module is NOT used"},
            {"id": "calendar_metadata", "claim_status":
             "calendar-metadata", "per_carrier": cal_meta,
             "source": ALLOWED_INPUTS[6]},
        ],
        "typed_absences": absences,
        "registry": {"pool_stations": len(stations),
                     "selected_stations": len(member_of)},
    }
    outdir = os.path.join(repo, OUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "geo2graph_geometry.geojson"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(geo, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    with open(os.path.join(outdir, "layers_manifest.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    html = HTML_TEMPLATE.replace("__GEOJSON__", json.dumps(geo)) \
        .replace("__MANIFEST__", json.dumps(manifest)) \
        .replace("__BANNER__", NON_CLAIMS)
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(html)
    print(f"exported {len(feats)} features "
          f"({n_plotted} stations + {len(absences)} typed absence, "
          f"{len(near)} edges, {len(segments)} segment boxes, "
          f"{n_fault} fault polygons) -> {OUT_DIR}/")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>geo2graph</title>
<link rel="stylesheet"
 href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;height:100%;font-family:system-ui,sans-serif}
 #banner{position:fixed;top:0;left:0;right:0;z-index:1200;
  background:#1a1a2e;color:#ffd166;padding:6px 12px;font-size:12px}
 #map{position:absolute;top:52px;bottom:0;left:0;right:0}
 #side{position:absolute;top:60px;right:8px;z-index:1100;background:#fff;
  border:1px solid #999;border-radius:6px;padding:8px 10px;font-size:12px;
  max-width:270px;box-shadow:0 1px 4px rgba(0,0,0,.3)}
 .sw{display:inline-block;width:10px;height:10px;border-radius:50%;
  margin-right:4px;vertical-align:middle}
</style></head><body>
<div id="banner"><b>GEOMETRY ONLY</b> &mdash; __BANNER__</div>
<div id="map"></div><div id="side"></div>
<script>
const GEO = __GEOJSON__;
const MAN = __MANIFEST__;
const CCOL = {istanbul_marmara:"#2563eb", socal_coachella:"#d97706",
              turkey_kahramanmaras:"#059669"};
const map = L.map("map");
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
 {attribution:"&copy; OpenStreetMap contributors"}).addTo(map);
function feats(layer){return GEO.features.filter(
 f=>f.properties.layer===layer && f.geometry);}
const groups = {};
function grp(name, maker){const g = L.layerGroup(maker()); groups[name]=g;
 g.addTo(map); return g;}
grp("fault polygons (legacy registry)", ()=>feats("fault_polygon").map(f=>
 L.polygon(f.geometry.coordinates[0].map(c=>[c[1],c[0]]),
  {color:"#b91c1c",weight:1,fillOpacity:0.06,dashArray:"2 4"})
  .bindPopup(`<b>${f.properties.name}</b><br>strike ${f.properties.strike}
   / dip ${f.properties.dip} / rake ${f.properties.rake}
   <br><i>legacy fault registry: context only</i>`)));
grp("segment boxes (pool polygons)", ()=>feats("segment_box").map(f=>
 L.polygon(f.geometry.coordinates[0].map(c=>[c[1],c[0]]),
  {color:CCOL[f.properties.carrier_key]||"#555",weight:1,
   fillOpacity:0.03,dashArray:"6 4"})
  .bindPopup(`<b>${f.properties.segment_name}</b><br>
   ${f.properties.carrier_key} &middot; ${f.properties.n_members}
   selected stations<br><i>coarse pool polygon, not a fault trace</i>`)));
grp("near edges (kNN, metres)", ()=>feats("near_edge").map(f=>
 L.polyline(f.geometry.coordinates.map(c=>[c[1],c[0]]),
  {color:CCOL[f.properties.carrier_key]||"#555",weight:1,opacity:0.55})
  .bindPopup(`${f.properties.station_a} &harr; ${f.properties.station_b}
   <br>${Math.round(f.properties.distance_m/100)/10} km
   <br><i>geometric proximity edge, NOT a coherence edge</i>`)));
grp("stations", ()=>feats("station").map(f=>{
 const p=f.properties, sel=p.registry_selected;
 return L.circleMarker([f.geometry.coordinates[1],
  f.geometry.coordinates[0]],
  {radius:sel?6:3.5, color:CCOL[p.carrier_key]||"#555", weight:1.5,
   fillColor:CCOL[p.carrier_key]||"#555", fillOpacity:sel?0.85:0.15})
  .bindPopup(`<b>${p.station_id}</b> (${p.network})<br>${p.carrier_key}
   &middot; ${p.segment_name}<br>${sel?"registry-selected":
   "pool member (not selected)"}`);}));
L.control.layers(null, groups, {collapsed:false}).addTo(map);
const pts = feats("station").map(f=>[f.geometry.coordinates[1],
 f.geometry.coordinates[0]]);
map.fitBounds(L.latLngBounds(pts).pad(0.15));
const carriers = Object.keys(CCOL);
const s = document.getElementById("side");
s.innerHTML = "<b>Carriers</b><br>" + carriers.map(c=>{
 const n = feats("station").filter(f=>f.properties.carrier_key===c);
 const sel = n.filter(f=>f.properties.registry_selected).length;
 return `<span class=sw style="background:${CCOL[c]}"></span>${c}
  &mdash; ${sel} selected / ${n.length} pool`;}).join("<br>")
 + `<br><br><b>Typed absences</b> (listed, never plotted):<br>`
 + MAN.typed_absences.map(a=>`${a.station_id} (${a.carrier_key})`)
   .join("<br>")
 + `<br><br><span style="color:#666">layers manifest: provenance +
  claim_status per layer; every source sha-bound</span>`;
carriers.forEach(c=>{}); // palette legend rendered above
</script></body></html>
"""

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
