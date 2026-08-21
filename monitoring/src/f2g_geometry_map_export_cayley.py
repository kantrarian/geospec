#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo2graph GEOMETRY-ONLY map export v2 (cayley) -- owner-directed
2026-08-21 (asylum quote sha bb94a28b... 'geometry build', publication
authorized quote sha 55ba6219...), repaired per the codex 1524Z
claims-hygiene pass:

  1. calendar_metadata layer + Phase-B calendar input REMOVED from the
     public payload; private repo paths removed from the public manifest;
     public provenance receipt only (provider identity, query URL,
     receipt sha, CRS, counts, typed absence, non-claims). Full private
     byte receipts go to layers_manifest_private.json (NEVER published).
  2. fault_polygon layer DROPPED (its nine geometries were exact
     duplicates of the coarse segment boxes -- interpretive ambiguity,
     zero geometric content). Locked by KAT.
  3. Upstream coordinate provenance supplied from the candidate pool's
     FDSN station-service receipts (KOERI EIDA net KO, SCEDC/Caltech net
     CI, KOERI net TU): provider, exact query URL (retrieval window in
     the URL), receipt sha256; per-station coordinate_source vocabulary
     (fdsn / v1+fdsn; the sole pure-v1 station is the typed absence with
     NO published coordinate, so every PLOTTED coordinate is
     provider-confirmed).
  4./5. wording: projected Euclidean distance in metres in the listed
     per-carrier UTM CRS (never 'true metres'); single-page Leaflet
     client requiring network access to the pinned CDN + OSM tiles.
  Forbidden-token KAT: the three public files must not contain
  temporal/calendar keys or private pipeline path tokens.

READ DISCIPLINE unchanged: allowlisted r-free sources only; snapshots/,
deltas/, and the sidecar ledger are NEVER opened.
Usage: exporter.py <repo_root>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

TABLES = (
    "data/phase_a_builder_artifact_v1/tables/station.jsonl",
    "data/phase_a_builder_artifact_v1/tables/near.jsonl",
    "data/phase_a_builder_artifact_v1/tables/segment.jsonl",
    "data/phase_a_builder_artifact_v1/tables/member_of.jsonl",
    "data/phase_a_builder_artifact_v1/tables/contains.jsonl",
)
ANCHOR = "docs/evidence_phase_a_result_anchor.json"
POOL_BLOB = "monitoring/src/d2_campaign_v2_candidate_pool.json"
CARRIERS = ("istanbul_marmara", "socal_coachella", "turkey_kahramanmaras")
NON_CLAIMS = ("seismic-network GEOMETRY only -- no measurement data; "
              "method validation status: INCONCLUSIVE; no earthquake "
              "forecast, precursor, or displacement claims; segment "
              "boxes are coarse station-grouping polygons, not fault "
              "traces")
DISCLOSURE = ("No measurement-valued source or measurement artifact is "
              "included in this publication. Source artifacts not "
              "included here cannot be independently reconstructed from "
              "these hashes alone.")
PROVIDER_NAMES = {
    "istanbul_marmara_KO": "KOERI/Bogazici University EIDA FDSN station "
                           "service (network KO)",
    "socal_coachella_CI": "SCEDC (Caltech) FDSN station service "
                          "(network CI)",
    "turkey_kahramanmaras_KO": "KOERI/Bogazici University EIDA FDSN "
                               "station service (network KO)",
}
FORBIDDEN_TOKENS = ("registered_days", "absent_days", "calendar",
                    "snapshot", "deltas", "pyg_sidecar",
                    "evidence_phase_a", "phase_b", "amendment",
                    "docs/", "monitoring/", "data/phase_a", "f2g_phase")
OUT_DIR = "docs/geo2graph_map"

_READ = {}


def _read(repo, rel):
    b = open(os.path.join(repo, rel), "rb").read()
    _READ[rel] = hashlib.sha256(b).hexdigest()
    return b


def _jsonl(repo, rel):
    return [json.loads(l) for l in
            _read(repo, rel).decode("utf-8").splitlines() if l.strip()]


def ring_latlon(poly):
    r = [[float(lon), float(lat)] for lat, lon in poly]
    if r and r[0] != r[-1]:
        r.append(list(r[0]))
    return r


def main(repo):
    stations = _jsonl(repo, TABLES[0])
    near = _jsonl(repo, TABLES[1])
    segments = _jsonl(repo, TABLES[2])
    member_of = _jsonl(repo, TABLES[3])
    _contains = _jsonl(repo, TABLES[4])
    anchor = json.loads(_read(repo, ANCHOR))
    pool_raw = subprocess.check_output(
        ["git", "cat-file", "blob", f"HEAD:{POOL_BLOB}"], cwd=repo)
    pool = json.loads(pool_raw)
    pool_sha = hashlib.sha256(pool_raw).hexdigest()
    src_of = {}
    for ck, c in pool["carriers"].items():
        for seg, rows in c["segments"].items():
            for r in rows:
                src_of[(ck, r["station_id"])] = r.get("source")
    receipts = pool.get("station_metadata_receipts", {})
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
                                       s["station_id"]) in selected,
                 "coordinate_source":
                     src_of.get((s["carrier_key"], s["station_id"]))}
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
                             "reason": "no published coordinate in the "
                                       "provider metadata (typed "
                                       "absence; never invented)"})
            feats.append({"type": "Feature", "geometry": None,
                          "properties": props})
    n_plotted = sum(1 for f in feats if f["geometry"])
    # every PLOTTED coordinate must be provider-confirmed (fdsn family)
    for f in feats:
        if f["geometry"] is not None:
            assert f["properties"]["coordinate_source"] in ("fdsn",
                                                            "v1+fdsn"), \
                f["properties"]
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
    geo = {"type": "FeatureCollection",
           "name": "geo2graph_geometry_v2",
           "features": feats}
    # ---- KATs ----
    assert len(stations) == 110 and n_plotted == 109
    assert [a["station_id"] for a in absences] == ["KO.KHMN"]
    assert excluded == ["KO.KHMN"]
    assert len(near) == 206 and len(segments) == 9
    layer_kinds = {f["properties"]["layer"] for f in feats}
    assert layer_kinds == {"station", "near_edge", "segment_box"}, \
        layer_kinds  # fault_polygon DROPPED (codex 1524 item 2)
    for f in feats:
        g = f["geometry"]
        if not g:
            continue
        pts = [g["coordinates"]] if g["type"] == "Point" else (
            g["coordinates"] if g["type"] == "LineString"
            else g["coordinates"][0])
        for lon, lat in pts:
            assert -125.0 <= lon <= 45.0 and 25.0 <= lat <= 45.0
    # codex 1554 map fix 2: provenance from the PLOTTED (carrier, network)
    # pairs only; every included receipt must be HTTP 200 with no error
    used_pairs = {(f["properties"]["carrier_key"],
                   f["properties"]["network"]) for f in feats
                  if f["properties"]["layer"] == "station"
                  and f["geometry"] is not None}
    used_keys = {f"{ck}_{net}" for ck, net in used_pairs}
    assert used_keys == {"istanbul_marmara_KO", "socal_coachella_CI",
                         "turkey_kahramanmaras_KO"}, used_keys
    prov = {}
    for key in sorted(used_keys):
        rec = receipts[key]
        assert rec["status"] == 200 and rec.get("error") is None, \
            (key, rec["status"], rec.get("error"))
        prov[key] = {"provider": PROVIDER_NAMES[key],
                     "fdsn_station_query_url": rec["url"],
                     "receipt_sha256": rec["sha256"],
                     "http_status": rec["status"]}
    self_lf = hashlib.sha256(open(__file__, "rb").read().replace(
        b"\r\n", b"\n")).hexdigest()
    manifest = {
        "schema": "geo2graph-map-public-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "non_claims": NON_CLAIMS,
        "disclosure": DISCLOSURE,
        "provenance": {
            "coordinate_sources": prov,
            "coordinate_source_vocabulary": {
                "fdsn": "coordinate from the provider's FDSN station "
                        "metadata (retrieval query + receipt hash above)",
                "v1+fdsn": "legacy internal value confirmed by the "
                           "provider's FDSN station metadata",
                "note": "the sole station without provider metadata "
                        "(KO.KHMN) is published as a typed absence with "
                        "NO coordinate -- every plotted coordinate is "
                        "provider-confirmed"},
            "attribution": "Station metadata: SCEDC (Caltech); "
                           "KOERI/Bogazici University EIDA. Map "
                           "tiles: (c) OpenStreetMap contributors. "
                           "Leaflet (BSD-2-Clause), loaded from the "
                           "pinned unpkg CDN at view time.",
        },
        "carrier_metric_crs": crs,
        "distance_semantics": "near_edge distance_m is the projected "
                              "Euclidean distance in metres in the "
                              "listed per-carrier UTM CRS",
        "layers": [
            {"id": "station", "claim_status": "geometry",
             "count_plotted": n_plotted,
             "count_typed_absent": len(absences)},
            {"id": "near_edge", "claim_status": "geometry",
             "count": len(near),
             "note": "k=3 nearest-neighbor geometric proximity edges; "
                     "NOT measurement edges"},
            {"id": "segment_box", "claim_status": "geometry",
             "count": len(segments),
             "note": "coarse station-grouping polygons, not fault "
                     "traces"},
        ],
        "typed_absences": absences,
        "registry": {"pool_stations": len(stations),
                     "selected_stations": len(member_of)},
    }
    outdir = os.path.join(repo, OUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    geo_b = (json.dumps(geo, sort_keys=True, separators=(",", ":"))
             + "\n").encode("utf-8")
    man_b = (json.dumps(manifest, indent=1, sort_keys=True)
             + "\n").encode("utf-8")
    html = HTML_TEMPLATE.replace("__GEOJSON__", json.dumps(geo)) \
        .replace("__MANIFEST__", json.dumps(manifest)) \
        .replace("__BANNER__", NON_CLAIMS)
    html_b = html.encode("utf-8")
    # forbidden-token KAT over ALL FIVE publication files (codex 1554
    # map fix 1: the committed blobs are the publication tree; README +
    # LICENSE must exist beside the generated three and scan clean)
    pub = [("geojson", geo_b), ("manifest", man_b), ("html", html_b)]
    for static in ("README.md", "LICENSE"):
        p = os.path.join(repo, OUT_DIR, static)
        assert os.path.exists(p), f"PUBLICATION_FILE_MISSING: {static}"
        pub.append((static, open(p, "rb").read()))
    for name, blob_ in pub:
        low = blob_.decode("utf-8").lower()
        for tok in FORBIDDEN_TOKENS:
            assert tok.lower() not in low, \
                f"FORBIDDEN_TOKEN in public {name}: {tok}"
    with open(os.path.join(outdir, "geo2graph_geometry.geojson"), "wb") \
            as f:
        f.write(geo_b)
    with open(os.path.join(outdir, "layers_manifest.json"), "wb") as f:
        f.write(man_b)
    with open(os.path.join(outdir, "index.html"), "wb") as f:
        f.write(html_b)
    # PRIVATE byte receipts -- never part of the publication
    private = {"schema": "geo2graph-map-private-receipts-v1",
               "exporter_lf_sha256": self_lf,
               "inputs": [{"path": p, "sha256": _READ[p]}
                          for p in list(TABLES) + [ANCHOR]],
               "pool_blob": {"ref": f"HEAD:{POOL_BLOB}",
                             "sha256": pool_sha},
               "outputs": {"geojson_sha256":
                           hashlib.sha256(geo_b).hexdigest(),
                           "manifest_sha256":
                           hashlib.sha256(man_b).hexdigest(),
                           "html_sha256":
                           hashlib.sha256(html_b).hexdigest()}}
    with open(os.path.join(outdir, "layers_manifest_private.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(private, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"exported {len(feats)} features ({n_plotted} stations + "
          f"{len(absences)} typed absence, {len(near)} edges, "
          f"{len(segments)} segment boxes; fault_polygon + calendar "
          "layers REMOVED) -> public 3 files + private receipts; "
          "forbidden-token KAT PASS")


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
grp("segment boxes (station grouping)", ()=>feats("segment_box").map(f=>
 L.polygon(f.geometry.coordinates[0].map(c=>[c[1],c[0]]),
  {color:CCOL[f.properties.carrier_key]||"#555",weight:1,
   fillOpacity:0.03,dashArray:"6 4"})
  .bindPopup(`<b>${f.properties.segment_name}</b><br>
   ${f.properties.carrier_key} &middot; ${f.properties.n_members}
   selected stations<br><i>coarse station-grouping polygon, not a fault
   trace</i>`)));
grp("proximity edges (kNN, projected)", ()=>feats("near_edge").map(f=>
 L.polyline(f.geometry.coordinates.map(c=>[c[1],c[0]]),
  {color:CCOL[f.properties.carrier_key]||"#555",weight:1,opacity:0.55})
  .bindPopup(`${f.properties.station_a} &harr; ${f.properties.station_b}
   <br>${Math.round(f.properties.distance_m/100)/10} km projected
   Euclidean distance (per-carrier UTM CRS)
   <br><i>geometric proximity edge, NOT a measurement edge</i>`)));
grp("stations", ()=>feats("station").map(f=>{
 const p=f.properties, sel=p.registry_selected;
 return L.circleMarker([f.geometry.coordinates[1],
  f.geometry.coordinates[0]],
  {radius:sel?6:3.5, color:CCOL[p.carrier_key]||"#555", weight:1.5,
   fillColor:CCOL[p.carrier_key]||"#555", fillOpacity:sel?0.85:0.15})
  .bindPopup(`<b>${p.station_id}</b> (${p.network})<br>${p.carrier_key}
   &middot; ${p.segment_name}<br>${sel?"registry-selected":
   "pool member (not selected)"}<br>coordinate source:
   ${p.coordinate_source}`);}));
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
 + `<br><br><span style="color:#666">provenance + per-layer claim
  status: layers_manifest.json (every coordinate provider-confirmed;
  receipts hashed)</span>`;
</script></body></html>
"""

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
