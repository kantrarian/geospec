"""fault2graph Phase A graph builder (cayley) -- codex contract `2e0c7a33` A3.

Builds the canonical graph-bearing surface from pinned identity inputs and
producer-verified matrix artifacts:

  * canonical station/segment/carrier node tables + typed edge tables
    (canonical UTF-8 JSON/JSONL: sorted keys, finite numbers only, one terminal
    LF -- the BYTE-AUTHORITATIVE surface; every export below is derived and may
    never redefine it);
  * daily snapshot G(day) + typed day-over-day delta table (delta_r only for the
    same (campaign_id, carrier_key, station-index, algorithm_id) carrier, same
    unordered pair, valid measurements on BOTH registered days -- otherwise a
    typed NOT_COMPARABLE row; zero is never imputed, absent days never bridged);
  * exports: GeoDataFrames, NetworkX, PyG HeteroData (identity-lossless round
    trips), map render labeled "seismic envelope coherence structure -- not
    displacement".

Boundary rules honored here (A1/A2/A3):
  * matrices enter ONLY through d2_f2g_matrix_producer.verify_matrix_artifact --
    verified immediately before use, reopened, verified again after (injectable
    `verifier` seam for hermetic bars; the PRODUCTION DEFAULT resolves the real
    module, and its absence is a typed refusal, never a silent pass-through);
  * `member_of` edges come ONLY from the selected registry; pool spares exist as
    nodes with typed flags and can carry neither a selected-registry membership
    nor a coherence edge;
  * `near` distances are computed in a PINNED projected metric CRS -- raw
    longitude/latitude degrees REFUSE; geometry and correlation stay separate
    typed fields;
  * `adjacent_to` derives only from the pinned topology input, never from
    visual proximity;
  * r remains a signed dimensionless correlation (unit=1). It is never called
    distance, displacement, motion, or movement.

Phase A only: read-only inputs, no acquisition, no production/registry mutation,
no claims; the Lambda_geo method remains INCONCLUSIVE.

Heavy geo deps (geopandas/shapely/pyproj/city2graph, torch for PyG) are imported
LAZILY so the canonical core stays runnable in a plain numpy environment; a
missing capability is a typed CapabilityUnavailable, never a crash or a silent
skip. city2graph is pinned at 3892a086 (BSD-3) by the environment lock.

REV 2 (codex builder review `17a550be` WORKS-WITH-FIX -- five bounded repairs):
  F1 CRITICAL: the builder now consumes the EXACT frozen v2 plan schema
     (station_registry = {carrier: [{segment_name, station_id,
     ordered_nslc_candidates}]}; no plan topology_version) and joins
     coordinates/network/NSLC from the DIGEST-PINNED candidate pool bytes
     (sha 15d0e32c...); every selected row must match its pool row exactly
     (carrier, segment, ordered NSLC) or the whole build refuses BEFORE output.
     Topology version comes only from the separately pinned topology object.
  F2 MAJOR: heterogeneous round trips no longer quotient relations away --
     NetworkX uses a keyed MultiGraph over station+segment+carrier nodes with
     ALL five A3 edge types and a full inverse; the PyG path routes through the
     pinned city2graph gdf_to_pyg/pyg_to_gdf hetero bridge (A5 DEPEND), never
     local replacement code.
  F3 MAJOR: ingest calls the producer seam with recompute=True immediately
     before consumption (the ONLY defense against an in-bounds self-consistent
     doctor) and recompute=False after (source rehash per producer P6).
  F4 MAJOR: station_index_digest rides every coherence edge and snapshot
     (exactly one per snapshot); a cross-frame delta is typed
     NOT_COMPARABLE/STATION_INDEX_MISMATCH -- same pair + same shape never
     repairs a changed index frame.
  F5 MAJOR: near edges build strictly PER CARRIER with that carrier's pinned
     projected CRS via the pinned city2graph knn builder; the horizontal-axis
     unit conversion factor is applied and recorded so distance_m is TRUE
     metres (EPSG:2263 US-survey-foot trap closed); carrier_key rides every
     edge and cross-carrier geometric edges are structurally impossible.
"""

import hashlib
import importlib
import json
import os

import numpy as np

SCHEMA = "f2g-graph-v1"
SOURCE_CRS = "EPSG:4326"
# A1 pin: the frozen 110-candidate pool (d2_campaign_v2_candidate_pool.json)
POOL_SHA256 = "15d0e32c51c027dc144c5c6d57ec5f100a59374f6248abfc5c56ee38628ddc67"
CITY2GRAPH_PIN = "3892a086fb21c7a7e774d5ab4020d052a827c3b5"
PLAN_ROW_KEYS = {"segment_name", "station_id", "ordered_nslc_candidates"}


class F2GRefusal(Exception):
    def __init__(self, reason_code, detail=""):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


class CapabilityUnavailable(Exception):
    pass


def canon_bytes(obj):
    """Canonical UTF-8 JSON: sorted keys, finite numbers only, one terminal LF."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8") + b"\n"


def canon_jsonl(rows):
    return b"".join(canon_bytes(r) for r in rows)


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---- identity tables (static) --------------------------------------------------------
def _pool_index(pool_bytes, pool_sha256):
    """Parse the DIGEST-PINNED candidate-pool bytes (F1). Returns
    {station_id: (carrier_key, pool_row)}; any drift from the pin refuses."""
    if not isinstance(pool_bytes, (bytes, bytearray)):
        raise F2GRefusal("POOL_BYTES_REQUIRED",
                         "pool must be the pinned BYTES, not a parsed object")
    if sha(bytes(pool_bytes)) != pool_sha256:
        raise F2GRefusal("POOL_DIGEST_MISMATCH",
                         "pool bytes do not hash to the registered pin")
    pool = json.loads(bytes(pool_bytes).decode("utf-8"))
    idx = {}
    for carrier, cinfo in pool["carriers"].items():
        for _segment, prows in cinfo["segments"].items():
            for row in prows:
                if row["station_id"] in idx:
                    raise F2GRefusal("DUPLICATE_POOL_STATION", row["station_id"])
                idx[row["station_id"]] = (carrier, row)
    return idx


def build_station_table(plan, pool_bytes, *, pool_sha256=POOL_SHA256):
    """Station nodes -- REV 2 (codex F1): EXACT v2 plan schema only.
    plan.station_registry maps carrier -> rows carrying ONLY
    {segment_name, station_id, ordered_nslc_candidates}; coordinates, network
    and NSLC truth come from the digest-pinned pool join, and every selected
    row must match its pool row exactly or the whole build refuses (never a
    partial table). Pool spares become typed-flag nodes with NO membership."""
    idx = _pool_index(pool_bytes, pool_sha256)
    reg = plan.get("station_registry")
    if not isinstance(reg, dict) or not all(isinstance(v, list)
                                            for v in reg.values()):
        raise F2GRefusal("PLAN_SCHEMA",
                         "station_registry must map carrier -> row list "
                         "(exact v2 schema)")
    selected = {}
    for carrier, rows in reg.items():
        for r in rows:
            if not isinstance(r, dict) or set(r.keys()) != PLAN_ROW_KEYS:
                raise F2GRefusal("PLAN_ROW_SCHEMA", str(sorted(r))[:80])
            sid = r["station_id"]
            if sid in selected:
                raise F2GRefusal("DUPLICATE_STATION", sid)
            if sid not in idx:
                raise F2GRefusal("POOL_JOIN_MISSING", sid)
            pcarrier, prow = idx[sid]
            if pcarrier != carrier:
                raise F2GRefusal("CARRIER_MISMATCH_POOL",
                                 f"{sid}: plan {carrier} vs pool {pcarrier}")
            if prow["segment"] != r["segment_name"]:
                raise F2GRefusal("SEGMENT_MISMATCH_POOL",
                                 f"{sid}: plan {r['segment_name']} vs pool "
                                 f"{prow['segment']}")
            if list(prow["ordered_nslc"]) != list(r["ordered_nslc_candidates"]):
                raise F2GRefusal("NSLC_MISMATCH_POOL", sid)
            selected[sid] = r["segment_name"]
    rows_out = []
    for sid in sorted(idx):
        carrier, prow = idx[sid]
        sel = sid in selected
        # REAL-DATA WRINKLE (disclosed for codex ratification): the frozen pool
        # carries one SELECTED station (KO.KHMN, turkey) with lon/lat = null.
        # Typed absence, never an invented coordinate: the node exists and may
        # carry membership/coherence (neither needs geometry) but is excluded
        # from the geometric backbone and the render.
        has_xy = prow.get("lon") is not None and prow.get("lat") is not None
        rows_out.append({
            "type": "station", "station_id": sid, "carrier_key": carrier,
            "segment_name": selected.get(sid),
            "pool_segment": prow["segment"], "network": prow["network"],
            "lon": float(prow["lon"]) if has_xy else None,
            "lat": float(prow["lat"]) if has_xy else None,
            "coordinates_available": has_xy,
            "ordered_nslc_candidates": list(prow["ordered_nslc"]),
            "pool_member": True, "registry_selected": sel,
        })
    return rows_out


def build_segment_table(plan, topology):
    """Segments from the EXACT v2 nested registry; topology_version only from
    the separately pinned topology object (F1: the plan has no such field)."""
    segs = {}
    for carrier, rows in plan["station_registry"].items():
        for r in rows:
            segs.setdefault((carrier, r["segment_name"]), []).append(
                r["station_id"])
    out = []
    for (carrier, seg), members in sorted(segs.items()):
        geom = (topology.get("segment_geometry") or {}).get(seg)
        out.append({"type": "segment", "carrier_key": carrier,
                    "segment_name": seg, "member_stations": sorted(members),
                    "topology_version": topology["topology_version"],
                    "polyline": geom})
    return out


def build_carrier_table(plan, topology, capsule_bindings):
    """Carrier nodes; topology_version from the topology object ONLY (F1)."""
    rows = []
    for carrier in sorted(plan["carriers"]):
        cb = capsule_bindings.get(carrier)
        rows.append({"type": "carrier", "carrier_key": carrier,
                     "topology_version": topology["topology_version"],
                     "capsule_expected_sha256": cb and cb["expected_sha256"],
                     "capsule_valid_through": cb and cb["valid_through"],
                     "capsule_threshold": cb and cb["threshold"]})
    return rows


def build_contains_edges(segment_table):
    """(carrier, contains, segment) from the segment table (design note 1.2)."""
    return [{"type": "contains", "carrier_key": s["carrier_key"],
             "segment_name": s["segment_name"]} for s in segment_table]


def build_member_of_edges(station_table):
    """(station, member_of, segment) -- selected-registry stations ONLY (A1)."""
    return [{"type": "member_of", "station_id": r["station_id"],
             "carrier_key": r["carrier_key"], "segment_name": r["segment_name"]}
            for r in station_table if r["registry_selected"]]


def build_adjacent_to_edges(topology, segment_table):
    """(segment, adjacent_to, segment) from the PINNED topology only (A3).
    Segment names resolve to (carrier, segment) via the segment table; an
    ambiguous or unknown name refuses."""
    by_name = {}
    for s in segment_table:
        by_name.setdefault(s["segment_name"], []).append(s["carrier_key"])
    def resolve(name):
        carriers = by_name.get(name)
        if not carriers:
            raise F2GRefusal("UNKNOWN_SEGMENT", name)
        if len(carriers) != 1:
            raise F2GRefusal("AMBIGUOUS_SEGMENT", name)
        return carriers[0]
    rows = []
    for a, b, attrs in topology.get("adjacency", []):
        ca, cb = resolve(a), resolve(b)
        if ca != cb:
            raise F2GRefusal("ADJACENCY_CARRIER_MIX", f"{a}({ca}) vs {b}({cb})")
        rows.append({"type": "adjacent_to", "carrier_key": ca,
                     "segment_a": min(a, b), "segment_b": max(a, b),
                     "along_strike_order": attrs.get("along_strike_order"),
                     "shared_fault": bool(attrs.get("shared_fault", False)),
                     "topology_version": topology["topology_version"]})
    rows.sort(key=lambda r: (r["segment_a"], r["segment_b"]))
    return rows


def build_near_edges(station_table, *, carrier_crs, k=3):
    """(station, near, station) geometric null backbone -- REV 2 (codex F5):
    built strictly PER CARRIER via the pinned city2graph knn builder in that
    carrier's pinned projected CRS. Geographic degrees REFUSE; the horizontal-
    axis unit conversion factor is applied and RECORDED so distance_m is true
    metres; carrier_key rides every edge and cross-carrier geometric edges are
    structurally impossible."""
    try:
        import geopandas as gpd
        from pyproj import CRS
        from shapely.geometry import Point
        import city2graph as c2g
    except ImportError as exc:
        raise CapabilityUnavailable(f"geo deps: {exc}")
    by_carrier = {}
    for r in station_table:
        if not r.get("coordinates_available", True):
            continue          # typed absence: no geometry, no geometric edge
        by_carrier.setdefault(r["carrier_key"], []).append(r)
    out = []
    for carrier in sorted(by_carrier):
        if carrier not in carrier_crs:
            raise F2GRefusal("NO_CRS_FOR_CARRIER", carrier)
        metric_crs = carrier_crs[carrier]
        crs = CRS.from_user_input(metric_crs)
        if crs.is_geographic:
            raise F2GRefusal("GEOGRAPHIC_CRS",
                             f"{carrier}: near needs a projected metric CRS, "
                             f"got {metric_crs}")
        factor = float(crs.axis_info[0].unit_conversion_factor)
        rows = by_carrier[carrier]
        gdf = gpd.GeoDataFrame(
            {"station_id": [r["station_id"] for r in rows]},
            geometry=[Point(r["lon"], r["lat"]) for r in rows],
            crs=SOURCE_CRS).set_index("station_id").to_crs(metric_crs)
        if len(rows) < 2:
            continue
        _nodes, eg = c2g.knn_graph(gdf, k=min(k, len(rows) - 1))
        for (a, b), row in eg.iterrows():
            a2, b2 = min(a, b), max(a, b)
            out.append({"type": "near", "carrier_key": carrier,
                        "station_a": a2, "station_b": b2,
                        "distance_m": round(float(row["weight"]) * factor, 3),
                        "unit_conversion_factor": factor,
                        "source_crs": SOURCE_CRS, "metric_crs": metric_crs,
                        "builder": f"city2graph.knn_graph:k={k}@{CITY2GRAPH_PIN[:8]}"})
    seen = {}
    for e in out:
        seen[(e["carrier_key"], e["station_a"], e["station_b"])] = e
    return [seen[k2] for k2 in sorted(seen)]


# ---- dynamic (per day) ---------------------------------------------------------------
def ingest_matrix(root, matrix_path, manifest_path, *, verifier=None):
    """Consume one producer artifact through the A2 consumer seam: verify
    IMMEDIATELY BEFORE use, reopen both artifacts, verify IMMEDIATELY AFTER.
    Production default resolves the real producer module; its absence is a
    typed refusal (never silent)."""
    if verifier is None:
        try:
            verifier = importlib.import_module(
                "d2_f2g_matrix_producer").verify_matrix_artifact
        except (ImportError, AttributeError) as exc:
            raise F2GRefusal("PRODUCER_SEAM_ABSENT", str(exc))
    # REV 2 (codex F3): pre-use verification MUST recompute -- derivation from
    # the bound source objects is the only defense against an in-bounds,
    # self-consistently rehashed doctor. Post-use recompute=False suffices
    # because that path still rehashes every source (producer P6).
    ok, reasons = verifier(root, matrix_path, manifest_path, recompute=True)
    if not ok:
        raise F2GRefusal("MATRIX_VERIFY_FAILED_PRE", str(reasons))
    with open(manifest_path, "rb") as fh:
        man = json.loads(fh.read().decode("utf-8"))
    r = np.load(matrix_path)
    ok, reasons = verifier(root, matrix_path, manifest_path, recompute=False)
    if not ok:
        raise F2GRefusal("MATRIX_VERIFY_FAILED_POST", str(reasons))
    return r, man


def build_coherence_edges(r, manifest, station_table):
    """(station, coheres_with, station) for one (carrier, day). Only finite
    upper-triangle cells become edges; absence stays absent (never weight 0);
    every endpoint must be a SELECTED station of the manifest's carrier."""
    ids = manifest["station_ids"]
    n = len(ids)
    if r.shape != (n, n):
        raise F2GRefusal("SHAPE_MISMATCH", f"{r.shape} vs {n} stations")
    selected = {s["station_id"] for s in station_table
                if s["registry_selected"]
                and s["carrier_key"] == manifest["carrier_key"]}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            v = r[i, j]
            if not np.isfinite(v):
                continue
            a, b = ids[i], ids[j]
            for s in (a, b):
                if s not in selected:
                    raise F2GRefusal("UNSELECTED_STATION_EDGE",
                                     f"{s} not in {manifest['carrier_key']} "
                                     f"selected registry")
            nov = manifest["n_overlap"][i][j]
            if not nov:
                raise F2GRefusal("EDGE_WITHOUT_SUPPORT", f"({a},{b}) n_overlap=0")
            edges.append({"type": "coheres_with",
                          "campaign_id": manifest["campaign_id"],
                          "carrier_key": manifest["carrier_key"],
                          "day": manifest["day"],
                          "algorithm_id": manifest["algorithm_config_digest"],
                          # REV 2 (codex F4): the index frame is PART of the
                          # comparison carrier and rides every edge
                          "station_index_digest": manifest["station_index_digest"],
                          "station_a": min(a, b), "station_b": max(a, b),
                          "r": float(v), "unit": 1, "n_overlap": int(nov)})
    edges.sort(key=lambda e: (e["station_a"], e["station_b"]))
    return edges


def build_snapshot(campaign_id, carrier_key, day, coherence_edges, station_states,
                   station_index_digest):
    """G(day): one canonical per-day snapshot document. REV 2 (codex F4):
    EXACTLY ONE station-index digest per snapshot; every edge must carry it."""
    for e in coherence_edges:
        if e["carrier_key"] != carrier_key or e["day"] != day \
                or e["campaign_id"] != campaign_id:
            raise F2GRefusal("SNAPSHOT_CARRIER_MIX",
                             f"edge {e['station_a']}-{e['station_b']}")
        if e["station_index_digest"] != station_index_digest:
            raise F2GRefusal("SNAPSHOT_INDEX_MIX",
                             f"edge {e['station_a']}-{e['station_b']} digest "
                             f"differs from the snapshot's")
    return {"schema": SCHEMA, "kind": "daily_snapshot",
            "campaign_id": campaign_id, "carrier_key": carrier_key, "day": day,
            "station_index_digest": station_index_digest,
            "station_states": dict(sorted(station_states.items())),
            "coheres_with": coherence_edges}


def build_delta_table(snap_prev, snap_curr, registered_days):
    """Typed day-over-day deltas (A3). delta_r exists ONLY for the same
    comparison carrier -- same campaign/carrier/algorithm, same unordered pair,
    valid measurement on BOTH days, and prev must be the PREVIOUS REGISTERED
    day. Anything else is a typed NOT_COMPARABLE row; zero is never imputed."""
    day, prev_day = snap_curr["day"], snap_prev["day"]
    if day not in registered_days or prev_day not in registered_days:
        raise F2GRefusal("UNREGISTERED_DAY", f"{prev_day}->{day}")
    idx = registered_days.index(day)
    if idx == 0 or registered_days[idx - 1] != prev_day:
        raise F2GRefusal("NOT_PREVIOUS_REGISTERED_DAY", f"{prev_day} !-> {day}")
    if (snap_prev["campaign_id"], snap_prev["carrier_key"]) != \
            (snap_curr["campaign_id"], snap_curr["carrier_key"]):
        raise F2GRefusal("CARRIER_MISMATCH")
    def key(e):
        return (e["station_a"], e["station_b"])
    prev = {key(e): e for e in snap_prev["coheres_with"]}
    curr = {key(e): e for e in snap_curr["coheres_with"]}
    # REV 2 (codex F4): the station-index frame is part of the comparison
    # carrier -- a changed frame makes EVERY pair typed NOT_COMPARABLE, no
    # matter that the pair labels and matrix shape look alike.
    frame_mismatch = (snap_prev["station_index_digest"]
                      != snap_curr["station_index_digest"])
    rows = []
    for k in sorted(set(prev) | set(curr)):
        a, b = k
        base = {"schema": SCHEMA, "kind": "delta",
                "campaign_id": snap_curr["campaign_id"],
                "carrier_key": snap_curr["carrier_key"],
                "day": day, "previous_day": prev_day,
                "station_a": a, "station_b": b}
        if frame_mismatch:
            rows.append({**base, "comparable": False, "delta_r": None,
                         "reason": "STATION_INDEX_MISMATCH"})
            continue
        if k not in prev:
            rows.append({**base, "comparable": False, "delta_r": None,
                         "reason": "PAIR_ABSENT_PREVIOUS_DAY"})
        elif k not in curr:
            rows.append({**base, "comparable": False, "delta_r": None,
                         "reason": "PAIR_ABSENT_CURRENT_DAY"})
        elif prev[k]["algorithm_id"] != curr[k]["algorithm_id"]:
            rows.append({**base, "comparable": False, "delta_r": None,
                         "reason": "ALGORITHM_MISMATCH"})
        else:
            rows.append({**base, "comparable": True,
                         "algorithm_id": curr[k]["algorithm_id"],
                         "delta_r": float(curr[k]["r"] - prev[k]["r"]),
                         "reason": None})
    return rows


# ---- exports (derived; never redefine the canonical surface) -------------------------
def to_geodataframes(station_table, near_edges):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise CapabilityUnavailable(f"geopandas/shapely: {exc}")
    sg = gpd.GeoDataFrame(
        station_table,
        geometry=[Point(r["lon"], r["lat"]) for r in station_table],
        crs=SOURCE_CRS)
    ng = gpd.GeoDataFrame(near_edges) if near_edges else None
    return {"station": sg, "near": ng}


def from_geodataframe(gdf):
    """Derived-export inverse for identity round trips: drop geometry and undo
    pandas' silent None->NaN coercion in OBJECT (identity) columns only. A NaN
    in a numeric column is left in place so the canonical surface's finite-only
    rule refuses it downstream -- the derived container never gets to redefine
    the canonical surface (A3)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise CapabilityUnavailable(f"pandas: {exc}")
    body = gdf.drop(columns="geometry", errors="ignore")
    numeric = {k: pd.api.types.is_numeric_dtype(body[k]) for k in body.columns}
    rows = []
    for rec in body.to_dict("records"):
        out = {}
        for k, v in rec.items():
            if not numeric[k] and isinstance(v, float) and pd.isna(v):
                out[k] = None            # undo coercion in identity columns only
            else:
                out[k] = v
        rows.append(out)
    return rows


def _node_key(kind, row):
    if kind == "station":
        return ("station", row["station_id"])
    if kind == "segment":
        return ("segment", f"{row['carrier_key']}|{row['segment_name']}")
    return ("carrier", row["carrier_key"])


def _edge_endpoints(name, e):
    """Endpoint node keys for every A3 relation (REV 2, codex F2)."""
    if name in ("coheres_with", "near"):
        return (("station", e["station_a"]), ("station", e["station_b"]))
    if name == "member_of":
        return (("station", e["station_id"]),
                ("segment", f"{e['carrier_key']}|{e['segment_name']}"))
    if name == "adjacent_to":
        return (("segment", f"{e['carrier_key']}|{e['segment_a']}"),
                ("segment", f"{e['carrier_key']}|{e['segment_b']}"))
    if name == "contains":
        return (("carrier", e["carrier_key"]),
                ("segment", f"{e['carrier_key']}|{e['segment_name']}"))
    raise F2GRefusal("UNKNOWN_EDGE_TABLE", name)


def to_networkx(node_tables, edge_tables):
    """REV 2 (codex F2): keyed MultiGraph over station+segment+carrier nodes
    with EVERY A3 edge type -- the relation type is the edge key, so a pair
    carrying both `near` and `coheres_with` keeps both, unmerged."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise CapabilityUnavailable(f"networkx: {exc}")
    G = nx.MultiGraph()
    for kind, rows in node_tables.items():
        for r in rows:
            G.add_node(_node_key(kind, r), **r)
    for name, rows in edge_tables.items():
        for e in rows:
            u, v = _edge_endpoints(name, e)
            for n in (u, v):
                if n not in G:
                    raise F2GRefusal("EDGE_ENDPOINT_ABSENT", f"{name}: {n}")
            G.add_edge(u, v, key=name, **{**e, "edge_table": name})
    return G


def from_networkx(G):
    """Full inverse: reconstructs ALL node and edge tables; canonical-byte
    equality with the originals is the round-trip bar."""
    node_tables = {"station": [], "segment": [], "carrier": []}
    for _n, d in G.nodes(data=True):
        node_tables[d["type"]].append(dict(d))
    node_tables["station"].sort(key=lambda r: r["station_id"])
    node_tables["segment"].sort(key=lambda r: (r["carrier_key"],
                                               r["segment_name"]))
    node_tables["carrier"].sort(key=lambda r: r["carrier_key"])
    edge_tables = {}
    for _u, _v, key, d in G.edges(data=True, keys=True):
        d = dict(d)
        name = d.pop("edge_table")
        if name != key:
            raise F2GRefusal("EDGE_KEY_DRIFT", f"{key} vs {name}")
        edge_tables.setdefault(name, []).append(d)
    sort_keys = {
        "coheres_with": lambda e: (e["station_a"], e["station_b"]),
        "near": lambda e: (e["carrier_key"], e["station_a"], e["station_b"]),
        "member_of": lambda e: (e["station_id"], e["segment_name"]),
        "adjacent_to": lambda e: (e["segment_a"], e["segment_b"]),
        "contains": lambda e: (e["carrier_key"], e["segment_name"]),
    }
    for name in edge_tables:
        edge_tables[name].sort(key=sort_keys[name])
    return node_tables, edge_tables


def _hetero_gdfs(node_tables, edge_tables):
    """The city2graph hetero convention: nodes = {type: gdf}, edges =
    {(src, rel, dst): gdf} -- built FROM the canonical tables (derived)."""
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
    except ImportError as exc:
        raise CapabilityUnavailable(f"geopandas: {exc}")
    st = node_tables["station"]
    nodes = {"station": gpd.GeoDataFrame(
        st, geometry=[Point(r["lon"], r["lat"])
                      if r.get("coordinates_available", True) else None
                      for r in st],
        crs=SOURCE_CRS).set_index("station_id", drop=False)}
    for kind, idcol in (("segment", "segment_name"), ("carrier", "carrier_key")):
        rows = node_tables.get(kind) or []
        if rows:
            nodes[kind] = gpd.GeoDataFrame(
                rows, geometry=[None] * len(rows),
                crs=SOURCE_CRS).set_index(idcol, drop=False)
    edges = {}
    trip = {"coheres_with": ("station", "coheres_with", "station"),
            "near": ("station", "near", "station"),
            "member_of": ("station", "member_of", "segment"),
            "adjacent_to": ("segment", "adjacent_to", "segment"),
            "contains": ("carrier", "contains", "segment")}
    endcols = {"coheres_with": ("station_a", "station_b"),
               "near": ("station_a", "station_b"),
               "member_of": ("station_id", "segment_name"),
               "adjacent_to": ("segment_a", "segment_b"),
               "contains": ("carrier_key", "segment_name")}
    for name, rows in edge_tables.items():
        if not rows:
            continue
        a, b = endcols[name]
        df = pd.DataFrame(rows)
        df.index = pd.MultiIndex.from_arrays([df[a], df[b]])
        edges[trip[name]] = gpd.GeoDataFrame(df, geometry=[None] * len(df),
                                             crs=SOURCE_CRS)
    return nodes, edges


def to_pyg(node_tables, edge_tables):
    """REV 2 (codex F2): the PyG path routes through the PINNED city2graph
    hetero bridge (A5 DEPEND) -- never local replacement code. Attributes
    (r, n_overlap, provenance, distance, CRS) ride as feature/metadata columns
    through gdf_to_pyg; pyg_to_gdf is the inverse."""
    try:
        import city2graph as c2g
    except ImportError as exc:
        raise CapabilityUnavailable(f"city2graph: {exc}")
    nodes, edges = _hetero_gdfs(node_tables, edge_tables)
    try:
        return c2g.gdf_to_pyg(nodes, edges)
    except ImportError as exc:
        raise CapabilityUnavailable(f"torch/torch_geometric: {exc}")


def from_pyg(data):
    try:
        import city2graph as c2g
    except ImportError as exc:
        raise CapabilityUnavailable(f"city2graph: {exc}")
    return c2g.pyg_to_gdf(data)


RENDER_LABEL = "seismic envelope coherence structure -- not displacement"


def render_map(station_table, coherence_edges, out_path, *, title_suffix=""):
    """Map render; the A3-mandated label is NON-NEGOTIABLE and always drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise CapabilityUnavailable(f"matplotlib: {exc}")
    pos = {r["station_id"]: (r["lon"], r["lat"]) for r in station_table
           if r.get("coordinates_available", True)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for e in coherence_edges:
        if e["station_a"] not in pos or e["station_b"] not in pos:
            continue          # typed absence: nothing is drawn at an invented spot
        (x0, y0), (x1, y1) = pos[e["station_a"]], pos[e["station_b"]]
        ax.plot([x0, x1], [y0, y1], lw=0.5 + 2.0 * abs(e["r"]),
                alpha=0.6, color="tab:blue" if e["r"] >= 0 else "tab:red")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=18, color="k", zorder=3)
    ax.set_title(f"{RENDER_LABEL}{(' | ' + title_suffix) if title_suffix else ''}",
                 fontsize=9)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def phase_a_result(*, input_digests, code_digests, output_digests, bar_results,
                   status):
    return {"schema": SCHEMA, "kind": "phase_a_result", "status": status,
            "input_digests": dict(sorted(input_digests.items())),
            "code_digests": dict(sorted(code_digests.items())),
            "output_digests": dict(sorted(output_digests.items())),
            "bar_results": bar_results,
            "non_claims": [
                "no forecast skill follows from representation",
                "seismic coherence is not displacement or tectonic movement",
                "Lambda_geo method remains INCONCLUSIVE",
                "registry-status artifacts only; no publication or claim",
            ]}
