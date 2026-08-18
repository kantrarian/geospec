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
"""

import hashlib
import importlib
import json
import os

import numpy as np

SCHEMA = "f2g-graph-v1"
SOURCE_CRS = "EPSG:4326"


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
def build_station_table(plan, pool=None):
    """Station nodes. The plan's selected registry is the ONLY membership
    authority; pool entries not in it become spare nodes (typed flags, no
    membership, no coherence edges)."""
    rows, seen = [], set()
    for r in plan["station_registry"]:
        key = r["station_id"]
        if key in seen:
            raise F2GRefusal("DUPLICATE_STATION", key)
        seen.add(key)
        rows.append({
            "type": "station", "station_id": key,
            "carrier_key": r["carrier_key"], "segment_name": r["segment_name"],
            "network": key.split(".")[0], "lon": float(r["lon"]),
            "lat": float(r["lat"]),
            "ordered_nslc_candidates": list(r["ordered_nslc_candidates"]),
            "pool_member": True, "registry_selected": True,
        })
    for p in (pool or []):
        if p["station_id"] in seen:
            continue
        rows.append({
            "type": "station", "station_id": p["station_id"],
            "carrier_key": p["carrier_key"], "segment_name": None,
            "network": p["station_id"].split(".")[0], "lon": float(p["lon"]),
            "lat": float(p["lat"]),
            "ordered_nslc_candidates": list(p.get("ordered_nslc_candidates", [])),
            "pool_member": True, "registry_selected": False,
        })
    rows.sort(key=lambda r: r["station_id"])
    return rows


def build_segment_table(plan, topology):
    segs = {}
    for r in plan["station_registry"]:
        segs.setdefault((r["carrier_key"], r["segment_name"]), []).append(
            r["station_id"])
    rows = []
    for (carrier, seg), members in sorted(segs.items()):
        geom = (topology.get("segment_geometry") or {}).get(seg)
        rows.append({"type": "segment", "carrier_key": carrier,
                     "segment_name": seg, "member_stations": sorted(members),
                     "topology_version": topology["topology_version"],
                     "polyline": geom})
    return rows


def build_carrier_table(plan, capsule_bindings):
    rows = []
    for carrier in sorted(plan["carriers"]):
        cb = capsule_bindings.get(carrier)
        rows.append({"type": "carrier", "carrier_key": carrier,
                     "topology_version": plan["topology_version"],
                     "capsule_expected_sha256": cb and cb["expected_sha256"],
                     "capsule_valid_through": cb and cb["valid_through"],
                     "capsule_threshold": cb and cb["threshold"]})
    return rows


def build_member_of_edges(station_table):
    """(station, member_of, segment) -- selected-registry stations ONLY (A1)."""
    return [{"type": "member_of", "station_id": r["station_id"],
             "carrier_key": r["carrier_key"], "segment_name": r["segment_name"]}
            for r in station_table if r["registry_selected"]]


def build_adjacent_to_edges(topology):
    """(segment, adjacent_to, segment) from the PINNED topology only (A3)."""
    rows = []
    for a, b, attrs in topology.get("adjacency", []):
        rows.append({"type": "adjacent_to", "segment_a": min(a, b),
                     "segment_b": max(a, b),
                     "along_strike_order": attrs.get("along_strike_order"),
                     "shared_fault": bool(attrs.get("shared_fault", False)),
                     "topology_version": topology["topology_version"]})
    rows.sort(key=lambda r: (r["segment_a"], r["segment_b"]))
    return rows


def build_near_edges(station_table, *, metric_crs, k=3):
    """(station, near, station) geometric null backbone -- KNN in a PINNED
    projected metric CRS. Raw geographic degrees REFUSE (A3)."""
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise CapabilityUnavailable(f"pyproj: {exc}")
    crs = CRS.from_user_input(metric_crs)
    if crs.is_geographic:
        raise F2GRefusal("GEOGRAPHIC_CRS",
                         f"near distances need a projected metric CRS, got {metric_crs}")
    tr = Transformer.from_crs(SOURCE_CRS, crs, always_xy=True)
    pts = {r["station_id"]: tr.transform(r["lon"], r["lat"])
           for r in station_table}
    ids = sorted(pts)
    edges = {}
    for s in ids:
        x0, y0 = pts[s]
        near = sorted(((((pts[t][0] - x0) ** 2 + (pts[t][1] - y0) ** 2) ** 0.5, t)
                       for t in ids if t != s))[:k]
        for d, t in near:
            a, b = min(s, t), max(s, t)
            edges.setdefault((a, b), {"type": "near", "station_a": a,
                                      "station_b": b,
                                      "distance_m": round(float(d), 3),
                                      "source_crs": SOURCE_CRS,
                                      "metric_crs": metric_crs,
                                      "builder": f"knn:k={k}"})
    return [edges[k2] for k2 in sorted(edges)]


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
    ok, reasons = verifier(root, matrix_path, manifest_path, recompute=False)
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
                          "station_a": min(a, b), "station_b": max(a, b),
                          "r": float(v), "unit": 1, "n_overlap": int(nov)})
    edges.sort(key=lambda e: (e["station_a"], e["station_b"]))
    return edges


def build_snapshot(campaign_id, carrier_key, day, coherence_edges, station_states):
    """G(day): one canonical per-day snapshot document."""
    for e in coherence_edges:
        if e["carrier_key"] != carrier_key or e["day"] != day \
                or e["campaign_id"] != campaign_id:
            raise F2GRefusal("SNAPSHOT_CARRIER_MIX",
                             f"edge {e['station_a']}-{e['station_b']}")
    return {"schema": SCHEMA, "kind": "daily_snapshot",
            "campaign_id": campaign_id, "carrier_key": carrier_key, "day": day,
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
    rows = []
    for k in sorted(set(prev) | set(curr)):
        a, b = k
        base = {"schema": SCHEMA, "kind": "delta",
                "campaign_id": snap_curr["campaign_id"],
                "carrier_key": snap_curr["carrier_key"],
                "day": day, "previous_day": prev_day,
                "station_a": a, "station_b": b}
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


def to_networkx(station_table, edge_tables):
    try:
        import networkx as nx
    except ImportError as exc:
        raise CapabilityUnavailable(f"networkx: {exc}")
    G = nx.Graph()
    for r in station_table:
        G.add_node(("station", r["station_id"]), **r)
    for name, rows in edge_tables.items():
        for e in rows:
            if name in ("coheres_with", "near"):
                G.add_edge(("station", e["station_a"]),
                           ("station", e["station_b"]),
                           **{**e, "edge_table": name})
    return G


def from_networkx(G):
    stations = sorted((d for n, d in G.nodes(data=True)
                       if d.get("type") == "station"),
                      key=lambda r: r["station_id"])
    edges = {}
    for _u, _v, d in G.edges(data=True):
        d = dict(d)
        name = d.pop("edge_table")
        edges.setdefault(name, []).append(d)
    for name in edges:
        edges[name].sort(key=lambda e: (e["station_a"], e["station_b"]))
    return stations, edges


def to_pyg(station_table, edge_tables):
    try:
        import torch  # noqa: F401
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise CapabilityUnavailable(f"torch/torch_geometric: {exc}")
    data = HeteroData()
    ids = [r["station_id"] for r in station_table]
    index = {s: i for i, s in enumerate(ids)}
    data["station"].node_id = ids
    for name, rows in edge_tables.items():
        if name not in ("coheres_with", "near"):
            continue
        pairs = [[index[e["station_a"]], index[e["station_b"]]] for e in rows]
        data["station", name, "station"].edge_index = \
            torch.tensor(pairs, dtype=torch.long).t().contiguous() \
            if pairs else torch.empty((2, 0), dtype=torch.long)
    return data


RENDER_LABEL = "seismic envelope coherence structure -- not displacement"


def render_map(station_table, coherence_edges, out_path, *, title_suffix=""):
    """Map render; the A3-mandated label is NON-NEGOTIABLE and always drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise CapabilityUnavailable(f"matplotlib: {exc}")
    pos = {r["station_id"]: (r["lon"], r["lat"]) for r in station_table}
    fig, ax = plt.subplots(figsize=(8, 6))
    for e in coherence_edges:
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
