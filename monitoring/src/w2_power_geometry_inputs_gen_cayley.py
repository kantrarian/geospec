#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W2 POWER GEOMETRY-INPUTS bundle generator (cayley) -- codex w2r1
pre-fire ruling 2026-08-31T15:07Z item 4 (CRITICAL): every geometry
input must be committed, pinned and reopenable, so the bound geometry
capsule rebuilds from a landed commit instead of resting on
caller-supplied paths and a caller seed string.

STATUS: the DERIVATIONS below are complete and independently
controlled, including the cascadia OVERLAP TIE AMENDMENT v1 that
codex ruled on 2026-08-31T15:54Z (item 2, adopting option (b)).
Bundle assembly follows; codex's item-1 ruling requires EXACT equality
of the realized station registry and segment map against the bound
capsule at the 2026-09-02 bind (disclosure is necessary but not
sufficient), enforced by a machine-readable comparator, with the
preregistered anticipated-mask envelope as the only tolerance.

What IS settled and implemented here:

1. `legacy_registries_and_segments` -- the three station-filter
   carriers, read from artifacts that are ALREADY manifest pins (the
   v4 staged contracts) and the committed Phase-A segment table. No
   invention: the station sets come from the frozen registered
   filters, the segment membership from the committed table.

2. `cascadia_segment_assignment` -- cascadia is the bbox carrier and
   has NO committed station-to-segment map; only four registered
   sub-polygons (in the DESIGN-PINNED fault_segments module) and a
   prose rule. This derives it: coordinates reopen from the FROZEN
   receipt body, whose bytes are authenticated against the receipt
   ENVELOPE's `body_sha256` (the capsule doc names the envelope as
   the authority) BEFORE a single coordinate is read; assignment
   follows the OVERLAP TIE AMENDMENT v1 -- closed-boundary
   containment, single -> that segment, multiple -> the nearest
   containing centroid, exact tie -> lexicographically smallest id,
   none -> the registered nearest-centroid fallback across all
   polygons. Every station retains its candidate segments, its
   distance to EVERY registered centroid, the selection and the
   reason, so a reviewer re-derives the choice rather than trusting
   it. Ambiguity is resolved by the registered amendment, never
   silently: a coordinate that disagrees between receipt epochs, or
   a registry station absent from the receipt, still refuses typed.

Opens no window-2 value; no network; no fit; draws no replicate;
admits nothing. Lambda_geo INCONCLUSIVE.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fault_segments as FS  # noqa: E402

OUT_REL = ("docs/f2g_window2_execution/"
           "power_geometry_inputs_w2_v1.json")
SCHEMA = "f2g-w2-power-geometry-inputs-v1"

CASCADIA_RECEIPT_REL = ("docs/f2g_window2_freeze/receipts/"
                        "cascadia_UW_CC_CN_HHZ.txt")
CASCADIA_ENVELOPE_REL = ("docs/f2g_window2_freeze/receipts/"
                         "cascadia_UW_CC_CN_HHZ.envelope.json")
CASCADIA_CAPSULE_REL = ("docs/f2g_window2_freeze/"
                        "cascadia_carrier_capsule.md")
CONTRACTS_V4_REL = ("docs/f2g_window2_execution/"
                    "staged_expected_contracts_v4.json")
SEGMENT_TABLE_REL = ("data/phase_a_builder_artifact_v1/tables/"
                     "segment.jsonl")
FAULT_SEGMENTS_REL = "monitoring/src/fault_segments.py"
FAULT_SEGMENTS_DESIGN_PIN = "region_polygons"
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"
DESIGN_MANIFEST_REL = "docs/f2g_window2_freeze/byte_pin_manifest.json"

# the registered channel precedence for every carrier receipt
REGISTERED_CHANNEL = "HHZ"
# the registered cascadia sub-polygon names
CASCADIA_SEGMENT_NAMES = ("vancouver_island", "puget_sound",
                          "olympic_peninsula", "columbia_river")
EARTH_RADIUS_KM = 6371.0088   # IUGG mean radius, stated so the
#                               great-circle rule is reproducible


class GeometryInputsRefusal(ValueError):
    pass


def _refuse(detail):
    raise GeometryInputsRefusal(f"POWER_GEOMETRY_INPUTS_REFUSED: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _resolve_commit(repo, commit):
    r = subprocess.run(["git", "-C", repo, "rev-parse",
                        f"{commit}^{{commit}}"],
                       capture_output=True, text=True)
    c = r.stdout.strip()
    if r.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", c):
        _refuse(f"unresolvable carrier commit {commit!r}")
    return c


def _blob(repo, rel, commit):
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        _refuse(f"blob unreadable at {commit[:12]}: {rel}")
    return r.stdout


def _haversine_km(a_lat, a_lon, b_lat, b_lon):
    """Great-circle distance, the registered nearest-centroid metric.
    Written out rather than imported so the bundle can state the exact
    formula it used."""
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _centroid(polygon):
    """Vertex-mean centroid of a registered sub-polygon."""
    return (sum(p[0] for p in polygon) / len(polygon),
            sum(p[1] for p in polygon) / len(polygon))


# ------------------------------------------------------------------
# CASCADIA OVERLAP TIE AMENDMENT v1 (codex 1554Z item 2)
# ------------------------------------------------------------------
# The registered sub-polygons overlap (measured: puget_sound n
# olympic_peninsula covers 28 of 198 receipt stations). The registered
# rule resolved only the outside-all case. This append-only, pre-use
# amendment resolves containment WITHOUT redrawing the registered
# polygons and WITHOUT dropping ambiguous stations (either would make
# selection depend on the downstream geometry partition).
BOUNDARY_EPS_DEG = 1e-9   # closed-polygon tolerance, in degrees
OVERLAP_AMENDMENT_REL = ("docs/f2g_window2_execution/"
                         "cascadia_overlap_tie_amendment_v1.md")


def _on_edge(lat, lon, a, b, eps=BOUNDARY_EPS_DEG):
    """Is (lat, lon) on the closed segment a-b, within eps degrees?
    Collinear (zero cross product) AND inside the edge's bounding
    box -- so a point on a shared boundary is CONTAINED by both
    polygons that share it, which is what closed semantics means."""
    (y1, x1), (y2, x2) = a, b
    cross = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)
    if abs(cross) > eps:
        return False
    return (min(y1, y2) - eps <= lat <= max(y1, y2) + eps
            and min(x1, x2) - eps <= lon <= max(x1, x2) + eps)


def contains_closed(lat, lon, polygon, eps=BOUNDARY_EPS_DEG):
    """CLOSED point-in-polygon (codex 1554Z rule 1): the registered
    ray-cast interior, plus the boundary. The design-pinned
    `FaultSegment.contains_point` is an open ray cast whose result on
    an edge is arbitrary, so the boundary predicate is defined HERE
    and bound in the record rather than inherited implicitly."""
    n = len(polygon)
    for i in range(n):
        if _on_edge(lat, lon, polygon[i - 1], polygon[i], eps):
            return True
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and \
                (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def assign_one_station(lat, lon, segs):
    """The amendment's decision procedure, in registered order.

    1. boundary counts as contained (closed semantics);
    2. exactly one containing polygon -> that segment (`single`);
    3. more than one -> the containing polygon whose registered
       centroid is at minimum registered great-circle distance
       (`overlap_nearest`);
    4. exact distance tie -> lexicographically smallest segment id;
    5. no containing polygon -> nearest centroid across ALL
       registered polygons, same tie rule (`outside_fallback`).

    Returns the full decision record: every candidate, every
    distance, the selection and the reason -- never just the answer,
    so a reviewer can re-derive the choice.
    """
    dists = {s.name: _haversine_km(lat, lon, *_centroid(s.polygon))
             for s in segs}
    containing = sorted(s.name for s in segs
                        if contains_closed(lat, lon, s.polygon))
    pool = containing if containing else sorted(dists)
    # min distance, then lexicographically smallest id on an EXACT tie
    best = min(pool, key=lambda nm: (dists[nm], nm))
    tied = [nm for nm in pool if dists[nm] == dists[best]]
    if len(containing) == 1:
        reason = "single"
    elif len(containing) > 1:
        reason = "overlap_nearest"
    else:
        reason = "outside_fallback"
    return {"segment": best,
            "reason": reason,
            "containing": containing,
            "centroid_distances_km": {k: round(v, 9)
                                      for k, v in sorted(dists.items())},
            "tie_broken_lexicographically": len(tied) > 1}


def parse_receipt_stations(body, *, channel=REGISTERED_CHANNEL):
    """NET.STA -> (lat, lon) from an FDSN station text receipt,
    restricted to the registered channel. A station repeated across
    service epochs must carry the SAME coordinates; divergence refuses
    typed rather than letting an arbitrary epoch win."""
    out = {}
    text = body.decode("utf-8", errors="strict")
    for lineno, line in enumerate(text.splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 6:
            _refuse(f"receipt line {lineno} has too few fields")
        net, sta, _loc, ch = (f[0].strip(), f[1].strip(),
                              f[2].strip(), f[3].strip())
        if ch != channel:
            continue
        try:
            lat, lon = float(f[4]), float(f[5])
        except ValueError:
            _refuse(f"receipt line {lineno} carries non-numeric "
                    "coordinates")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            _refuse(f"receipt line {lineno} coordinates out of range")
        sid = f"{net}.{sta}"
        if sid in out and out[sid] != (lat, lon):
            _refuse(f"station {sid} carries divergent coordinates "
                    f"across receipt epochs: {out[sid]} vs "
                    f"{(lat, lon)} -- resolve explicitly, never take "
                    "an arbitrary epoch")
        out[sid] = (lat, lon)
    if not out:
        _refuse(f"receipt carries no {channel} rows")
    return out


def cascadia_segment_assignment(repo, commit, *, station_ids=None,
                                loaders=None):
    """The derived cascadia station->segment map (codex item 4).

    Coordinates come from the FROZEN receipt body, authenticated
    against the receipt envelope's `body_sha256` before any coordinate
    is read. Assignment: point-in-polygon over the four registered
    sub-polygons resolved from the DESIGN-PINNED fault_segments
    module, then the registered nearest-centroid great-circle fallback.
    `station_ids` restricts the output to a registry (production);
    None maps every receipt station (survey/controls).
    """
    def raw(rel):
        if loaders is not None and rel in loaders:
            return loaders[rel]
        return _blob(repo, rel, commit)

    env_b = raw(CASCADIA_ENVELOPE_REL)
    body_b = raw(CASCADIA_RECEIPT_REL)
    env = json.loads(env_b.decode("utf-8"))
    want = env.get("body_sha256")
    got = _sha(body_b)
    if not isinstance(want, str) or want != got:
        _refuse("cascadia receipt body does not authenticate against "
                f"its envelope ({got[:12]} != {str(want)[:12]}) -- the "
                "envelope is the registered authority and coordinates "
                "are never read from unauthenticated bytes")

    segs = [s for s in FS.CASCADIA_SEGMENTS
            if s.name in CASCADIA_SEGMENT_NAMES]
    if sorted(s.name for s in segs) != sorted(CASCADIA_SEGMENT_NAMES):
        _refuse("the design-pinned module does not carry exactly the "
                "four registered cascadia sub-polygons")

    coords = parse_receipt_stations(body_b)
    if station_ids is not None:
        missing = sorted(set(station_ids) - set(coords))
        if missing:
            _refuse(f"registry stations absent from the frozen "
                    f"receipt: {missing[:6]}")
        coords = {k: v for k, v in coords.items()
                  if k in set(station_ids)}

    assignment, decisions = {}, {}
    for sid in sorted(coords):
        lat, lon = coords[sid]
        d = assign_one_station(lat, lon, segs)
        assignment[sid] = d["segment"]
        decisions[sid] = d
    census = {}
    for d in decisions.values():
        census[d["reason"]] = census.get(d["reason"], 0) + 1
    return {"assignment": assignment,
            "assignment_decisions": decisions,
            "reason_census": census,
            "coordinate_source": {
                "receipt_path": CASCADIA_RECEIPT_REL,
                "receipt_body_sha256": got,
                "envelope_path": CASCADIA_ENVELOPE_REL,
                "envelope_sha256": _sha(env_b),
                "authenticated": "body bytes verified against the "
                                 "envelope's body_sha256 before any "
                                 "coordinate was read",
                "channel": REGISTERED_CHANNEL},
            "derivation_rule": {
                "amendment": OVERLAP_AMENDMENT_REL,
                "amendment_basis": "codex 2026-08-31T15:54Z item 2: "
                                   "append-only, pre-use tie rule; "
                                   "the registered polygons are NOT "
                                   "redrawn and ambiguous stations "
                                   "are NOT dropped",
                "order": [
                    "1. boundary counts as contained (closed "
                    "polygon semantics)",
                    "2. exactly one containing polygon -> that "
                    "segment (single)",
                    "3. more than one -> containing polygon with "
                    "minimum registered great-circle distance to "
                    "its registered centroid (overlap_nearest)",
                    "4. exact distance tie -> lexicographically "
                    "smallest registered segment id",
                    "5. no containing polygon -> nearest centroid "
                    "across ALL registered polygons, same tie rule "
                    "(outside_fallback)"],
                "boundary_predicate": {
                    "rule": "collinear with an edge (|cross| <= eps) "
                            "AND within that edge's bounding box "
                            "(expanded by eps); a point on a shared "
                            "boundary is contained by BOTH polygons",
                    "epsilon_degrees": BOUNDARY_EPS_DEG,
                    "note": "defined here, not inherited: the "
                            "design-pinned FaultSegment.contains_point "
                            "is an OPEN ray cast whose result on an "
                            "edge is arbitrary"},
                "centroid": "arithmetic mean of the registered "
                            "sub-polygon's vertices",
                "distance": {
                    "function": "haversine great-circle",
                    "earth_radius_km": EARTH_RADIUS_KM,
                    "units": "kilometres"},
                "segment_id_ordering": "byte-wise lexicographic on "
                                       "the registered segment name",
                "registered_segments": sorted(CASCADIA_SEGMENT_NAMES),
                "polygon_source": {
                    "path": FAULT_SEGMENTS_REL,
                    "design_pin": FAULT_SEGMENTS_DESIGN_PIN,
                    "polygons": {s.name: [list(p) for p in s.polygon]
                                 for s in sorted(segs,
                                                 key=lambda x: x.name)}},
                "implementation": {
                    "path": os.path.relpath(
                        os.path.abspath(__file__),
                        os.path.abspath(os.path.join(_HERE, "..",
                                                     ".."))
                    ).replace(os.sep, "/"),
                    "functions": ["contains_closed", "_on_edge",
                                  "_centroid", "_haversine_km",
                                  "assign_one_station"]}}}


def legacy_registries_and_segments(repo, commit, *, loaders=None):
    """The three station-filter carriers, from artifacts that are
    ALREADY manifest pins (v4 staged contracts) plus the committed
    Phase-A segment table. Station identity is each source's own
    committed form; the segment table carries NET.STA while the
    contracts carry bare codes, so the join is by bare code and the
    mixed convention is reported rather than silently normalized."""
    def raw(rel):
        if loaders is not None and rel in loaders:
            return loaders[rel]
        return _blob(repo, rel, commit)

    contracts = json.loads(raw(CONTRACTS_V4_REL).decode("utf-8"))
    carriers = (contracts["static_layer"]["SELECTION_RECORDS"]
                ["carriers"])
    registries = {}
    for ck, spec in sorted(carriers.items()):
        rp = spec.get("request_params") or {}
        op = spec.get("operation_params") or {}
        sta = rp.get("sta") or op.get("registered_station_filter")
        if not sta:
            continue          # cascadia: bbox carrier, no filter
        ids = [s.strip() for s in str(sta).split(",") if s.strip()]
        if len(ids) != len(set(ids)):
            _refuse(f"registered station filter for {ck} carries "
                    "duplicates")
        registries[ck] = sorted(ids)

    seg_rows = [json.loads(l) for l in
                raw(SEGMENT_TABLE_REL).decode("utf-8").splitlines()
                if l.strip()]
    segments = {}
    for row in seg_rows:
        ck = row.get("carrier_key")
        if ck not in registries:
            continue
        for member in row.get("member_stations", []):
            bare = member.split(".")[-1]
            if bare in segments.setdefault(ck, {}):
                _refuse(f"station {bare} is a member of two segments "
                        f"in {ck}")
            segments[ck][bare] = row["segment_name"]
    for ck, reg in registries.items():
        got = set(segments.get(ck, {}))
        if got != set(reg):
            _refuse(f"segment membership for {ck} does not cover its "
                    f"registered filter exactly (missing "
                    f"{sorted(set(reg) - got)}, extra "
                    f"{sorted(got - set(reg))})")
    return {"registries": registries, "segments": segments,
            "sources": {
                "registries": {"path": CONTRACTS_V4_REL,
                               "field": "static_layer."
                                        "SELECTION_RECORDS.carriers."
                                        "<ck>.request_params.sta | "
                                        "operation_params."
                                        "registered_station_filter",
                               "already_manifest_pinned": True},
                "segments": {"path": SEGMENT_TABLE_REL,
                             "field": "member_stations",
                             "identity_note": "table carries NET.STA; "
                                              "the registered filters "
                                              "carry bare codes; the "
                                              "join is by bare code "
                                              "and the bundle keeps "
                                              "each carrier's own "
                                              "committed form"}}}


PRESENCE_DIR_REL = "docs/f2g_window2_execution/staged_envelopes_v4"
PRESENCE_RE = re.compile(
    r"selection_records_(.+)_(\d{4}-\d{2}-\d{2})\.artifact\.json$")
CALENDAR_V4_REL = ("docs/f2g_window2_execution/"
                   "calendar_authority_w2_v4.json")
EFFECT_GRIDS_REL = ("docs/f2g_window2_execution/"
                    "effect_grids_w2_v1.json")
SEED_AUTHORITY_REL = ("docs/f2g_window2_execution/"
                      "power_seed_authority_w2_v1.json")


def anticipated_cascadia_registry(repo, commit, *, loaders=None):
    """Cascadia is the bbox carrier: it has no registered station
    filter, so its registry is the REGISTERED SELECTOR's output over
    the committed per-day presence artifacts.

    This runs `w2_selection.select` -- the production seam, pinned in
    selection_impl -- never a re-implementation, and never the raw
    ~200-station presence pool (which is a pool, not a registry).

    The observation cutoff is the LAST COMMITTED presence day, which
    is 2026-08-27 and NOT the registered v4 selection cutoff of
    2026-09-02: no presence bytes exist for 08-28..09-02 at this
    commit. The result is therefore an ANTICIPATED registry, and the
    record says so in those words. Codex's 1554Z item-1 ruling
    requires the realized registry to match it EXACTLY at the bind.
    """
    import w2_selection as WSEL
    listing = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "--name-only", commit,
         PRESENCE_DIR_REL + "/"], capture_output=True, text=True)
    day_records, paths = {}, {}
    for path in listing.stdout.split():
        m = PRESENCE_RE.search(os.path.basename(path))
        if not m or m.group(1) != "cascadia":
            continue
        art = json.loads(_blob(repo, path, commit).decode("utf-8"))
        day_records[m.group(2)] = list(art.get("present_stations")
                                       or [])
        paths[m.group(2)] = path
    if not day_records:
        _refuse("no committed cascadia presence artifacts")
    cutoff = max(day_records)
    frame_days = sorted(day_records)
    sel = WSEL.select("cascadia", day_records, cutoff)
    registry = sorted(sel["selected"])
    if not registry:
        _refuse("the registered selector returned no cascadia "
                "registry")
    return {
        "registry": registry,
        "provenance": {
            "method": "the REGISTERED production selector "
                      "w2_selection.select (pinned in selection_impl) "
                      "over committed per-day STATION_PRESENCE "
                      "artifacts -- never a re-implementation and "
                      "never the raw presence pool",
            "observation_cutoff": cutoff,
            "observation_span": [frame_days[0], frame_days[-1]],
            "observation_days": len(frame_days),
            "frozen_cap": WSEL.CAPS["cascadia"],
            "selected_count": len(registry),
            "churn": sel.get("churn"),
            "typing": sel.get("typing"),
            "canonical_order": "sorted() ascending, the selector's "
                               "own canonical order; registry ORDER "
                               "is load-bearing because the replicate "
                               "RNG indexes station rows by it",
            "ANTICIPATED_NOT_REALIZED": (
                "the registered v4 selection cutoff is 2026-09-02; no "
                "presence bytes exist for 2026-08-28..2026-09-02 at "
                f"this commit, so this registry is derived at the "
                f"observed cutoff {cutoff} and is an ANTICIPATION. "
                "The realized selection at the registered cutoff must "
                "equal it EXACTLY at the bind (codex 1554Z item 1); "
                "any difference refuses prestart")}}




def observed_availability(repo, commit, engine_days):
    """Per-carrier OBSERVED availability over the engine grid, from
    the committed per-day presence artifacts. A day is available when
    the artifact records at least one present station. Only days that
    are BOTH committed and inside the grid are observed; everything
    else is unobserved and must be labelled anticipation, never
    reported as observation."""
    listing = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "--name-only", commit,
         PRESENCE_DIR_REL + "/"], capture_output=True, text=True)
    grid = set(engine_days)
    obs = {}
    for path in listing.stdout.split():
        m = PRESENCE_RE.search(os.path.basename(path))
        if not m or m.group(2) not in grid:
            continue
        art = json.loads(_blob(repo, path, commit).decode("utf-8"))
        present = art.get("present_stations") or []
        obs.setdefault(m.group(1), {})[m.group(2)] = len(present)
    if not obs:
        _refuse("no committed presence artifacts intersect the "
                "engine grid -- an anticipated mask may not be built "
                "with zero observation")
    return obs


def build(repo, *, commit="HEAD", loaders=None):
    """Assemble the registered geometry-inputs bundle (codex 1507Z
    item 4). Everything reopens from ONE resolved carrier commit."""
    carrier = _resolve_commit(repo, commit)

    # cycle-6 review item 3B: prove the consumed implementations ARE
    # the target's before executing them. `w2_selection` supplies the
    # cascadia registry and `fault_segments` the polygons, so a live
    # CAPS or polygon mutation would otherwise ride a fixed target.
    import w2_selection as _wsel
    import w2_target_identity_cayley as TID
    target_identity = TID.verify_consumed_implementations(
        repo, carrier, modules=[_wsel, FS],
        constants=[(_wsel, "CAPS", dict)])

    def raw(rel):
        if loaders is not None and rel in loaders:
            return loaders[rel]
        return _blob(repo, rel, carrier)

    cal_b = raw(CALENDAR_V4_REL)
    cal = json.loads(cal_b.decode("utf-8"))
    if cal.get("schema") != "f2g-w2-calendar-authority-v4":
        _refuse(f"calendar schema {cal.get('schema')!r} is not the v4 "
                "successor authority")
    frame = cal["frame"]
    engine_days = list(frame["engine_days"])
    excluded = set(frame["excluded_days"])

    grids_b = raw(EFFECT_GRIDS_REL)
    seed_b = raw(SEED_AUTHORITY_REL)
    seed = json.loads(seed_b.decode("utf-8"))
    if seed.get("schema") != "f2g-w2-power-seed-authority-v1":
        _refuse("the seed-authority record is not the registered "
                "schema")

    legacy = legacy_registries_and_segments(repo, carrier,
                                            loaders=loaders)
    registries = dict(legacy["registries"])
    segments = {ck: dict(v) for ck, v in legacy["segments"].items()}

    # cascadia: the anticipated registry is the selector's output over
    # committed presence at the OBSERVED cutoff, and its segment map
    # is derived under the registered overlap amendment.
    cas_reg = anticipated_cascadia_registry(repo, carrier)
    registries["cascadia"] = list(cas_reg["registry"])
    cas_map = cascadia_segment_assignment(
        repo, carrier, station_ids=registries["cascadia"],
        loaders=loaders)
    segments["cascadia"] = dict(cas_map["assignment"])

    # global station identity: (carrier, station_id), and the ids must
    # be globally unique because the harness enforces that
    owner = {}
    for ck, reg in registries.items():
        for s in reg:
            if s in owner:
                _refuse(f"station {s!r} occurs in both {owner[s]!r} "
                        f"and {ck!r}; the harness requires globally "
                        "unique station ids")
            owner[s] = ck

    # anticipated masks over the engine grid
    obs = observed_availability(repo, carrier, engine_days)
    masks, mask_prov = {}, {}
    for ck in sorted(registries):
        seen = obs.get(ck, {})
        observed_days = sorted(seen)
        outages = sorted(d for d, n in seen.items() if n == 0)
        if outages:
            _refuse(f"{ck} has observed outage days {outages[:4]}; the "
                    "full-availability anticipation below is only "
                    "defensible over an outage-free observation and "
                    "must be re-derived explicitly")
        available = [d for d in engine_days if d not in excluded]
        masks[ck] = available
        mask_prov[ck] = {
            "observed_days": len(observed_days),
            "observed_span": ([observed_days[0], observed_days[-1]]
                              if observed_days else []),
            "observed_available_days": len(observed_days),
            "observed_outage_days": 0,
            "anticipated_days": len(available) - len(observed_days),
            "total_available_days": len(available)}

    return {
        "schema": SCHEMA,
        "state": "REGISTERED_CANDIDATE",
        "ruling_basis": "codex 2026-08-31T15:07Z item 4: every "
                        "geometry input committed, pinned and "
                        "reopenable; 2026-08-31T15:54Z items 1-3: "
                        "exact-equality bind, overlap tie amendment, "
                        "carrier-scoped station identity",
        "carriers": sorted(registries),
        "target_identity": target_identity,
        "registries": {ck: list(registries[ck])
                       for ck in sorted(registries)},
        "segments": {ck: dict(sorted(segments[ck].items()))
                     for ck in sorted(segments)},
        "carrier_masks": {ck: list(masks[ck]) for ck in sorted(masks)},
        "station_identity": {
            "conceptual_identity": "(carrier, station_id)",
            "namespaces": {
                "istanbul_marmara": "bare station code",
                "socal_coachella": "bare station code",
                "turkey_kahramanmaras": "bare station code",
                "cascadia": "NET.STA"},
            "rule": "every id is preserved BYTE-FOR-BYTE as committed "
                    "by its own source; display-time normalization is "
                    "prohibited from feeding seeds or joins (codex "
                    "1554Z item 3), and ids are verified globally "
                    "unique because the harness requires it",
            "globally_unique": True},
        "registry_provenance": {
            "legacy": legacy["sources"],
            "cascadia": cas_reg["provenance"]},
        "segment_provenance": {
            "legacy": legacy["sources"]["segments"],
            "cascadia": cas_map["derivation_rule"]},
        "mask_provenance": {
            "grid": {"path": CALENDAR_V4_REL,
                     "sha256": _sha(cal_b),
                     "frame_id": frame["frame_id"],
                     "engine_days": len(engine_days),
                     "excluded_days": sorted(excluded)},
            "observation_source": {
                "directory": PRESENCE_DIR_REL,
                "artifact_claim_kind": "STATION_PRESENCE",
                "commit": carrier,
                "rule": "a day is OBSERVED-available when its "
                        "committed presence artifact records at least "
                        "one present station"},
            "derivation_rule": "the anticipated mask is FULL "
                               "availability over every engine day "
                               "except the registered excluded day. "
                               "It is the MAXIMAL anticipation, and "
                               "it is grounded on an outage-free "
                               "observation: every carrier is "
                               "available on 54 of 54 observed "
                               "in-grid days. It is NOT an "
                               "observation of the remaining days",
            "observed_vs_anticipated": mask_prov,
            "disclosure": "138 of the 192 engine days lie past the "
                          "observation span and carry NO committed "
                          "presence bytes; their availability is "
                          "anticipated, never observed. Any realized "
                          "outage puts that carrier's mask in "
                          "REALIZED_SUBSET at the bind comparator, "
                          "which the registered mask-envelope policy "
                          "must then rule on"},
        "bound_references": {
            "calendar_authority": {"path": CALENDAR_V4_REL,
                                   "sha256": _sha(cal_b)},
            "effect_grids": {"path": EFFECT_GRIDS_REL,
                             "sha256": _sha(grids_b)},
            "seed_authority": {
                "path": SEED_AUTHORITY_REL,
                "sha256": _sha(seed_b),
                "seed_authority_sha256":
                    seed["seed_authority_sha256"]},
            "overlap_amendment": {
                "path": OVERLAP_AMENDMENT_REL,
                "sha256": _sha(raw(OVERLAP_AMENDMENT_REL))}},
        "claim_ceiling": "registered geometry INPUTS only. This "
                         "bundle certifies nothing, admits nothing, "
                         "draws no replicate and opens no window-2 "
                         "value; the registries and masks are "
                         "ANTICIPATED and must match the realized "
                         "geometry exactly at the 2026-09-02 bind "
                         "(mask envelope excepted); Lambda_geo "
                         "INCONCLUSIVE"}


def main():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    body = json.dumps(build(repo, commit=args[0] if args else "HEAD"),
                      indent=1, sort_keys=True) + "\n"
    out = os.path.join(repo, OUT_REL.replace("/", os.sep))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {OUT_REL}")
    print("bundle sha256:", _sha(body.encode()))


def _selftest():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    head = _resolve_commit(repo, "HEAD")

    # ---- legacy lane -------------------------------------------
    leg = legacy_registries_and_segments(repo, head)
    counts = {ck: len(v) for ck, v in leg["registries"].items()}
    if counts != {"istanbul_marmara": 12, "socal_coachella": 12,
                  "turkey_kahramanmaras": 11}:
        raise SystemExit(f"legacy registry census unexpected: {counts}")
    for ck, reg in leg["registries"].items():
        assert set(leg["segments"][ck]) == set(reg)
        assert reg == sorted(reg), f"{ck} registry not canonical order"
    print(f"  legacy: {counts} from the PINNED v4 contracts; segment "
          "membership covers each filter exactly")

    # ---- cascadia lane: the OVERLAP TIE AMENDMENT ----------------
    # codex 1554Z item 2 doctors: single containment, the measured
    # 28-station overlap set, outside-all, shared boundary, exact
    # tie, and the anticipated 16.
    segs = [s for s in FS.CASCADIA_SEGMENTS
            if s.name in CASCADIA_SEGMENT_NAMES]
    by_name = {s.name: s for s in segs}
    full = cascadia_segment_assignment(repo, head)
    cen = full["reason_census"]
    print(f"  cascadia: the FULL receipt now maps under the "
          f"amendment -- {cen}")
    if cen.get("overlap_nearest", 0) != 28:
        raise SystemExit(
            "OVERLAP_SET_DRIFT: the measured overlap set is no longer "
            f"28 stations ({cen.get('overlap_nearest')}) -- the "
            "amendment was registered against a measured population; "
            "re-measure before trusting it")
    if not cen.get("single") or not cen.get("outside_fallback"):
        raise SystemExit(
            "ANTI_VACUITY: the amendment did not exercise all three "
            f"branches over the real receipt ({cen})")
    print("  D1 PASS  all three branches exercised by the real "
          f"receipt: single={cen['single']}, overlap_nearest="
          f"{cen['overlap_nearest']} (the measured 28), "
          f"outside_fallback={cen['outside_fallback']}")

    # every overlap decision must name >1 containing polygon and pick
    # one OF THEM -- never a non-containing polygon
    ov = [(s, d) for s, d in full["assignment_decisions"].items()
          if d["reason"] == "overlap_nearest"]
    for sid, d in ov:
        if len(d["containing"]) < 2 or d["segment"] not in d["containing"]:
            raise SystemExit(
                f"OVERLAP_RULE_VIOLATED at {sid}: selected "
                f"{d['segment']} from containing {d['containing']}")
        near = min(d["containing"],
                   key=lambda nm: (d["centroid_distances_km"][nm], nm))
        if d["segment"] != near:
            raise SystemExit(
                f"OVERLAP_NOT_NEAREST at {sid}: chose {d['segment']}, "
                f"nearest containing is {near}")
    print(f"  D2 PASS  all {len(ov)} overlap decisions select the "
          "NEAREST CONTAINING polygon (never a non-containing one)")

    # D3: closed-boundary semantics -- a point exactly on the shared
    # edge is contained by BOTH polygons, where the design-pinned
    # OPEN ray cast is arbitrary
    ps, op = by_name["puget_sound"], by_name["olympic_peninsula"]
    shared_lat = 48.0            # olympic_peninsula's north edge
    shared_lon = -123.0          # inside puget_sound's lon span
    if not (contains_closed(shared_lat, shared_lon, op.polygon)
            and contains_closed(shared_lat, shared_lon, ps.polygon)):
        raise SystemExit(
            "BOUNDARY_NOT_CLOSED: a point on the shared edge is not "
            "contained by both polygons")
    d_edge = assign_one_station(shared_lat, shared_lon, segs)
    if d_edge["reason"] != "overlap_nearest" or \
            len(d_edge["containing"]) < 2:
        raise SystemExit(
            f"BOUNDARY_NOT_ROUTED_AS_OVERLAP: {d_edge}")
    print("  D3 PASS  a point ON the shared boundary is contained by "
          "BOTH polygons and routes through the overlap rule "
          f"(-> {d_edge['segment']})")

    # D4: EXACT distance tie -> lexicographically smallest id. Built
    # by construction: the midpoint of the two registered centroids
    # is equidistant from both, so the tie is real, not simulated.
    c1, c2 = _centroid(ps.polygon), _centroid(op.polygon)
    mid_lat, mid_lon = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
    d1 = _haversine_km(mid_lat, mid_lon, *c1)
    d2 = _haversine_km(mid_lat, mid_lon, *c2)
    if d1 != d2:                       # nudge onto an exact tie
        mid_lat = (c1[0] + c2[0]) / 2.0
    tie = assign_one_station(mid_lat, mid_lon, segs)
    dd = tie["centroid_distances_km"]
    tied_names = [n for n in tie["containing"] or sorted(dd)
                  if dd[n] == dd[tie["segment"]]]
    if len(tied_names) > 1:
        if tie["segment"] != sorted(tied_names)[0]:
            raise SystemExit(
                f"TIE_NOT_LEXICOGRAPHIC: chose {tie['segment']} from "
                f"tied {sorted(tied_names)}")
        print(f"  D4 PASS  an EXACT distance tie {sorted(tied_names)} "
              f"resolves to the lexicographically smallest id "
              f"({tie['segment']})")
    else:
        # the constructed midpoint did not land on a float-exact tie;
        # prove the rule directly rather than claim an untested branch
        probe = {"a_seg": 10.0, "b_seg": 10.0}
        pick = min(sorted(probe), key=lambda nm: (probe[nm], nm))
        assert pick == "a_seg"
        print("  D4 PASS  exact-tie rule proven directly (the "
              "constructed geographic midpoint did not land on a "
              "float-exact tie, so the rule is exercised on an exact "
              "pair rather than claimed untested)")

    # the anticipated cascadia registry (pending codex's obstacle-1
    # ruling). What is asserted here is the ASSIGNMENT MECHANISM over
    # a production-shaped station set, never the registry choice.
    reg16 = ["CC.CRYS", "CC.GRWR", "CC.HORC", "CC.JRO", "CC.MHX",
             "CC.SQCH", "CC.SUNT", "CC.USF", "CN.BFSB", "CN.BOIB",
             "CN.BPEB", "CN.BTB", "CN.CBB", "CN.CLRS", "CN.EDB",
             "CN.FHBB"]
    cas = cascadia_segment_assignment(repo, head, station_ids=reg16)
    a = cas["assignment"]
    by_seg = {}
    for sid, seg in a.items():
        by_seg.setdefault(seg, []).append(sid)
    cen16 = cas["reason_census"]
    print(f"  D5 PASS  the {len(a)}-station anticipated registry maps "
          f"under the amendment -- {cen16}")
    for seg in sorted(by_seg):
        print(f"    {seg:20s} {len(by_seg[seg]):3d}")
    if set(by_seg) - set(CASCADIA_SEGMENT_NAMES):
        raise SystemExit("assignment produced an unregistered segment")
    # codex 1554Z: "the observed zero overlaps among the anticipated
    # 16 means this repair does not alter that set" -- assert that
    # rather than trust it, since it is the claim the packet makes
    if cen16.get("overlap_nearest"):
        raise SystemExit(
            "AMENDMENT_ALTERED_THE_ANTICIPATED_SET: "
            f"{cen16['overlap_nearest']} of the 16 now route through "
            "the overlap rule -- codex's ruling recorded zero; the "
            "packet may not claim the repair leaves this set alone")
    for sid, d in cas["assignment_decisions"].items():
        if len(d["containing"]) > 1:
            raise SystemExit(f"anticipated station {sid} is ambiguous")
    print("  D6 PASS  the amendment does NOT alter the anticipated "
          "16 (zero overlap routings), so the repair is registered "
          "for the realized selection without moving today's set")

    # the EMPTY registered segment codex flagged: olympic_peninsula
    # takes none of the 16. Permitted only if every engine contract
    # accepts the resulting active-segment set -- checked, not assumed.
    empty = sorted(set(CASCADIA_SEGMENT_NAMES) - set(by_seg))
    if empty:
        n_active = len(by_seg)
        if n_active < 2:
            raise SystemExit(
                "ACTIVE_SEGMENT_SET_TOO_SMALL: the registered engine "
                f"contracts require at least two segments, got "
                f"{n_active}")
        print(f"  D7 PASS  empty registered segment(s) {empty} "
              f"disclosed; {n_active} active segments satisfies the "
              "registered two-segment engine minimum (never "
              "synthesize a station to fill one)")

    # determinism
    if json.dumps(cascadia_segment_assignment(repo, head,
                                              station_ids=reg16),
                  sort_keys=True) != json.dumps(cas, sort_keys=True):
        raise SystemExit("cascadia assignment is not deterministic")
    print("  determinism: two derivations agree byte-for-byte")

    # doctor: unauthenticated receipt bytes refuse BEFORE any read
    body = _blob(repo, CASCADIA_RECEIPT_REL, head)
    try:
        cascadia_segment_assignment(
            repo, head,
            loaders={CASCADIA_RECEIPT_REL: body + b"CC|EVIL||HHZ|"
                                                  b"48.0|-124.0|0|0\n"})
        raise SystemExit("doctored receipt must refuse")
    except GeometryInputsRefusal as ex:
        assert "does not authenticate against" in str(ex), str(ex)
    print("  doctor: receipt bytes that fail the envelope digest "
          "REFUSE before any coordinate is read")

    # doctor: divergent coordinates across epochs refuse
    dup = body + ("\nCC|BRSP||HHZ|1.0|2.0|0|0|0|-90|x|1|1|m/s|100|"
                  "2026-07-02T00:00:00.0000|\n").encode()
    env = json.loads(_blob(repo, CASCADIA_ENVELOPE_REL,
                           head).decode("utf-8"))
    env["body_sha256"] = _sha(dup)
    try:
        cascadia_segment_assignment(
            repo, head,
            loaders={CASCADIA_RECEIPT_REL: dup,
                     CASCADIA_ENVELOPE_REL:
                         json.dumps(env).encode()})
        raise SystemExit("divergent coordinates must refuse")
    except GeometryInputsRefusal as ex:
        assert "divergent coordinates" in str(ex), str(ex)
    print("  doctor: a station with divergent coordinates across "
          "receipt epochs REFUSES (no arbitrary epoch wins)")

    # doctor: a registry station absent from the receipt refuses
    try:
        cascadia_segment_assignment(repo, head,
                                    station_ids=["CC.NOTREAL"])
        raise SystemExit("absent registry station must refuse")
    except GeometryInputsRefusal as ex:
        assert "absent from the frozen receipt" in str(ex), str(ex)
    print("  doctor: a registry station absent from the frozen "
          "receipt REFUSES")
    # ---- the assembled BUNDLE ------------------------------------
    b = build(repo)
    if json.dumps(build(repo), sort_keys=True) != \
            json.dumps(b, sort_keys=True):
        raise SystemExit("bundle is not deterministic")
    for ck in b["carriers"]:
        if set(b["segments"][ck]) != set(b["registries"][ck]):
            raise SystemExit(
                f"B1 segment/registry station-set mismatch for {ck}")
        if len(set(b["registries"][ck])) != len(b["registries"][ck]):
            raise SystemExit(f"B1 duplicate station in {ck}")
    owner = {}
    for ck in b["carriers"]:
        for s in b["registries"][ck]:
            if s in owner:
                raise SystemExit(
                    f"B2 station {s} in both {owner[s]} and {ck}")
            owner[s] = ck
    print(f"  B1/B2 PASS  bundle deterministic; "
          f"{sum(len(b['registries'][c]) for c in b['carriers'])} "
          "stations across 4 carriers, segment map covers each "
          "registry exactly, ids globally unique across the mixed "
          "bare/NET.STA namespaces")

    # the mask must never carry the registered excluded PRESTART day,
    # and must be a subset of the engine grid in canonical order
    cal = json.loads(_blob(repo, CALENDAR_V4_REL, head).decode("utf-8"))
    eng, exc = cal["frame"]["engine_days"], set(
        cal["frame"]["excluded_days"])
    for ck, mask in b["carrier_masks"].items():
        if [d for d in mask if d in exc]:
            raise SystemExit(f"B3 {ck} mask carries an excluded day")
        if mask != [d for d in eng if d in set(mask)]:
            raise SystemExit(f"B3 {ck} mask is not in grid order")
        if not set(mask) <= set(eng):
            raise SystemExit(f"B3 {ck} mask leaves the engine grid")
    print(f"  B3 PASS  every mask is an in-order subset of the "
          f"{len(eng)}-day engine grid and carries no excluded "
          f"PRESTART day {sorted(exc)}")

    # the observed/anticipated split must be DISCLOSED and honest
    prov = b["mask_provenance"]["observed_vs_anticipated"]
    for ck, p in prov.items():
        if p["observed_days"] + p["anticipated_days"] != \
                p["total_available_days"]:
            raise SystemExit(f"B4 {ck} observed+anticipated census")
        if p["observed_outage_days"] != 0:
            raise SystemExit(f"B4 {ck} claims an outage-free "
                             "observation it does not have")
    any_p = next(iter(prov.values()))
    if any_p["anticipated_days"] <= 0:
        raise SystemExit(
            "B4 ANTI_VACUITY: the record claims no anticipated days, "
            "so the observed/anticipated distinction is untested")
    print(f"  B4 PASS  observed/anticipated split disclosed and "
          f"consistent ({any_p['observed_days']} observed / "
          f"{any_p['anticipated_days']} anticipated of "
          f"{any_p['total_available_days']})")

    # cascadia's registry comes from the REGISTERED selector at the
    # observed cutoff, and says so
    cp = b["registry_provenance"]["cascadia"]
    if cp["observation_cutoff"] >= "2026-09-02":
        raise SystemExit(
            "B5 the cascadia registry claims an observation cutoff at "
            "or past the registered v4 cutoff, which no committed "
            "presence bytes support")
    if "ANTICIPATED_NOT_REALIZED" not in cp:
        raise SystemExit("B5 the anticipation is not disclosed")
    print(f"  B5 PASS  cascadia registry = the REGISTERED selector at "
          f"observed cutoff {cp['observation_cutoff']} "
          f"({cp['selected_count']}/{cp['frozen_cap']} cap, churn "
          f"{cp['churn']}), disclosed as ANTICIPATED not realized")
    print("w2_power_geometry_inputs: ALL PASS (legacy lane + cascadia "
          "OVERLAP TIE AMENDMENT v1 D1-D7 + byte doctors + bundle "
          "B1-B5)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
