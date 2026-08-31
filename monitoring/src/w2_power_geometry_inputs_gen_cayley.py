#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W2 POWER GEOMETRY-INPUTS bundle generator (cayley) -- codex w2r1
pre-fire ruling 2026-08-31T15:07Z item 4 (CRITICAL): every geometry
input must be committed, pinned and reopenable, so the bound geometry
capsule rebuilds from a landed commit instead of resting on
caller-supplied paths and a caller seed string.

STATUS: the two DERIVATIONS below are complete and independently
controlled. Bundle assembly waits on codex's obstacle-1 ruling (routed
2026-08-31T15:25Z): the registered v4 selection cutoff is 2026-09-02,
no presence bytes exist past 2026-08-27, and the prestart schedule
regenerates registries ON prestart day -- after the certification
campaign must already have run. Nothing here is pinned or admitted
until that ruling lands.

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
   the authority) BEFORE a single coordinate is read; assignment is
   point-in-polygon, with the registered nearest-centroid great-circle
   fallback for in-bbox stations outside every sub-polygon; every
   station records WHICH rule placed it and, for the fallback, its
   distance. Ambiguity is never resolved silently: a station inside
   two sub-polygons, or a coordinate that disagrees between receipt
   epochs, refuses typed.

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

    assignment, reasons = {}, {}
    for sid in sorted(coords):
        lat, lon = coords[sid]
        inside = [s.name for s in segs if s.contains_point(lat, lon)]
        if len(inside) > 1:
            _refuse(f"station {sid} falls inside multiple registered "
                    f"sub-polygons {inside} -- the registered rule "
                    "does not resolve overlap, so this refuses rather "
                    "than picking one")
        if inside:
            assignment[sid] = inside[0]
            reasons[sid] = {"rule": "inside_polygon",
                            "segment": inside[0]}
            continue
        best, best_km = None, None
        for s in segs:
            c_lat, c_lon = _centroid(s.polygon)
            d = _haversine_km(lat, lon, c_lat, c_lon)
            if best_km is None or d < best_km:
                best, best_km = s.name, d
        assignment[sid] = best
        reasons[sid] = {"rule": "nearest_centroid",
                        "segment": best,
                        "great_circle_km": round(best_km, 6)}
    return {"assignment": assignment,
            "assignment_reasons": reasons,
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
                "primary": "point-in-polygon over the four registered "
                           "cascadia sub-polygons",
                "fallback": "nearest sub-polygon centroid by "
                            "great-circle distance, for in-bbox "
                            "stations outside every sub-polygon "
                            "(cascadia_carrier_capsule.md)",
                "centroid": "arithmetic mean of the sub-polygon's "
                            "vertices",
                "great_circle": "haversine on a sphere of mean radius "
                                f"{EARTH_RADIUS_KM} km",
                "overlap": "a station inside two sub-polygons REFUSES "
                           "typed; the registered rule does not "
                           "resolve overlap",
                "polygon_source": {
                    "path": FAULT_SEGMENTS_REL,
                    "design_pin": FAULT_SEGMENTS_DESIGN_PIN}}}


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

    # ---- cascadia lane ------------------------------------------
    # MEASURED at the landed tip: the four registered sub-polygons
    # OVERLAP (puget_sound lat 47.0-48.5 / lon -124.0..-122.0 against
    # olympic_peninsula lat 46.5-48.0 / lon -125.0..-122.5), and 28 of
    # the 198 receipt stations fall in that band. The registered rule
    # does not resolve overlap, so mapping the whole receipt REFUSES.
    # That refusal is the first control -- it is a real defect of the
    # registered geometry, routed to codex, not a synthetic doctor.
    try:
        cascadia_segment_assignment(repo, head)
        raise SystemExit(
            "OVERLAP_CONTROL_INERT: the full receipt no longer "
            "refuses -- the registered sub-polygon overlap this "
            "control exists to prove has changed; re-measure before "
            "trusting any cascadia map")
    except GeometryInputsRefusal as ex:
        assert "multiple registered sub-polygons" in str(ex), str(ex)
    print("  cascadia: mapping the FULL receipt REFUSES on the "
          "measured sub-polygon overlap (28/198 stations; registered "
          "rule does not resolve it) -- routed to codex")

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
    n_inside = sum(1 for r in cas["assignment_reasons"].values()
                   if r["rule"] == "inside_polygon")
    n_near = len(a) - n_inside
    print(f"  cascadia: the {len(a)}-station anticipated registry "
          f"assigns CLEANLY ({n_inside} inside_polygon, {n_near} "
          "nearest_centroid; zero overlap hits)")
    for seg in sorted(by_seg):
        print(f"    {seg:20s} {len(by_seg[seg]):3d}")
    if set(by_seg) - set(CASCADIA_SEGMENT_NAMES):
        raise SystemExit("assignment produced an unregistered segment")

    # BOTH branches must be exercised, or the fallback is unproven
    if n_inside == 0 or n_near == 0:
        raise SystemExit(
            "ANTI_VACUITY: the assignment did not exercise both the "
            f"point-in-polygon and nearest-centroid branches "
            f"(inside={n_inside}, nearest={n_near})")
    print("  anti-vacuity: BOTH assignment branches exercised by the "
          "real registry over the frozen receipt")

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
    print("w2_power_geometry_inputs derivations: ALL PASS "
          "(bundle assembly awaits codex's obstacle-1 ruling)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(
            "POWER_GEOMETRY_INPUTS_NOT_ASSEMBLABLE: the registry "
            "source is pending codex's obstacle-1 ruling (routed "
            "2026-08-31T15:25Z); run --selftest to exercise the "
            "settled derivations")
