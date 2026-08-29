#!/usr/bin/env python3
"""Regenerate docs/atlas.html from registered repo geometry + the
daily ensemble output.

PRESENTATION ONLY. Reads: monitoring/config/regions.yaml,
monitoring/src/d2_campaign_v2_candidate_pool.json, the frozen
campaign-v2 plan (selected registry), the mag frame capsules, and
docs/ensemble_latest.json (which run_and_publish.ps1 step 3 has just
copied). Writes exactly one file: docs/atlas.html, from
monitoring/atlas_template.html. No network, no evidence path, no
claim -- the page carries the standing ceiling verbatim (seismic
envelope coherence structure, not displacement; Lambda_geo
INCONCLUSIVE; no predictive claim). Exit nonzero on any failure so
the runner can skip staging a broken page (fail-soft, never abort
the publish).
"""
import hashlib
import io
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "atlas_template.html")
OUT = os.path.join(REPO, "docs", "atlas.html")


def parse_regions():
    txt = io.open(os.path.join(HERE, "config", "regions.yaml"),
                  encoding="utf-8").read()
    body = txt.split("\nregions:\n", 1)[1]
    body = body.split("\n# Data sources")[0]
    out = []
    for m in re.finditer(r"^  (\w+):\n((?:    .*\n?)+)", body, re.M):
        rid, blk = m.group(1), m.group(2)
        name = re.search(r'name:\s*"([^"]+)"', blk)
        enabled = re.search(r"enabled:\s*(\w+)", blk)
        notes = re.search(r'notes:\s*"([^"]+)"', blk)
        poly = [[float(a), float(b)] for a, b in
                re.findall(r"- \[([\d.\-]+), ([\d.\-]+)\]", blk)]
        out.append({"id": rid, "name": name.group(1),
                    "enabled": enabled.group(1) == "true",
                    "notes": notes.group(1) if notes else "",
                    "polygon": poly})
    return out


def selected_registry():
    sys.path.insert(0, os.path.join(HERE, "src"))
    import d2_campaign_v2_plan as P
    bundle = open(os.path.join(HERE, "src", "campaign_v2_phase05",
                               "phase0_bundle.json"), "rb").read()
    plan = P.build_v2_campaign_plan(bundle)
    if isinstance(plan, tuple):
        plan = plan[0]
    if isinstance(plan, (bytes, str)):
        plan = json.loads(plan)
    return {car: {r["station_id"].split(".", 1)[1] for r in rows}
            for car, rows in plan["station_registry"].items()}


def carriers():
    pool = json.load(open(os.path.join(
        HERE, "src", "d2_campaign_v2_candidate_pool.json"),
        encoding="utf-8"))
    sel = selected_registry()
    out = {}
    for car, obj in pool["carriers"].items():
        segs = {}
        for seg, sts in obj["segments"].items():
            rows = []
            for st in sts:
                lat, lon = st.get("lat"), st.get("lon")
                located = (isinstance(lat, (int, float))
                           and isinstance(lon, (int, float))
                           and math.isfinite(lat)
                           and math.isfinite(lon))
                # codex atlas fix 2: a coordinate-unknown station
                # stays IN the census, flagged, and is never
                # plotted -- null must not coerce to (0,0)
                rows.append({"code": st["code"],
                             "lat": lat if located else None,
                             "lon": lon if located else None,
                             "located": located,
                             "sel": st["code"] in sel.get(
                                 car, set()),
                             "assign": st.get("assignment", "")})
            segs[seg] = rows
        out[car] = {"provider": obj.get("provider", ""),
                    "segments": segs,
                    "polygons": obj["segment_polygons"]}
    return out


def mag_capsules():
    capdir = os.path.join(REPO, "docs", "f2g_window2_execution",
                          "mag_capsules")
    mags = []
    if not os.path.isdir(capdir):
        return mags
    for f in sorted(os.listdir(capdir)):
        p = os.path.join(capdir, f)
        if not (f.endswith(".json") and os.path.isfile(p)):
            continue
        c = json.load(open(p, encoding="utf-8"))
        if "coordinates_lat_lon" not in c:
            continue
        lat, lon = c["coordinates_lat_lon"]
        mags.append({"code": c["iaga_code"],
                     "name": c.get("observatory", c["iaga_code"]),
                     "lat": lat, "lon": lon,
                     "carrier": c.get("carrier", "")})
    return mags


# display-only nominal localities for daily regions without a
# registered polygon (markers, not geometry; flagged on the page)
NOMINAL = {"ridgecrest": [35.65, -117.65],
           "campi_flegrei": [40.83, 14.14],
           "kaikoura": [-42.40, 173.68],
           "anchorage": [61.20, -149.90],
           "kumamoto": [32.80, 130.70],
           "hualien": [23.97, 121.60],
           "turkey_kahramanmaras": [37.60, 37.00]}


def daily(regions):
    ens = json.load(open(os.path.join(REPO, "docs",
                                      "ensemble_latest.json"),
                         encoding="utf-8"))

    def cent(poly):
        la = sum(p[0] for p in poly) / len(poly)
        lo = sum(p[1] for p in poly) / len(poly)
        return [round(la, 3), round(lo, 3)]
    reg_cent = {r["id"]: cent(r["polygon"]) for r in regions}
    d_regions = []
    for rid, rv in ens["regions"].items():
        c = reg_cent.get(rid) or NOMINAL.get(rid)
        if c is None:
            continue
        d_regions.append({"id": rid, "tier": rv.get("tier"),
                          "tier_name": rv.get("tier_name", ""),
                          "risk": round(
                              rv.get("combined_risk") or 0, 3),
                          "lat": c[0], "lon": c[1],
                          "nominal": rid not in reg_cent})
    seen = {}
    for rid, ev in (ens.get("earthquake_events") or {}).items():
        le = (ev or {}).get("largest_event")
        if not le:
            continue
        eid = le.get("event_id")
        if eid in seen:
            seen[eid]["regions"].append(rid)
            continue
        seen[eid] = {"id": eid, "lat": le["latitude"],
                     "lon": le["longitude"],
                     "mag": round(le["magnitude"], 2),
                     "place": le.get("place", ""),
                     "time": le.get("time", "")[:16],
                     "regions": [rid],
                     "count": ev.get("event_count")}
    # codex atlas fix 5: the lag is OBSERVED from the data, never
    # a template constant
    lag_days = None
    try:
        import datetime as _dt
        d0 = _dt.date.fromisoformat(str(ens.get("date")))
        t0 = _dt.datetime.fromisoformat(
            str(ens.get("timestamp"))[:19])
        lag_days = (t0.date() - d0).days
    except (ValueError, TypeError):
        pass
    return {"date": ens.get("date"),
            "generated": ens.get("timestamp", "")[:16],
            "lag_days": lag_days,
            "summary": ens.get("summary", {}),
            "regions": sorted(
                d_regions,
                key=lambda r: (-(r["tier"] if r["tier"] is not None
                                 else -9), -r["risk"], r["id"])),
            "events": list(seen.values())}


def _fin(x):
    return (isinstance(x, (int, float))
            and not isinstance(x, bool) and math.isfinite(x))


def _latlon_ok(lat, lon):
    return (_fin(lat) and _fin(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180)


def validate_bundle(bundle):
    """codex atlas fix 3: every plotted numeric field is typed,
    finite, and in range BEFORE serialization -- a NaN or coerced
    string refuses here, never publishing a page whose script
    cannot start."""
    def bad(msg):
        raise SystemExit("ATLAS_VALIDATE_REFUSED: " + msg)
    for rg in bundle["regions"]:
        for pt in rg["polygon"]:
            if not _latlon_ok(pt[0], pt[1]):
                bad(f"region {rg['id']} polygon point {pt}")
    for car, obj in bundle["carriers"].items():
        for seg, poly in obj["polygons"].items():
            for pt in poly:
                if not _latlon_ok(pt[0], pt[1]):
                    bad(f"{car}/{seg} polygon point {pt}")
        for seg, rows in obj["segments"].items():
            for st in rows:
                if st["located"]:
                    if not _latlon_ok(st["lat"], st["lon"]):
                        bad(f"station {st['code']} coords "
                            f"{st['lat']},{st['lon']}")
                elif st["lat"] is not None or \
                        st["lon"] is not None:
                    bad(f"station {st['code']} unlocated but "
                        "carries coordinates")
    for m in bundle["mags"]:
        if not _latlon_ok(m["lat"], m["lon"]):
            bad(f"mag {m['code']} coords")
    for r in bundle["daily"]["regions"]:
        if not _latlon_ok(r["lat"], r["lon"]) or \
                not _fin(r["risk"]):
            bad(f"daily region {r['id']}")
        if r["tier"] is not None and \
                not isinstance(r["tier"], int):
            bad(f"daily region {r['id']} tier type")
    for ev in bundle["daily"]["events"]:
        if not _latlon_ok(ev["lat"], ev["lon"]) or \
                not _fin(ev["mag"]):
            bad(f"event {ev['id']}")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    """codex atlas fix 4: published source identifiers are the FULL
    sha256 of the bytes actually read, recomputed at generation --
    never a hand-written literal."""
    srcs = [("regions",
             os.path.join(HERE, "config", "regions.yaml")),
            ("candidate_pool", os.path.join(
                HERE, "src",
                "d2_campaign_v2_candidate_pool.json")),
            ("phase0_bundle", os.path.join(
                HERE, "src", "campaign_v2_phase05",
                "phase0_bundle.json")),
            ("ensemble", os.path.join(REPO, "docs",
                                      "ensemble_latest.json")),
            ("template", TEMPLATE)]
    capdir = os.path.join(REPO, "docs", "f2g_window2_execution",
                          "mag_capsules")
    if os.path.isdir(capdir):
        for f in sorted(os.listdir(capdir)):
            fp = os.path.join(capdir, f)
            if f.endswith(".json") and os.path.isfile(fp):
                srcs.append(("mag_capsule:" + f, fp))
    out = []
    for name, fp in srcs:
        out.append({"name": name,
                    "path": os.path.relpath(fp, REPO)
                    .replace(os.sep, "/"),
                    "sha256": _sha256_file(fp)})
    out.append({"name": "coastline",
                "path": "Natural Earth 110m land (public "
                        "domain), baked into the template",
                "sha256": None})
    return out


def _extract_block(html, sid):
    m = re.search('<script id="' + sid
                  + '"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise SystemExit(
            f"ATLAS_RENDER_REFUSED: block {sid} absent")
    return m.group(1).replace("<\\/", "</")


def build_bundle():
    regions = parse_regions()
    cars = carriers()
    census = {"selected": 0, "selected_located": 0,
              "pool": 0, "pool_located": 0, "unlocated_codes": []}
    for obj in cars.values():
        for rows in obj["segments"].values():
            for st in rows:
                k = "selected" if st["sel"] else "pool"
                census[k] += 1
                if st["located"]:
                    census[k + "_located"] += 1
                else:
                    census["unlocated_codes"].append(st["code"])
    census["unlocated_codes"].sort()
    return {"regions": regions, "carriers": cars,
            "mags": mag_capsules(), "daily": daily(regions),
            "census": census, "provenance": provenance()}


def main():
    bundle = build_bundle()
    validate_bundle(bundle)
    data = json.dumps(bundle, separators=(",", ":"),
                      allow_nan=False)
    data = data.replace("</", "<\\/")
    tpl = io.open(TEMPLATE, encoding="utf-8").read()
    if "__GEODATA__" not in tpl:
        raise SystemExit("ATLAS_TEMPLATE_PLACEHOLDER_MISSING")
    html = tpl.replace("__GEODATA__", data, 1)
    # codex atlas fix 3: both embedded blocks must reparse from the
    # RENDERED page (and again from the written temp bytes) before
    # the old page is replaced; a refusal deletes the temp file and
    # leaves the old page standing
    tmp = OUT + ".tmp"
    try:
        json.loads(_extract_block(html, "geodata"))
        json.loads(_extract_block(html, "landdata"))
        io.open(tmp, "w", encoding="utf-8",
                newline="").write(html)
        reread = io.open(tmp, encoding="utf-8").read()
        json.loads(_extract_block(reread, "geodata"))
        json.loads(_extract_block(reread, "landdata"))
        os.replace(tmp, OUT)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    c = bundle["census"]
    print(f"atlas: {os.path.getsize(OUT)} bytes; "
          f"date={bundle['daily']['date']} "
          f"lag={bundle['daily']['lag_days']}d "
          f"regions={len(bundle['regions'])} "
          f"selected={c['selected']} "
          f"({c['selected_located']} located) "
          f"pool={c['pool']} ({c['pool_located']} located) "
          f"unlocated={c['unlocated_codes']} "
          f"mags={len(bundle['mags'])} "
          f"daily={len(bundle['daily']['regions'])}r/"
          f"{len(bundle['daily']['events'])}ev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
