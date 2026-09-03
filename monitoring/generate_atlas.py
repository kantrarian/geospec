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
import subprocess
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


# 2026-09-01 (grassmann program review B2/B4): the three ensemble
# components and the CLOSED status vocabulary the page renders. "live"
# is the ensemble's own definition of a counted method (available and
# not frozen -- ensemble.py methods_available); the unavailable classes
# are read from the component's own notes so the page says WHY a
# component is dark instead of silently renormalising over the rest.
COMPONENTS = ("lambda_geo", "fault_correlation", "seismic_thd")
COMPONENT_STATUS = ("live", "frozen", "stale", "no_registry", "no_data")


def component_status(c):
    if not isinstance(c, dict):
        return "no_data"
    # `frozen` is an explicit disposition and takes precedence over
    # availability. Real ensemble rows can be both unavailable and
    # frozen; classifying those from their explanatory note loses the
    # stronger state on the public surface.
    if c.get("frozen"):
        return "frozen"
    if c.get("available"):
        return "live"
    notes = str(c.get("notes") or "").lower()
    if "stale" in notes or "valid_through" in notes:
        return "stale"
    if "no registry entry" in notes:
        return "no_registry"
    return "no_data"


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
        comps = rv.get("components") or {}
        status = {k: component_status(comps.get(k)) for k in COMPONENTS}
        pers = rv.get("persistence") or {}
        weights = rv.get("effective_weights") or {}
        d_regions.append({"id": rid, "tier": rv.get("tier"),
                          "tier_name": rv.get("tier_name", ""),
                          "risk": round(
                              rv.get("combined_risk") or 0, 3),
                          "lat": c[0], "lon": c[1],
                          "nominal": rid not in reg_cent,
                          # qualifier fields (B2): a tier is never
                          # rendered without how many methods carried it
                          "methods_available": rv.get("methods_available"),
                          "agreement": rv.get("agreement") or "",
                          "confirmed": bool(pers.get("is_confirmed", False)),
                          "components": status,
                          "weights": {k: round(float(v), 3)
                                      for k, v in weights.items()}})
    # per-component live/dark census over the rendered regions (B4):
    # derived from the SAME status field the rows carry
    components_live = {}
    for k in COMPONENTS:
        counts = {s: 0 for s in COMPONENT_STATUS}
        for r in d_regions:
            counts[r["components"][k]] += 1
        counts["n"] = len(d_regions)
        components_live[k] = counts
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
            "components_live": components_live,
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
        # B2/B4 qualifier fields: present, closed vocabulary, and
        # CONSISTENT -- methods_available must equal the number of
        # live components and the effective weights must be carried
        # by exactly those components (ensemble.py renormalises over
        # available, non-frozen components; a divergence here is an
        # ensemble inconsistency the page must not paper over)
        for key in ("methods_available", "agreement", "confirmed",
                    "components", "weights"):
            if key not in r:
                bad(f"daily region {r['id']} lacks {key}")
        m = r["methods_available"]
        if not isinstance(m, int) or isinstance(m, bool) or \
                not 0 <= m <= len(COMPONENTS):
            bad(f"daily region {r['id']} methods_available type")
        if not isinstance(r["agreement"], str):
            bad(f"daily region {r['id']} agreement type")
        if not isinstance(r["confirmed"], bool):
            bad(f"daily region {r['id']} confirmed type")
        comps = r["components"]
        if not isinstance(comps, dict) or \
                set(comps) != set(COMPONENTS):
            bad(f"daily region {r['id']} components keys")
        for k, s in comps.items():
            if s not in COMPONENT_STATUS:
                bad(f"daily region {r['id']} component {k} "
                    f"status {s!r}")
        live = {k for k, s in comps.items() if s == "live"}
        w = r["weights"]
        if not isinstance(w, dict) or not set(w) <= set(COMPONENTS):
            bad(f"daily region {r['id']} weights keys")
        for k, v in w.items():
            if not _fin(v) or not 0 <= v <= 1:
                bad(f"daily region {r['id']} weight {k}={v!r}")
        if m != len(live):
            bad(f"daily region {r['id']} methods_available={m} but "
                f"{len(live)} live components {sorted(live)}")
        if set(w) != live:
            bad(f"daily region {r['id']} weights carried by "
                f"{sorted(w)} but live components are {sorted(live)}")
    cl = bundle["daily"].get("components_live")
    if not isinstance(cl, dict) or set(cl) != set(COMPONENTS):
        bad("components_live census absent or mis-keyed")
    expected_cl = {}
    for k in COMPONENTS:
        counts = {s: 0 for s in COMPONENT_STATUS}
        for r in bundle["daily"]["regions"]:
            counts[r["components"][k]] += 1
        counts["n"] = len(bundle["daily"]["regions"])
        expected_cl[k] = counts
    if cl != expected_cl:
        bad("components_live census does not derive from daily rows")
    for ev in bundle["daily"]["events"]:
        if not _latlon_ok(ev["lat"], ev["lon"]) or \
                not _fin(ev["mag"]):
            bad(f"event {ev['id']}")


def _committed_blob(rel):
    """The bytes git holds for `rel` at HEAD, or None when the path is
    absent from HEAD (a new file) or git cannot be run. None means
    "cannot compare", never "compared and matched"."""
    try:
        p = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                            "HEAD:" + rel], capture_output=True)
    except OSError:
        return None
    return p.stdout if p.returncode == 0 else None


def _refuse_eol_view(name, rel, path):
    """grassmann 1933Z MEDIUM. Pinning a path `-text` does NOT rewrite
    a copy already on disk: a checkout made before the pin keeps its
    CRLF bytes, git never re-smudges an unchanged file when attributes
    change, and this table then publishes digests that name no
    committed bytes -- which is exactly the defect on master today.

    The pin fixes the repository; it does not fix a checkout. So the
    check is here, where the digest is taken, and not only in a
    landing recipe that has to be remembered.

    REFUSE only when the live bytes differ from the committed blob
    ONLY in line endings. A genuine content change is left alone --
    the daily runner rewrites docs/ensemble_latest.json before this
    runs, and normalising that does NOT reproduce the blob, so the
    daily path is unaffected. A refusal leaves the old page standing.
    """
    blob = _committed_blob(rel)
    if blob is None:
        return
    with open(path, "rb") as f:
        live = f.read()
    if live == blob or live.replace(b"\r\n", b"\n") != blob:
        return
    raise SystemExit(
        "ATLAS_PROVENANCE_EOL_VIEW: " + rel + " (" + name + ") differs "
        "from its committed blob only in line endings, so its recorded "
        "digest would name no committed bytes. This checkout predates "
        "the -text pin; the pin does not rewrite files already on "
        "disk. First require an empty git status --porcelain, then "
        "remove and restore only the pinned inputs, then regenerate: "
        "git -C <repo> rm -q -- monitoring/config/regions.yaml "
        "monitoring/atlas_template.html docs/ensemble_latest.json "
        "docs/f2g_window2_execution/mag_capsules/"
        "mag_capsule_frn.json docs/f2g_window2_execution/mag_capsules/"
        "mag_capsule_izn.json docs/f2g_window2_execution/mag_capsules/"
        "mag_capsule_tuc.json docs/f2g_window2_execution/mag_capsules/"
        "mag_capsule_vic.json && git -C <repo> checkout HEAD -- "
        "monitoring/config/regions.yaml monitoring/atlas_template.html "
        "docs/ensemble_latest.json docs/f2g_window2_execution/"
        "mag_capsules/mag_capsule_frn.json docs/f2g_window2_execution/"
        "mag_capsules/mag_capsule_izn.json docs/f2g_window2_execution/"
        "mag_capsules/mag_capsule_tuc.json docs/f2g_window2_execution/"
        "mag_capsules/mag_capsule_vic.json")


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
        try:
            rel = os.path.relpath(fp, REPO).replace(os.sep, "/")
        except ValueError:
            # a source on another mount cannot be named relative to the
            # repository. Refuse: publishing the absolute path instead
            # would leak a local filesystem path onto a public page.
            raise SystemExit(
                "ATLAS_PROVENANCE_SOURCE_OUTSIDE_REPO: " + name +
                " resolves to " + fp + ", which shares no mount with " +
                REPO)
        if rel.startswith("../"):
            raise SystemExit(
                "ATLAS_PROVENANCE_SOURCE_OUTSIDE_REPO: " + name +
                " resolves outside the repository (" + rel + ")")
        _refuse_eol_view(name, rel, fp)
        out.append({"name": name, "path": rel,
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
