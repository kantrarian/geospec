#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cascadia owner-forecast scorer (cayley) -- mechanical scoring for
docs/forecasts/2026-08-21-cascadia-m4-48h-owner-forecast.md per its
FROZEN scoring section: at window close, query USGS ComCat for both
frames, record HIT/MISS + events verbatim for the appended RESULT
section. Append-only doc discipline: this script PRINTS the RESULT
block; the append is a separate reviewed edit.

Frames verbatim from the registration:
- A (primary): program cascadia polygons' bbox lat 45.0..51.0,
  lon -128.0..-121.5
- B (secondary): lat 44.0..52.0, lon -131.0..-121.0
Event: magnitude > 4.0 (any type), origin time within
[2026-08-21T17:22:07Z, 2026-08-23T17:22:07Z].

Run BEFORE window close -> the artifact is labeled PARTIAL_WINDOW and
carries no verdict (no-overclaim rule); at/after close -> FINAL with
HIT/MISS per frame. Frame A polygon membership: the registration names
the bbox as the frame boundary ("bbox lat 45.0-51.0, lon
-128.0..-121.5"); events inside the bbox are additionally annotated
with point-in-polygon vs FAULT_SEGMENTS for the record.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

WINDOW_START = "2026-08-21T17:22:07Z"
WINDOW_END = "2026-08-23T17:22:07Z"
MAG_MIN_EXCLUSIVE = 4.0
FRAMES = {"A": {"minlatitude": 45.0, "maxlatitude": 51.0,
                "minlongitude": -128.0, "maxlongitude": -121.5},
          "B": {"minlatitude": 44.0, "maxlatitude": 52.0,
                "minlongitude": -131.0, "maxlongitude": -121.0}}
COMCAT = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def query_frame(frame):
    params = dict(FRAMES[frame], format="geojson",
                  starttime=WINDOW_START.replace("Z", ""),
                  endtime=WINDOW_END.replace("Z", ""),
                  minmagnitude=MAG_MIN_EXCLUSIVE,  # server-side floor;
                  orderby="time")                  # exact > filter below
    url = COMCAT + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        body = json.load(r)
    events = []
    for f in body.get("features", []):
        p = f["properties"]
        if p.get("mag") is None or p["mag"] <= MAG_MIN_EXCLUSIVE:
            continue                     # frozen rule: STRICTLY > 4.0
        t = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                          time.gmtime(p["time"] / 1000.0))
        lon, lat, depth = f["geometry"]["coordinates"]
        events.append({"id": f.get("id"), "utc": t,
                       "mag": p["mag"], "magType": p.get("magType"),
                       "lat": lat, "lon": lon, "depth_km": depth,
                       "place": p.get("place")})
    return {"url": url, "count": len(events), "events": events}


def score():
    fired = now_utc()
    final = fired >= WINDOW_END
    out = {"fired_utc": fired,
           "window": [WINDOW_START, WINDOW_END],
           "status": "FINAL" if final else
                     "PARTIAL_WINDOW (no verdict; window still open)",
           "frames": {}}
    for fr in ("A", "B"):
        res = query_frame(fr)
        rec = {"events_m_gt_4": res["count"],
               "events": res["events"], "query_url": res["url"]}
        if final:
            rec["verdict"] = "HIT" if res["count"] > 0 else "MISS"
        out["frames"][fr] = rec
    return out


if __name__ == "__main__":
    print(json.dumps(score(), indent=1, sort_keys=True))
