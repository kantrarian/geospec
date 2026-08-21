#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NET-1 v1 -- network integrity probe (cayley, owner-directed
2026-08-21). OPERATIONAL TELEMETRY, no-claims lane: measures whether the
seismic stations the program depends on are currently delivering data,
and how stale each one is. No science, no prediction, no measurement
values retained -- byte counts and HTTP statuses only.

v1 scope: the 35 selected registry stations (KO + CI networks). Per
station, two gentle FDSN dataselect probes (last 1h, then last 24h only
if the 1h window is empty) classify:
  LIVE          bytes in the last hour
  STALE_24H     no last-hour bytes, but bytes within 24h
  NO_DATA_24H   reachable but no bytes in 24h
  HTTP_ERROR    endpoint error/timeout
Per-carrier coverage index = live fraction over selected stations.
Output: docs/net1/net1_latest.json + a dated snapshot. Intended to run
from the daily runner; alerts + map layer iterate later per the roadmap.
Usage: probe.py <repo_root>
"""
import json
import os
import sys
import time
import urllib.request

ENDPOINTS = {
    "KO": "http://eida.koeri.boun.edu.tr/fdsnws/dataselect/1/query",
    "CI": "https://service.scedc.caltech.edu/fdsnws/dataselect/1/query",
}
CARRIERS = ("istanbul_marmara", "socal_coachella", "turkey_kahramanmaras")
TIMEOUT = 25


def utc(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))


def fetch_bytes(net, sta, t0, t1):
    url = (f"{ENDPOINTS[net]}?net={net}&sta={sta}&loc=*&cha=HHZ"
           f"&starttime={utc(t0)}&endtime={utc(t1)}")
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "geo2graph-net1/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, len(r.read())
    except urllib.error.HTTPError as e:
        return e.code, 0
    except Exception:
        return None, 0


def main(repo):
    root = os.path.join(repo, "data/phase_a_builder_artifact_v1/tables")
    mo = [json.loads(l) for l in open(os.path.join(root,
                                                   "member_of.jsonl"),
                                      encoding="utf-8") if l.strip()]
    now = time.time()
    stations = []
    for m in sorted(mo, key=lambda x: (x["carrier_key"],
                                       x["station_id"])):
        net, sta = m["station_id"].split(".", 1)
        if net not in ENDPOINTS:
            stations.append({"station_id": m["station_id"],
                             "carrier": m["carrier_key"],
                             "status": "NO_ENDPOINT"})
            continue
        s1, b1 = fetch_bytes(net, sta, now - 3600, now)
        b24 = None
        s24 = None
        if b1 > 0:
            status = "LIVE"
        else:
            s24, b24 = fetch_bytes(net, sta, now - 86400, now)
            if b24 and b24 > 0:
                status = "STALE_24H"
            elif s24 in (200, 204, 404):
                status = "NO_DATA_24H"
            else:
                status = "HTTP_ERROR"
        row = {"station_id": m["station_id"],
               "carrier": m["carrier_key"], "network": net,
               "status": status, "http_1h": s1, "bytes_1h": b1}
        if b24 is not None:
            row["http_24h"] = s24
            row["bytes_24h"] = b24
        stations.append(row)
        print(f"[{m['carrier_key']}] {m['station_id']}: {status} "
              f"(1h {b1}B)", flush=True)
    regions = {}
    for ck in CARRIERS:
        rows = [s for s in stations if s["carrier"] == ck]
        live = sum(1 for s in rows if s["status"] == "LIVE")
        regions[ck] = {"selected": len(rows), "live": live,
                       "stale_24h": sum(1 for s in rows
                                        if s["status"] == "STALE_24H"),
                       "unreachable_or_empty":
                           sum(1 for s in rows if s["status"] in
                               ("NO_DATA_24H", "HTTP_ERROR")),
                       "coverage": round(live / len(rows), 3)
                       if rows else None}
    out = {"schema": "net1-integrity-snapshot-v1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "lane": "operational-telemetry; no scientific claims; no "
                   "measurement values retained (byte counts + HTTP "
                   "status only)",
           "probe": "FDSN dataselect HHZ, 1h window, 24h fallback",
           "regions": regions, "stations": stations}
    outdir = os.path.join(repo, "docs", "net1")
    os.makedirs(outdir, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    for name in (f"net1_snapshot_{day}.json", "net1_latest.json"):
        with open(os.path.join(outdir, name), "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
    print(json.dumps(regions, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "."))
