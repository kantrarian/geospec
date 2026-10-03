#!/usr/bin/env python3
"""thd_bootstrap_fetch.py - bounded raw acquisition for a THD bootstrap (grassmann 2026-10-03; owner go in-session
2026-10-03T16:30Z "go ahead with the strict fetch"; codex MEASUREMENT_SUPPORT_PACKET_REVIEW finding 1).

Fetches EXACTLY the intervals of an acquisition spec (thd_bootstrap.acquisition_spec, STRICT or MINIMAL) for one
NSLC from an FDSN data centre, one UTC day per request, and writes RAW miniSEED (no merge, no detrend, no resampling)
plus a manifest with the request parameters, the response byte count, sha256, trace count and gaps per file. The
output directory must be outside the runner tree (refused otherwise), and an existing file is never overwritten.
This is acquisition only; qualification and estimation stay in thd_bootstrap.

    python thd_bootstrap_fetch.py --spec <thd_bootstrap result json> --which strict --station IU.SNZO --location 00 \
        --out-dir E:/GeoSpec/thd_bootstrap_fetch_20261003 [--client IRIS] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thd_bootstrap import EXPECTED_CHANNEL, iso, parse_utc  # noqa: E402

SCHEMA = "geospec.thd-bootstrap-fetch.v1"
RUNNER_ROOT = Path(__file__).resolve().parent.parent.parent


def day_requests(intervals):
    """Split [start, end) intervals into UTC-day requests [D 00:00, min(D+1 00:00, end)); never beyond an interval."""
    reqs = []
    for a, b in intervals:
        s, e = parse_utc(a), parse_utc(b)
        cur = s.replace(hour=0, minute=0, second=0, microsecond=0)
        while cur < e:
            nxt = cur + timedelta(days=1)
            reqs.append((max(cur, s), min(nxt, e)))
            cur = nxt
    return reqs


def fetch(spec_path, which, station, location, out_dir, client_name="IRIS", dry_run=False, timeout=120, pause=1.0):
    spec = json.load(open(spec_path, encoding="utf-8"))["acquisition_spec"][which + ("_full_windows" if which == "strict" else "_splice")]
    net, sta = station.split(".", 1)
    out = Path(out_dir)
    try:
        if out.resolve().is_relative_to(RUNNER_ROOT.resolve()):
            raise SystemExit("REFUSED: out-dir inside the runner tree: %s" % out)
    except AttributeError:
        if str(out.resolve()).startswith(str(RUNNER_ROOT.resolve())):
            raise SystemExit("REFUSED: out-dir inside the runner tree: %s" % out)
    out.mkdir(parents=True, exist_ok=True)
    reqs = day_requests(spec["intervals"])
    manifest = dict(schema=SCHEMA, spec_source=os.path.abspath(spec_path), spec=which, station=station, location=location,
                    channel=EXPECTED_CHANNEL, client=client_name, intervals=spec["intervals"], n_requests=len(reqs),
                    started_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), files=[], refusals=[])
    print("FETCH", which, station, location, "requests", len(reqs), "dry_run" if dry_run else client_name)
    if dry_run:
        for s, e in reqs:
            print("  would request", iso(s), "->", iso(e))
        return manifest
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    client = Client(client_name, timeout=timeout)
    for i, (s, e) in enumerate(reqs, 1):
        name = f"{net}.{sta}.{location}.{EXPECTED_CHANNEL}.{s.strftime('%Y%m%dT%H%M%S')}.mseed"
        path = out / name
        rec = dict(file=name, request=dict(network=net, station=sta, location=location, channel=EXPECTED_CHANNEL,
                                            starttime=iso(s), endtime=iso(e)))
        if path.exists():
            rec.update(status="EXISTS_SKIPPED", bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest["files"].append(rec); print("  skip (exists)", name); continue
        t0 = time.time()
        try:
            st = client.get_waveforms(network=net, station=sta, location=location, channel=EXPECTED_CHANNEL,
                                      starttime=UTCDateTime(s), endtime=UTCDateTime(e))
        except Exception as exc:          # recorded by name; no retry loop
            rec.update(status="REFUSED", error=type(exc).__name__ + ": " + str(exc)[:200], seconds=round(time.time() - t0, 1))
            manifest["refusals"].append(rec); print("  REFUSED", name, rec["error"][:80]); time.sleep(pause); continue
        if len(st) == 0:
            rec.update(status="EMPTY", seconds=round(time.time() - t0, 1)); manifest["refusals"].append(rec)
            print("  EMPTY", name); time.sleep(pause); continue
        st.write(str(path), format="MSEED")       # raw: no merge, no detrend
        raw = path.read_bytes()
        gaps = st.get_gaps()
        rec.update(status="WRITTEN", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), n_traces=len(st),
                   npts=int(sum(tr.stats.npts for tr in st)), sampling_rates=sorted({float(tr.stats.sampling_rate) for tr in st}),
                   first=str(st[0].stats.starttime), last=str(st[-1].stats.endtime), n_gaps=len(gaps),
                   gap_seconds=round(float(sum(abs(g[6]) for g in gaps)), 3), seconds=round(time.time() - t0, 1))
        manifest["files"].append(rec)
        print("  %3d/%d %s %d B traces=%d gaps=%d %.1fs" % (i, len(reqs), name, len(raw), len(st), len(gaps), rec["seconds"]))
        with open(out / "FETCH_MANIFEST.partial.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        time.sleep(pause)
    manifest["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["total_bytes"] = sum(f.get("bytes", 0) for f in manifest["files"])
    with open(out / "FETCH_MANIFEST.json", "x", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    try:
        os.remove(out / "FETCH_MANIFEST.partial.json")
    except OSError:
        pass
    print("DONE files=%d refusals=%d total_bytes=%d" % (len(manifest["files"]), len(manifest["refusals"]), manifest["total_bytes"]))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True); ap.add_argument("--which", choices=("strict", "minimal"), required=True)
    ap.add_argument("--station", required=True); ap.add_argument("--location", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--client", default="IRIS")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    fetch(a.spec, a.which, a.station, a.location, a.out_dir, a.client, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
