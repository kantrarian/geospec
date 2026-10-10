"""Replay of the IU.SNZO bound operator, v1 (as served, public commit 23aa8658) against v2 (this tree), over RETAINED bytes
only (cayley 2026-10-10; codex b7a29458 correction 2: "retained 91-day replay where bytes actually exist, with counts
recovered/still-refused/unknown"). Reads miniSEED already on disk; makes no request of any provider.

For every requested day D (the operator's daily window [D - 25 h, D 00:00 UTC), exactly as daily_measurement asks) it
runs BOTH stitch_window versions on the same retained IU.SNZO.00.BHZ records and classifies:
  RECOVERED       v1 refused, v2 accepted (named: the v2 identical-overlap samples joined)
  STILL_REFUSED   both refuse; v2's refusal kind and facts are reported
  BOTH_ACCEPT     both accept; their supports MUST be byte-identical (sha256 compared) -- a difference is a defect
  V2_ONLY_REFUSES v1 accepted, v2 refuses (expected only for masked/nonfinite input) -- reported, never hidden
  UNKNOWN         no retained bytes cover any of the window: not measured, never counted as either

Usage (host, pinned venv, cwd = <geospec tree>/monitoring/src):
  python -B thd_bound_operator_replay_cayley_20261010.py --retained <dir of miniSEED> --days 2026-06-11:2026-09-09 --out <json>
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V1_COMMIT = "23aa86586b6bc321ec6330cc7cfa5e3982cf942b"


def load_v1():
    source = subprocess.run(["git", "-C", str(HERE), "show", "%s:monitoring/src/thd_bound_station_operator.py" % V1_COMMIT],
                            capture_output=True, check=True).stdout
    tmp = tempfile.NamedTemporaryFile("wb", suffix="_v1_operator.py", delete=False)
    tmp.write(source); tmp.close()
    spec = importlib.util.spec_from_file_location("thd_bound_station_operator_v1", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)
    return module


def days(spec):
    lo, hi = spec.split(":")
    d, end = datetime.fromisoformat(lo), datetime.fromisoformat(hi)
    while d <= end:
        yield d
        d += timedelta(days=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retained", required=True, help="directory of retained miniSEED (read only)")
    ap.add_argument("--days", required=True, help="first:last calibration/scored day, inclusive (YYYY-MM-DD:YYYY-MM-DD)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from obspy import Stream, UTCDateTime, read
    import thd_bound_station_operator as V2
    V1 = load_v1()
    spec = V2.BOUND_STATIONS["IU.SNZO"]
    files = sorted(p for p in Path(a.retained).rglob("*") if p.is_file())
    retained = Stream()
    unreadable = []
    for p in files:
        try:
            retained += read(str(p)).select(network="IU", station="SNZO", location=spec["location"], channel=spec["channel"])
        except Exception as exc:  # noqa: BLE001 -- a non-miniSEED file is listed, never guessed at
            unreadable.append(dict(path=str(p), error=type(exc).__name__))
    epochs = V2._retained_epochs("IU.SNZO.%s.%s" % (spec["location"], spec["channel"]))
    rows, counts = [], dict(RECOVERED=0, STILL_REFUSED=0, BOTH_ACCEPT=0, V2_ONLY_REFUSES=0, UNKNOWN=0, BOTH_ACCEPT_BYTES_DIFFER=0)
    for day in days(a.days):
        end = day.replace(tzinfo=timezone.utc)
        start = end - timedelta(hours=25)
        records = [tr.copy() for tr in retained if tr.stats.endtime > UTCDateTime(start) and tr.stats.starttime < UTCDateTime(end)]
        if not records:
            rows.append(dict(day=day.date().isoformat(), outcome="UNKNOWN", records=0)); counts["UNKNOWN"] += 1
            continue
        kw = dict(network="IU", station="SNZO", location=spec["location"], channel=spec["channel"],
                  expected_rate=spec["expected_rate_hz"], start=start.replace(tzinfo=None), end=end.replace(tzinfo=None))
        d1, _, r1 = V1.stitch_window([tr.copy() for tr in records], **kw)
        d2, _, r2 = V2.stitch_window([tr.copy() for tr in records], response_epochs=epochs, **kw)
        ok1, ok2 = d1 is not None, d2 is not None
        row = dict(day=day.date().isoformat(), records=len(records), v1_refusal=r1.get("refusal"), v2_refusal=r2.get("refusal"),
                   v2_discontinuity=r2.get("discontinuity"), v2_coverage=r2.get("coverage"),
                   v2_identical_overlap_samples=r2.get("identical_overlap_samples"))
        if ok1 and ok2:
            same = (r1.get("support_sha256") == r2.get("support_sha256"))
            row.update(outcome="BOTH_ACCEPT", support_identical=same)
            counts["BOTH_ACCEPT"] += 1
            if not same:
                counts["BOTH_ACCEPT_BYTES_DIFFER"] += 1
        elif ok2:
            row.update(outcome="RECOVERED"); counts["RECOVERED"] += 1
        elif ok1:
            row.update(outcome="V2_ONLY_REFUSES"); counts["V2_ONLY_REFUSES"] += 1
        else:
            row.update(outcome="STILL_REFUSED"); counts["STILL_REFUSED"] += 1
        rows.append(row)
    receipt = dict(schema="cayley.thd-bound-operator-replay.v1", generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   retained_dir=str(a.retained), files=len(files), unreadable=unreadable, days=a.days,
                   v1=dict(commit=V1_COMMIT, operator_identity=V1.operator_identity("IU.SNZO")),
                   v2=dict(operator_identity=V2.operator_identity("IU.SNZO"), version=spec["operator_version"]),
                   counts=counts, rows=rows,
                   not_claimed=["no provider was contacted", "UNKNOWN days are unmeasured, not refused or accepted",
                                "a RECOVERED day is an operator outcome, not a calibration or a score"])
    Path(a.out).write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))
    return 1 if counts["BOTH_ACCEPT_BYTES_DIFFER"] else 0


if __name__ == "__main__":
    sys.exit(main())
