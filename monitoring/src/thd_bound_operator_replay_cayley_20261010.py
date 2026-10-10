"""Replay of the IU.SNZO bound operator, v1 (as served, public commit 23aa8658) against v2 (this tree), over RETAINED bytes
only (cayley 2026-10-10; codex b7a29458 correction 2: "retained 91-day replay where bytes actually exist, with counts
recovered/still-refused/unknown"). Reads bytes already on disk; makes no request of any provider.

For every requested day D (the operator's daily window [D - 25 h, D 00:00 UTC), exactly as daily_measurement asks) it
runs BOTH stitch_window versions on the same retained IU.SNZO.00.BHZ records and classifies:
  RECOVERED       v1 refused, v2 accepted (named: the v2 identical-overlap samples joined)
  STILL_REFUSED   both refuse; v2's refusal kind and facts are reported
  BOTH_ACCEPT     both accept; their supports MUST be byte-identical (sha256 compared) -- a difference is a defect
  V2_ONLY_REFUSES v1 accepted, v2 refuses (expected only for masked/nonfinite input) -- reported, never hidden
  UNKNOWN         the retained bytes do not hold the window's input: not measured, never counted as either

Two retained layouts:
  --retained <dir>        miniSEED files (any layout); every IU.SNZO.00.BHZ record overlapping the window is handed over.
  --runner-cache <dir>    the runner's monitoring/data/seismic_cache: <dir>/<region>/<YYYYMMDD>/<file>.pkl, a pickled
                          {"IU.SNZO": obspy Stream} per cache day (the fault-correlation cache; grassmann e5a34e44 used
                          the same bytes for the parity). For day D the traces of cache days D-2 and D-1 are handed over
                          whole; the operator cuts the window. A missing cache day is UNKNOWN (CACHE_DAY_MISSING).
     INPUT KIND (grassmann e5a34e44 s.2): a cache day is an independent 24 h fetch, so the window is assembled from TWO
     fetches -- not the one continuous provider response production received. Every row says so
     (input_kind = RUNNER_CACHE_TWO_INDEPENDENT_FETCHES) and labels its cache runs: MIDNIGHT_START (the bytes reproduced
     the issued values in grassmann's parity) or NON_MIDNIGHT_START (a different fetch time than production; not the
     issued input). Operator outcomes on these rows are facts about these bytes, not about the issued run.

Usage (host, pinned venv, cwd = <geospec tree>/monitoring/src):
  python -B thd_bound_operator_replay_cayley_20261010.py --runner-cache C:/GeoSpec/geospec_runner/monitoring/data/seismic_cache --days 2026-06-11:2026-09-09 --out <json>
  python -B thd_bound_operator_replay_cayley_20261010.py --retained <dir of miniSEED> --days A:B --out <json>
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V1_COMMIT = "23aa86586b6bc321ec6330cc7cfa5e3982cf942b"
CACHE_REGION, CACHE_FILE, CACHE_KEY = "kaikoura", "hope_fault_waveforms.pkl", "IU.SNZO"
INPUT_KIND_CACHE = "RUNNER_CACHE_TWO_INDEPENDENT_FETCHES"
INPUT_KIND_MSEED = "RETAINED_MINISEED_AS_FOUND"


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


def served_id(spec):
    return "IU.SNZO.%s.%s" % (spec["location"], spec["channel"])


class MiniSeedSource:
    kind = INPUT_KIND_MSEED

    def __init__(self, root, spec):
        from obspy import Stream, read
        self.files = sorted(p for p in Path(root).rglob("*") if p.is_file())
        self.unreadable, self.stream = [], Stream()
        for p in self.files:
            try:
                self.stream += read(str(p)).select(network="IU", station="SNZO", location=spec["location"], channel=spec["channel"])
            except Exception as exc:  # noqa: BLE001 -- a non-miniSEED file is listed, never guessed at
                self.unreadable.append(dict(path=str(p), error=type(exc).__name__))

    def records(self, day, start, end):
        from obspy import UTCDateTime
        recs = [tr.copy() for tr in self.stream if tr.stats.endtime > UTCDateTime(start) and tr.stats.starttime < UTCDateTime(end)]
        return (recs, {}) if recs else (None, dict(unknown_reason="NO_RETAINED_RECORD_OVERLAPS_THE_WINDOW"))


class RunnerCacheSource:
    kind = INPUT_KIND_CACHE

    def __init__(self, root, spec):
        self.root, self.spec, self.unreadable, self.files = Path(root), spec, [], []

    def cache_day(self, day):
        p = self.root / CACHE_REGION / day.strftime("%Y%m%d") / CACHE_FILE
        if not p.is_file():
            return None, dict(day=day.date().isoformat(), status="FILE_ABSENT")
        raw = p.read_bytes()
        self.files.append(str(p))
        try:
            obj = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 -- a bad pickle is a typed status
            self.unreadable.append(dict(path=str(p), error=type(exc).__name__))
            return None, dict(day=day.date().isoformat(), status="UNREADABLE")
        stream = obj.get(CACHE_KEY) if isinstance(obj, dict) else None
        traces = [tr for tr in (stream or []) if tr.id == served_id(self.spec)]
        if not traces:
            return None, dict(day=day.date().isoformat(), status="KEY_OR_TRACE_ABSENT")
        first = min(tr.stats.starttime for tr in traces)
        return traces, dict(day=day.date().isoformat(), status="PRESENT", sha256=hashlib.sha256(raw).hexdigest(),
                            n_traces=len(traces), start=str(first),
                            cache_run=("MIDNIGHT_START" if first.hour == 0 else "NON_MIDNIGHT_START"))

    def records(self, day, start, end):
        recs, meta = [], dict(cache_days=[])
        for back in (2, 1):
            traces, facts = self.cache_day(day - timedelta(days=back))
            meta["cache_days"].append(facts)
            if traces is None:
                return None, dict(meta, unknown_reason="CACHE_DAY_MISSING")
            recs.extend(tr.copy() for tr in traces)
        runs = {f["cache_run"] for f in meta["cache_days"]}
        meta["issued_input"] = ("CACHE_RUNS_MIDNIGHT_START" if runs == {"MIDNIGHT_START"}
                                else "CACHE_RUN_NON_MIDNIGHT_START_NOT_THE_ISSUED_INPUT")
        return recs, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--retained", help="directory of retained miniSEED (read only)")
    src.add_argument("--runner-cache", help="the runner's monitoring/data/seismic_cache directory (read only)")
    ap.add_argument("--days", required=True, help="first:last calibration/scored day, inclusive (YYYY-MM-DD:YYYY-MM-DD)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import thd_bound_station_operator as V2
    V1 = load_v1()
    spec = V2.BOUND_STATIONS["IU.SNZO"]
    source = MiniSeedSource(a.retained, spec) if a.retained else RunnerCacheSource(a.runner_cache, spec)
    epochs = V2._retained_epochs(served_id(spec))
    rows, counts = [], dict(RECOVERED=0, STILL_REFUSED=0, BOTH_ACCEPT=0, V2_ONLY_REFUSES=0, UNKNOWN=0, BOTH_ACCEPT_BYTES_DIFFER=0)
    by_issued = {}
    for day in days(a.days):
        end = day.replace(tzinfo=timezone.utc)
        start = end - timedelta(hours=25)
        records, meta = source.records(day, start, end)
        if records is None:
            rows.append(dict(day=day.date().isoformat(), outcome="UNKNOWN", input_kind=source.kind, **meta))
            counts["UNKNOWN"] += 1
            continue
        kw = dict(network="IU", station="SNZO", location=spec["location"], channel=spec["channel"],
                  expected_rate=spec["expected_rate_hz"], start=start.replace(tzinfo=None), end=end.replace(tzinfo=None))
        d1, _, r1 = V1.stitch_window([tr.copy() for tr in records], **kw)
        d2, _, r2 = V2.stitch_window([tr.copy() for tr in records], response_epochs=epochs, **kw)
        ok1, ok2 = d1 is not None, d2 is not None
        row = dict(day=day.date().isoformat(), input_kind=source.kind, records=len(records), v1_refusal=r1.get("refusal"),
                   v2_refusal=r2.get("refusal"), v2_discontinuity=r2.get("discontinuity"), v2_coverage=r2.get("coverage"),
                   v2_identical_overlap_samples=r2.get("identical_overlap_samples"), **meta)
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
        issued = meta.get("issued_input", "NOT_CLASSIFIED")
        by_issued.setdefault(issued, {}).setdefault(row["outcome"], 0)
        by_issued[issued][row["outcome"]] += 1
        rows.append(row)
    receipt = dict(schema="cayley.thd-bound-operator-replay.v2", generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   source=dict(kind=source.kind, root=str(a.retained or a.runner_cache), files=len(source.files),
                               unreadable=source.unreadable),
                   days=a.days, v1=dict(commit=V1_COMMIT, operator_identity=V1.operator_identity("IU.SNZO")),
                   v2=dict(operator_identity=V2.operator_identity("IU.SNZO"), version=spec["operator_version"]),
                   counts=counts, counts_by_issued_input=by_issued, rows=rows,
                   not_claimed=["no provider was contacted", "UNKNOWN days are unmeasured, not refused or accepted",
                                "a RECOVERED day is an operator outcome, not a calibration or a score",
                                "a runner-cache window is assembled from two independent fetches, not the production response"])
    Path(a.out).write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(dict(counts=counts, by_issued_input=by_issued), sort_keys=True))
    return 1 if counts["BOTH_ACCEPT_BYTES_DIFFER"] else 0


if __name__ == "__main__":
    sys.exit(main())
