#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CERTIFICATION CAMPAIGN RUNNER (cayley) -- per-point
process parallelism under codex's BINDING 1544Z ack.

THE ACK'S TERMS, implemented structurally:
- The unit of dispatch is the WHOLE POINT: one worker process runs one
  certification start-to-finish, so one point's replicate sequence can
  NEVER split across workers (there is no replicate-level dispatch
  surface in this module at all).
- Same-family points deliberately share common random numbers (the
  registered seed grammar omits the point); scheduling is irrelevant
  because every worker reconstructs its own (authority, family, r)
  sequence -- this runner adds NO randomness and NO seed handling.
- The PRE-FIRE INVOCATION RECORD is written BEFORE any worker starts:
  argv verbatim, process count, interpreter (version + executable),
  host, the exact ordered point list + its digest, the manifest
  commit/geometry ref, and a live-UTC fire timestamp.

Points file (JSON, ordered): [{"family": "B1B", "point": {...},
"entry": "detection"|"specificity"}, ...]. Detection points route to
run_point_certification; specificity points to
run_b1b_specificity_certification (the codex-repaired boundary).
The runner passes NO overrides -- certification constructs its own
R / n_draws / seed authority (POWER_CERTIFICATION_CONFIG_UNBOUND
stays live).

Outputs: <outdir>/invocation_record.json (pre-fire),
<outdir>/point_<idx>.json per point, <outdir>/campaign_summary.json
(post-run; per-point record digests; refusals recorded verbatim).

No certification runs at import or selftest; the selftest exercises
the scheduler + record shapes with stub workers only. Opens no
window-2 value.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

POINT_ENTRIES = {"detection", "specificity"}


class RunnerRefusal(ValueError):
    pass


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _digest(obj):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def load_points(path):
    with open(path, encoding="utf-8") as f:
        pts = json.load(f)
    if not isinstance(pts, list) or not pts:
        raise RunnerRefusal("RUNNER_POINTS_INVALID: empty/non-list")
    seen = set()
    for i, p in enumerate(pts):
        if not isinstance(p, dict) or \
                set(p) != {"family", "point", "entry"}:
            raise RunnerRefusal(
                f"RUNNER_POINTS_INVALID: entry {i} schema not closed")
        if p["entry"] not in POINT_ENTRIES:
            raise RunnerRefusal(
                f"RUNNER_POINTS_INVALID: entry {i} kind {p['entry']!r}")
        if p["entry"] == "specificity" and (
                p["family"] != "B1B" or set(p["point"]) != {"gain"}):
            raise RunnerRefusal(
                f"RUNNER_POINTS_INVALID: entry {i} specificity must "
                "be a B1B {gain} point")
        key = _canon(p)
        if key in seen:
            raise RunnerRefusal(
                f"RUNNER_POINTS_INVALID: entry {i} duplicates an "
                "earlier point")
        seen.add(key)
    return pts


def write_invocation_record(outdir, points, manifest_commit,
                            geometry_path, n_procs, argv):
    """codex 1544Z: recorded PRE-FIRE, before any worker starts."""
    rec = {
        "schema": "f2g-w2-cert-invocation-v1",
        "fired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime()),
        "argv": list(argv),
        "process_count": int(n_procs),
        "interpreter": {"executable": sys.executable,
                        "version": sys.version,
                        "platform": platform.platform()},
        "host": platform.node(),
        "manifest_commit": str(manifest_commit),
        "geometry_path": str(geometry_path),
        "ordered_points": points,
        "ordered_points_sha256": _digest(points),
        "dispatch_rule": "whole-point-per-process; one point's "
                         "replicate sequence never splits across "
                         "workers",
        "overrides": None}
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "invocation_record.json")
    if os.path.exists(p):
        raise RunnerRefusal(
            "RUNNER_INVOCATION_EXISTS: refusing to overwrite a "
            "fired campaign's record")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def run_worker(repo, manifest_commit, geometry_path, points_path,
               idx, outdir):
    """ONE point, start to finish, in THIS process."""
    import w2_power_harness_cayley as PH
    pts = load_points(points_path)
    if not 0 <= idx < len(pts):
        raise RunnerRefusal(f"RUNNER_POINT_INDEX_INVALID: {idx}")
    spec = pts[idx]
    ref = {"manifest_commit": str(manifest_commit),
           "path": str(geometry_path)}
    out = {"index": idx, "spec": spec}
    try:
        if spec["entry"] == "specificity":
            rec = PH.run_b1b_specificity_certification(
                repo, ref, dict(spec["point"]))
        else:
            rec = PH.run_point_certification(
                repo, ref, spec["family"], dict(spec["point"]))
        out["record"] = rec
        out["refusal"] = None
    except PH.PowerHarnessError as e:
        out["record"] = None
        out["refusal"] = str(e)
    body = json.dumps(out, indent=1, sort_keys=True) + "\n"
    with open(os.path.join(outdir, f"point_{idx:03d}.json"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(body)
    return out


def run_campaign(repo, manifest_commit, geometry_path, points_path,
                 n_procs, outdir, argv=None, spawn=None):
    """Parent: pre-fire record, then at most n_procs concurrent
    whole-point workers over the ordered list. `spawn` is injectable
    for the selftest; production spawns this module as a subprocess
    per point."""
    points = load_points(points_path)
    write_invocation_record(outdir, points, manifest_commit,
                            geometry_path, n_procs,
                            argv if argv is not None else sys.argv)

    def _spawn(idx):
        return subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker",
             repo, str(manifest_commit), geometry_path, points_path,
             str(idx), outdir])
    spawn = spawn or _spawn
    running = {}
    order_started = []
    idx = 0
    while idx < len(points) or running:
        while idx < len(points) and len(running) < int(n_procs):
            running[idx] = spawn(idx)
            order_started.append(idx)
            idx += 1
        done = [i for i, h in running.items()
                if h.poll() is not None]
        if not done:
            time.sleep(0.2)
            continue
        for i in done:
            if running[i].returncode != 0:
                raise RunnerRefusal(
                    f"RUNNER_WORKER_FAILED: point {i} exit "
                    f"{running[i].returncode}")
            del running[i]
    results = []
    for i in range(len(points)):
        with open(os.path.join(outdir, f"point_{i:03d}.json"),
                  encoding="utf-8") as f:
            results.append(json.load(f))
    summary = {
        "schema": "f2g-w2-cert-campaign-summary-v1",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "n_points": len(points),
        "order_started": order_started,
        "per_point": [{"index": r["index"],
                       "family": r["spec"]["family"],
                       "entry": r["spec"]["entry"],
                       "status": (r["record"] or {}).get("status"),
                       "refusal": r["refusal"],
                       "record_sha256":
                           _digest(r["record"]) if r["record"]
                           else None}
                      for r in results],
        "ordered_points_sha256": _digest(points)}
    with open(os.path.join(outdir, "campaign_summary.json"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="w2runner_")

    # points-file doctors
    pp = os.path.join(tmp, "pts.json")
    good = [{"family": "B2B", "point": {"m": 2}, "entry": "detection"},
            {"family": "B1B", "point": {"gain": 3.0},
             "entry": "specificity"},
            {"family": "B1B", "point": {"k": 3, "n_e": 2,
                                        "delta_lat": 0.2},
             "entry": "detection"}]
    with open(pp, "w") as f:
        json.dump(good, f)
    assert len(load_points(pp)) == 3
    for bad, label in (
            ([], "empty"),
            ([{"family": "B2B", "point": {}}], "open schema"),
            ([dict(good[0], entry="warmup")], "unknown entry"),
            ([{"family": "B2A", "point": {"gain": 1.0},
               "entry": "specificity"}], "non-B1B specificity"),
            (good + [good[0]], "duplicate")):
        bp = os.path.join(tmp, "bad.json")
        with open(bp, "w") as f:
            json.dump(bad, f)
        try:
            load_points(bp)
            raise AssertionError(f"{label} must refuse")
        except RunnerRefusal as e:
            assert "RUNNER_POINTS_INVALID" in str(e), label

    # pre-fire record: shape, digest, written before workers, and
    # never overwritten
    od = os.path.join(tmp, "out1")
    rec = write_invocation_record(od, good, "deadbeef", "docs/x.json",
                                  3, ["argv0", "--fire"])
    assert rec["ordered_points_sha256"] == _digest(good)
    assert rec["process_count"] == 3 and rec["overrides"] is None
    assert os.path.exists(os.path.join(od, "invocation_record.json"))
    try:
        write_invocation_record(od, good, "deadbeef", "x", 3, [])
        raise AssertionError("second record must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_INVOCATION_EXISTS" in str(e)

    # scheduler: stub workers; whole-point dispatch (each worker sees
    # ONE index); concurrency never exceeds n_procs; ordered start
    od2 = os.path.join(tmp, "out2")
    os.makedirs(od2)
    live = {"n": 0, "max": 0}

    class StubProc:
        def __init__(self, idx):
            self.idx = idx
            self.returncode = None
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])
            with open(os.path.join(od2, f"point_{idx:03d}.json"),
                      "w") as f:
                json.dump({"index": idx, "spec": good[idx],
                           "record": {"status": "STUB"},
                           "refusal": None}, f)

        def poll(self):
            if self.returncode is None:
                self.returncode = 0
                live["n"] -= 1
            return self.returncode

    s = run_campaign(".", "deadbeef", "docs/x.json", pp, 2, od2,
                     argv=["kat"], spawn=StubProc)
    assert s["n_points"] == 3
    assert s["order_started"] == [0, 1, 2]      # registered order
    assert live["max"] <= 2                     # cap respected
    assert all(pp_["status"] == "STUB" for pp_ in s["per_point"])
    assert s["ordered_points_sha256"] == _digest(good)

    # failed worker refuses the campaign (never silently absorbed)
    od3 = os.path.join(tmp, "out3")
    os.makedirs(od3)

    class FailProc(StubProc):
        def poll(self):
            if self.returncode is None:
                self.returncode = 1
                live["n"] -= 1
            return self.returncode
    try:
        run_campaign(".", "deadbeef", "docs/x.json", pp, 2, od3,
                     argv=["kat"], spawn=FailProc)
        raise AssertionError("failed worker must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_WORKER_FAILED" in str(e)

    print("w2_cert_runner selftest: ALL PASS (stub workers only; "
          "no certification executed)")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _, _, repo, mc, gp, pp, idx, od = sys.argv
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        run_worker(repo, mc, gp, pp, int(idx), od)
        return
    if len(sys.argv) == 1:
        _selftest()
        return
    repo, mc, gp, pp, n, od = sys.argv[1:7]
    summary = run_campaign(os.path.abspath(repo), mc, gp, pp, int(n),
                           od)
    print(json.dumps({k: summary[k] for k in
                      ("n_points", "completed_utc",
                       "ordered_points_sha256")}, indent=1))


if __name__ == "__main__":
    main()
