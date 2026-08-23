#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CERTIFICATION CAMPAIGN RUNNER (cayley) -- per-point
process parallelism under codex's BINDING 1544Z ack, REV 2 folding
the codex 1909Z items 2-4 repairs.

THE ACK'S TERMS, implemented structurally:
- The unit of dispatch is the WHOLE POINT: one worker process runs one
  certification start-to-finish; there is no replicate-level dispatch
  surface to misuse.
- Same-family points deliberately share common random numbers (the
  registered seed grammar omits the point); scheduling is irrelevant
  because every worker reconstructs its own (authority, family, r)
  sequence -- this runner adds NO randomness and NO seed handling.

codex 1909Z repairs folded:
- item 2 (TOCTOU): the ordered point list comes ONLY from the
  COMMITTED SELECTOR ARTIFACT (digest recomputed against its own
  recorded ordered_points_sha256); the manifest commit is resolved to
  the exact 40-hex BEFORE the invocation writes; workers read the
  INVOCATION RECORD ITSELF (never the caller's mutable file) and
  refuse a points-digest mismatch; after workers finish the parent
  requires result.index == i, result.spec == invocation point i, and
  the returned record's family/point identity to equal that spec.
- item 3 (refusal absorption): a typed harness refusal writes its
  diagnostic then EXITS NONZERO; the parent refuses the campaign on
  any nonzero worker, refusal, missing record, or identity mismatch,
  TERMINATES + JOINS every still-live worker, and writes a typed
  campaign_aborted artifact.
- item 4 (fire-input validation): strict integer
  1 <= process_count <= n_points, resolvable manifest commit, and a
  campaign-artifact-free output directory are all required BEFORE the
  invocation record exists.

Outputs: <outdir>/invocation_record.json (pre-fire),
<outdir>/point_<idx>.json per point, <outdir>/campaign_summary.json,
or <outdir>/campaign_aborted.json on refusal.

No certification runs at import or selftest (stub workers only).
Opens no window-2 value.
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
CAMPAIGN_ARTIFACTS = ("invocation_record.json",
                      "campaign_summary.json",
                      "campaign_aborted.json")
WORKER_REFUSAL_EXIT = 3


class RunnerRefusal(ValueError):
    pass


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _digest(obj):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def validate_points(pts, where="selector"):
    if not isinstance(pts, list) or not pts:
        raise RunnerRefusal(
            f"RUNNER_POINTS_INVALID: {where} empty/non-list")
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


def load_selector(selector_path):
    """codex item 2: the ordered list comes from the COMMITTED
    selector artifact; its recorded digest must recompute."""
    with open(selector_path, "rb") as f:
        raw = f.read()
    art = json.loads(raw.decode("utf-8"))
    if not isinstance(art, dict) or \
            art.get("schema") != "f2g-w2-tier-selector-v1":
        raise RunnerRefusal(
            "RUNNER_SELECTOR_INVALID: not a selector artifact")
    pts = validate_points(art.get("ordered_points"))
    if _digest(pts) != art.get("ordered_points_sha256"):
        raise RunnerRefusal(
            "RUNNER_SELECTOR_INVALID: ordered_points digest does not "
            "recompute")
    return art, pts, hashlib.sha256(raw).hexdigest()


def resolve_manifest_commit(repo, mc):
    p = subprocess.run(
        ["git", "-C", repo, "rev-parse", f"{mc}^{{commit}}"],
        capture_output=True)
    full = p.stdout.decode().strip()
    if p.returncode != 0 or len(full) != 40 or \
            any(c not in "0123456789abcdef" for c in full):
        raise RunnerRefusal(
            f"RUNNER_MANIFEST_UNRESOLVABLE: {mc!r}")
    return full


def _validate_fire_inputs(repo, manifest_commit, n_procs, points,
                          outdir):
    """codex item 4: everything validated BEFORE the invocation
    record exists."""
    if type(n_procs) is not int or not \
            1 <= n_procs <= len(points):
        raise RunnerRefusal(
            f"RUNNER_PROCESS_COUNT_INVALID: {n_procs!r} not a strict "
            f"integer in [1, {len(points)}]")
    if os.path.isdir(outdir):
        stale = [n for n in os.listdir(outdir)
                 if n in CAMPAIGN_ARTIFACTS
                 or (n.startswith("point_") and n.endswith(".json"))]
        if stale:
            raise RunnerRefusal(
                f"RUNNER_OUTDIR_STALE: {sorted(stale)[:4]} present -- "
                "an aborted run never mixes with a new one")
    return resolve_manifest_commit(repo, manifest_commit)


def write_invocation_record(outdir, points, manifest_commit_full,
                            geometry_path, n_procs, argv,
                            selector_path, selector_sha256):
    """codex 1544Z: recorded PRE-FIRE, before any worker starts.
    Workers read THIS record; it is the single points carrier."""
    rec = {
        "schema": "f2g-w2-cert-invocation-v2",
        "fired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime()),
        "argv": list(argv),
        "process_count": int(n_procs),
        "interpreter": {"executable": sys.executable,
                        "version": sys.version,
                        "platform": platform.platform()},
        "host": platform.node(),
        "manifest_commit": str(manifest_commit_full),
        "geometry_path": str(geometry_path),
        "selector_path": str(selector_path),
        "selector_sha256": str(selector_sha256),
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


def _load_invocation(outdir, expected_points_sha):
    with open(os.path.join(outdir, "invocation_record.json"),
              encoding="utf-8") as f:
        inv = json.load(f)
    pts = inv["ordered_points"]
    if _digest(pts) != inv["ordered_points_sha256"] or \
            inv["ordered_points_sha256"] != expected_points_sha:
        raise RunnerRefusal(
            "RUNNER_POINTS_DIGEST_MISMATCH: invocation points "
            "diverge from the fired digest")
    return inv, pts


def run_worker(repo, outdir, idx, expected_points_sha):
    """ONE point, start to finish, in THIS process. The spec comes
    from the INVOCATION RECORD (codex item 2), never a caller file.
    A typed harness refusal writes its diagnostic then exits nonzero
    (codex item 3)."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import w2_power_harness_cayley as PH
    inv, pts = _load_invocation(outdir, expected_points_sha)
    idx = int(idx)
    if not 0 <= idx < len(pts):
        raise RunnerRefusal(f"RUNNER_POINT_INDEX_INVALID: {idx}")
    spec = pts[idx]
    ref = {"manifest_commit": inv["manifest_commit"],
           "path": inv["geometry_path"]}
    out = {"index": idx, "spec": spec}
    refused = False
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
        refused = True
    body = json.dumps(out, indent=1, sort_keys=True) + "\n"
    with open(os.path.join(outdir, f"point_{idx:03d}.json"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(body)
    if refused:
        sys.exit(WORKER_REFUSAL_EXIT)
    return out


def _record_identity_ok(spec, rec):
    if not isinstance(rec, dict):
        return False
    if rec.get("family") != spec["family"] or \
            rec.get("point") != spec["point"]:
        return False
    if spec["entry"] == "specificity" and \
            rec.get("class") != "B1B_GAIN_STEP_SPECIFICITY":
        return False
    return True


def _abort(outdir, running, reason, detail):
    """codex item 3: terminate + join every live worker, write the
    typed abort artifact, refuse."""
    for h in running.values():
        try:
            h.terminate()
        except Exception:
            pass
    for h in running.values():
        try:
            h.wait()
        except Exception:
            pass
    art = {"schema": "f2g-w2-campaign-aborted-v1",
           "aborted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime()),
           "reason": reason, "detail": detail}
    with open(os.path.join(outdir, "campaign_aborted.json"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(art, indent=1, sort_keys=True) + "\n")
    raise RunnerRefusal(f"{reason}: {detail}")


def run_campaign(repo, manifest_commit, geometry_path, selector_path,
                 n_procs, outdir, argv=None, spawn=None):
    """Parent: validate ALL fire inputs, write the pre-fire record,
    then at most n_procs concurrent whole-point workers over the
    selector's ordered list. `spawn` is injectable for the selftest;
    production spawns this module as a subprocess per point."""
    selector, points, selector_sha = load_selector(selector_path)
    mc_full = _validate_fire_inputs(repo, manifest_commit, n_procs,
                                    points, outdir)
    inv = write_invocation_record(
        outdir, points, mc_full, geometry_path, n_procs,
        argv if argv is not None else sys.argv, selector_path,
        selector_sha)
    psha = inv["ordered_points_sha256"]

    def _spawn(idx):
        return subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker",
             repo, outdir, str(idx), psha])
    spawn = spawn or _spawn
    running = {}
    order_started = []
    idx = 0
    while idx < len(points) or running:
        while idx < len(points) and len(running) < n_procs:
            running[idx] = spawn(idx)
            order_started.append(idx)
            idx += 1
        done = [i for i, h in running.items()
                if h.poll() is not None]
        if not done:
            time.sleep(0.2)
            continue
        for i in done:
            h = running.pop(i)
            if h.returncode != 0:
                _abort(outdir, running, "RUNNER_WORKER_FAILED",
                       f"point {i} exit {h.returncode}")
    results = []
    for i in range(len(points)):
        p = os.path.join(outdir, f"point_{i:03d}.json")
        if not os.path.exists(p):
            _abort(outdir, {}, "RUNNER_RESULT_MISSING", f"point {i}")
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    # codex item 2/3: identity + refusal checks over EVERY result
    for i, r in enumerate(results):
        if r.get("index") != i or r.get("spec") != points[i]:
            _abort(outdir, {}, "RUNNER_RESULT_IDENTITY_MISMATCH",
                   f"point {i} result does not match the invocation")
        if r.get("refusal") is not None or r.get("record") is None:
            _abort(outdir, {}, "RUNNER_WORKER_REFUSED",
                   f"point {i}: {r.get('refusal')}")
        if not _record_identity_ok(points[i], r["record"]):
            _abort(outdir, {}, "RUNNER_RESULT_IDENTITY_MISMATCH",
                   f"point {i} certification record family/point "
                   "diverges from the invocation spec")
    summary = {
        "schema": "f2g-w2-cert-campaign-summary-v2",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "n_points": len(points),
        "order_started": order_started,
        "selector_sha256": selector_sha,
        "manifest_commit": mc_full,
        "per_point": [{"index": r["index"],
                       "family": r["spec"]["family"],
                       "entry": r["spec"]["entry"],
                       "point": r["spec"]["point"],
                       "status": r["record"].get("status"),
                       "record_sha256": _digest(r["record"])}
                      for r in results],
        "ordered_points_sha256": psha}
    with open(os.path.join(outdir, "campaign_summary.json"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="w2runner_")
    repo_g = os.path.abspath(os.path.join(_HERE, "..", ".."))

    good = [{"family": "B2B", "point": {"m": 2}, "entry": "detection"},
            {"family": "B1B", "point": {"gain": 3.0},
             "entry": "specificity"},
            {"family": "B1B", "point": {"k": 3, "n_e": 2,
                                        "delta_lat": 0.2},
             "entry": "detection"}]

    def mk_selector(pts, path, doctor=None):
        pts = json.loads(json.dumps(pts))     # isolate the fixture
        art = {"schema": "f2g-w2-tier-selector-v1",
               "ordered_points": pts,
               "ordered_points_sha256": _digest(pts)}
        if doctor:
            doctor(art)
        with open(path, "w") as f:
            json.dump(art, f)
        return path

    sp = mk_selector(good, os.path.join(tmp, "sel.json"))
    art, pts, sha = load_selector(sp)
    assert pts == good and len(sha) == 64

    # selector doctors: wrong schema, digest mismatch, bad points
    for doctor, label in (
            (lambda a: a.update(schema="x"), "schema"),
            (lambda a: a.update(ordered_points_sha256="0" * 64),
             "digest"),
            (lambda a: a["ordered_points"].append(
                a["ordered_points"][0]), "duplicate")):
        bp = mk_selector(good, os.path.join(tmp, "bad.json"), doctor)
        try:
            load_selector(bp)
            raise AssertionError(f"{label} selector must refuse")
        except RunnerRefusal:
            pass

    # item 4: fire-input doctors BEFORE any invocation exists
    hexmc = subprocess.run(["git", "-C", repo_g, "rev-parse", "HEAD"],
                           capture_output=True).stdout.decode().strip()
    for n, label in ((0, "zero"), (-2, "negative"), ("2", "string"),
                     (99, "over")):
        try:
            _validate_fire_inputs(repo_g, hexmc, n, good,
                                  os.path.join(tmp, "ov"))
            raise AssertionError(f"{label} n_procs must refuse")
        except RunnerRefusal as e:
            assert "RUNNER_PROCESS_COUNT_INVALID" in str(e)
    try:
        _validate_fire_inputs(repo_g, "no-such-ref-xyz", 2, good,
                              os.path.join(tmp, "ov"))
        raise AssertionError("unresolvable manifest must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_MANIFEST_UNRESOLVABLE" in str(e)
    assert len(_validate_fire_inputs(repo_g, hexmc[:12], 2, good,
                                     os.path.join(tmp, "ov"))) == 40
    stale = os.path.join(tmp, "stale")
    os.makedirs(stale)
    open(os.path.join(stale, "point_000.json"), "w").close()
    try:
        _validate_fire_inputs(repo_g, hexmc, 2, good, stale)
        raise AssertionError("stale outdir must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_OUTDIR_STALE" in str(e)

    # pre-fire record: shape + no-overwrite
    od = os.path.join(tmp, "out1")
    rec = write_invocation_record(od, good, "f" * 40, "docs/x.json",
                                  2, ["argv0"], sp, sha)
    assert rec["ordered_points_sha256"] == _digest(good)
    assert rec["selector_sha256"] == sha
    try:
        write_invocation_record(od, good, "f" * 40, "x", 2, [], sp,
                                sha)
        raise AssertionError("second record must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_INVOCATION_EXISTS" in str(e)

    # item 2: workers read the INVOCATION; mutating the selector file
    # after fire cannot reach them (stub workers copy spec from the
    # written invocation record, exactly like the real worker)
    live = {"n": 0, "max": 0}

    def stub_writer(outdir, transform=None, rc=0):
        class StubProc:
            def __init__(self, idx):
                self.idx = idx
                self.returncode = None
                self.terminated = False
                live["n"] += 1
                live["max"] = max(live["max"], live["n"])
                with open(os.path.join(outdir,
                                       "invocation_record.json"),
                          encoding="utf-8") as f:
                    inv = json.load(f)
                out = {"index": idx,
                       "spec": inv["ordered_points"][idx],
                       "record": {
                           "status": "STUB",
                           "family":
                               inv["ordered_points"][idx]["family"],
                           "point":
                               inv["ordered_points"][idx]["point"],
                           "class": "B1B_GAIN_STEP_SPECIFICITY"},
                       "refusal": None}
                if transform:
                    transform(idx, out)
                with open(os.path.join(outdir,
                                       f"point_{idx:03d}.json"),
                          "w") as f:
                    json.dump(out, f)
                self._rc = rc

            def poll(self):
                if self.returncode is None:
                    self.returncode = self._rc
                    live["n"] -= 1
                return self.returncode

            def terminate(self):
                self.terminated = True

            def wait(self):
                return self.returncode
        return StubProc

    od2 = os.path.join(tmp, "out2")
    sp2 = mk_selector(good, os.path.join(tmp, "sel2.json"))

    def fire(outdir, selector, transform=None, rc=0, n=2,
             mutate_after=False):
        Stub = stub_writer(outdir, transform, rc)

        def spawn(idx):
            if mutate_after and idx == 0:
                mk_selector([good[0]], selector)   # caller mutates
            return Stub(idx)
        return run_campaign(repo_g, hexmc[:12], "docs/x.json",
                            selector, n, outdir, argv=["kat"],
                            spawn=spawn)
    s = fire(od2, sp2, mutate_after=True)
    assert s["n_points"] == 3 and s["order_started"] == [0, 1, 2]
    assert live["max"] <= 2
    assert [pp["point"] for pp in s["per_point"]] == \
        [p["point"] for p in good]        # mutation never reached work
    assert len(s["manifest_commit"]) == 40

    # item 3: a refusal-writing worker (nonzero exit) aborts the
    # campaign, terminates siblings, writes the typed artifact
    od3 = os.path.join(tmp, "out3")
    sp3 = mk_selector(good, os.path.join(tmp, "sel3.json"))
    try:
        fire(od3, sp3, transform=lambda i, o: o.update(
            record=None, refusal="POWER_GEOMETRY_UNBOUND: kat")
            if i == 1 else None, rc=WORKER_REFUSAL_EXIT)
        raise AssertionError("refusing worker must abort campaign")
    except RunnerRefusal as e:
        assert "RUNNER_WORKER_FAILED" in str(e)
    assert os.path.exists(os.path.join(od3, "campaign_aborted.json"))
    assert not os.path.exists(os.path.join(od3,
                                           "campaign_summary.json"))

    # item 3 (zero-exit path): a refusal that somehow exits 0 is
    # still caught by the parent's result checks
    od4 = os.path.join(tmp, "out4")
    sp4 = mk_selector(good, os.path.join(tmp, "sel4.json"))
    try:
        fire(od4, sp4, transform=lambda i, o: o.update(
            record=None, refusal="POWER_X: kat") if i == 2 else None)
        raise AssertionError("zero-exit refusal must abort")
    except RunnerRefusal as e:
        assert "RUNNER_WORKER_REFUSED" in str(e)

    # identity mismatch: record family diverging from spec aborts
    od5 = os.path.join(tmp, "out5")
    sp5 = mk_selector(good, os.path.join(tmp, "sel5.json"))
    try:
        fire(od5, sp5, transform=lambda i, o: o["record"].update(
            family="B3A") if i == 0 else None)
        raise AssertionError("identity mismatch must abort")
    except RunnerRefusal as e:
        assert "RUNNER_RESULT_IDENTITY_MISMATCH" in str(e)

    # direct worker: a REAL typed harness refusal writes its
    # diagnostic and exits nonzero (codex item 3, worker level)
    od6 = os.path.join(tmp, "out6")
    write_invocation_record(od6, good, hexmc, "docs/no-such.json", 1,
                            ["kat"], sp, sha)
    try:
        run_worker(repo_g, od6, 0, _digest(good))
        raise AssertionError("worker must exit nonzero on refusal")
    except SystemExit as e:
        assert e.code == WORKER_REFUSAL_EXIT
    with open(os.path.join(od6, "point_000.json")) as f:
        d = json.load(f)
    assert d["record"] is None and "POWER_GEOMETRY" in d["refusal"]
    # worker refuses a points-digest mismatch before any work
    try:
        run_worker(repo_g, od6, 0, "0" * 64)
        raise AssertionError("digest mismatch must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_POINTS_DIGEST_MISMATCH" in str(e)

    print("w2_cert_runner selftest: ALL PASS (stub workers + typed "
          "refusal paths; no certification executed)")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _, _, repo, outdir, idx, psha = sys.argv
        run_worker(repo, outdir, int(idx), psha)
        return
    if len(sys.argv) == 1:
        _selftest()
        return
    repo, mc, gp, sel, n, od = sys.argv[1:7]
    summary = run_campaign(os.path.abspath(repo), mc, gp, sel,
                           int(n), od)
    print(json.dumps({k: summary[k] for k in
                      ("n_points", "completed_utc", "manifest_commit",
                       "ordered_points_sha256")}, indent=1))


if __name__ == "__main__":
    main()
