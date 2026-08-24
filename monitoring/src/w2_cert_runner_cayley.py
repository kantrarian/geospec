#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CERTIFICATION CAMPAIGN RUNNER (cayley) -- per-point
process parallelism under codex's BINDING 1544Z ack; REV 3 folding
the codex 2235Z items 2-4 (runner half) onto the REV 2 repairs
(1909Z items 2-4).

STRUCTURAL TERMS (unchanged): the unit of dispatch is the WHOLE POINT
(no replicate-level surface exists); the runner adds no randomness
and no seed handling.

REV 3 (codex 2235Z):
- item 2: the selector is a COMMITTED git object -- the fire input is
  (selector_commit, selector_path); the blob reopens via git cat-file
  and `verify_selector_artifact` reruns the registered rule against
  the artifact's BOUND smoke/effect-grid carriers and enforces the
  exact 14-point shape with gains 3 then 10. A fabricated local file
  has no entry path.
- item 3: workers authenticate the COMPLETE INVOCATION, not the point
  list: the closed invocation core is canonically hashed
  (invocation_sha256); every worker receives the expected digest and
  recomputes it on reopen BEFORE reading any field -- manifest-only or
  geometry-only post-write mutation refuses. Spawn/poll/read failures
  inside the parent all route through terminate+join+typed abort.
- item 4 (runner half): the invocation, summary, and abort artifacts
  publish via an ATOMIC CREATE-ONCE primitive (unique same-directory
  temp + os.link, which never replaces); a pre-existing destination
  refuses -- check-then-overwrite is gone.

REV 2 terms retained: pre-fire invocation record; ordered scheduling
with a strict cap; typed harness refusals exit nonzero; the parent
refuses on any refusal/missing record/identity mismatch; strict
fire-input validation (process count, resolvable 40-hex manifest,
stale-outdir).

No certification runs at import or selftest (stub workers only).
Opens no window-2 value.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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


def _publish_once(path, body_text):
    """codex 2235Z item 4 (+ 0130Z refinement): atomic CREATE-ONCE
    publication -- PER-CALL-UNIQUE same-directory temp (mkstemp,
    O_EXCL) + os.link (which never replaces). An existing destination
    refuses typed (this primitive never reuses, even on identical
    bytes -- a second campaign must not silently share an outdir);
    after a successful link the destination is REOPENED and compared
    against this caller's bytes before returning."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body_text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise RunnerRefusal(
                f"RUNNER_PUBLISH_EXISTS: {os.path.basename(path)} "
                "already published (create-once; never replaced)")
        with open(path, "r", encoding="utf-8", newline="") as f:
            if f.read() != body_text:
                raise RunnerRefusal(
                    f"RUNNER_PUBLISH_EXISTS: {os.path.basename(path)}"
                    " diverged after publication (mutated carrier)")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


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


def _git_blob(repo, commit, path):
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{path}"], capture_output=True)
    if p.returncode != 0:
        raise RunnerRefusal(
            f"RUNNER_SELECTOR_INVALID: {path} unreadable at "
            f"{commit} (only a COMMITTED selector can fire)")
    return p.stdout


def load_selector_committed(repo, selector_commit, selector_path,
                            *, blob_reader=None):
    """codex 2235Z item 2: the selector comes from a COMMITTED git
    object and is verified by an independent rerun against its bound
    carriers -- a self-consistent digest alone never fires."""
    import w2_tier_selector_cayley as TS
    if blob_reader is None:
        def blob_reader(commit, path):
            return _git_blob(repo, commit, path)
    raw = blob_reader(selector_commit, selector_path)
    art = json.loads(raw.decode("utf-8"))
    try:
        TS.verify_selector_artifact(repo, art,
                                    blob_reader=blob_reader)
    except TS.SelectorRefusal as e:
        raise RunnerRefusal(f"RUNNER_SELECTOR_INVALID: {e}")
    pts = validate_points(art["ordered_points"])
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
    """codex 1909Z item 4: everything validated BEFORE the invocation
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


def _invocation_digest(rec):
    """The digest of the COMPLETE closed invocation core (every field
    except the digest itself)."""
    return _digest({k: v for k, v in rec.items()
                    if k != "invocation_sha256"})


def write_invocation_record(outdir, points, manifest_commit_full,
                            geometry_path, n_procs, argv,
                            selector_commit, selector_path,
                            selector_sha256, admitted_carriers=None):
    """codex 1544Z: recorded PRE-FIRE, before any worker starts;
    2235Z item 3: the whole closed core is hashed
    (invocation_sha256), and workers authenticate THAT; 2235Z item 4:
    published atomically create-once."""
    rec = {
        "schema": "f2g-w2-cert-invocation-v3",
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
        "selector_commit": str(selector_commit),
        "selector_path": str(selector_path),
        "selector_sha256": str(selector_sha256),
        "admitted_carriers": admitted_carriers,
        "ordered_points": points,
        "ordered_points_sha256": _digest(points),
        "dispatch_rule": "whole-point-per-process; one point's "
                         "replicate sequence never splits across "
                         "workers",
        "overrides": None}
    rec["invocation_sha256"] = _invocation_digest(rec)
    _publish_once(os.path.join(outdir, "invocation_record.json"),
                  json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def _load_invocation(outdir, expected_invocation_sha):
    """2235Z item 3: recompute the digest of the COMPLETE closed core
    and require it to equal BOTH the stored field and the expected
    value BEFORE any field is consumed."""
    with open(os.path.join(outdir, "invocation_record.json"),
              encoding="utf-8") as f:
        inv = json.load(f)
    got = _invocation_digest(inv)
    if got != inv.get("invocation_sha256") or \
            got != expected_invocation_sha:
        raise RunnerRefusal(
            "RUNNER_INVOCATION_DIGEST_MISMATCH: the invocation "
            "carrier diverges from the fired digest")
    pts = inv["ordered_points"]
    if _digest(pts) != inv["ordered_points_sha256"]:
        raise RunnerRefusal(
            "RUNNER_POINTS_DIGEST_MISMATCH: invocation points "
            "diverge from their recorded digest")
    return inv, pts


def run_worker(repo, outdir, idx, expected_invocation_sha):
    """ONE point, start to finish, in THIS process. Every consumed
    field comes from the digest-authenticated invocation. A typed
    harness refusal writes its diagnostic then exits nonzero."""
    import w2_power_harness_cayley as PH
    inv, pts = _load_invocation(outdir, expected_invocation_sha)
    idx = int(idx)
    if not 0 <= idx < len(pts):
        raise RunnerRefusal(f"RUNNER_POINT_INDEX_INVALID: {idx}")
    spec = pts[idx]
    ref = {"manifest_commit": inv["manifest_commit"],
           "path": inv["geometry_path"]}
    out = {"index": idx, "spec": spec,
           "invocation_sha256": inv["invocation_sha256"]}
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
    """Terminate + join every live worker, publish the typed abort
    artifact (create-once; a racing second abort tolerates the
    winner), refuse."""
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
    try:
        _publish_once(os.path.join(outdir, "campaign_aborted.json"),
                      json.dumps(art, indent=1, sort_keys=True)
                      + "\n")
    except RunnerRefusal:
        pass                      # first abort artifact wins
    raise RunnerRefusal(f"{reason}: {detail}")


def run_campaign(repo, manifest_commit, geometry_path,
                 selector_commit, selector_path, n_procs, outdir,
                 argv=None, spawn=None, blob_reader=None,
                 git_resolve=None):
    """Parent: committed-selector verification, fire-input
    validation, atomic pre-fire record, then at most n_procs
    concurrent whole-point workers. Every parent-side failure --
    spawn, poll, read, refusal, identity -- routes through _abort."""
    selector, points, selector_sha = load_selector_committed(
        repo, selector_commit, selector_path,
        blob_reader=blob_reader)
    mc_full = _validate_fire_inputs(repo, manifest_commit, n_procs,
                                    points, outdir)
    # codex 0238Z item 3: committed is not ADMITTED -- the selector's
    # carriers must be the manifest-pinned grids + the closed output
    # of the admitted Tier-S invocation; the admitted identities bind
    # into the invocation core digest below
    import w2_tier_selector_cayley as TS
    try:
        admitted = TS.verify_selector_admission(
            repo, selector, mc_full, blob_reader=blob_reader,
            git_resolve=git_resolve)
    except TS.SelectorRefusal as e:
        raise RunnerRefusal(f"RUNNER_SELECTOR_INVALID: {e}")
    inv = write_invocation_record(
        outdir, points, mc_full, geometry_path, n_procs,
        argv if argv is not None else sys.argv, selector_commit,
        selector_path, selector_sha, admitted)
    isha = inv["invocation_sha256"]

    def _spawn(idx):
        return subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker",
             repo, outdir, str(idx), isha])
    spawn = spawn or _spawn
    running = {}
    order_started = []
    idx = 0
    while idx < len(points) or running:
        try:
            while idx < len(points) and len(running) < n_procs:
                running[idx] = spawn(idx)
                order_started.append(idx)
                idx += 1
            done = [i for i, h in running.items()
                    if h.poll() is not None]
        except RunnerRefusal:
            raise
        except Exception as e:
            _abort(outdir, running, "RUNNER_SCHEDULER_FAILED",
                   f"{type(e).__name__}: {e}")
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
        try:
            with open(p, encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            _abort(outdir, {}, "RUNNER_RESULT_UNREADABLE",
                   f"point {i}: {e}")
    for i, r in enumerate(results):
        if r.get("index") != i or r.get("spec") != points[i] or \
                r.get("invocation_sha256") != isha:
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
        "schema": "f2g-w2-cert-campaign-summary-v3",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "n_points": len(points),
        "order_started": order_started,
        "invocation_sha256": isha,
        "manifest_commit": mc_full,
        "selector_commit": str(selector_commit),
        "selector_path": str(selector_path),
        "selector_sha256": selector_sha,
        "geometry_path": str(geometry_path),
        "per_point": [{"index": r["index"],
                       "family": r["spec"]["family"],
                       "entry": r["spec"]["entry"],
                       "point": r["spec"]["point"],
                       "status": r["record"].get("status"),
                       "record_sha256": _digest(r["record"])}
                      for r in results],
        "ordered_points_sha256": inv["ordered_points_sha256"]}
    _publish_once(os.path.join(outdir, "campaign_summary.json"),
                  json.dumps(summary, indent=1, sort_keys=True)
                  + "\n")
    return summary


# ---------------------------------------------------------------- selftest
def _selftest():
    import copy
    import tempfile as _tf
    import w2_tier_selector_cayley as TS
    tmp = _tf.mkdtemp(prefix="w2runner_")
    repo_g = os.path.abspath(os.path.join(_HERE, "..", ".."))

    # a VALID committed-style 14-point selector over an in-memory
    # blob store (the registered rule applied to a hand fixture)
    def outs(k):
        return [True] * k + [False] * (50 - k)
    grids = {
        "B2A": [{"m": 1}, {"m": 2}, {"m": 3}],
        "B2B": [{"m": 1}, {"m": 2}, {"m": 3},
                {"m": 2, "dropout": 0.1}, {"m": 2, "dropout": 0.25}],
        "B3A": [{"delta": d} for d in range(1, 5)],
        "B1B": [{"delta_lat": 0.1 * j, "k": 3, "n_e": 2}
                for j in range(1, 10)]
               + [{"gain": 3.0}, {"gain": 10.0}]}
    fams = {}
    for fam in ("B2A", "B2B", "B1B", "B3A"):
        det = [(i, p) for i, p in enumerate(grids[fam])
               if "gain" not in p]
        entries = []
        for k, (gi, gp) in enumerate(det):
            e = {"point": dict(gp), "outcomes": outs(30 + k)}
            if fam == "B1B":
                e["post_loco_outcomes"] = None
            entries.append(e)
        fams[fam] = entries
    # B1B: give the top-8 (by pre-LOCO rank) post-LOCO outcomes
    b1b_counts = [(30 + k, k) for k in range(9)]
    keep = sorted(range(9), key=lambda k: (-(30 + k), k))[:8]
    for k in keep:
        fams["B1B"][k]["post_loco_outcomes"] = outs(20 + k)
    # the ADMITTED-carrier fixture world (codex 0238Z item 3): a
    # fixture manifest pins grids.json; the smoke is the closed
    # Tier-S output with a reopenable invocation record
    ts_inv = {"schema": "f2g-w2-tier-s-invocation-v1",
              "purpose": "kat"}
    ts_inv["invocation_sha256"] = _digest(
        {k: v for k, v in ts_inv.items()
         if k != "invocation_sha256"})
    grids_raw = json.dumps({"schema": "f2g-w2-effect-grids-v1",
                            "grids": grids}).encode()
    smoke = {"schema": "f2g-w2-tier-s-smoke-v1",
             "quality": {"R": 50, "n_draws": 999},
             "geometry_capsule_digest": "kat",
             "effect_grids_sha256": _digest(grids),
             "invocation_ref": {"commit": "d" * 40,
                                "path": "ts_invocation.json"},
             "invocation_sha256": ts_inv["invocation_sha256"],
             "families": fams}
    fix_man = {"slots": {"power_harness": {"pins": [
        {"path": "grids.json", "commit": "d" * 40,
         "blob_sha256": hashlib.sha256(grids_raw).hexdigest()}]}}}
    store = {("d" * 40, "smoke.json"): json.dumps(smoke).encode(),
             ("d" * 40, "grids.json"): grids_raw,
             ("d" * 40, "ts_invocation.json"):
                 json.dumps(ts_inv).encode()}
    refs = {"smoke_ref": {"commit": "d" * 40, "path": "smoke.json"},
            "effect_grids_ref": {"commit": "d" * 40,
                                 "path": "grids.json"}}
    art = TS.select_candidates(smoke, grids, **refs)
    store[("d" * 40, "selector.json")] = json.dumps(art).encode()

    def reader(commit, path):
        if path.endswith("execution_manifest.json"):
            return json.dumps(fix_man).encode()
        try:
            return store[(commit, path)]
        except KeyError:
            raise RunnerRefusal(
                f"RUNNER_SELECTOR_INVALID: {path} unreadable at "
                f"{commit} (only a COMMITTED selector can fire)")

    def gresolve(c):
        return c if len(str(c)) == 40 else             resolve_manifest_commit(repo_g, c)

    art2, pts, sha = load_selector_committed(
        ".", "d" * 40, "selector.json", blob_reader=reader)
    assert len(pts) == 14

    # item 2 locks: fabricated minimal selector; altered points;
    # uncommitted path (real git)
    fab = {"schema": "f2g-w2-tier-selector-v1",
           "ordered_points": [{"family": "B2A", "point": {"m": 999},
                               "entry": "detection"}]}
    fab["ordered_points_sha256"] = _digest(fab["ordered_points"])
    store[("d" * 40, "fab.json")] = json.dumps(fab).encode()
    try:
        load_selector_committed(".", "d" * 40, "fab.json",
                                blob_reader=reader)
        raise AssertionError("fabricated selector must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_SELECTOR_INVALID" in str(e)
    try:
        load_selector_committed(repo_g, "HEAD",
                                "docs/no-such-selector.json")
        raise AssertionError("uncommitted selector must refuse")
    except RunnerRefusal as e:
        assert "unreadable" in str(e)

    hexmc = subprocess.run(["git", "-C", repo_g, "rev-parse", "HEAD"],
                           capture_output=True).stdout.decode().strip()

    # item 4 (runner): create-once -- a pre-existing invocation
    # refuses regardless of bytes
    od0 = os.path.join(tmp, "race")
    os.makedirs(od0)
    with open(os.path.join(od0, "invocation_record.json"), "w") as f:
        f.write("{}")
    try:
        write_invocation_record(od0, pts, "f" * 40, "g.json", 2,
                                ["kat"], "d" * 40, "selector.json",
                                sha)
        raise AssertionError("existing invocation must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_PUBLISH_EXISTS" in str(e)

    # stub workers that read + authenticate the invocation like the
    # real worker
    live = {"n": 0, "max": 0}

    def stub_writer(outdir, isha_holder, transform=None, rc=0,
                    raise_on=None):
        class StubProc:
            def __init__(self, idx):
                if raise_on is not None and idx == raise_on:
                    raise OSError("spawn blew up")
                self.idx = idx
                self.returncode = None
                self.terminated = False
                live["n"] += 1
                live["max"] = max(live["max"], live["n"])
                inv, ipts = _load_invocation(outdir,
                                             isha_holder["sha"])
                out = {"index": idx, "spec": ipts[idx],
                       "invocation_sha256": inv["invocation_sha256"],
                       "record": {"status": "STUB",
                                  "family": ipts[idx]["family"],
                                  "point": ipts[idx]["point"],
                                  "class":
                                      "B1B_GAIN_STEP_SPECIFICITY"},
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

    def fire(outdir, transform=None, rc=0, raise_on=None, n=3):
        holder = {"sha": None}
        Stub = stub_writer(outdir, holder, transform, rc, raise_on)

        def spawn(idx):
            if holder["sha"] is None:
                with open(os.path.join(outdir,
                                       "invocation_record.json"),
                          encoding="utf-8") as f:
                    holder["sha"] = json.load(
                        f)["invocation_sha256"]
            return Stub(idx)
        return run_campaign(repo_g, hexmc[:12], "docs/g.json",
                            "d" * 40, "selector.json", n, outdir,
                            argv=["kat"], spawn=spawn,
                            blob_reader=reader, git_resolve=gresolve)

    od1 = os.path.join(tmp, "ok")
    s = fire(od1)
    assert s["n_points"] == 14 and live["max"] <= 3
    assert s["invocation_sha256"] and len(s["manifest_commit"]) == 40
    assert s["selector_sha256"] == sha

    # item 3 locks: manifest-only + geometry-only post-write mutation
    # refuse at the worker
    for field, val in (("manifest_commit", "c" * 40),
                       ("geometry_path", "docs/geometry-B.json")):
        odm = os.path.join(tmp, f"mut_{field}")
        os.makedirs(odm)
        rec = write_invocation_record(odm, pts, "a" * 40,
                                      "docs/geometry-A.json", 2,
                                      ["kat"], "d" * 40,
                                      "selector.json", sha)
        with open(os.path.join(odm, "invocation_record.json"),
                  encoding="utf-8") as f:
            doc = json.load(f)
        doc[field] = val                      # points untouched
        with open(os.path.join(odm, "invocation_record.json"), "w",
                  encoding="utf-8") as f:
            json.dump(doc, f)
        try:
            _load_invocation(odm, rec["invocation_sha256"])
            raise AssertionError(f"{field} mutation must refuse")
        except RunnerRefusal as e:
            assert "RUNNER_INVOCATION_DIGEST_MISMATCH" in str(e)

    # item 3 lock: second spawn raises while the first is live ->
    # typed abort artifact + sibling terminated
    od2 = os.path.join(tmp, "spawnfail")
    try:
        fire(od2, raise_on=1, n=2)
        raise AssertionError("spawn failure must abort")
    except RunnerRefusal as e:
        assert "RUNNER_SCHEDULER_FAILED" in str(e)
    assert os.path.exists(os.path.join(od2, "campaign_aborted.json"))

    # refusal + identity locks (REV 2 semantics retained)
    od3 = os.path.join(tmp, "refusal")
    try:
        fire(od3, transform=lambda i, o: o.update(
            record=None, refusal="POWER_X: kat") if i == 4 else None,
            rc=0)
        raise AssertionError("zero-exit refusal must abort")
    except RunnerRefusal as e:
        assert "RUNNER_WORKER_REFUSED" in str(e)
    od4 = os.path.join(tmp, "ident")
    try:
        fire(od4, transform=lambda i, o: o["record"].update(
            family="B3A") if i == 0 else None)
        raise AssertionError("identity mismatch must abort")
    except RunnerRefusal as e:
        assert "RUNNER_RESULT_IDENTITY_MISMATCH" in str(e)

    # fire-input validation retained (REV 2)
    for n, label in ((0, "zero"), (-2, "negative"), ("2", "string"),
                     (99, "over")):
        try:
            _validate_fire_inputs(repo_g, hexmc, n, pts,
                                  os.path.join(tmp, "ov"))
            raise AssertionError(f"{label} n_procs must refuse")
        except RunnerRefusal as e:
            assert "RUNNER_PROCESS_COUNT_INVALID" in str(e)
    try:
        _validate_fire_inputs(repo_g, "no-such-ref-xyz", 2, pts,
                              os.path.join(tmp, "ov"))
        raise AssertionError("unresolvable manifest must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_MANIFEST_UNRESOLVABLE" in str(e)

    # direct worker: a REAL typed harness refusal exits nonzero
    od6 = os.path.join(tmp, "worker")
    rec6 = write_invocation_record(od6, pts, hexmc,
                                   "docs/no-such.json", 1, ["kat"],
                                   "d" * 40, "selector.json", sha)
    try:
        run_worker(repo_g, od6, 0, rec6["invocation_sha256"])
        raise AssertionError("worker must exit nonzero on refusal")
    except SystemExit as e:
        assert e.code == WORKER_REFUSAL_EXIT
    with open(os.path.join(od6, "point_000.json")) as f:
        d = json.load(f)
    assert d["record"] is None and "POWER_GEOMETRY" in d["refusal"]
    try:
        run_worker(repo_g, od6, 0, "0" * 64)
        raise AssertionError("digest mismatch must refuse")
    except RunnerRefusal as e:
        assert "RUNNER_INVOCATION_DIGEST_MISMATCH" in str(e)

    print("w2_cert_runner selftest: ALL PASS (stub workers + typed "
          "refusal paths; no certification executed)")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _, _, repo, outdir, idx, isha = sys.argv
        run_worker(repo, outdir, int(idx), isha)
        return
    if len(sys.argv) == 1:
        _selftest()
        return
    repo, mc, gp, sel_c, sel_p, n, od = sys.argv[1:8]
    summary = run_campaign(os.path.abspath(repo), mc, gp, sel_c,
                           sel_p, int(n), od)
    print(json.dumps({k: summary[k] for k in
                      ("n_points", "completed_utc", "manifest_commit",
                       "invocation_sha256")}, indent=1))


if __name__ == "__main__":
    main()
