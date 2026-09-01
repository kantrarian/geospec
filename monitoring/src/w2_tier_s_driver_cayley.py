#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 TIER-S PRODUCTION DRIVER (cayley) -- the operator
entrypoint that `w2_tier_s_runner_cayley` deliberately does not
provide.

The runner is a LIBRARY: its `__main__` runs a selftest with stub
smoke functions, and nothing in the landed tree calls `fire_pre` or
`run_smoke_point`. This module is that missing caller and nothing
more. It contributes NO new chronology, NO new schema, NO new
identity and NO statistical logic -- every carrier is published by the
runner's own create-once primitives, every identity comes from the
runner's pre-invocation, and every p-value comes from
`w2_power_harness_cayley.run_point_smoke`. If this file computed
anything the runner does not, that would itself be the defect.

WHAT IT ADDS, and why each is here rather than in the runner:

- **Phase separation with NO automatic commits.** The runner's
  contract is that "commits between phases are the operator's, so the
  chain commits are chronologically real". A driver that committed for
  you would forge exactly that chronology. Each phase therefore ends
  by printing the commit the operator must make, and the next phase
  refuses until the carrier it needs exists.

- **A HOST GUARD.** `_check_point_capsule` validates index, family,
  point, pre-digest, quality, geometry and seed -- but NOT `host`.
  A run split across machines therefore VERIFIES CLEAN while the
  pre-invocation's single `host` field is false for every point
  produced elsewhere (cayley 0241Z, endorsed by grassmann). Until that
  is closed properly in the runner, this driver refuses to touch a
  pre-invocation fired on another machine. It is an enforcement, not a
  fix: the underlying field is still unchecked by the verifier chain,
  and a different caller could still do what this one refuses to.

- **Resume-safety.** `_publish_once` is create-once and refuses an
  existing destination, so a naive re-run after a crash dies on the
  first completed point. Phase 1 is 80 points and hours long; it must
  survive a reboot. Already-published points are SKIPPED, never
  rewritten -- the driver never deletes or replaces a carrier.

- **Live progress**, because a multi-hour run with no output is
  indistinguishable from a hung one.

Tier-S output is PRELIMINARY_SMOKE. Nothing here certifies anything,
opens any window-2 value, or licenses a scientific claim. The panels
are fully synthetic (numpy draws from the registered seed grammar);
this is a power simulation, not an analysis of observations.
"""
import json
import os
import platform
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_tier_s_runner_cayley as TSR
from w2_cert_runner_cayley import RunnerRefusal

# The REGISTERED production paths. They are module constants rather
# than CLI arguments on purpose: a caller-supplied grid or geometry
# path is precisely the shape the geometry-admission ruling closed.
GRIDS_REL = "docs/f2g_window2_execution/effect_grids_w2_v1.json"
GEOMETRY_REL = ("docs/f2g_window2_execution/"
                "bound_geometry_capsule_v2.json")
IMPL_REL = "monitoring/src/w2_power_harness_cayley.py"
PRE_NAME = "tier_s_pre_invocation.json"


class DriverRefusal(RuntimeError):
    pass


def _refuse(detail):
    raise DriverRefusal(f"TIER_S_DRIVER_REFUSED: {detail}")


def _blob_reader(repo):
    def read(commit, path):
        p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                            f"{commit}:{path}"], capture_output=True)
        if p.returncode != 0:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: {path} unreadable at "
                f"{commit}")
        return p.stdout
    return read


def _load_pre_checked(outdir):
    """Reopen the pre-invocation and enforce the host guard.

    The runner's own `_load_pre` needs the expected digest; here the
    digest IS what we are reading, so this reads first and then hands
    the value back through the runner's verifier, which recomputes it.
    """
    p = os.path.join(outdir, PRE_NAME)
    if not os.path.exists(p):
        _refuse(f"no pre-invocation at {p} -- fire the pre first")
    with open(p, encoding="utf-8") as f:
        pre = json.load(f)
    sha = pre.get("invocation_sha256")
    if not isinstance(sha, str) or not sha:
        _refuse("the pre-invocation carries no invocation digest")
    # the runner recomputes and refuses on divergence
    pre = TSR._load_pre(outdir, sha)
    here = platform.node()
    if pre.get("host") != here:
        _refuse(
            f"this pre-invocation was fired on host {pre.get('host')!r} "
            f"and this is {here!r}. The per-point capsule check does "
            "NOT verify host, so a cross-host run would verify clean "
            "while the carrier's host field became false. Run every "
            "phase on the firing host, or fire a new campaign here.")
    return pre, sha


def _loco_registry(repo, pre):
    """The LOCO fold set, resolved from the PINNED geometry capsule
    the pre-invocation already bound -- never a caller list."""
    cap = json.loads(_blob_reader(repo)(
        pre["geometry"]["commit"],
        pre["geometry"]["path"]).decode("utf-8"))
    carrier = cap["loco_registry_carrier"]
    return sorted(cap["registries"][carrier])


def _capsule_path(outdir, idx, loco):
    name = (f"smoke_loco_{idx:03d}.json" if loco
            else f"smoke_point_{idx:03d}.json")
    return os.path.join(outdir, name)


# ---------------------------------------------------------------- phases
def cmd_fire(repo, outdir, manifest_commit):
    """Phase 0. Publishes the closed pre-invocation create-once."""
    if os.path.exists(os.path.join(outdir, PRE_NAME)):
        _refuse("a pre-invocation already exists in this outdir; "
                "create-once is never reused -- use a fresh outdir "
                "for a new campaign")
    pre, points = TSR.fire_pre(
        repo, manifest_commit, GRIDS_REL, GEOMETRY_REL, IMPL_REL,
        outdir, blob_reader=_blob_reader(repo), argv=list(sys.argv))
    print(f"pre_invocation_sha256 {pre['invocation_sha256']}")
    print(f"host                  {pre['host']}")
    print(f"manifest_commit       {pre['manifest_commit']}")
    print(f"quality               R={pre['quality']['R']} "
          f"n_draws={pre['quality']['n_draws']}")
    print(f"detection points      {len(points)}")
    print(f"output_root           {pre['output_root']}")
    print("\nNEXT: commit the pre-invocation, then run phase1.")
    return 0


def _run_one(repo, outdir, sha, idx, loco):
    TSR.run_smoke_point(repo, outdir, idx, sha, _blob_reader(repo),
                        with_loco=loco)


def cmd_worker(repo, outdir, idx, loco):
    """One point, in its own process. Re-verifies the pre digest and
    the host guard independently of the parent."""
    pre, sha = _load_pre_checked(outdir)
    _run_one(repo, outdir, int(idx), loco)
    return 0


def _drive(repo, outdir, indices, loco, procs):
    """Spawn one worker process per point, `procs` at a time.

    Separate processes rather than threads because the runner's own
    isolation argument applies here too: a worker that shares this
    interpreter shares every monkeypatchable callable in it.
    """
    pending = list(indices)
    done = skipped = 0
    for i in list(pending):
        if os.path.exists(_capsule_path(outdir, i, loco)):
            pending.remove(i)
            skipped += 1
    total = len(pending)
    if skipped:
        print(f"  resuming: {skipped} already published, {total} to go")
    running, t0 = {}, time.time()
    while pending or running:
        while pending and len(running) < procs:
            i = pending.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--worker", repo, outdir, str(i)]
            if loco:
                cmd.append("--loco")
            running[i] = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT)
        for i, proc in list(running.items()):
            if proc.poll() is None:
                continue
            out = proc.stdout.read().decode("utf-8", errors="replace")
            del running[i]
            if proc.returncode != 0:
                for p in running.values():
                    p.kill()
                _refuse(f"point {i} failed (rc={proc.returncode}): "
                        f"{out.strip()[-400:]}")
            done += 1
            el = time.time() - t0
            rate = el / done if done else 0
            print(f"  [{done}/{total}] point {i:3d} ok  "
                  f"elapsed {el/60:5.1f}m  eta "
                  f"{(total-done)*rate/60:5.1f}m", flush=True)
        if running:
            time.sleep(0.5)
    return done, skipped


def cmd_phase1(repo, outdir, procs):
    pre, sha = _load_pre_checked(outdir)
    points = TSR.derive_points(pre, _blob_reader(repo))
    print(f"phase1: {len(points)} detection points, {procs} processes")
    done, skipped = _drive(repo, outdir, range(len(points)), False,
                           procs)
    print(f"phase1 complete: {done} run, {skipped} already present")
    print("\nNEXT: run rank.")
    return 0


def cmd_rank(repo, outdir):
    pre, sha = _load_pre_checked(outdir)
    top8 = TSR.rank_stage1_b1b(outdir, sha, _blob_reader(repo))
    print("top8 " + ",".join(str(i) for i in top8))
    print("\nNEXT: run phase2 with exactly this list.")
    return 0


def cmd_phase2(repo, outdir, procs, top8):
    """Stage 2: LOCO folds for the stage-1 top-8 B1B points.

    The list is re-derived here and compared against the caller's, so
    a hand-edited selection cannot enter the run.
    """
    pre, sha = _load_pre_checked(outdir)
    derived = TSR.rank_stage1_b1b(outdir, sha, _blob_reader(repo))
    if list(top8) != list(derived):
        _refuse(f"the supplied top-8 {list(top8)} is not the "
                f"deterministic stage-1 ranking {derived}")
    print(f"phase2: {len(derived)} LOCO points, {procs} processes")
    done, skipped = _drive(repo, outdir, derived, True, procs)
    print(f"phase2 complete: {done} run, {skipped} already present")
    print("\nNEXT: run aggregate.")
    return 0


def cmd_aggregate(repo, outdir):
    pre, sha = _load_pre_checked(outdir)
    reader = _blob_reader(repo)
    top8 = TSR.rank_stage1_b1b(outdir, sha, reader)
    registry = _loco_registry(repo, pre)
    results, comp, smoke = TSR.aggregate(repo, outdir, sha, top8,
                                         registry, reader)
    print(f"results_blob_sha256   {smoke['results_blob_sha256']}")
    print(f"completion_sha256     {smoke['completion_sha256']}")
    print(f"loco fold set         {len(registry)} stations")
    print("\nNEXT: commit results+completion+draft smoke, then run "
          "finalize with the pre commit and the results commit.")
    return 0


def cmd_finalize(repo, outdir, pre_commit, results_commit, rel_dir):
    """Reopens all three carriers from their COMMITS and publishes the
    final smoke. The digests must already agree; this proves the
    committed bytes are the ones the draft described."""
    pre, sha = _load_pre_checked(outdir)
    reader = _blob_reader(repo)
    r_rel = f"{rel_dir}/tier_s_results.json"
    r_sha = __import__("hashlib").sha256(
        reader(results_commit, r_rel)).hexdigest()
    smoke = TSR.finalize_smoke(
        outdir,
        {"commit": pre_commit, "path": f"{rel_dir}/{PRE_NAME}"},
        {"commit": results_commit,
         "path": f"{rel_dir}/tier_s_completion.json"},
        {"commit": results_commit, "path": r_rel,
         "blob_sha256": r_sha},
        reader)
    print(f"final smoke published, results {r_sha[:12]}")
    print("\nNEXT: commit the final smoke, then run select.")
    return 0


def _usage():
    return (
        "TIER_S_DRIVER_USAGE:\n"
        "  driver.py fire      <repo> <outdir> <manifest_commit>\n"
        "  driver.py phase1    <repo> <outdir> <procs>\n"
        "  driver.py rank      <repo> <outdir>\n"
        "  driver.py phase2    <repo> <outdir> <procs> <i,i,...>\n"
        "  driver.py aggregate <repo> <outdir>\n"
        "  driver.py finalize  <repo> <outdir> <pre_commit> "
        "<results_commit> <rel_dir>\n"
        "  driver.py --worker  <repo> <outdir> <idx> [--loco]\n"
        "  driver.py --selftest\n"
        "Operator commits happen BETWEEN phases and are not made by "
        "this driver.")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(_usage())
    cmd = argv[1]
    if cmd == "--selftest":
        return _selftest()
    if cmd == "--worker":
        return cmd_worker(argv[2], argv[3], argv[4],
                          "--loco" in argv[5:])
    if cmd == "fire":
        return cmd_fire(os.path.abspath(argv[2]), argv[3], argv[4])
    if cmd == "phase1":
        return cmd_phase1(os.path.abspath(argv[2]), argv[3],
                          int(argv[4]))
    if cmd == "rank":
        return cmd_rank(os.path.abspath(argv[2]), argv[3])
    if cmd == "phase2":
        return cmd_phase2(os.path.abspath(argv[2]), argv[3],
                          int(argv[4]),
                          [int(x) for x in argv[5].split(",") if x])
    if cmd == "aggregate":
        return cmd_aggregate(os.path.abspath(argv[2]), argv[3])
    if cmd == "finalize":
        return cmd_finalize(os.path.abspath(argv[2]), argv[3],
                            argv[4], argv[5], argv[6])
    raise SystemExit(_usage())


# ---------------------------------------------------------------- selftest
def _selftest():
    """Exercises the driver's OWN behaviour -- the host guard, the
    resume skip, the phase-ordering refusals and the stage-2 ranking
    lock -- against the REAL runner with stub smoke functions. It
    fires no real point and publishes nothing outside a temp tree.
    """
    import hashlib
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="tier-s-driver-selftest-")
    try:
        repo = os.path.join(tmp, "r")
        os.makedirs(repo)

        def g(*a):
            return subprocess.run(["git", "-C", repo] + list(a),
                                  capture_output=True, check=True)
        g("init", "-q", "-b", "main")
        g("config", "user.email", "s@t")
        g("config", "user.name", "s")

        def wf(rel, body):
            p = os.path.join(repo, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)

        grids = {"B2A": [{"m": 1}], "B2B": [{"m": 2}],
                 "B1B": [{"k": 1}, {"k": 2}, {"k": 3}],
                 "B3A": [{"m": 3}]}
        wf(GRIDS_REL, json.dumps({"grids": grids}, sort_keys=True))
        cap = {"capsule_digest": "c" * 64,
               "seed_authority_sha256": "b" * 64,
               "loco_registry_carrier": "cascadia",
               "registries": {"cascadia": ["S1", "S0"]}}
        wf(GEOMETRY_REL, json.dumps(cap, sort_keys=True))
        wf(IMPL_REL, "# impl\n")
        g("add", "-A")
        g("commit", "-qm", "artifacts")
        c1 = g("rev-parse", "HEAD").stdout.decode().strip()

        def sha_at(rel):
            return hashlib.sha256(
                subprocess.run(["git", "-C", repo, "cat-file", "blob",
                                f"{c1}:{rel}"],
                               capture_output=True).stdout).hexdigest()

        man = {"slots": {"s": {"status": "BOUND", "pins": [
            {"path": r, "commit": c1, "blob_sha256": sha_at(r)}
            for r in (GRIDS_REL, GEOMETRY_REL, IMPL_REL)]}}}
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(man, sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "manifest")
        c2 = g("rev-parse", "HEAD").stdout.decode().strip()

        outdir = os.path.join(repo, "tier_s")
        pre, points = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, outdir,
            blob_reader=_blob_reader(repo), argv=["selftest"])
        assert len(points) == 6, len(points)
        print(f"  D-0 PASS  pre fired over {len(points)} points")

        # ---- D-1 the HOST GUARD ------------------------------------
        _load_pre_checked(outdir)          # positive: same host
        p = os.path.join(outdir, PRE_NAME)
        with open(p, encoding="utf-8") as f:
            body = json.load(f)
        real_host = body["host"]
        body["host"] = "some-other-machine"
        body["invocation_sha256"] = TSR._digest(
            {k: v for k, v in body.items()
             if k != "invocation_sha256"})
        os.remove(p)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(body, indent=1, sort_keys=True) + "\n")
        try:
            _load_pre_checked(outdir)
        except DriverRefusal as e:
            assert "host" in str(e), str(e)
        else:
            raise AssertionError(
                "D-1 FAILED: a pre fired on another host was accepted")
        body["host"] = real_host
        body["invocation_sha256"] = TSR._digest(
            {k: v for k, v in body.items()
             if k != "invocation_sha256"})
        os.remove(p)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(body, indent=1, sort_keys=True) + "\n")
        _load_pre_checked(outdir)
        print("  D-1 PASS  a pre-invocation fired on another host "
              "REFUSES here, and the same-host positive still loads "
              "(so the guard is not refusing everything)")

        # ---- D-2 phase ordering ------------------------------------
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        try:
            _load_pre_checked(empty)
        except DriverRefusal as e:
            assert "no pre-invocation" in str(e), str(e)
        else:
            raise AssertionError("D-2 FAILED: ran without a pre")
        try:
            cmd_fire(repo, outdir, c2)
        except DriverRefusal as e:
            assert "already exists" in str(e), str(e)
        else:
            raise AssertionError(
                "D-2 FAILED: re-fired over a live pre-invocation")
        print("  D-2 PASS  a phase without a pre refuses, and a "
              "second fire over a live outdir refuses (create-once is "
              "never reused)")

        # ---- D-3 resume skip ---------------------------------------
        sha = pre["invocation_sha256"]
        reader = _blob_reader(repo)

        def stub(fam, point, folds):
            rec = {"family": fam, "point": point,
                   "quality": dict(pre["quality"]),
                   "geometry_capsule_digest":
                       pre["geometry"]["capsule_digest"],
                   "seed_authority_sha256":
                       pre["seed_authority_sha256"],
                   "certifiable": False,
                   "replicates": [{"p_values": {
                       "B1B": 0.001, "B2A": 0.5, "B2B": 0.5,
                       "B3A": 0.5}}]}
            if folds:
                rec["loco_folds"] = [{"S0": 0.001, "S1": 0.001}]
            return rec

        for i in range(len(points)):
            TSR.run_smoke_point(repo, outdir, i, sha, reader,
                                smoke_fn=stub)
        before = sorted(os.listdir(outdir))
        done, skipped = _drive(repo, outdir, range(len(points)),
                               False, 2)
        assert (done, skipped) == (0, len(points)), (done, skipped)
        assert sorted(os.listdir(outdir)) == before, \
            "D-3 FAILED: a resume rewrote or added a carrier"
        print(f"  D-3 PASS  a resumed phase SKIPS all {skipped} "
              "published points, spawns nothing, and leaves every "
              "carrier byte-untouched")

        # ---- D-4 the stage-2 ranking lock --------------------------
        derived = TSR.rank_stage1_b1b(outdir, sha, reader)
        try:
            cmd_phase2(repo, outdir, 1, [derived[-1]] if derived
                       else [0])
        except DriverRefusal as e:
            assert "deterministic stage-1 ranking" in str(e), str(e)
        else:
            raise AssertionError(
                "D-4 FAILED: a hand-edited top-8 was accepted")
        print("  D-4 PASS  a hand-supplied stage-2 selection that is "
              "not the deterministic ranking REFUSES")

        # ---- D-5 the spawn path is REAL, and a worker failure
        # aborts rather than being swallowed ---------------------
        # D-3 proves the resume SKIPS, but on its own it would pass
        # even if `_drive` never spawned anything at all. This fires
        # a genuine worker subprocess against a fresh outdir: the
        # child must actually start, load the pre, clear the host
        # guard, reach the REAL harness -- and die there, because
        # this selftest repo's geometry is a fixture. That failure
        # is the point twice over: it proves the process really ran,
        # and it proves the parent turns a worker failure into a
        # typed refusal instead of continuing with a missing point.
        out2 = os.path.join(repo, "tier_s_spawn")
        pre2, pts2 = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out2,
            blob_reader=_blob_reader(repo), argv=["selftest-spawn"])
        assert not os.path.exists(_capsule_path(out2, 0, False))
        try:
            _drive(repo, out2, [0], False, 1)
        except DriverRefusal as e:
            assert "point 0 failed" in str(e), str(e)
            # the child got past argument parsing, the pre load and
            # the host guard -- it failed inside the runner/harness
            assert "TIER_S_DRIVER_REFUSED: this pre-invocation"                 not in str(e), "the worker died on the host guard"
        else:
            raise AssertionError(
                "D-5 FAILED: a worker that cannot produce a point "
                "reported success -- a silent hole in a multi-hour "
                "run")
        assert not os.path.exists(_capsule_path(out2, 0, False)),             "D-5 FAILED: a failed worker left a carrier behind"
        print("  D-5 PASS  `_drive` really SPAWNS (the child reached "
              "the real harness before failing on fixture geometry), "
              "a worker failure becomes a typed refusal rather than "
              "a missing point, and nothing partial is left behind")

        print("w2 tier-s driver selftest: ALL PASS "
              "(driver behaviour only; stub smoke; nothing fired). "
              "NOT covered here: a worker SUCCEEDING end-to-end, "
              "which needs real pinned geometry and is exercised "
              "only by the real phase-1 run.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
