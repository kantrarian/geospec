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
SELECTOR_REL = "monitoring/src/w2_tier_selector_cayley.py"
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
    if pre.get("schema") != TSR.PRE_SCHEMA:
        _refuse(
            f"the pre-invocation carries schema {pre.get('schema')!r}; "
            f"this production path admits only {TSR.PRE_SCHEMA!r}. A v1 "
            "carrier has no driver pin and no execution capsule, so "
            "accepting one here would reopen exactly the unbound "
            "surfaces v2 closes -- there is no downgrade path.")
    # codex 0314Z point 4: the LIVE runtime identity, recomputed here
    # rather than read from the carrier. My earlier host-only guard
    # was defense-in-depth that same-host interpreter drift walked
    # straight through; this names whichever field drifted.
    TSR.require_live_execution(pre)
    return pre, sha


def _lf_sha(path):
    """LF-normalized digest of a working-tree file.

    The manifest's `blob_sha256` is the git blob, which is LF; the
    checkout on this box is CRLF. Comparing raw bytes would refuse
    every file on Windows for a reason that has nothing to do with
    provenance.
    """
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(
            f.read().replace(b"\r\n", b"\n")).hexdigest()


def _require_bound_sources(repo, pre, manifest_commit=None):
    """codex 0314Z point 2: "the driver, runner, and harness used by
    each process must match their bound source identities before any
    work begins."

    The pre binds the driver and the harness directly; the runner is
    resolved from the manifest the pre binds. This checks the files
    THIS process will actually import, so a reviewed driver cannot
    fire and a locally edited one cannot quietly take over mid-run.
    """
    man = json.loads(_blob_reader(repo)(
        pre["manifest_commit"] if pre else manifest_commit,
        "docs/f2g_window2_execution/execution_manifest.json"
    ).decode("utf-8"))

    def _mp(path):
        try:
            return TSR._pin_for(man, path)
        except TSR.RunnerRefusal as e:
            _refuse(str(e))

    # With a carrier, compare against what the RUN claims; before one
    # exists (cmd_fire) the manifest is the only authority there is.
    d_ref = pre["driver"] if pre else _mp(TSR.DRIVER_REL)
    i_ref = pre["implementation"] if pre else _mp(IMPL_REL)
    checks = [(d_ref["path"], d_ref["blob_sha256"], "driver"),
              (i_ref["path"], i_ref["blob_sha256"], "harness")]
    try:
        runner_pin = TSR._pin_for(man, TSR.RUNNER_REL)
    except TSR.RunnerRefusal as e:
        _refuse(str(e))
    checks.append((TSR.RUNNER_REL, runner_pin["blob_sha256"],
                   "runner"))
    # codex item 4: the SELECTOR is an executable implementation the
    # later commands run, and `verify-select` uses it to adjudicate
    # the artifact under test. A tampered live selector could admit a
    # bad chain, so it is bound here rather than trusted.
    # through codex's BOUND-only helper, not a private re-scan: my
    # first version walked every slot without checking `status`,
    # which is precisely the defect their finding 1 closed.
    try:
        sel_pin = TSR._pin_for(man, SELECTOR_REL)
    except TSR.RunnerRefusal as e:
        _refuse(str(e))
    checks.append((SELECTOR_REL, sel_pin["blob_sha256"], "selector"))
    for rel, want, label in checks:
        live = os.path.join(repo, rel.replace("/", os.sep))
        if not os.path.exists(live):
            _refuse(f"the {label} {rel} is absent from the worktree")
        got = _lf_sha(live)
        if got != want:
            _refuse(
                f"the {label} on disk ({got[:12]}) is not the source "
                f"identity the pre binds ({str(want)[:12]}) -- the "
                "reviewed bytes are not the bytes that would run")
    return True


def _loco_registry(repo, pre):
    """The LOCO fold set, resolved from the PINNED geometry capsule
    the pre-invocation already bound -- never a caller list."""
    cap = json.loads(_blob_reader(repo)(
        pre["geometry"]["commit"],
        pre["geometry"]["path"]).decode("utf-8"))
    carrier = cap["loco_registry_carrier"]
    return sorted(cap["registries"][carrier])


def _report_after_publish(kind, report):
    """Run a reporting closure whose artifact is ALREADY published.

    Returns 0 either way, because the postcondition -- the artifact
    exists -- has already been met. If the summary itself fails, that
    is stated as a typed status rather than being allowed to surface
    as a failed command, because the operator's next decision depends
    on whether the artifact exists, not on whether it got printed.
    """
    try:
        report()
    except Exception as exc:                             # noqa: BLE001
        print(f"{kind}_PUBLISHED_REPORTING_FAILED: the artifact IS "
              f"published and valid; only this summary failed "
              f"({type(exc).__name__}: {str(exc)[:160]}). Do NOT "
              "re-run this command -- the create-once artifact "
              "exists.")
    return 0


def _capsule_path(outdir, idx, loco):
    name = (f"smoke_loco_{idx:03d}.json" if loco
            else f"smoke_point_{idx:03d}.json")
    return os.path.join(outdir, name)


def _repo_rel(repo, outdir):
    """The outdir must live INSIDE the repo, because the whole
    chronology argument rests on the carriers being committable. A
    campaign writing outside the tree can never have a committed pre
    to check against, which is exactly how finding 2 slipped past."""
    r = os.path.abspath(repo)
    o = os.path.abspath(outdir)
    try:
        common = os.path.commonpath([r, o])
    except ValueError:                      # different drives
        common = None
    if common != r or r == o:
        _refuse(f"the output root {o} is not inside the repository "
                f"{r}; a carrier that cannot be committed cannot have "
                "its pre-fire commit verified")
    return os.path.relpath(o, r).replace(os.sep, "/")


def _is_ancestor(repo, a, b):
    return subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", a, b],
        capture_output=True).returncode == 0


def _require_committed_pre(repo, outdir, pre_commit):
    """codex 0257Z finding 2 (CRITICAL). The claimed `fire -> commit
    -> phase1` boundary was advisory: phase 1 needed only a live pre
    FILE, so it ran happily before the operator commit -- with no git
    repository at all -- while reporting success. The chronology the
    runner's contract rests on was therefore not enforced anywhere.

    Every phase and every worker now proves the pre it is about to
    act on is the one that was COMMITTED: same repo-relative path,
    byte-identical to the live create-once carrier, and binding a
    manifest that is an ancestor of the pre commit."""
    rel = _repo_rel(repo, outdir)
    live_path = os.path.join(outdir, PRE_NAME)
    if not os.path.exists(live_path):
        _refuse(f"no pre-invocation at {live_path}")
    with open(live_path, "rb") as f:
        live = f.read()
    try:
        committed = _blob_reader(repo)(pre_commit, f"{rel}/{PRE_NAME}")
    except Exception as exc:                             # noqa: BLE001
        _refuse(f"the pre-invocation is not committed at "
                f"{str(pre_commit)[:12]}:{rel}/{PRE_NAME} "
                f"({str(exc)[:120]}) -- commit it before this phase")
    if committed != live:
        _refuse("the committed pre-invocation differs byte-for-byte "
                "from the live carrier; the chronology would be false")
    pre = json.loads(live.decode("utf-8"))
    mc = pre.get("manifest_commit")
    if not _is_ancestor(repo, mc, pre_commit):
        _refuse(f"the pre binds manifest {str(mc)[:12]} which is not "
                f"an ancestor of the pre commit {str(pre_commit)[:12]}"
                " -- unrelated lineage")
    return pre


def _validate_published(repo, outdir, pre, points, idx, loco):
    """codex 0257Z finding 4 (MAJOR). Resume trusted a FILENAME: any
    bytes at the expected path counted as a finished point, so
    `not-json` read as complete and a forged carrier could steer the
    rank path. Nothing is a skip until it has been reopened and has
    passed the same closed validation aggregate applies.

    This is the ONE validator; resume, rank and aggregate all call
    it, so the three paths cannot drift apart. Invalid bytes REFUSE
    and are never overwritten -- create-once means a bad carrier is
    an operator decision, not something a driver silently repairs."""
    path = _capsule_path(outdir, idx, loco)
    try:
        with open(path, encoding="utf-8") as f:
            cap = json.load(f)
    except ValueError as exc:
        _refuse(f"the published carrier {os.path.basename(path)} is "
                f"not JSON ({str(exc)[:80]}) -- refusing rather than "
                "counting it complete or overwriting it")
    fam, point = points[idx]
    # the RUNNER's own closed check, not a re-implementation
    TSR._check_point_capsule(cap, idx, fam, point, pre)
    rec = cap["record"]
    reps = rec.get("replicates")
    want_r = pre["quality"]["R"]
    if not isinstance(reps, list) or len(reps) != want_r:
        _refuse(f"{os.path.basename(path)} carries "
                f"{len(reps) if isinstance(reps, list) else 'no'} "
                f"replicates, the pre binds R={want_r}")
    for j, rep in enumerate(reps):
        pv = rep.get("p_values") if isinstance(rep, dict) else None
        if not isinstance(pv, dict) or \
                set(pv) != {"B1B", "B2A", "B2B", "B3A"}:
            _refuse(f"{os.path.basename(path)} replicate {j} is not a "
                    "closed four-family p-vector")
    if loco:
        folds = rec.get("loco_folds")
        if not isinstance(folds, list) or len(folds) != want_r:
            _refuse(f"{os.path.basename(path)} is a LOCO carrier with "
                    "no per-replicate fold map")
        registry = set(_loco_registry(repo, pre))
        for j, fr in enumerate(folds):
            if not isinstance(fr, dict) or set(fr) != registry:
                _refuse(f"{os.path.basename(path)} replicate {j} fold "
                        "set is not the registered LOCO station set")
    elif rec.get("loco_folds") is not None:
        # the harness ALWAYS emits this key; for a detection point it
        # is None. Refusing on presence rejected every real carrier.
        _refuse(f"{os.path.basename(path)} is a detection carrier "
                "carrying LOCO folds")
    return cap


# ---------------------------------------------------------------- phases
def cmd_fire(repo, outdir, manifest_commit):
    """Phase 0. Publishes the closed pre-invocation create-once."""
    if os.path.exists(os.path.join(outdir, PRE_NAME)):
        _refuse("a pre-invocation already exists in this outdir; "
                "create-once is never reused -- use a fresh outdir "
                "for a new campaign")
    # The binding is CREATED here, so this is the one place an
    # unbound driver could mint a pre asserting the pinned identity,
    # after which every later phase would verify happily against a
    # claim minted by unbound code. Checked BEFORE publication: a
    # refusal after fire_pre would strand a create-once carrier.
    _require_bound_sources(repo, None, manifest_commit)
    pre, points = TSR.fire_pre(
        repo, manifest_commit, GRIDS_REL, GEOMETRY_REL, IMPL_REL,
        outdir, blob_reader=_blob_reader(repo), argv=list(sys.argv))
    # ---- PUBLICATION HAS HAPPENED. Everything below is reporting.
    # codex 1550Z finding 4: the exit status must describe the
    # POSTCONDITION, not whether the summary printed. The first
    # version of this function published the create-once carrier and
    # then died on `pre['host']` -- a field v2 had moved into the
    # execution capsule -- exiting non-zero. An operator trusting
    # that status retries, hits RUNNER_PUBLISH_EXISTS, and now has a
    # live carrier they believe does not exist. A reporting
    # exception must never masquerade as an absent artifact.
    return _report_after_publish(
        "TIER_S_PRE", lambda: (
            print(f"pre_invocation_sha256 {pre['invocation_sha256']}"),
            print(f"host                  "
                  f"{pre['execution']['host']}"),
            print(f"interpreter           "
                  f"{pre['execution']['interpreter_implementation']} "
                  f"{pre['execution']['numpy_version']}"),
            print(f"manifest_commit       {pre['manifest_commit']}"),
            print(f"driver                "
                  f"{pre['driver']['commit'][:12]} / "
                  f"{pre['driver']['blob_sha256'][:12]}"),
            print(f"quality               R={pre['quality']['R']} "
                  f"n_draws={pre['quality']['n_draws']}"),
            print(f"detection points      {len(points)}"),
            print(f"output_root           {pre['output_root']}"),
            print("\nNEXT: commit the pre-invocation ALONE, then run "
                  "phase1 with that commit.")))


def _run_one(repo, outdir, sha, idx, loco):
    TSR.run_smoke_point(repo, outdir, idx, sha, _blob_reader(repo),
                        with_loco=loco)


def cmd_worker(repo, outdir, idx, loco, pre_commit):
    """One point, in its own process. Repeats EVERY gate the parent
    ran -- the host guard, the pre digest, and the committed-pre
    proof -- because a child that trusts its parent is a child the
    parent's caller can steer."""
    pre, sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    _run_one(repo, outdir, sha, int(idx), loco)
    return 0


def _drive(repo, outdir, indices, loco, procs, pre, points,
           pre_commit):
    """Spawn one worker process per point, `procs` at a time.

    Separate processes rather than threads because the runner's own
    isolation argument applies here too: a worker that shares this
    interpreter shares every monkeypatchable callable in it.
    """
    if isinstance(procs, bool) or not isinstance(procs, int) or \
            procs < 1:
        _refuse(f"process count {procs!r} is not a positive integer "
                "(codex 0257Z finding 5: procs=0 entered a "
                "non-progressing busy loop)")
    cap = (os.cpu_count() or 1) * 2
    if procs > cap:
        _refuse(f"process count {procs} exceeds this host's cap "
                f"{cap} ({os.cpu_count()} logical cores x2)")
    pending = list(indices)
    done = skipped = 0
    for i in list(pending):
        if os.path.exists(_capsule_path(outdir, i, loco)):
            # NOT a skip until it has been reopened and validated
            _validate_published(repo, outdir, pre, points, i, loco)
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
                   "--worker", repo, outdir, str(i), pre_commit]
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


def cmd_phase1(repo, outdir, procs, pre_commit):
    pre, sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    points = TSR.derive_points(pre, _blob_reader(repo))
    print(f"phase1: {len(points)} detection points, {procs} processes")
    done, skipped = _drive(repo, outdir, range(len(points)), False,
                           procs, pre, points, pre_commit)
    print(f"phase1 complete: {done} run, {skipped} already present")
    print("\nNEXT: run rank.")
    return 0


def cmd_rank(repo, outdir, pre_commit):
    pre, sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    points = TSR.derive_points(pre, _blob_reader(repo))
    # codex 0257Z finding 4: rank reads every B1B carrier, so it gets
    # the SAME closed validation as resume and aggregate. Ranking off
    # a carrier nothing reopened is how a forged point would steer
    # the stage-2 selection.
    for i, (fam, _p) in enumerate(points):
        if fam == "B1B":
            _validate_published(repo, outdir, pre, points, i, False)
    top8 = TSR.rank_stage1_b1b(outdir, sha, _blob_reader(repo))
    print("top8 " + ",".join(str(i) for i in top8))
    print("\nNEXT: run phase2 with exactly this list.")
    return 0


def cmd_phase2(repo, outdir, procs, top8, pre_commit):
    """Stage 2: LOCO folds for the stage-1 top-8 B1B points.

    The list is re-derived here and compared against the caller's, so
    a hand-edited selection cannot enter the run.
    """
    pre, sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    points = TSR.derive_points(pre, _blob_reader(repo))
    for i, (fam, _p) in enumerate(points):
        if fam == "B1B":
            _validate_published(repo, outdir, pre, points, i, False)
    derived = TSR.rank_stage1_b1b(outdir, sha, _blob_reader(repo))
    if list(top8) != list(derived):
        _refuse(f"the supplied top-8 {list(top8)} is not the "
                f"deterministic stage-1 ranking {derived}")
    print(f"phase2: {len(derived)} LOCO points, {procs} processes")
    done, skipped = _drive(repo, outdir, derived, True, procs, pre,
                           points, pre_commit)
    print(f"phase2 complete: {done} run, {skipped} already present")
    print("\nNEXT: run aggregate.")
    return 0


def cmd_aggregate(repo, outdir, pre_commit):
    pre, sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    reader = _blob_reader(repo)
    points = TSR.derive_points(pre, reader)
    for i, (fam, _p) in enumerate(points):
        _validate_published(repo, outdir, pre, points, i, False)
    top8 = TSR.rank_stage1_b1b(outdir, sha, reader)
    for i in top8:
        _validate_published(repo, outdir, pre, points, i, True)
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
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
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


def cmd_select(repo, outdir, smoke_commit, grids_commit, rel_dir,
               pre_commit):
    """codex 0257Z finding 5: the driver used to stop at `finalize`
    and print "run select" for a command that did not exist -- the
    selector module's `__main__` is a selftest only, so the promised
    smoke-to-selector route ended in an unreviewed manual link.

    This is that link, governed: the final smoke and the effect grids
    are reopened FROM THEIR COMMITS, the selector artifact is
    published create-once, and nothing is taken from caller state."""
    import w2_tier_selector_cayley as TS
    pre, _sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    reader = _blob_reader(repo)
    smoke_rel = f"{rel_dir}/tier_s_smoke_final.json"
    smoke = json.loads(reader(smoke_commit,
                              smoke_rel).decode("utf-8"))
    if "pre_invocation_ref" not in smoke:
        _refuse("the committed smoke is a DRAFT (no pre_invocation "
                "ref) -- run finalize and commit before select")
    grids_art = json.loads(reader(grids_commit,
                                  GRIDS_REL).decode("utf-8"))
    grids = grids_art.get("grids", grids_art)
    art = TS.select_candidates(
        smoke, grids,
        smoke_ref={"commit": smoke_commit, "path": smoke_rel},
        effect_grids_ref={"commit": grids_commit,
                          "path": GRIDS_REL})
    TSR._publish_once(os.path.join(outdir, "selector.json"),
                      json.dumps(art, indent=1, sort_keys=True,
                                 allow_nan=False) + "\n")
    print("selector published create-once")
    print("\nNEXT: commit the selector, then run verify-select with "
          "its commit.")
    return 0


def cmd_verify_select(repo, outdir, selector_commit, manifest_commit,
                      rel_dir, pre_commit):
    """The post-commit half: reopen the COMMITTED selector and put it
    through the production admission function. A selector that only
    exists on disk has never been proven to be the one that was
    published."""
    import hashlib
    import w2_tier_selector_cayley as TS
    pre, _sha = _load_pre_checked(outdir)
    _require_committed_pre(repo, outdir, pre_commit)
    _require_bound_sources(repo, pre)
    reader = _blob_reader(repo)
    rel = f"{rel_dir}/selector.json"
    raw = reader(selector_commit, rel)
    art = json.loads(raw.decode("utf-8"))
    adm = TS.verify_selector_admission(
        repo, art, manifest_commit,
        selector_identity={"commit": selector_commit, "path": rel,
                           "blob_sha256":
                               hashlib.sha256(raw).hexdigest()})
    print(f"selector ADMITTED at manifest {adm['manifest_commit'][:12]}"
          f", pre {adm['pre_invocation']['commit'][:12]}")
    return 0


def _usage():
    return (
        "TIER_S_DRIVER_USAGE:\n"
        "  driver.py fire       <repo> <outdir> <manifest_commit>\n"
        "  driver.py phase1     <repo> <outdir> <procs> <pre_commit>\n"
        "  driver.py rank       <repo> <outdir> <pre_commit>\n"
        "  driver.py phase2     <repo> <outdir> <procs> <i,i,...> "
        "<pre_commit>\n"
        "  driver.py aggregate  <repo> <outdir> <pre_commit>\n"
        "  driver.py finalize   <repo> <outdir> <pre_commit> "
        "<results_commit> <rel_dir>\n"
        "  driver.py select     <repo> <outdir> <smoke_commit> "
        "<grids_commit> <rel_dir> <pre_commit>\n"
        "  driver.py verify-select <repo> <outdir> <selector_commit> "
        "<manifest_commit> <rel_dir> <pre_commit>\n"
        "  driver.py --worker   <repo> <outdir> <idx> <pre_commit> "
        "[--loco]\n"
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
        return cmd_worker(argv[2], argv[3], argv[4], "--loco" in argv,
                          argv[5])
    if cmd == "fire":
        return cmd_fire(os.path.abspath(argv[2]), argv[3], argv[4])
    if cmd == "phase1":
        return cmd_phase1(os.path.abspath(argv[2]), argv[3],
                          int(argv[4]), argv[5])
    if cmd == "rank":
        return cmd_rank(os.path.abspath(argv[2]), argv[3], argv[4])
    if cmd == "phase2":
        return cmd_phase2(os.path.abspath(argv[2]), argv[3],
                          int(argv[4]),
                          [int(x) for x in argv[5].split(",") if x],
                          argv[6])
    if cmd == "aggregate":
        return cmd_aggregate(os.path.abspath(argv[2]), argv[3],
                             argv[4])
    if cmd == "select":
        return cmd_select(os.path.abspath(argv[2]), argv[3], argv[4],
                          argv[5], argv[6], argv[7])
    if cmd == "verify-select":
        return cmd_verify_select(os.path.abspath(argv[2]), argv[3],
                                 argv[4], argv[5], argv[6], argv[7])
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
    import time

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

        # The REGISTERED effect grids, verbatim. A synthetic grid
        # cannot be used any more: the pre-run gate (codex 0314Z,
        # from grassmann's pre-registration) refuses any order that
        # is not the registered one, and a selftest that could opt
        # out of a production gate would not be testing production.
        real_grids = os.path.join(
            os.path.dirname(os.path.dirname(_HERE)),
            GRIDS_REL.replace("/", os.sep))
        with open(real_grids, encoding="utf-8") as f:
            grids_body = f.read()
        wf(GRIDS_REL, grids_body)
        grids = json.loads(grids_body)["grids"]
        # the driver and runner are pinned too, because
        # `_require_bound_sources` checks the bytes that would run
        wf(TSR.DRIVER_REL, "# driver fixture\n")
        wf(TSR.RUNNER_REL, "# runner fixture\n")
        wf(SELECTOR_REL, "# selector fixture\n")
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
            for r in (GRIDS_REL, GEOMETRY_REL, IMPL_REL,
                      TSR.DRIVER_REL, TSR.RUNNER_REL,
                      SELECTOR_REL)]}}}
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(man, sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "manifest")
        c2 = g("rev-parse", "HEAD").stdout.decode().strip()

        outdir = os.path.join(repo, "tier_s")
        pre, points = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, outdir,
            blob_reader=_blob_reader(repo), argv=["selftest"])
        assert len(points) == 80, len(points)
        assert pre["schema"] == TSR.PRE_SCHEMA
        assert set(pre["driver"]) == {"commit", "path", "blob_sha256"}
        TSR.validate_execution(pre["execution"])
        _require_bound_sources(repo, pre)
        g("add", "-A")
        g("commit", "-qm", "pre-invocation")
        c3 = g("rev-parse", "HEAD").stdout.decode().strip()
        print(f"  D-0 PASS  v2 pre fired over {len(points)} REAL "
              "registered points, carrying the driver pin and the "
              "closed execution capsule, with the bound sources "
              "verified on disk")

        # ---- D-1 EXECUTION DRIFT, field by field ----------------
        # The v1 driver guarded `host` only, and codex showed
        # same-host interpreter drift walking straight through it.
        # Every field of the closed capsule is now load-bearing, so
        # every field gets its own refusal AND the untouched positive
        # is re-run between them -- a guard that refuses everything
        # would otherwise look identical to one that works.
        _load_pre_checked(outdir)                      # positive
        p = os.path.join(outdir, PRE_NAME)

        def _repre(mut):
            with open(p, encoding="utf-8") as f:
                body = json.load(f)
            mut(body)
            body["invocation_sha256"] = TSR._digest(
                {k: v for k, v in body.items()
                 if k != "invocation_sha256"})
            os.remove(p)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(body, indent=1, sort_keys=True)
                        + "\n")

        real_ex = dict(pre["execution"])
        for field, bogus in (
                ("host", "some-other-machine"),
                ("interpreter_executable", "X:/not-this-python.exe"),
                ("interpreter_implementation", "NotCPython"),
                ("interpreter_version", "different-python"),
                ("numpy_version", "0.0.0"),
                ("numpy_config_sha256", "f" * 64)):
            _repre(lambda b, _f=field, _v=bogus: b["execution"].update(
                {_f: _v}))
            try:
                _load_pre_checked(outdir)
            except TSR.RunnerRefusal as e:
                assert "EXECUTION_DRIFT" in str(e) and field in str(e), \
                    (field, str(e))
            else:
                raise AssertionError(
                    f"D-1 FAILED: execution drift on {field} accepted")
            _repre(lambda b, _e=real_ex: b.__setitem__("execution",
                                                       dict(_e)))
            _load_pre_checked(outdir)      # positive between each
        # and a v1 carrier has no way back in
        _repre(lambda b: b.__setitem__(
            "schema", "f2g-w2-tier-s-pre-invocation-v1"))
        try:
            _load_pre_checked(outdir)
        except DriverRefusal as e:
            assert "no downgrade path" in str(e), str(e)
        else:
            raise AssertionError(
                "D-1 FAILED: a v1 pre was admitted by the v2 path")
        _repre(lambda b: b.__setitem__("schema", TSR.PRE_SCHEMA))
        _load_pre_checked(outdir)
        print("  D-1 PASS  all six execution-identity fields drift-"
              "refuse INDIVIDUALLY (host, interpreter path, "
              "implementation, version, NumPy version, NumPy build "
              "config), a v1 downgrade refuses, and the untouched "
              "positive re-passes between every one")

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
                   # the REAL harness record always carries
                   # loco_folds (None for detection). The stub must
                   # match it, or stub-based controls cannot see a
                   # shape defect that the real path hits.
                   "loco_folds": None,
                   "replicates": [{"p_values": {
                       "B1B": 0.001, "B2A": 0.5, "B2B": 0.5,
                       "B3A": 0.5}}
                       for _ in range(pre["quality"]["R"])]}
            if folds:
                rec["loco_folds"] = [{"S0": 0.001, "S1": 0.001}
                                     for _ in range(
                                         pre["quality"]["R"])]
            return rec

        for i in range(len(points)):
            TSR.run_smoke_point(repo, outdir, i, sha, reader,
                                smoke_fn=stub)
        before = sorted(os.listdir(outdir))
        done, skipped = _drive(repo, outdir, range(len(points)),
                               False, 2, pre, points, c3)
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
                       else [0], c3)
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
        g("add", "-A")
        g("commit", "-qm", "spawn pre")
        c3b = g("rev-parse", "HEAD").stdout.decode().strip()
        # (a) the worker must reach the HARNESS. Asserting only
        # that "something failed" is what let the previous
        # version of this control stay green while the child was
        # dying on a TypeError in cmd_worker, never reaching the
        # harness at all -- and the routed packet repeated the
        # strong reading. The refusal must now NAME the harness.
        HARNESS_MARK = "w2_power_harness_cayley"
        try:
            _drive(repo, out2, [0], False, 1, pre2, pts2, c3b)
        except DriverRefusal as e:
            assert "point 0 failed" in str(e), str(e)
            assert (HARNESS_MARK in str(e)
                    or "POWER_" in str(e)), (
                "D-5 FAILED: the worker died BEFORE reaching the "
                "harness, so this control proves nothing about "
                "the spawn path -- " + str(e)[:200])
        else:
            raise AssertionError(
                "D-5 FAILED: a worker that cannot produce a "
                "point reported success -- a silent hole in a "
                "multi-hour run")
        assert not os.path.exists(_capsule_path(out2, 0, False)), (
            "D-5 FAILED: a failed worker left a carrier behind")

        # (b) DISCRIMINATION: the same assertion must go RED when
        # the worker dies EARLIER, inside the driver's own gates.
        # Without this, (a) could be satisfied by accident. Here
        # the child gets a pre commit that does not carry this
        # outdir, so it refuses in _require_committed_pre --
        # driver code, before the harness is ever imported.
        try:
            _drive(repo, out2, [0], False, 1, pre2, pts2, c3)
        except DriverRefusal as e2:
            early = str(e2)
            assert "point 0 failed" in early, early
            assert (HARNESS_MARK not in early
                    and "POWER_" not in early), (
                "D-5 FAILED: an early driver-gate failure is "
                "indistinguishable from reaching the harness, so "
                "the marker cannot discriminate -- " + early[:200])
        else:
            raise AssertionError(
                "D-5 FAILED: a worker with a wrong pre commit "
                "reported success")
        print("  D-5 PASS  `_drive` really SPAWNS and the child "
              "REACHES THE HARNESS (refusal names "
              "w2_power_harness_cayley/POWER_*), a worker failure "
              "becomes a typed refusal leaving no partial carrier "
              "-- AND the same assertion goes RED for a worker "
              "that dies earlier in the driver's own gates, so it "
              "discriminates rather than accepting any failure")

        # ---- D-6 (codex R-2, CRITICAL): the committed-pre gate ---
        # R-2 ran phase 1 with no git repository at all. The claimed
        # fire -> commit -> phase1 boundary was advisory; it is now
        # proved against the committed bytes.
        out3 = os.path.join(repo, "tier_s_uncommitted")
        pre3, pts3 = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out3,
            blob_reader=_blob_reader(repo), argv=["selftest-nc"])
        try:
            _require_committed_pre(repo, out3, c3)
        except DriverRefusal as e:
            assert "not committed" in str(e), str(e)
        else:
            raise AssertionError(
                "D-6 FAILED: an UNCOMMITTED pre was accepted -- the "
                "chronology boundary is advisory again")
        g("add", "-A")
        g("commit", "-qm", "uncommitted-pre now committed")
        c3c = g("rev-parse", "HEAD").stdout.decode().strip()
        _require_committed_pre(repo, out3, c3c)      # positive
        # the SAME carrier under a wrong commit still refuses
        try:
            _require_committed_pre(repo, out3, c3)
        except DriverRefusal as e:
            assert "not committed" in str(e), str(e)
        else:
            raise AssertionError(
                "D-6 FAILED: a pre commit that does not carry this "
                "outdir was accepted")
        # an outdir outside the repo can never be committed
        try:
            _repo_rel(repo, tmp)
        except DriverRefusal as e:
            assert "not inside the repository" in str(e), str(e)
        else:
            raise AssertionError(
                "D-6 FAILED: an outdir outside the repo was accepted")
        print("  D-6 PASS  an UNCOMMITTED pre refuses, the same "
              "carrier under the wrong commit refuses, an outdir "
              "outside the repo refuses -- and the committed positive "
              "still passes")

        # ---- D-7 (codex R-3, MAJOR): resume validates, not stats ---
        out4 = os.path.join(repo, "tier_s_forged")
        pre4, pts4 = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out4,
            blob_reader=_blob_reader(repo), argv=["selftest-forge"])
        g("add", "-A")
        g("commit", "-qm", "forge pre")
        c3d = g("rev-parse", "HEAD").stdout.decode().strip()
        with open(os.path.join(out4, "smoke_point_000.json"), "w",
                  encoding="utf-8") as f:
            f.write("not-json")
        try:
            _drive(repo, out4, [0], False, 1, pre4, pts4, c3d)
        except DriverRefusal as e:
            assert "not JSON" in str(e), str(e)
        else:
            raise AssertionError(
                "D-7 FAILED: malformed bytes at the expected path "
                "counted as a completed point")
        assert open(os.path.join(out4, "smoke_point_000.json"),
                    encoding="utf-8").read() == "not-json", \
            "D-7 FAILED: the driver OVERWROTE an existing carrier"
        # a structurally valid capsule with the WRONG replicate count
        # must also refuse -- filename plus parseability is not enough
        # NOTE the capsule is otherwise WELL FORMED under v2 -- closed
        # schema, right identities, right execution digest -- so the
        # refusal isolates the replicate-count check rather than
        # tripping an earlier one.
        short = {"index": 0, "family": pts4[0][0], "point": pts4[0][1],
                 "pre_invocation_sha256": pre4["invocation_sha256"],
                 "execution_sha256": "PLACEHOLDER",
                 "record": {"replicates": [],
                            "loco_folds": None,
                            "certifiable": False}}
        out5 = os.path.join(repo, "tier_s_short")
        pre5, pts5 = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out5,
            blob_reader=_blob_reader(repo), argv=["selftest-short"])
        g("add", "-A")
        g("commit", "-qm", "short pre")
        c3e = g("rev-parse", "HEAD").stdout.decode().strip()
        short["pre_invocation_sha256"] = pre5["invocation_sha256"]
        short["execution_sha256"] = TSR.execution_digest(
            pre5["execution"])
        with open(os.path.join(out5, "smoke_point_000.json"), "w",
                  encoding="utf-8") as f:
            json.dump(short, f)
        try:
            _drive(repo, out5, [0], False, 1, pre5, pts5, c3e)
        except DriverRefusal as e:
            assert "replicates" in str(e), str(e)
        else:
            raise AssertionError(
                "D-7 FAILED: a carrier with the wrong replicate count "
                "counted as complete")
        # and a point produced under a DIFFERENT runtime identity
        # refuses even though everything else about it is right
        drift = dict(short, execution_sha256="a" * 64)
        drift["record"] = {"replicates": [
            {"p_values": {"B1B": 0.5, "B2A": 0.5, "B2B": 0.5,
                          "B3A": 0.5}}
            for _ in range(pre5["quality"]["R"])],
            "loco_folds": None, "certifiable": False}
        out6 = os.path.join(repo, "tier_s_exdrift")
        pre6, pts6 = TSR.fire_pre(
            repo, c2, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out6,
            blob_reader=_blob_reader(repo), argv=["selftest-exd"])
        g("add", "-A")
        g("commit", "-qm", "exdrift pre")
        c3f = g("rev-parse", "HEAD").stdout.decode().strip()
        drift["pre_invocation_sha256"] = pre6["invocation_sha256"]
        with open(os.path.join(out6, "smoke_point_000.json"), "w",
                  encoding="utf-8") as f:
            json.dump(drift, f)
        try:
            _drive(repo, out6, [0], False, 1, pre6, pts6, c3f)
        except TSR.RunnerRefusal as e:
            assert "EXECUTION_DRIFT" in str(e), str(e)
        else:
            raise AssertionError(
                "D-7 FAILED: a point carrying a foreign execution "
                "digest counted as complete")
        print("  D-7 PASS  malformed bytes, a wrong-replicate-count "
              "carrier, and a point bearing a FOREIGN execution digest "
              "all refuse instead of counting as complete, and none is "
              "overwritten")

        # ---- D-8 (codex R-4, MODERATE): process count ------------
        for bad in (0, -1, 1.5, True, "4"):
            try:
                _drive(repo, outdir, [], False, bad, pre, points, c3)
            except DriverRefusal as e:
                assert "process count" in str(e), (bad, str(e))
            else:
                raise AssertionError(
                    f"D-8 FAILED: procs={bad!r} accepted (R-4 was a "
                    "non-progressing busy loop)")
        print("  D-8 PASS  0, -1, 1.5, True and \"4\" are each "
              "refused as a process count, and the host cap is "
              "enforced")

        # ---- D-9 (codex R-6): the select surface exists -----------
        u = _usage()
        assert "select" in u and "verify-select" in u, u
        for name in ("cmd_select", "cmd_verify_select"):
            assert name in globals(), name
        print("  D-9 PASS  a governed select phase and its post-commit "
              "verification exist on the operator surface (R-6 was "
              "'no production select command exists')")

        # ---- D-10 the SOURCE identity is load-bearing ------------
        # codex 0257Z finding 1 was that the firing artifact sat
        # outside the admitted set. Two halves close it: it must BE
        # in the manifest, and the bytes on disk must be the bytes
        # that were admitted. Both get a refusal, and the untouched
        # positive is re-run after, so neither is a guard that simply
        # refuses everything.
        man_no_drv = {"slots": {"s": {"status": "BOUND", "pins": [
            {"path": r, "commit": c1, "blob_sha256": sha_at(r)}
            for r in (GRIDS_REL, GEOMETRY_REL, IMPL_REL,
                      TSR.RUNNER_REL, SELECTOR_REL)]}}}
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(man_no_drv, sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "manifest without the driver")
        c2_nodrv = g("rev-parse", "HEAD").stdout.decode().strip()
        try:
            TSR.fire_pre(repo, c2_nodrv, GRIDS_REL, GEOMETRY_REL,
                         IMPL_REL, os.path.join(repo, "tier_s_nodrv"),
                         blob_reader=_blob_reader(repo),
                         argv=["selftest-nodrv"])
        except TSR.RunnerRefusal as e:
            assert TSR.DRIVER_REL in str(e), str(e)
        else:
            raise AssertionError(
                "D-10 FAILED: a pre was fired binding a driver that "
                "the manifest does not admit")
        # restore the good manifest
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(man, sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "restore manifest")
        c2_ok = g("rev-parse", "HEAD").stdout.decode().strip()
        for label, rel in (("driver", TSR.DRIVER_REL),
                           ("runner", TSR.RUNNER_REL),
                           ("harness", IMPL_REL)):
            live = os.path.join(repo, rel.replace("/", os.sep))
            with open(live, encoding="utf-8") as f:
                keep = f.read()
            with open(live, "w", encoding="utf-8",
                      newline="\n") as f:
                f.write(keep + "# locally edited after admission\n")
            try:
                _require_bound_sources(repo, pre)
            except DriverRefusal as e:
                assert label in str(e) and "not the source identity" \
                    in str(e), (label, str(e))
            else:
                raise AssertionError(
                    f"D-10 FAILED: an edited {label} on disk was "
                    "accepted -- the reviewed bytes are not the bytes "
                    "that would run")
            with open(live, "w", encoding="utf-8",
                      newline="\n") as f:
                f.write(keep)
            _require_bound_sources(repo, pre)      # positive again
        print("  D-10 PASS  a driver absent from the manifest cannot "
              "fire, and an edited driver, runner or harness on disk "
              "each refuse by name -- with the untouched positive "
              "re-passing after every one")

        # ---- D-12 (codex item 3): EVERY CLI wrapper, via main() -
        # Success and refusal through main(argv) for every public
        # command, asserting exit status, the typed transition, the
        # exact artifact keyset, and reopen validation. Two of the
        # three defects in this arc reached public master because the
        # CLI was never composed end to end.
        argv0 = sys.argv[0]

        def cli(*a):
            return main([argv0] + [str(x) for x in a])

        def cli_refuses(where, *a):
            try:
                cli(*a)
            except (DriverRefusal, TSR.RunnerRefusal, SystemExit,
                    IndexError) as exc:
                return str(exc)
            raise AssertionError(
                f"D-12 {where}: main({a[0]}) returned instead of "
                "refusing")

        # -- refusal surface, one per command ---------------------
        cli_refuses("phase1 needs a committed pre", "phase1", repo,
                    out3, 1, c3)
        cli_refuses("rank needs a committed pre", "rank", repo, out3,
                    c3)
        cli_refuses("aggregate needs a committed pre", "aggregate",
                    repo, out3, c3)
        cli_refuses("worker needs a committed pre", "--worker", repo,
                    out3, 0, c3)
        cli_refuses("phase2 rejects a hand list", "phase2", repo,
                    outdir, 1, "0", c3)
        cli_refuses("unknown command", "not-a-command")
        cli_refuses("fire refuses a live outdir", "fire", repo,
                    outdir, c2)
        print("  D-12a PASS  every command refuses through main(argv) "
              "when its precondition is absent -- committed pre, "
              "deterministic ranking, create-once outdir, unknown "
              "verb")

        # -- success surface, on the stub campaign ----------------
        # rank: exit 0 and the deterministic top-8
        assert cli("rank", repo, outdir, c3) == 0, "D-12 rank"
        top = TSR.rank_stage1_b1b(outdir, sha, reader)
        # phase 2's LOCO carriers, published through the runner with
        # the stub, so aggregate's success path is reachable here
        for i in top:
            fam_i, pt_i = points[i]
            TSR.run_smoke_point(repo, outdir, i, sha, reader,
                                with_loco=True,
                                smoke_fn=stub)
        before_keys = set(os.listdir(outdir))
        assert cli("aggregate", repo, outdir, c3) == 0, "D-12 aggregate"
        new_keys = set(os.listdir(outdir)) - before_keys
        assert new_keys == {"tier_s_results.json",
                            "tier_s_completion.json",
                            "tier_s_smoke.json"}, sorted(new_keys)
        for nm in sorted(new_keys):
            with open(os.path.join(outdir, nm), encoding="utf-8") as f:
                json.load(f)          # reopen validation
        g("add", "-A")
        g("commit", "-qm", "d12 results")
        c_res = g("rev-parse", "HEAD").stdout.decode().strip()
        rel_dir = os.path.relpath(outdir, repo).replace(os.sep, "/")
        assert cli("finalize", repo, outdir, c3, c_res, rel_dir) == 0
        assert os.path.exists(os.path.join(
            outdir, "tier_s_smoke_final.json")), "D-12 finalize"
        g("add", "-A")
        g("commit", "-qm", "d12 final smoke")
        c_sm = g("rev-parse", "HEAD").stdout.decode().strip()
        assert cli("select", repo, outdir, c_sm, c1, rel_dir, c3) == 0
        assert os.path.exists(os.path.join(outdir, "selector.json")), \
            "D-12 select"
        g("add", "-A")
        g("commit", "-qm", "d12 selector")
        c_sel = g("rev-parse", "HEAD").stdout.decode().strip()
        # verify-select drives PRODUCTION selector admission,
        # which resolves the geometry capsule for real. A
        # fixture capsule cannot satisfy that, so its SUCCESS
        # path is reachable only from a real campaign. What is
        # asserted here is the honest half: it must REFUSE the
        # fixture chain, and for the geometry reason -- which
        # is itself a control, because a verify-select that
        # accepted a fixture chain would be worthless.
        # its own handler rather than widening cli_refuses: the
        # generic helper deliberately catches a NARROW tuple, and
        # broadening it to admit SelectorRefusal would weaken every
        # other refusal check in D-12a to buy one assertion here.
        import w2_tier_selector_cayley as _TSsel
        try:
            cli("verify-select", repo, outdir, c_sel, c2, rel_dir, c3)
        except _TSsel.SelectorRefusal as _vsx:
            vs = str(_vsx)
            assert "SELECTOR_UNADMITTED" in vs, vs[:200]
            assert "POWER_GEOMETRY_UNBOUND" in vs, (
                "D-12: verify-select refused, but not for the geometry "
                f"reason -- {vs[:200]}")
        else:
            raise AssertionError(
                "D-12 FAILED: verify-select ADMITTED a fixture chain; "
                "production admission must resolve the geometry "
                "capsule for real")
        print("  D-12b PASS  rank, aggregate, finalize and "
              "select each SUCCEED through main(argv) with exit "
              "0, publishing exactly their declared artifact "
              "keyset, every one of which reopens and parses. "
              "verify-select REFUSES the fixture chain on "
              "geometry, as it must; its success path needs a "
              "real campaign and is NOT covered here")


        # ---- D-11 (codex item 1): THE PROCESS BOUNDARY, real ----
        # The defect that killed the campaign lived in the argument
        # handoff between cmd_worker and _run_one -- a seam every
        # in-process test skipped. This drives the whole chain for
        # real, on one point, against the REAL pinned geometry:
        #   main(argv) -> _drive -> child -> main(argv) --worker
        #   -> cmd_worker -> _run_one -> TSR.run_smoke_point
        # and requires the child to EXIT 0 leaving one carrier that
        # reopens and validates. ~10 minutes; that cost is the point.
        real_repo = os.path.dirname(os.path.dirname(_HERE))
        pbt = tempfile.mkdtemp(prefix="d11-process-boundary-")
        pwt = os.path.join(pbt, "t")
        try:
            add = subprocess.run(
                ["git", "-C", real_repo, "worktree", "add", "--detach",
                 pwt, "HEAD"], capture_output=True)
            if add.returncode:
                raise AssertionError(
                    "D-11 could not materialise a detached worktree: "
                    + add.stderr.decode(errors="replace")[:200])
            pout = os.path.join(pwt, "docs", "f2g_window2_execution",
                                "d11_boundary")

            def pg(*a):
                return subprocess.run(["git", "-C", pwt] + list(a),
                                      capture_output=True, check=True)
            pg("config", "user.email", "d11@t")
            pg("config", "user.name", "d11")

            # (1) fire through main(argv) -- exit 0, one carrier
            rc = main([sys.argv[0], "fire", pwt, pout, "HEAD"])
            assert rc == 0, f"D-11: fire returned {rc}, expected 0"
            assert sorted(os.listdir(pout)) == [PRE_NAME], (
                "D-11: fire must leave exactly the pre-invocation, "
                f"found {sorted(os.listdir(pout))}")

            # (2) commit it alone, so the committed-pre gate has an
            # exact artifact to prove the child against
            pg("add", os.path.relpath(
                os.path.join(pout, PRE_NAME), pwt).replace(os.sep, "/"))
            pg("commit", "-qm", "d11 pre")
            pc = subprocess.run(["git", "-C", pwt, "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True).stdout.strip()

            # (3) ONE point through a REAL child process
            bpre, bsha = _load_pre_checked(pout)
            bpts = TSR.derive_points(bpre, _blob_reader(pwt))
            fam0, _pt0 = bpts[0]
            print(f"  D-11 .... running ONE REAL point ({fam0}) "
                  "through a real child process, ~10 min")
            t0 = time.time()
            done, skipped = _drive(pwt, pout, [0], False, 1, bpre,
                                   bpts, pc)
            el = time.time() - t0
            assert (done, skipped) == (1, 0), (done, skipped)

            # (4) the carrier must exist and REOPEN valid
            cap_p = _capsule_path(pout, 0, False)
            assert os.path.exists(cap_p), (
                "D-11 FAILED: the child reported success but left no "
                "carrier")
            cap = _validate_published(pwt, pout, bpre, bpts, 0, False)
            assert len(cap["record"]["replicates"]) == \
                bpre["quality"]["R"]
            assert cap["execution_sha256"] == \
                TSR.execution_digest(bpre["execution"])
            # ---- D-11b (codex item 2): the four mutations ----
            # A positive that cannot go red proves nothing. Each of
            # these must break the boundary FOR ITS OWN REASON, and
            # each expectation was measured by running it, not read
            # off the source. Index 1 is unpublished, and all four
            # fail before the harness, so they are seconds not
            # minutes.
            with open(__file__, encoding="utf-8") as _df:
                drv_src = _df.read()
            MUTS = [
                ("omit sha",
                 "_run_one(repo, outdir, sha, int(idx), loco)",
                 "_run_one(repo, outdir, int(idx), loco)",
                 "missing 1 required positional argument"),
                ("misorder sha",
                 "_run_one(repo, outdir, sha, int(idx), loco)",
                 "_run_one(repo, outdir, int(idx), sha, loco)",
                 "RUNNER_TIER_S_PRE_DIGEST_MISMATCH"),
                ("prevent runner entry",
                 "import w2_tier_s_runner_cayley as TSR",
                 "raise ImportError('MUTATION: runner entry "
                 "prevented')\nimport w2_tier_s_runner_cayley as TSR",
                 "MUTATION: runner entry prevented"),
                ("break committed-pre loading",
                 "if committed != live:",
                 "if True:",
                 "the committed pre-invocation differs byte-for-byte"),
            ]
            menv = dict(os.environ, PYTHONPATH=_HERE)
            for mname, mold, mnew, needle in MUTS:
                assert drv_src.count(mold) >= 1, f"{mname}: anchor"
                mpath = os.path.join(pbt, f"mut_{abs(hash(mname))}.py")
                with open(mpath, "w", encoding="utf-8",
                          newline=chr(10)) as _mf:
                    _mf.write(
                        drv_src.replace(mold, mnew, 1))
                mr = subprocess.run(
                    [sys.executable, mpath, "--worker", pwt, pout,
                     "1", pc], capture_output=True, text=True,
                    cwd=_HERE, env=menv, timeout=900)
                mblob = mr.stdout + mr.stderr
                assert mr.returncode != 0, (
                    f"D-11b {mname}: the mutated worker SUCCEEDED")
                assert needle in mblob, (
                    f"D-11b {mname}: failed for the wrong reason -- "
                    f"expected {needle!r}, got "
                    f"{mblob.strip()[-200:]!r}")
                assert "w2_power_harness_cayley" not in mblob and \
                    "POWER_" not in mblob, (
                        f"D-11b {mname}: reached the harness anyway, "
                        "so the positive's marker does not "
                        "discriminate")
                assert not os.path.exists(
                    _capsule_path(pout, 1, False)), (
                        f"D-11b {mname}: left a carrier behind")
            print(f"  D-11b PASS  all {len(MUTS)} mutations break the "
                  "boundary, each for its OWN measured reason (missing "
                  "argument / pre-digest mismatch / runner entry / "
                  "committed-pre bytes), none reaches the harness, "
                  "and none leaves a carrier -- so the positive above "
                  "is falsifiable rather than decorative")


            print(f"  D-11 PASS  the FULL process boundary works for "
                  f"real: main(argv) -> _drive -> child -> --worker "
                  f"-> cmd_worker -> _run_one -> run_smoke_point, "
                  f"child exit 0, one carrier with "
                  f"{len(cap['record']['replicates'])} replicates "
                  f"that reopens and validates ({el/60:.1f} min). "
                  "This is the seam the arity defect lived in, and no "
                  "in-process control could have reached it")
        finally:
            subprocess.run(["git", "-C", real_repo, "worktree",
                            "remove", "--force", pwt],
                           capture_output=True)
            subprocess.run(["git", "-C", real_repo, "worktree",
                            "prune"], capture_output=True)
            shutil.rmtree(pbt, ignore_errors=True)

        print("w2 tier-s driver selftest: ALL PASS "
              "(driver behaviour only; stub smoke; nothing fired). "
              "D-11 covers a worker SUCCEEDING end-to-end against "
              "REAL pinned geometry through a real child "
              "process, and D-11b proves that positive is "
              "falsifiable by four measured mutations.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
