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

- **An IDENTITY JOIN over every executable this process can run**
  (codex 2112Z finding 1). The first source guard hashed
  `repo/<path>` -- repository FILES, not the code executing -- so a
  byte-modified external copy of this driver, handed a clean repository
  as `<repo>`, passed and minted a pre claiming the pinned identity;
  a PYTHONPATH-shadowed runner, cert-runner, harness or selector ran
  unchecked; and a self-hashed committed pre could point `driver` /
  `implementation` at any live path whose bytes it also named. Now,
  before any publication or work: exactly one BOUND pin per registered
  path (resolved by THIS module, not the runner it is about to verify);
  the module actually loaded (`module.__file__`) must BE
  `repo/<registered path>` (samefile) -- so a module from anywhere else
  refuses even when the repository copy is pristine; that file's LF
  digest must equal the pin; and the pre's `driver` and
  `implementation` references must EQUAL their pins exactly. codex
  2303Z: the resolution is stdlib-only and NON-EXECUTING
  (`PathFinder.find_spec`) and precedes every project import, so no
  module acts before it is proven; the phase-B stats engine that
  computes two of the four p-values is the ninth bound executable.

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

# codex 2303Z finding 1 (CRITICAL): NO project-local import before the
# provenance preflight. Module top-levels execute on import, so a module
# imported here would act before it was proven. The runner and the
# cert-runner's typed refusal are bound by `_bootstrap` after EVERY
# executable's origin has been resolved without executing it and checked
# against its BOUND pin.
TSR = None
RunnerRefusal = None
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"
DRIVER_REL = "monitoring/src/w2_tier_s_driver_cayley.py"
RUNNER_REL = "monitoring/src/w2_tier_s_runner_cayley.py"

# The REGISTERED production paths. They are module constants rather
# than CLI arguments on purpose: a caller-supplied grid or geometry
# path is precisely the shape the geometry-admission ruling closed.
GRIDS_REL = "docs/f2g_window2_execution/effect_grids_w2_v1.json"
GEOMETRY_REL = ("docs/f2g_window2_execution/"
                "bound_geometry_capsule_v2.json")
IMPL_REL = "monitoring/src/w2_power_harness_cayley.py"
SELECTOR_REL = "monitoring/src/w2_tier_selector_cayley.py"
# the remaining executables a Tier-S process loads (codex 2112Z finding
# 1: "for every loaded executable module"): the cert-runner supplies
# canonicalisation, create-once publication and manifest resolution;
# the harness imports the two adapters at import time and the
# target-identity guard at run time. All are BOUND manifest pins.
CERT_REL = "monitoring/src/w2_cert_runner_cayley.py"
TARGET_ID_REL = "monitoring/src/w2_target_identity_cayley.py"
B1B_REL = "monitoring/src/w2_b1b.py"
B2B_REL = "monitoring/src/w2_b2b.py"
# codex 2303Z finding 1 half A: the phase-B stats engine -- imported by
# the harness at import time, called for the B2A and B3A p-values, and
# pinned NOWHERE until this cycle. Ninth executable.
STATS_REL = "monitoring/src/d2_f2g_phase_b_stats.py"
# EVERY executable a Tier-S process can run: (label, registered path,
# module name). Order is the refusal order: the driver first, then the
# runner whose helpers the rest of the process would otherwise trust.
EXECUTABLES = (
    ("driver", DRIVER_REL, "w2_tier_s_driver_cayley"),
    ("runner", RUNNER_REL, "w2_tier_s_runner_cayley"),
    ("cert-runner", CERT_REL, "w2_cert_runner_cayley"),
    ("harness", IMPL_REL, "w2_power_harness_cayley"),
    ("selector", SELECTOR_REL, "w2_tier_selector_cayley"),
    ("target-identity", TARGET_ID_REL, "w2_target_identity_cayley"),
    ("b1b adapter", B1B_REL, "w2_b1b"),
    ("b2b adapter", B2B_REL, "w2_b2b"),
    ("phase-b stats", STATS_REL, "d2_f2g_phase_b_stats"),
)
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
            raise (RunnerRefusal or DriverRefusal)(
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


def _read_blob(repo, commit, path):
    """stdlib-only committed-blob read for the preflight: nothing
    project-local may run before provenance is established."""
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{path}"], capture_output=True)
    if p.returncode != 0:
        _refuse(f"{path} unreadable at {str(commit)[:12]}")
    return p.stdout


def _bound_pin(man, rel):
    """Exactly one BOUND manifest pin for `rel`, resolved by THIS module.
    The runner is one of the executables the join is about to verify,
    so its `_pin_for` cannot be trusted before its own identity is."""
    matches = [pin for slot in man.get("slots", {}).values()
               if slot.get("status") == "BOUND"
               for pin in (slot.get("pins", ()) or ())
               if isinstance(pin, dict) and pin.get("path") == rel]
    if len(matches) != 1:
        _refuse(f"{rel} is not exactly one BOUND manifest pin "
                f"(found {len(matches)})")
    return matches[0]


def _ref_short(ref):
    if not isinstance(ref, dict):
        return repr(ref)[:60]
    return (f"{{{str(ref.get('commit'))[:12]} / {ref.get('path')} / "
            f"{str(ref.get('blob_sha256'))[:12]}}}")


def _bootstrap(repo, manifest_commit):
    """codex 2303Z finding 1 (CRITICAL): provenance BEFORE execution.

    Two halves were open. (A) The harness imports the phase-B stats
    engine and calls it for two of the four p-values; it was pinned
    nowhere and outside the join, so a foreign copy returning p=0.0
    passed while the gate attested the pristine repository file. (B)
    The driver imported the runner and cert-runner at module top and
    the join imported the rest BEFORE checking them; module top-levels
    run on import, so a shadow's import-time code had already acted by
    the time it was refused.

    Now, stdlib-only and in this order:
      1. read the manifest at `manifest_commit` (git cat-file);
      2. for EVERY executable in EXECUTABLES (nine, the stats engine
         included) resolve the origin the interpreter WOULD load --
         `importlib.machinery.PathFinder.find_spec` over `sys.path`,
         which executes nothing -- or, if the name is already bound in
         `sys.modules` (imported before this process reached the
         preflight), take that module's `__file__`; require the origin
         to BE `repo/<registered path>` (samefile) and its LF digest to
         equal exactly one BOUND pin -- before importing anything;
      3. only then import, and re-check each module's `__file__`
         against the same pin (defence in depth);
      4. bind the proven runner and the cert-runner's refusal class.
    Idempotent: later calls re-run every check against the same pins.
    """
    global TSR, RunnerRefusal
    import importlib
    import importlib.machinery as _mach
    repo = os.path.abspath(repo)
    man = json.loads(_read_blob(repo, manifest_commit,
                                MANIFEST_REL).decode("utf-8"))
    resolved = []
    for label, rel, modname in EXECUTABLES:
        fixed = os.path.join(repo, rel.replace("/", os.sep))
        if not os.path.exists(fixed):
            _refuse(f"the {label} {rel} is absent from the worktree")
        pin = _bound_pin(man, rel)
        preloaded = None if label == "driver" else sys.modules.get(modname)
        if label == "driver":
            origin = os.path.abspath(__file__)
            how = "executing in this process was loaded from"
        elif preloaded is not None:
            # bound before this preflight (a launcher, a .pth, a
            # sitecustomize): its import-time code has already run in
            # this process; it is refused by name and never used
            origin = getattr(preloaded, "__file__", None)
            how = "executing in this process was loaded from"
        else:
            spec = _mach.PathFinder.find_spec(modname, list(sys.path))
            origin = getattr(spec, "origin", None) \
                if spec is not None else None
            how = "would be loaded from"
        if not origin or not os.path.exists(origin):
            _refuse(f"the {label} ({modname}) resolves to no source "
                    "file on this interpreter's path")
        if not os.path.samefile(origin, fixed):
            _refuse(f"the {label} {how} {os.path.abspath(origin)}, not "
                    f"the repository's {rel} -- a module from anywhere "
                    "else refuses even when the repository copy is "
                    "pristine")
        got = _lf_sha(origin)
        if got != pin["blob_sha256"]:
            _refuse(
                f"the {label} on disk ({got[:12]}) is not the source "
                f"identity the manifest binds "
                f"({str(pin['blob_sha256'])[:12]}) -- the reviewed "
                "bytes are not the bytes that would run")
        resolved.append((label, rel, modname, fixed, pin))
    # every origin is proven; NOW import, and re-check what was loaded
    mods = {}
    for label, rel, modname, fixed, pin in resolved:
        if label == "driver":
            mod = sys.modules[__name__]
        else:
            mod = importlib.import_module(modname)
        mf = getattr(mod, "__file__", None)
        if not mf or not os.path.exists(mf) or \
                not os.path.samefile(mf, fixed) or \
                _lf_sha(mf) != pin["blob_sha256"]:
            _refuse(f"the {label} changed identity between preflight "
                    f"and import ({mf})")
        mods[label] = mod
    TSR = mods["runner"]
    RunnerRefusal = mods["cert-runner"].RunnerRefusal
    if TSR.DRIVER_REL != DRIVER_REL or TSR.RUNNER_REL != RUNNER_REL:
        _refuse("the bound runner registers different driver/runner "
                "paths than this driver")
    return mods


def _loaded_executables():
    """(label, registered path, module) for every bound executable,
    AFTER `_bootstrap` has proven and imported them."""
    if TSR is None:
        _refuse("executables requested before the provenance preflight "
                "bound them")
    return [(label, rel, sys.modules[__name__] if label == "driver"
             else sys.modules[modname])
            for label, rel, modname in EXECUTABLES]


def _enter(repo, outdir, pre_commit):
    """codex 0149Z finding 1 (CRITICAL): the trust root is the COMMITTED
    pre-invocation, never the live pointer. The previous gate read the
    LIVE pre's `manifest_commit` and ran the preflight against it, so a
    substituted live file naming a local manifest that pinned a
    marker-bearing runner got that runner imported before the
    committed-pre check ever ran.

    Stdlib-only until the root is proven, in this order:
      1. the outdir must be inside the repository (its committed path
         is derivable);
      2. reopen `pre_commit:<rel>/tier_s_pre_invocation.json` with
         `git cat-file`; the LIVE bytes must EQUAL the committed bytes;
      3. parse the COMMITTED bytes; `manifest_commit` must be a 40-hex
         commit that is an ancestor of `pre_commit`;
      4. only then run the provenance preflight against THAT manifest
         and bind the executables.
    The later digest / reference / committed-pre checks in every caller
    stay as defence in depth. Returns the committed pre."""
    rel = _repo_rel(repo, outdir)
    live_path = os.path.join(outdir, PRE_NAME)
    if not os.path.exists(live_path):
        _refuse(f"no pre-invocation at {live_path} -- fire the pre first")
    with open(live_path, "rb") as f:
        live = f.read()
    committed = _read_blob(repo, pre_commit, f"{rel}/{PRE_NAME}")
    if committed != live:
        _refuse("the live pre-invocation differs byte-for-byte from the "
                f"one committed at {str(pre_commit)[:12]} -- the trust "
                "root is the COMMITTED carrier; nothing is imported "
                "against a live pointer")
    try:
        pre = json.loads(committed.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        _refuse("the committed pre-invocation is not JSON")
    mc = pre.get("manifest_commit") if isinstance(pre, dict) else None
    if not isinstance(mc, str) or len(mc) != 40 or \
            any(c not in "0123456789abcdef" for c in mc):
        _refuse("the committed pre-invocation names no 40-hex manifest "
                "commit")
    if not _is_ancestor(repo, mc, pre_commit):
        _refuse(f"the committed pre binds manifest {mc[:12]} which is "
                f"not an ancestor of the pre commit {str(pre_commit)[:12]}"
                " -- unrelated lineage")
    _bootstrap(repo, mc)
    return pre


def _require_bound_sources(repo, pre, manifest_commit=None):
    """The identity join: `_bootstrap` (stdlib preflight -> import ->
    recheck) for the manifest the pre binds, then the pre's own
    `driver` and `implementation` references must EQUAL their resolved
    pins exactly ({commit, path, blob_sha256}) -- a reference that
    merely names live bytes is not an identity."""
    mc = pre["manifest_commit"] if pre else manifest_commit
    _bootstrap(repo, mc)
    if pre is not None:
        man = json.loads(_read_blob(repo, mc, MANIFEST_REL)
                         .decode("utf-8"))
        drv_pin = _bound_pin(man, DRIVER_REL)
        want_d = {"commit": drv_pin["commit"], "path": DRIVER_REL,
                  "blob_sha256": drv_pin["blob_sha256"]}
        if pre["driver"] != want_d:
            _refuse(f"the pre's driver reference "
                    f"{_ref_short(pre['driver'])} is not the BOUND "
                    f"manifest pin {_ref_short(want_d)} -- a reference "
                    "that merely names live bytes is not an identity")
        impl_pin = _bound_pin(man, IMPL_REL)
        want_i = {"commit": impl_pin["commit"], "path": IMPL_REL,
                  "blob_sha256": impl_pin["blob_sha256"]}
        if pre["implementation"] != want_i:
            _refuse(f"the pre's implementation reference "
                    f"{_ref_short(pre['implementation'])} is not the "
                    f"BOUND manifest pin {_ref_short(want_i)} -- a "
                    "reference that merely names live bytes is not an "
                    "identity")
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

    codex 2112Z finding 3: EVERY create-once publisher (fire,
    aggregate, finalize, select) routes its whole reporting tail
    through here, the diagnostic is best-effort on stderr, and this
    helper cannot rethrow -- a broken output stream after publication
    must still exit 0, because the artifact exists and a non-zero exit
    invites the retry that create-once then refuses.
    """
    try:
        report()
    except Exception as exc:                             # noqa: BLE001
        try:
            sys.stderr.write(
                f"{kind}_PUBLISHED_REPORTING_FAILED: the artifact IS "
                f"published and valid; only this summary failed "
                f"({type(exc).__name__}: {str(exc)[:160]}). Do NOT "
                "re-run this command -- the create-once artifact "
                "exists.\n")
            sys.stderr.flush()
        except Exception:                                # noqa: BLE001
            pass       # a broken stream must not fail a published step
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
    _enter(repo, outdir, pre_commit)
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
    # grassmann 2124Z lock: children are spawned from THIS file, so this
    # file must BE the repository's pinned driver -- asserted HERE, at
    # the spawn, not only in the join the caller ran, so a shadowed or
    # external parent can never launch a pristine child while itself
    # unreviewed
    spawn = os.path.abspath(__file__)
    fixed = os.path.join(os.path.abspath(repo),
                         DRIVER_REL.replace("/", os.sep))
    if not (os.path.exists(spawn) and os.path.exists(fixed)
            and os.path.samefile(spawn, fixed)):
        _refuse(f"the spawn target {spawn} is not the repository's "
                f"pinned driver {fixed} -- a parent from anywhere else "
                "may not launch children")
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
    _enter(repo, outdir, pre_commit)
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
    _enter(repo, outdir, pre_commit)
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
    _enter(repo, outdir, pre_commit)
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
    _enter(repo, outdir, pre_commit)
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
    # ---- PUBLICATION HAS HAPPENED (three create-once artifacts).
    # codex 2112Z finding 3: the exit status describes the
    # postcondition, never whether the summary printed.
    return _report_after_publish(
        "TIER_S_AGGREGATE", lambda: (
            print(f"results_blob_sha256   {smoke['results_blob_sha256']}"),
            print(f"completion_sha256     {smoke['completion_sha256']}"),
            print(f"loco fold set         {len(registry)} stations"),
            print("\nNEXT: commit results+completion+draft smoke, then "
                  "run finalize with the pre commit and the results "
                  "commit.")))


def cmd_finalize(repo, outdir, pre_commit, results_commit, rel_dir):
    """Reopens all three carriers from their COMMITS and publishes the
    final smoke. The digests must already agree; this proves the
    committed bytes are the ones the draft described."""
    _enter(repo, outdir, pre_commit)
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
    # ---- PUBLICATION HAS HAPPENED (codex 2112Z finding 3)
    return _report_after_publish(
        "TIER_S_FINAL_SMOKE", lambda: (
            print(f"final smoke published, results {r_sha[:12]}"),
            print("\nNEXT: commit the final smoke, then run select.")))


def cmd_select(repo, outdir, smoke_commit, grids_commit, rel_dir,
               pre_commit):
    """codex 0257Z finding 5: the driver used to stop at `finalize`
    and print "run select" for a command that did not exist -- the
    selector module's `__main__` is a selftest only, so the promised
    smoke-to-selector route ended in an unreviewed manual link.

    This is that link, governed: the final smoke and the effect grids
    are reopened FROM THEIR COMMITS, the selector artifact is
    published create-once, and nothing is taken from caller state."""
    _enter(repo, outdir, pre_commit)
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
    # ---- PUBLICATION HAS HAPPENED (codex 2112Z finding 3)
    return _report_after_publish(
        "TIER_S_SELECTOR", lambda: (
            print("selector published create-once"),
            print("\nNEXT: commit the selector, then run verify-select "
                  "with its commit.")))


def cmd_verify_select(repo, outdir, selector_commit, manifest_commit,
                      rel_dir, pre_commit):
    """The post-commit half: reopen the COMMITTED selector and put it
    through the production admission function. A selector that only
    exists on disk has never been proven to be the one that was
    published."""
    import hashlib
    _enter(repo, outdir, pre_commit)
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
        "  driver.py --selftest-in-repo <real_repo>   (internal: the "
        "body, run by --selftest from a fixture repository's own "
        "driver copy)\n"
        "Operator commits happen BETWEEN phases and are not made by "
        "this driver.")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(_usage())
    cmd = argv[1]
    if cmd == "--selftest":
        return _selftest_launch()
    if cmd == "--selftest-in-repo":
        return _selftest(argv[2])
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
def _cli(argv, env=None, timeout=1800):
    """Run one driver command in a REAL child interpreter and return
    (returncode, combined output). The CLI is the operator surface; a
    control that calls the wrapper function in-process is not a
    control of the CLI."""
    p = subprocess.run([sys.executable] + [str(a) for a in argv],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _selftest_launch():
    """`--selftest`: materialise a FIXTURE REPOSITORY that carries the
    REAL executable bytes at their registered paths, then run the
    selftest body FROM THAT REPOSITORY'S OWN DRIVER COPY.

    codex 2112Z finding 1: the source guard now requires the module
    executing in this process to BE `repo/<registered path>` (samefile)
    with the pinned bytes. Running the body from this file against a
    fixture repository would therefore be REFUSED by the join --
    correctly, since this file is not that repository's driver. That
    refusal is the property under test, so it is not bypassed: the body
    is executed by the fixture's copy, for which the join holds exactly
    as it does in production, and this file's D-11 exercises the REAL
    checkout through its own driver in a real child process.
    """
    import shutil
    import tempfile
    real_repo = os.path.dirname(os.path.dirname(_HERE))
    tmp = tempfile.mkdtemp(prefix="tier-s-driver-selftest-")
    try:
        repo = os.path.join(tmp, "r")
        dst = os.path.join(repo, "monitoring", "src")
        os.makedirs(dst)
        n = 0
        for name in sorted(os.listdir(_HERE)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(_HERE, name), "rb") as s:
                body = s.read().replace(b"\r\n", b"\n")
            with open(os.path.join(dst, name), "wb") as d:
                d.write(body)
            n += 1
        print(f"  selftest: fixture repository carries {n} real source "
              "files at their registered paths; the body runs from ITS "
              "driver copy so the identity join holds as in production")
        p = subprocess.run(
            [sys.executable,
             os.path.join(dst, os.path.basename(__file__)),
             "--selftest-in-repo", real_repo], cwd=repo)
        return p.returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _selftest(real_repo):
    """The selftest BODY (run by `_selftest_launch` from a fixture
    repository's own driver copy; `real_repo` is the real checkout,
    used only by D-11 for a detached worktree over real geometry).

    Exercises the driver's OWN behaviour -- the identity join, the host
    guard, the resume skip, the phase-ordering refusals and the stage-2
    ranking lock -- against the REAL runner with stub smoke functions,
    and the FULL operator CLI against real pinned geometry through real
    child processes. It publishes nothing outside temp trees.
    """
    import hashlib
    import shutil
    import tempfile
    import time

    repo = os.path.dirname(os.path.dirname(_HERE))
    real_repo = os.path.abspath(real_repo)
    if os.path.abspath(repo) == real_repo:
        raise AssertionError(
            "the selftest body must run from a FIXTURE repository, "
            "never from the real checkout (use --selftest)")
    tmp = os.path.dirname(os.path.abspath(repo))
    try:
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

        # The REGISTERED effect grids, verbatim from the real checkout.
        # A synthetic grid cannot be used any more: the pre-run gate
        # (codex 0314Z, from grassmann's pre-registration) refuses any
        # order that is not the registered one, and a selftest that
        # could opt out of a production gate would not be testing
        # production.
        with open(os.path.join(real_repo, GRIDS_REL.replace("/", os.sep)),
                  encoding="utf-8") as f:
            grids_body = f.read()
        wf(GRIDS_REL, grids_body)
        grids = json.loads(grids_body)["grids"]
        cap = {"capsule_digest": "c" * 64,
               "seed_authority_sha256": "b" * 64,
               "loco_registry_carrier": "cascadia",
               "registries": {"cascadia": ["S1", "S0"]}}
        wf(GEOMETRY_REL, json.dumps(cap, sort_keys=True))
        # The executables are the REAL bytes the launcher copied in at
        # their registered paths -- no stub modules any more, because
        # the identity join binds the module EXECUTING, and the module
        # executing here is this very file.
        EXEC_RELS = tuple(rel for _l, rel, _m in EXECUTABLES)
        for rel in EXEC_RELS:
            assert os.path.exists(os.path.join(
                repo, rel.replace("/", os.sep))), rel
        g("add", "-A")
        g("commit", "-qm", "artifacts")
        c1 = g("rev-parse", "HEAD").stdout.decode().strip()

        def sha_at(rel, commit=None):
            return hashlib.sha256(
                subprocess.run(["git", "-C", repo, "cat-file", "blob",
                                f"{commit or c1}:{rel}"],
                               capture_output=True).stdout).hexdigest()

        PINNED = (GRIDS_REL, GEOMETRY_REL) + EXEC_RELS

        def manifest_for(overrides=None):
            pins = []
            for r in PINNED:
                cm = (overrides or {}).get(r, c1)
                pins.append({"path": r, "commit": cm,
                             "blob_sha256": sha_at(r, cm)})
            return {"slots": {"s": {"status": "BOUND", "pins": pins}}}
        man = manifest_for()
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(man, sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "manifest")
        c2 = g("rev-parse", "HEAD").stdout.decode().strip()
        # the provenance preflight against the fixture manifest binds
        # TSR and RunnerRefusal for everything below -- the FIRST
        # project-local import in this process happens here
        assert TSR is None, "a project module was bound before the preflight"
        _bootstrap(repo, c2)
        assert TSR is not None and RunnerRefusal is not None

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
              "closed execution capsule, with the identity join "
              f"holding for all {len(EXEC_RELS)} bound executables "
              "(module.__file__ IS repo/<path>, bytes == pin, pre refs "
              "== pins)")

        # ---- D-1 EXECUTION DRIFT, field by field ----------------
        # The v1 driver guarded `host` only, and codex showed
        # same-host interpreter drift walking straight through it.
        # Every field of the closed capsule is now load-bearing, so
        # every field gets its own refusal AND the untouched positive
        # is re-run between them -- a guard that refuses everything
        # would otherwise look identical to one that works.
        _load_pre_checked(outdir)                      # positive
        p = os.path.join(outdir, PRE_NAME)

        def _repre_at(path, mut):
            with open(path, encoding="utf-8") as f:
                body = json.load(f)
            mut(body)
            body["invocation_sha256"] = TSR._digest(
                {k: v for k, v in body.items()
                 if k != "invocation_sha256"})
            os.remove(path)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(body, indent=1, sort_keys=True)
                        + "\n")

        def _repre(mut):
            _repre_at(p, mut)

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
        # child must actually start, load the pre, clear the identity
        # join, reach the REAL harness -- and die there, because
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
            pin for pin in man["slots"]["s"]["pins"]
            if pin["path"] != DRIVER_REL]}}}
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
            assert DRIVER_REL in str(e), str(e)
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
        # EVERY bound executable, edited on disk after import, refuses
        # BY NAME -- the join hashes module.__file__, which for a module
        # loaded from the repository is the edited file
        for label, rel, _mod in _loaded_executables():
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
              f"fire, and each of the {len(EXEC_RELS)} bound "
              "executables edited on disk refuses by name -- with the "
              "untouched positive re-passing after every one")

        # ---- D-10b (codex 2112Z finding 1, second half): a FORGED
        # committed pre cannot point `driver` / `implementation` at
        # any live path -- the references must EQUAL the BOUND pins
        out7 = os.path.join(repo, "tier_s_forgedref")
        pre7, pts7 = TSR.fire_pre(
            repo, c2_ok, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out7,
            blob_reader=_blob_reader(repo), argv=["selftest-forgedref"])
        g("add", "-A")
        g("commit", "-qm", "forgedref pre")
        p7 = os.path.join(out7, PRE_NAME)
        # a REAL live module the fixture manifest does not pin (the
        # stats engine that used to play this role is now the ninth
        # bound executable -- codex 2303Z)
        UNPINNED = "monitoring/src/w2_cascadia.py"
        assert os.path.exists(os.path.join(repo, UNPINNED.replace(
            "/", os.sep))), UNPINNED
        assert not any(pin["path"] == UNPINNED
                       for pin in man["slots"]["s"]["pins"])
        forgeries = [
            ("driver", "driver -> an UNPINNED live path (self-hashed, "
                       "bytes match)",
             lambda b: b.__setitem__("driver", {
                 "commit": c1, "path": UNPINNED,
                 "blob_sha256": sha_at(UNPINNED)})),
            ("implementation", "implementation -> an UNPINNED live "
                               "path (self-hashed, bytes match)",
             lambda b: b.__setitem__("implementation", {
                 "commit": c1, "path": UNPINNED,
                 "blob_sha256": sha_at(UNPINNED)})),
            ("driver", "driver -> a DIFFERENT bound pin (the runner's)",
             lambda b: b.__setitem__("driver", {
                 "commit": c1, "path": RUNNER_REL,
                 "blob_sha256": sha_at(RUNNER_REL)})),
            ("driver", "driver -> right path and bytes, wrong commit",
             lambda b: b["driver"].__setitem__("commit", c2_ok)),
        ]
        for key, label, mut in forgeries:
            _repre_at(p7, mut)
            g("add", "-A")
            g("commit", "-qm", f"forged {key}")
            cf = g("rev-parse", "HEAD").stdout.decode().strip()
            forged, _fs = _load_pre_checked(out7)     # self-consistent
            _require_committed_pre(repo, out7, cf)    # and committed
            try:
                _require_bound_sources(repo, forged)
            except DriverRefusal as e:
                assert f"the pre's {key} reference" in str(e) and \
                    "is not the BOUND manifest pin" in str(e), \
                    (label, str(e))
            else:
                raise AssertionError(
                    f"D-10b FAILED: forged pre reference accepted "
                    f"({label})")
            # and through a COMMAND: the committed-pre gate passes
            # (it IS committed), the join must still refuse
            try:
                cmd_phase1(repo, out7, 1, cf)
            except DriverRefusal as e:
                assert "is not the BOUND manifest pin" in str(e), \
                    (label, str(e))
            else:
                raise AssertionError(
                    f"D-10b FAILED: phase1 ran on a forged pre "
                    f"({label})")
            _repre_at(p7, lambda b: b.update(
                {"driver": dict(pre7["driver"]),
                 "implementation": dict(pre7["implementation"])}))
            g("add", "-A")
            g("commit", "-qm", "restore refs")
            restored, _rs = _load_pre_checked(out7)
            _require_bound_sources(repo, restored)    # positive again
        print(f"  D-10b PASS  {len(forgeries)} forged, self-hashed, "
              "COMMITTED pre references (driver/implementation -> an "
              "unpinned live path, a different bound pin, a wrong "
              "commit) each refuse as 'not the BOUND manifest pin' "
              "both in the join and through phase1, and the restored "
              "positive re-passes after every one")

        # ---- D-10c (codex 2112Z finding 1): the module EXECUTING is
        # the identity -- an external driver copy (pristine OR
        # modified) and a shadowed runner / cert-runner / harness /
        # selector each refuse BY NAME before any outdir exists
        shadow_dir = os.path.join(tmp, "shadow")
        os.makedirs(shadow_dir, exist_ok=True)
        launcher = os.path.join(shadow_dir, "shadow_launch.py")
        with open(launcher, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join([
                "import importlib, os, runpy, sys",
                "name, src, drv = sys.argv[1], sys.argv[2], sys.argv[3]",
                "here = os.path.dirname(os.path.abspath(__file__))",
                "# the shadow directory FIRST, then the repository's "
                "sources so every other import resolves for real",
                "sys.path[:0] = [here, src]",
                "importlib.import_module(name)   # binds the SHADOW",
                "sys.argv = [drv] + sys.argv[4:]",
                "runpy.run_path(drv, run_name='__main__')", ""]))
        drv_path = os.path.join(repo, DRIVER_REL.replace("/", os.sep))
        with open(drv_path, "rb") as f:
            pristine = f.read()
        src_dir = os.path.join(repo, "monitoring", "src")
        env_src = dict(os.environ, PYTHONPATH=src_dir)
        n_ext = 0
        for tag, body in (("pristine", pristine),
                          ("byte-modified",
                           pristine + b"\n# externally modified\n")):
            ext = os.path.join(shadow_dir, f"external_{tag}_driver.py")
            with open(ext, "wb") as f:
                f.write(body)
            out_x = os.path.join(repo, f"tier_s_external_{tag}")
            rc, blob = _cli([ext, "fire", repo, out_x, c2_ok],
                            env=env_src)
            assert rc != 0, f"D-10c: {tag} EXTERNAL driver fired"
            assert "TIER_S_DRIVER_REFUSED" in blob and \
                "the driver executing in this process was loaded from" \
                in blob, (tag, blob[-400:])
            assert not os.path.exists(out_x), (
                f"D-10c: {tag} external driver created the outdir "
                "before refusing")
            n_ext += 1
        shadowed = [(l, r, m.__name__) for l, r, m in _loaded_executables()
                    if l != "driver"]
        for label, rel, modname in shadowed:
            with open(os.path.join(repo, rel.replace("/", os.sep)),
                      "rb") as f:
                real_bytes = f.read()
            sh = os.path.join(shadow_dir, modname + ".py")
            with open(sh, "wb") as f:
                f.write(real_bytes)          # pristine bytes, wrong place
            out_s = os.path.join(repo, f"tier_s_shadow_{modname}")
            rc, blob = _cli([launcher, modname, src_dir, drv_path,
                             "fire", repo, out_s, c2_ok])
            os.remove(sh)
            assert rc != 0, f"D-10c: shadowed {label} fired"
            assert "TIER_S_DRIVER_REFUSED" in blob and \
                f"the {label} executing in this process was loaded from" \
                in blob and shadow_dir.replace("\\", "/") in \
                blob.replace("\\", "/"), (label, blob[-500:])
            assert not os.path.exists(out_s), (
                f"D-10c: shadowed {label} created the outdir")
        out_pos = os.path.join(repo, "tier_s_positive_after_shadows")
        rc, blob = _cli([drv_path, "fire", repo, out_pos, c2_ok])
        assert rc == 0 and sorted(os.listdir(out_pos)) == [PRE_NAME], \
            blob[-400:]
        print(f"  D-10c PASS  {n_ext} EXTERNAL driver copies (pristine "
              f"bytes AND modified) and {len(shadowed)} shadowed "
              "executables (pristine bytes at a foreign path, bound "
              "under the real name before the driver imports it) each "
              "refuse BY NAME with no outdir created; the untouched "
              "positive re-passes through the real CLI")

        # ---- D-10d (grassmann 2124Z lock): the SPAWN TARGET must be
        # the pinned repository driver, asserted at the spawn itself.
        # Mutation: this module's __file__ is pointed at the pristine
        # external copy (same bytes, foreign path); _drive must refuse
        # typed BEFORE validating or spawning anything, and the
        # restored positive must re-pass.
        real_file = globals()["__file__"]
        ext_pristine = os.path.join(shadow_dir,
                                    "external_pristine_driver.py")
        assert os.path.exists(ext_pristine)
        globals()["__file__"] = ext_pristine
        try:
            _drive(repo, out2, [0], False, 1, pre2, pts2, c3b)
        except DriverRefusal as e:
            assert "spawn target" in str(e) and \
                "not the repository's pinned driver" in str(e), str(e)
        else:
            raise AssertionError(
                "D-10d FAILED: _drive spawned from a foreign __file__")
        finally:
            globals()["__file__"] = real_file
        assert not os.path.exists(_capsule_path(out2, 0, False)), (
            "D-10d FAILED: the refused spawn left a carrier")
        done_d, skipped_d = _drive(repo, outdir, range(len(points)),
                                   False, 2, pre, points, c3)
        assert (done_d, skipped_d) == (0, len(points)), (done_d, skipped_d)
        print("  D-10d PASS  a parent whose __file__ is a foreign copy of "
              "the driver (same bytes) is refused AT THE SPAWN, typed, "
              "before validation or launch; the restored positive "
              "re-passes")

        # ---- D-10e (codex 2303Z finding 1, half B): provenance BEFORE
        # execution. The in-tree runner and the in-tree phase-B stats
        # engine each get an IMPORT-TIME side effect (write a marker
        # file) prepended. A child driver must refuse each by name on
        # the byte digest -- and the marker must NOT exist afterwards,
        # proving the module was resolved and rejected without ever
        # being imported. (The previous driver imported the runner at
        # module top: the marker would have been written first.)
        for label, rel, modname in (("runner", RUNNER_REL,
                                     "w2_tier_s_runner_cayley"),
                                    ("phase-b stats", STATS_REL,
                                     "d2_f2g_phase_b_stats")):
            live = os.path.join(repo, rel.replace("/", os.sep))
            marker = os.path.join(tmp, f"import_marker_{modname}.txt")
            with open(live, "rb") as f:
                keep_b = f.read()
            side_effect = ("import os as _mo\n_mo.makedirs(%r, exist_ok="
                           "True)\nopen(%r, 'w').write('EXECUTED')\n"
                           % (os.path.dirname(marker), marker))
            with open(live, "wb") as f:
                f.write(side_effect.encode("utf-8") + keep_b)
            out_m = os.path.join(repo, f"tier_s_marker_{modname}")
            rc, blob = _cli([drv_path, "fire", repo, out_m, c2_ok])
            with open(live, "wb") as f:
                f.write(keep_b)
            assert rc != 0 and f"the {label}" in blob and \
                "not the source identity" in blob, (label, blob[-400:])
            assert not os.path.exists(marker), (
                f"D-10e FAILED: the {label}'s import-time code EXECUTED "
                "before the identity check")
            assert not os.path.exists(out_m), label
        rc, blob = _cli([drv_path, "rank", repo, outdir, c3])
        assert rc == 0 and "top8 " in blob, blob[-300:]
        print("  D-10e PASS  an in-tree runner and an in-tree phase-B "
              "stats engine carrying IMPORT-TIME side effects are each "
              "refused by name in a child driver and their markers "
              "never appear -- resolved and rejected without being "
              "imported; the restored positive re-passes")

        # ---- D-10f (codex 2303Z finding 1, half A + its control): a
        # FOREIGN phase-B stats engine that would return p=0.0 for B2A
        # and B3A is bound under the real name before the driver runs;
        # the worker must refuse it BY NAME before any harness call,
        # leaving no carrier.
        sh = os.path.join(shadow_dir, "d2_f2g_phase_b_stats.py")
        with open(sh, "w", encoding="utf-8", newline="\n") as f:
            f.write("def b2a_family(*a, **k):\n"
                    "    return {'p_value': 0.0, 'source': 'MALICIOUS'}\n"
                    "\n\ndef b3a_family(*a, **k):\n"
                    "    return {'p_value': 0.0, 'source': 'MALICIOUS'}\n")
        out_fs = os.path.join(repo, "tier_s_foreign_stats")
        TSR.fire_pre(repo, c2_ok, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out_fs,
                     blob_reader=_blob_reader(repo),
                     argv=["selftest-foreign-stats"])
        g("add", "-A")
        g("commit", "-qm", "foreign-stats pre")
        c_fs = g("rev-parse", "HEAD").stdout.decode().strip()
        rc, blob = _cli([launcher, "d2_f2g_phase_b_stats", src_dir,
                         drv_path, "--worker", repo, out_fs, "0", c_fs])
        os.remove(sh)
        assert rc != 0 and "the phase-b stats executing in this process " \
            "was loaded from" in blob, blob[-500:]
        assert HARNESS_MARK not in blob and "POWER_" not in blob, (
            "D-10f FAILED: the harness ran with a foreign stats engine "
            "bound -- " + blob[-300:])
        assert not os.path.exists(_capsule_path(out_fs, 0, False))
        print("  D-10f PASS  a foreign phase-B stats engine (p=0.0 for "
              "B2A/B3A) bound under the real name is refused BY NAME "
              "before any harness call, leaving no carrier -- the ninth "
              "executable is inside the join")

        # ---- D-10g (codex 0149Z finding 1): the trust root is the
        # COMMITTED pre. Reproduces codex's counterexample as a real
        # child: a local commit whose manifest pins an in-tree runner
        # carrying an IMPORT-TIME marker; the LIVE pre file is replaced
        # by a pointer at that manifest while the command names the clean
        # committed pre. The gate must refuse the live-vs-committed
        # divergence BEFORE any bootstrap: marker absent, no harness, no
        # carrier. Then the pointer is restored and the positive re-passes.
        rn_live = os.path.join(repo, RUNNER_REL.replace("/", os.sep))
        with open(rn_live, "rb") as f:
            keep_runner = f.read()
        g_marker = os.path.join(tmp, "anchor_marker_runner.txt")
        g_side = ("import os as _mo\n_mo.makedirs(%r, exist_ok=True)\n"
                  "open(%r, 'w').write('runner imported')\n"
                  % (os.path.dirname(g_marker), g_marker))
        with open(rn_live, "wb") as f:
            f.write(g_side.encode("utf-8") + keep_runner)
        g("add", "-A")
        g("commit", "-qm", "marker runner (D-10g)")
        c_rmark = g("rev-parse", "HEAD").stdout.decode().strip()
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(manifest_for({RUNNER_REL: c_rmark}), sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "manifest pinning the marker runner (D-10g)")
        c_manmark = g("rev-parse", "HEAD").stdout.decode().strip()
        out_g = os.path.join(repo, "tier_s_trustroot")
        TSR.fire_pre(repo, c2_ok, GRIDS_REL, GEOMETRY_REL, IMPL_REL, out_g,
                     blob_reader=_blob_reader(repo),
                     argv=["selftest-trustroot"])
        g("add", "-A")
        g("commit", "-qm", "clean committed pre (D-10g)")
        pc_g = g("rev-parse", "HEAD").stdout.decode().strip()
        p_g = os.path.join(out_g, PRE_NAME)
        with open(p_g, "rb") as f:
            committed_pre_bytes = f.read()
        with open(p_g, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"manifest_commit": c_manmark}) + "\n")
        # the marker runner IS on disk and IS pinned by the pointed-at
        # manifest: the old gate imported it here
        rc, blob = _cli([drv_path, "phase1", repo, out_g, "1", pc_g])
        assert rc != 0 and "differs byte-for-byte from the one committed" \
            in blob and "the trust root is the COMMITTED carrier" in blob, \
            blob[-500:]
        assert not os.path.exists(g_marker), (
            "D-10g FAILED: the marker runner was IMPORTED before the "
            "committed-pre check -- the live pointer selected the pins")
        assert HARNESS_MARK not in blob and "POWER_" not in blob
        assert not os.path.exists(_capsule_path(out_g, 0, False))
        assert "not the source identity" not in blob, (
            "D-10g: refused at the pin, not at the trust root -- ordering "
            "not proven: " + blob[-300:])
        # restore the pointer and the pristine runner; positive re-passes
        with open(p_g, "wb") as f:
            f.write(committed_pre_bytes)
        with open(rn_live, "wb") as f:
            f.write(keep_runner)
        g("add", "-A")
        g("commit", "-qm", "restore runner (D-10g)")
        got_pre = _enter(repo, out_g, pc_g)
        assert got_pre["manifest_commit"] == c2_ok, got_pre["manifest_commit"]
        rc, blob = _cli([drv_path, "rank", repo, outdir, c3])
        assert rc == 0 and "top8 " in blob, blob[-300:]
        print("  D-10g PASS  a LIVE pre pointer at a local manifest pinning "
              "an import-marker runner (the runner on disk, pinned) is "
              "refused for live-vs-committed divergence BEFORE any "
              "bootstrap -- marker never written, no harness, no carrier; "
              "the restored committed pre re-enters and the positive "
              "re-passes")

        # ---- D-10h (codex 0149Z finding 1, second half): the runner's
        # registration is COMPARED, not tautologically. An admitted
        # mutant runner registering a different driver path must be
        # refused by the bound-registration check (the previous guard
        # compared a constant with itself and could never refuse).
        rn_src = keep_runner.decode("utf-8")
        rn_old = 'DRIVER_REL = "monitoring/src/w2_tier_s_driver_cayley.py"'
        assert rn_src.count(rn_old) == 1, rn_src.count(rn_old)
        with open(rn_live, "wb") as f:
            f.write(rn_src.replace(
                rn_old, 'DRIVER_REL = "monitoring/src/NOT_THE_DRIVER.py"',
                1).encode("utf-8"))
        g("add", "-A")
        g("commit", "-qm", "runner registering another driver (D-10h)")
        c_rreg = g("rev-parse", "HEAD").stdout.decode().strip()
        wf("docs/f2g_window2_execution/execution_manifest.json",
           json.dumps(manifest_for({RUNNER_REL: c_rreg}), sort_keys=True))
        g("add", "-A")
        g("commit", "-qm", "manifest admitting the mis-registering runner")
        c_manreg = g("rev-parse", "HEAD").stdout.decode().strip()
        out_h = os.path.join(repo, "tier_s_misregistered")
        rc, blob = _cli([drv_path, "fire", repo, out_h, c_manreg])
        assert rc != 0 and "registers different driver/runner paths" \
            in blob, blob[-400:]
        assert not os.path.exists(out_h)
        with open(rn_live, "wb") as f:
            f.write(keep_runner)
        g("add", "-A")
        g("commit", "-qm", "restore runner (D-10h)")
        rc, blob = _cli([drv_path, "rank", repo, outdir, c3])
        assert rc == 0 and "top8 " in blob, blob[-300:]
        print("  D-10h PASS  an ADMITTED runner registering a different "
              "driver path is refused by the registration comparison "
              "(fire refuses, no outdir); the restored positive re-passes")

        # ---- D-11b (codex item 2, re-based): the boundary mutations
        # as ADMITTED MUTANTS. Under a source-identity join, an
        # external mutated copy refuses at the join -- correct, but it
        # says nothing about the seam the mutation targets. So each
        # mutant is COMMITTED and PINNED (admitted) in this fixture,
        # fires its own pre, and its worker is required to fail FOR
        # THE MUTATION'S OWN MEASURED REASON, before the harness,
        # leaving no carrier. The join is about identity, not virtue.
        MUTS = [
            ("omit sha",
             "_run_one(repo, outdir, sha, int(idx), loco)",
             "_run_one(repo, outdir, int(idx), loco)",
             "missing 1 required positional argument", True),
            ("misorder sha",
             "_run_one(repo, outdir, sha, int(idx), loco)",
             "_run_one(repo, outdir, int(idx), sha, loco)",
             "RUNNER_TIER_S_PRE_DIGEST_MISMATCH", True),
            ("prevent runner entry",
             "            mod = importlib.import_module(modname)",
             "            raise ImportError('MUTATION: runner entry "
             "prevented')\n            mod = "
             "importlib.import_module(modname)",
             "MUTATION: runner entry prevented", False),
            # the FIRST `if committed != live:` in this file is now the
            # trust-root gate in _enter (codex 0149Z), which precedes
            # _require_committed_pre; breaking it refuses with _enter's
            # measured text (v4 run 03:18Z), before any import
            ("break committed-pre loading (trust-root gate)",
             "if committed != live:",
             "if True:",
             "the live pre-invocation differs byte-for-byte", True),
        ]
        drv_src = pristine.decode("utf-8")

        def _commit_driver(body):
            with open(drv_path, "wb") as f:
                f.write(body)
            g("add", "-A")
            g("commit", "-qm", "driver bytes")
            return g("rev-parse", "HEAD").stdout.decode().strip()

        for k, (mname, mold, mnew, needle, fires) in enumerate(MUTS):
            assert drv_src.count(mold) >= 1, f"{mname}: anchor"
            cD = _commit_driver(
                drv_src.replace(mold, mnew, 1).encode("utf-8"))
            wf("docs/f2g_window2_execution/execution_manifest.json",
               json.dumps(manifest_for({DRIVER_REL: cD}), sort_keys=True))
            g("add", "-A")
            g("commit", "-qm", "manifest admits the mutant")
            cM = g("rev-parse", "HEAD").stdout.decode().strip()
            outM = os.path.join(repo, f"tier_s_mutant_{k}")
            if not fires:
                rc, blob = _cli([drv_path, "fire", repo, outM, cM])
                assert rc != 0 and needle in blob, (mname, blob[-300:])
                assert not os.path.exists(outM), mname
            else:
                rc, blob = _cli([drv_path, "fire", repo, outM, cM])
                assert rc == 0, (
                    f"D-11b {mname}: the ADMITTED mutant did not fire "
                    f"-- {blob[-400:]}")
                g("add", "-A")
                g("commit", "-qm", "mutant pre")
                pcM = g("rev-parse", "HEAD").stdout.decode().strip()
                rc, blob = _cli([drv_path, "--worker", repo, outM, "0",
                                 pcM])
                assert rc != 0, (
                    f"D-11b {mname}: the mutated worker SUCCEEDED")
                assert needle in blob, (
                    f"D-11b {mname}: failed for the wrong reason -- "
                    f"expected {needle!r}, got {blob.strip()[-300:]!r}")
                assert HARNESS_MARK not in blob and \
                    "POWER_" not in blob, (
                        f"D-11b {mname}: reached the harness anyway, "
                        "so the positive's marker does not "
                        "discriminate")
                assert not os.path.exists(
                    _capsule_path(outM, 0, False)), (
                        f"D-11b {mname}: left a carrier behind")
            # restore the pristine driver (same blob as c1's pin) and
            # re-pass the untouched positive through the real CLI
            _commit_driver(pristine)
            rc, blob = _cli([drv_path, "rank", repo, outdir, c3])
            assert rc == 0 and "top8 " in blob, (mname, blob[-300:])
        print(f"  D-11b PASS  all {len(MUTS)} ADMITTED mutants break "
              "the boundary, each for its OWN measured reason (missing "
              "argument / pre-digest mismatch / runner entry / "
              "committed-pre bytes), none reaches the harness, none "
              "leaves a carrier, and the pristine positive re-passes "
              "through the real CLI after every one")

        # ---- D-12 (codex item 3 + 2112Z finding 3): EVERY CLI
        # wrapper, via main(argv), and every create-once publisher's
        # exit status describing the POSTCONDITION even when the
        # output stream is broken
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

        class _BrokenStream:
            def write(self, s):
                raise OSError("MUTATION: output stream broken")

            def flush(self):
                raise OSError("MUTATION: flush on a broken stream")

        def with_broken_stdout(fn):
            real = sys.stdout
            sys.stdout = _BrokenStream()
            try:
                return fn()
            finally:
                sys.stdout = real

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

        # -- the helper itself cannot rethrow -----------------------
        real_out, real_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _BrokenStream()
        try:
            rc_h = _report_after_publish("KAT", lambda: print("x"))
        finally:
            sys.stdout, sys.stderr = real_out, real_err
        assert rc_h == 0, rc_h
        # -- fire with a broken stdout: published, exit 0, then
        # create-once refuses the retry the old exit status invited
        out_fb = os.path.join(repo, "tier_s_fire_broken")
        rc = with_broken_stdout(lambda: cli("fire", repo, out_fb, c2_ok))
        assert rc == 0 and sorted(os.listdir(out_fb)) == [PRE_NAME], rc
        assert "already exists" in cli_refuses(
            "fire create-once", "fire", repo, out_fb, c2_ok)

        # -- success surface, on the stub campaign ----------------
        # rank: exit 0 and the deterministic top-8
        assert cli("rank", repo, outdir, c3) == 0, "D-12 rank"
        top = TSR.rank_stage1_b1b(outdir, sha, reader)
        # phase 2's LOCO carriers, published through the runner with
        # the stub, so aggregate's success path is reachable here
        for i in top:
            TSR.run_smoke_point(repo, outdir, i, sha, reader,
                                with_loco=True,
                                smoke_fn=stub)
        # phase2 through main over the validated pre-existing LOCO
        # carriers: the wrapper transition, exit 0, nothing spawned
        assert cli("phase2", repo, outdir, 1, ",".join(
            str(i) for i in top), c3) == 0, "D-12 phase2"
        before_keys = set(os.listdir(outdir))
        # aggregate publishes THREE create-once artifacts; its
        # reporting tail runs on a BROKEN stdout and the exit status
        # must still describe the postcondition
        rc = with_broken_stdout(lambda: cli("aggregate", repo, outdir,
                                            c3))
        assert rc == 0, f"D-12 aggregate exit {rc} with broken stdout"
        new_keys = set(os.listdir(outdir)) - before_keys
        assert new_keys == {"tier_s_aggregate_envelope.json",
                            "tier_s_results.json",
                            "tier_s_completion.json",
                            "tier_s_smoke.json"}, sorted(new_keys)
        for nm in sorted(new_keys):
            with open(os.path.join(outdir, nm), encoding="utf-8") as f:
                json.load(f)          # reopen validation
        assert "RUNNER_PUBLISH_EXISTS" in cli_refuses(
            "aggregate create-once", "aggregate", repo, outdir, c3)
        g("add", "-A")
        g("commit", "-qm", "d12 results")
        c_res = g("rev-parse", "HEAD").stdout.decode().strip()
        rel_dir = os.path.relpath(outdir, repo).replace(os.sep, "/")
        rc = with_broken_stdout(lambda: cli("finalize", repo, outdir, c3,
                                            c_res, rel_dir))
        assert rc == 0, f"D-12 finalize exit {rc} with broken stdout"
        with open(os.path.join(outdir, "tier_s_smoke_final.json"),
                  encoding="utf-8") as f:
            json.load(f)
        assert "RUNNER_PUBLISH_EXISTS" in cli_refuses(
            "finalize create-once", "finalize", repo, outdir, c3, c_res,
            rel_dir)
        g("add", "-A")
        g("commit", "-qm", "d12 final smoke")
        c_sm = g("rev-parse", "HEAD").stdout.decode().strip()
        rc = with_broken_stdout(lambda: cli("select", repo, outdir, c_sm,
                                            c1, rel_dir, c3))
        assert rc == 0, f"D-12 select exit {rc} with broken stdout"
        with open(os.path.join(outdir, "selector.json"),
                  encoding="utf-8") as f:
            json.load(f)
        assert "RUNNER_PUBLISH_EXISTS" in cli_refuses(
            "select create-once", "select", repo, outdir, c_sm, c1,
            rel_dir, c3)
        g("add", "-A")
        g("commit", "-qm", "d12 selector")
        c_sel = g("rev-parse", "HEAD").stdout.decode().strip()
        # verify-select drives PRODUCTION selector admission,
        # which resolves the geometry capsule for real. A
        # fixture capsule cannot satisfy that, so its SUCCESS
        # path lives in D-11 over the real bound geometry. What is
        # asserted here is the honest half: it must REFUSE the
        # fixture chain, and for the geometry reason -- which
        # is itself a control, because a verify-select that
        # accepted a fixture chain would be worthless.
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
        print("  D-12b PASS  phase2, rank, aggregate, finalize and "
              "select each SUCCEED through main(argv) with exit 0 -- "
              "fire, aggregate, finalize and select with their "
              "reporting tail on a BROKEN stdout -- publishing exactly "
              "their declared artifact keyset, every artifact "
              "reopening, and every create-once publisher REFUSING "
              "the retry; the helper itself returns 0 with both "
              "streams broken. verify-select REFUSES the fixture "
              "chain on geometry, as it must; its success is D-11's")

        # ---- D-13 (codex 2303Z finding 2): aggregate publishes NOTHING
        # until every artifact is built and validated in memory, and a
        # partial publication is RECOVERABLE from the create-once
        # envelope, byte-exactly, never overwriting a divergent member.
        AGG = {"tier_s_aggregate_envelope.json", "tier_s_results.json",
               "tier_s_completion.json", "tier_s_smoke.json"}

        def _agg_files(o):
            return {n for n in os.listdir(o) if n in AGG}

        def _fresh_campaign(tag):
            o = os.path.join(repo, f"tier_s_agg_{tag}")
            pz, pts_z = TSR.fire_pre(
                repo, c2_ok, GRIDS_REL, GEOMETRY_REL, IMPL_REL, o,
                blob_reader=_blob_reader(repo), argv=["selftest-agg-" + tag])
            g("add", "-A")
            g("commit", "-qm", f"agg {tag} pre")
            cz = g("rev-parse", "HEAD").stdout.decode().strip()
            shz = pz["invocation_sha256"]

            def stub_z(fam, point, folds):
                rec = {"family": fam, "point": point,
                       "quality": dict(pz["quality"]),
                       "geometry_capsule_digest":
                           pz["geometry"]["capsule_digest"],
                       "seed_authority_sha256": pz["seed_authority_sha256"],
                       "certifiable": False, "loco_folds": None,
                       "replicates": [{"p_values": {
                           "B1B": 0.001, "B2A": 0.5, "B2B": 0.5,
                           "B3A": 0.5}} for _ in range(pz["quality"]["R"])]}
                if folds:
                    rec["loco_folds"] = [{"S0": 0.001, "S1": 0.001}
                                         for _ in range(pz["quality"]["R"])]
                return rec
            for i in range(len(pts_z)):
                TSR.run_smoke_point(repo, o, i, shz, reader, smoke_fn=stub_z)
            for i in TSR.rank_stage1_b1b(o, shz, reader):
                TSR.run_smoke_point(repo, o, i, shz, reader, with_loco=True,
                                    smoke_fn=stub_z)
            return o, cz

        # (a) a bad VALUE in one carrier refuses typed with ZERO
        # aggregate files -- string, NaN, out-of-range, bool, negative
        o_bad, c_bad = _fresh_campaign("badval")
        cp0 = os.path.join(o_bad, "smoke_point_000.json")
        with open(cp0, encoding="utf-8") as f:
            good_txt = f.read()
        for vlabel, bad in (("string", '"0.5"'), ("NaN", "NaN"),
                            ("out-of-range", "1.5"), ("bool", "true"),
                            ("negative", "-0.001")):
            capd = json.loads(good_txt)
            capd["record"]["replicates"][0]["p_values"]["B2A"] = "__BAD__"
            txt = json.dumps(capd, indent=1, sort_keys=True).replace(
                '"__BAD__"', bad)
            with open(cp0, "w", encoding="utf-8", newline="\n") as f:
                f.write(txt)
            msg = cli_refuses(f"aggregate {vlabel} p-value", "aggregate",
                              repo, o_bad, c_bad)
            assert "RUNNER_TIER_S_UNADMITTED" in msg and "p-value" in msg, \
                (vlabel, msg[:200])
            assert _agg_files(o_bad) == set(), (vlabel, _agg_files(o_bad))
        with open(cp0, "w", encoding="utf-8", newline="\n") as f:
            f.write(good_txt)
        assert cli("aggregate", repo, o_bad, c_bad) == 0
        assert _agg_files(o_bad) == AGG, _agg_files(o_bad)

        # (b) an injected failure on member 2 and on member 3: the
        # envelope is durable, the retry completes the missing members
        # with the envelope's exact bytes, and a further call refuses
        def _failing_on(member, real_pub):
            def failing(path, body):
                if os.path.basename(path) == member:
                    raise OSError(f"MUTATION: publication of {member} "
                                  "failed")
                return real_pub(path, body)
            return failing
        for member in ("tier_s_completion.json", "tier_s_smoke.json"):
            o_i, c_i = _fresh_campaign("inject_" + member[7:-5])
            real_pub = TSR._publish_once
            TSR._publish_once = _failing_on(member, real_pub)
            try:
                try:
                    cli("aggregate", repo, o_i, c_i)
                    raise AssertionError(
                        "D-13 FAILED: the injected publication failure "
                        "did not surface")
                except OSError as e:
                    assert "MUTATION" in str(e), str(e)
            finally:
                TSR._publish_once = real_pub
            present = _agg_files(o_i)
            assert "tier_s_aggregate_envelope.json" in present and \
                member not in present, (member, present)
            with open(os.path.join(o_i, "tier_s_aggregate_envelope.json"),
                      encoding="utf-8") as f:
                env = json.load(f)
            assert cli("aggregate", repo, o_i, c_i) == 0, member
            assert _agg_files(o_i) == AGG, (member, _agg_files(o_i))
            for name in ("tier_s_results.json", "tier_s_completion.json",
                         "tier_s_smoke.json"):
                with open(os.path.join(o_i, name), "r", encoding="utf-8",
                          newline="") as f:
                    assert f.read() == env["members"][name]["body"], name
            assert "RUNNER_PUBLISH_EXISTS" in cli_refuses(
                "aggregate complete", "aggregate", repo, o_i, c_i)

        # (c) a member that DIVERGED between attempts is never overwritten
        o_d, c_d = _fresh_campaign("divergent")
        real_pub = TSR._publish_once
        TSR._publish_once = _failing_on("tier_s_smoke.json", real_pub)
        try:
            try:
                cli("aggregate", repo, o_d, c_d)
            except OSError:
                pass
        finally:
            TSR._publish_once = real_pub
        rp = os.path.join(o_d, "tier_s_results.json")
        with open(rp, "r", encoding="utf-8", newline="") as f:
            r_live = f.read()
        with open(rp, "w", encoding="utf-8", newline="") as f:
            f.write(r_live + " ")
        msg = cli_refuses("aggregate divergent member", "aggregate", repo,
                          o_d, c_d)
        assert "RUNNER_TIER_S_AGGREGATE_DIVERGENT" in msg, msg[:200]
        with open(rp, "r", encoding="utf-8", newline="") as f:
            assert f.read() == r_live + " ", "D-13: divergent member rewritten"
        assert not os.path.exists(os.path.join(o_d, "tier_s_smoke.json"))
        # (d) codex 0149Z finding 2: a MALFORMED recovery envelope is a
        # typed, no-write refusal -- never a traceback. On a campaign
        # left partial (smoke missing), the envelope file is mutated in
        # place seven ways; each must refuse RUNNER_TIER_S_UNADMITTED
        # naming the member, publish no missing member, and leave every
        # existing byte untouched; the restored envelope then completes.
        o_m, c_m = _fresh_campaign("malformed")
        real_pub = TSR._publish_once
        TSR._publish_once = _failing_on("tier_s_smoke.json", real_pub)
        try:
            try:
                cli("aggregate", repo, o_m, c_m)
            except OSError:
                pass
        finally:
            TSR._publish_once = real_pub
        env_p = os.path.join(o_m, "tier_s_aggregate_envelope.json")
        with open(env_p, "r", encoding="utf-8", newline="") as f:
            env_txt = f.read()
        env_ok = json.loads(env_txt)
        before_bytes = {}
        for nm in ("tier_s_results.json", "tier_s_completion.json"):
            with open(os.path.join(o_m, nm), "rb") as f:
                before_bytes[nm] = f.read()
        assert not os.path.exists(os.path.join(o_m, "tier_s_smoke.json"))
        good_sha = env_ok["members"]["tier_s_results.json"]["sha256"]
        good_body = env_ok["members"]["tier_s_results.json"]["body"]

        def _mut_env(fn):
            e = json.loads(env_txt)
            fn(e)
            with open(env_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(e, indent=1, sort_keys=True) + "\n")
        MALFORMED = [
            ("member is {}", lambda e: e["members"].__setitem__(
                "tier_s_smoke.json", {})),
            ("member is not a dict", lambda e: e["members"].__setitem__(
                "tier_s_smoke.json", ["body", "sha256"])),
            ("body is not text", lambda e: e["members"]["tier_s_smoke.json"]
             .__setitem__("body", 123)),
            ("digest is not hex", lambda e: e["members"]["tier_s_smoke.json"]
             .__setitem__("sha256", "zz" * 32)),
            ("digest does not recompute", lambda e:
             e["members"]["tier_s_smoke.json"].__setitem__(
                 "sha256", good_sha if good_sha !=
                 e["members"]["tier_s_smoke.json"]["sha256"]
                 else "0" * 64)),
            ("extra key", lambda e: e["members"]["tier_s_smoke.json"]
             .__setitem__("note", "x")),
            ("body is not JSON", lambda e: e["members"]["tier_s_smoke.json"]
             .update({"body": "not json", "sha256": __import__("hashlib")
                      .sha256(b"not json").hexdigest()})),
        ]
        for mlabel, mut in MALFORMED:
            _mut_env(mut)
            msg = cli_refuses(f"aggregate malformed envelope ({mlabel})",
                              "aggregate", repo, o_m, c_m)
            assert "RUNNER_TIER_S_UNADMITTED" in msg and \
                "envelope member" in msg, (mlabel, msg[:200])
            assert not os.path.exists(os.path.join(o_m, "tier_s_smoke.json")), \
                (mlabel, "wrote a member from a malformed envelope")
            for nm, b0 in before_bytes.items():
                with open(os.path.join(o_m, nm), "rb") as f:
                    assert f.read() == b0, (mlabel, nm, "existing member changed")
        del good_body
        with open(env_p, "w", encoding="utf-8", newline="") as f:
            f.write(env_txt)
        assert cli("aggregate", repo, o_m, c_m) == 0
        assert _agg_files(o_m) == AGG
        print("  D-13 PASS  a string / NaN / out-of-range / bool / negative "
              "p-value refuses typed with ZERO aggregate files; injected "
              "failures on member 2 and member 3 leave a durable envelope "
              "and the retry completes the missing members byte-exactly, "
              "then refuses create-once; a diverged member is refused and "
              f"never overwritten; {len(MALFORMED)} malformed envelopes "
              "(empty member, non-dict, non-text body, non-hex / wrong "
              "digest, extra key, non-JSON body) each refuse TYPED naming "
              "the member with no write and every existing byte untouched, "
              "and the restored envelope completes")

        # ---- D-11 (codex item 1 + 2112Z finding 2): THE FULL CLI
        # over REAL pinned geometry, every command through main(argv)
        # in a REAL child process running the real checkout's own
        # driver:
        #   fire -> commit -> phase1 (79 prepublished fixtures + ONE
        #   real point) -> rank -> phase2 (validated LOCO carriers) ->
        #   aggregate -> commit -> finalize -> commit -> select ->
        #   commit -> verify-select == 0
        # plus the identity join's locks on the real tree. ~11 minutes;
        # that cost is the point.
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
            pdrv = os.path.join(pwt, DRIVER_REL.replace("/", os.sep))
            prel = "docs/f2g_window2_execution/d11_boundary"
            pout = os.path.join(pwt, prel.replace("/", os.sep))

            def pg(*a):
                return subprocess.run(["git", "-C", pwt] + list(a),
                                      capture_output=True, check=True)
            pg("config", "user.email", "d11@t")
            pg("config", "user.name", "d11")

            # (1) fire through the REAL CLI -- exit 0, exactly the pre
            rc, blob = _cli([pdrv, "fire", pwt, pout, "HEAD"])
            assert rc == 0, f"D-11 fire rc={rc}: {blob[-600:]}"
            assert sorted(os.listdir(pout)) == [PRE_NAME], (
                "D-11: fire must leave exactly the pre-invocation, "
                f"found {sorted(os.listdir(pout))}")
            # (2) commit it alone, so the committed-pre gate has an
            # exact artifact to prove every child against
            pg("add", f"{prel}/{PRE_NAME}")
            pg("commit", "-qm", "d11 pre")
            pc = subprocess.run(["git", "-C", pwt, "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True).stdout.strip()
            bpre, bsha = _load_pre_checked(pout)
            preader = _blob_reader(pwt)
            bpts = TSR.derive_points(bpre, preader)
            assert len(bpts) == 80, len(bpts)
            breg = _loco_registry(pwt, bpre)

            def bstub(fam, point, folds):
                rec = {"family": fam, "point": point,
                       "quality": dict(bpre["quality"]),
                       "geometry_capsule_digest":
                           bpre["geometry"]["capsule_digest"],
                       "seed_authority_sha256":
                           bpre["seed_authority_sha256"],
                       "certifiable": False, "loco_folds": None,
                       "replicates": [{"p_values": {
                           "B1B": 0.001, "B2A": 0.5, "B2B": 0.5,
                           "B3A": 0.5}}
                           for _ in range(bpre["quality"]["R"])]}
                if folds:
                    rec["loco_folds"] = [{s: 0.001 for s in breg}
                                         for _ in range(
                                             bpre["quality"]["R"])]
                return rec
            # (3) 79 structurally valid detection fixtures, point 0
            # ABSENT -- each must pass the very validator the phase
            # applies to a skip
            for i in range(1, len(bpts)):
                TSR.run_smoke_point(pwt, pout, i, bsha, preader,
                                    smoke_fn=bstub)
            for i in range(1, len(bpts)):
                _validate_published(pwt, pout, bpre, bpts, i, False)
            # (4) phase1 through main(argv) in a real process: 79
            # validated skips and ONE real point through
            # main -> cmd_phase1 -> _drive -> child --worker ->
            # cmd_worker -> _run_one -> run_smoke_point
            fam0, _pt0 = bpts[0]
            print(f"  D-11 .... phase1 through the real CLI: 79 "
                  f"fixtures to validate + ONE REAL point ({fam0}) "
                  "through a real child process, ~10 min", flush=True)
            t0 = time.time()
            rc, blob = _cli([pdrv, "phase1", pwt, pout, "1", pc],
                            timeout=5400)
            el = time.time() - t0
            assert rc == 0, f"D-11 phase1 rc={rc}: {blob[-800:]}"
            assert "resuming: 79 already published, 1 to go" in blob \
                and "phase1 complete: 1 run, 79 already present" in \
                blob, blob[-600:]
            cap = _validate_published(pwt, pout, bpre, bpts, 0, False)
            assert len(cap["record"]["replicates"]) == \
                bpre["quality"]["R"]
            assert cap["execution_sha256"] == \
                TSR.execution_digest(bpre["execution"])
            # (5) rank through main
            rc, blob = _cli([pdrv, "rank", pwt, pout, pc])
            assert rc == 0, f"D-11 rank rc={rc}: {blob[-400:]}"
            top_line = [ln for ln in blob.splitlines()
                        if ln.startswith("top8 ")]
            assert len(top_line) == 1, blob[-400:]
            top = [int(x) for x in top_line[0][5:].split(",") if x]
            assert top == TSR.rank_stage1_b1b(pout, bsha, preader), top
            # (6) phase2 through main over validated pre-existing LOCO
            # carriers (the wrapper transition; nothing spawned)
            for i in top:
                TSR.run_smoke_point(pwt, pout, i, bsha, preader,
                                    with_loco=True, smoke_fn=bstub)
            rc, blob = _cli([pdrv, "phase2", pwt, pout, "1",
                             ",".join(str(i) for i in top), pc])
            assert rc == 0, f"D-11 phase2 rc={rc}: {blob[-400:]}"
            assert f"resuming: {len(top)} already published, 0 to go" \
                in blob and f"phase2 complete: 0 run, {len(top)} " \
                "already present" in blob, blob[-400:]
            # (7) aggregate -> commit -> finalize -> commit -> select
            # -> commit -> verify-select, all through main, exit 0
            before_p = set(os.listdir(pout))
            rc, blob = _cli([pdrv, "aggregate", pwt, pout, pc])
            assert rc == 0, f"D-11 aggregate rc={rc}: {blob[-400:]}"
            assert set(os.listdir(pout)) - before_p == {
                "tier_s_aggregate_envelope.json",
                "tier_s_results.json", "tier_s_completion.json",
                "tier_s_smoke.json"}
            pg("add", "-A")
            pg("commit", "-qm", "d11 results")
            c_res_p = subprocess.run(
                ["git", "-C", pwt, "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip()
            rc, blob = _cli([pdrv, "finalize", pwt, pout, pc, c_res_p,
                             prel])
            assert rc == 0, f"D-11 finalize rc={rc}: {blob[-400:]}"
            assert os.path.exists(os.path.join(
                pout, "tier_s_smoke_final.json"))
            pg("add", "-A")
            pg("commit", "-qm", "d11 final smoke")
            c_sm_p = subprocess.run(
                ["git", "-C", pwt, "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip()
            rc, blob = _cli([pdrv, "select", pwt, pout, c_sm_p,
                             bpre["effect_grids"]["commit"], prel, pc])
            assert rc == 0, f"D-11 select rc={rc}: {blob[-400:]}"
            assert os.path.exists(os.path.join(pout, "selector.json"))
            pg("add", "-A")
            pg("commit", "-qm", "d11 selector")
            c_sel_p = subprocess.run(
                ["git", "-C", pwt, "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip()
            rc, blob = _cli([pdrv, "verify-select", pwt, pout, c_sel_p,
                             bpre["manifest_commit"], prel, pc])
            assert rc == 0 and "selector ADMITTED" in blob, (
                f"D-11 verify-select rc={rc}: {blob[-800:]}")
            # ---- D-11c (2112Z finding 1 on the REAL tree): an
            # external driver copy -- pristine bytes, foreign path --
            # and a shadowed runner refuse before any outdir exists;
            # the untouched positive re-passes
            with open(pdrv, "rb") as f:
                p_pristine = f.read()
            psrc = os.path.join(pwt, "monitoring", "src")
            for tag, body in (("pristine", p_pristine),
                              ("byte-modified",
                               p_pristine + b"\n# externally modified\n")):
                ext = os.path.join(pbt, f"external_{tag}.py")
                with open(ext, "wb") as f:
                    f.write(body)
                out_x = os.path.join(pwt, "docs", "f2g_window2_execution",
                                     f"d11_external_{tag}")
                rc, blob = _cli([ext, "fire", pwt, out_x, "HEAD"],
                                env=dict(os.environ, PYTHONPATH=psrc))
                assert rc != 0 and "the driver executing in this " \
                    "process was loaded from" in blob, (tag, blob[-400:])
                assert not os.path.exists(out_x), tag
            with open(os.path.join(psrc, "w2_tier_s_runner_cayley.py"),
                      "rb") as f:
                sh_body = f.read()
            sh = os.path.join(shadow_dir, "w2_tier_s_runner_cayley.py")
            with open(sh, "wb") as f:
                f.write(sh_body)
            out_s = os.path.join(pwt, "docs", "f2g_window2_execution",
                                 "d11_shadow_runner")
            rc, blob = _cli([launcher, "w2_tier_s_runner_cayley", psrc,
                             pdrv, "fire", pwt, out_s, "HEAD"])
            os.remove(sh)
            assert rc != 0 and "the runner executing in this process " \
                "was loaded from" in blob, blob[-400:]
            assert not os.path.exists(out_s)
            out_ok = os.path.join(pwt, "docs", "f2g_window2_execution",
                                  "d11_positive_after_locks")
            rc, blob = _cli([pdrv, "fire", pwt, out_ok, "HEAD"])
            assert rc == 0 and sorted(os.listdir(out_ok)) == [PRE_NAME], \
                blob[-400:]
            print("  D-11c PASS  on the REAL tree: a pristine external "
                  "driver copy, a modified one, and a shadowed runner "
                  "each refuse BY NAME with no outdir created; the "
                  "untouched positive re-passes")
            print(f"  D-11 PASS  the FULL operator CLI works for real "
                  "over real pinned geometry, every command through "
                  "main(argv) in a real child process: fire -> phase1 "
                  "(79 validated fixtures + ONE real point, child exit "
                  f"0, {len(cap['record']['replicates'])} replicates "
                  f"reopened, {el/60:.1f} min) -> rank -> phase2 -> "
                  "aggregate -> finalize -> select -> verify-select "
                  "ADMITTED (exit 0). Composition class closed: no "
                  "wrapper handoff is untested")
        finally:
            subprocess.run(["git", "-C", real_repo, "worktree",
                            "remove", "--force", pwt],
                           capture_output=True)
            subprocess.run(["git", "-C", real_repo, "worktree",
                            "prune"], capture_output=True)
            shutil.rmtree(pbt, ignore_errors=True)

        print("w2 tier-s driver selftest: ALL PASS "
              "(driver behaviour only; stub smoke; nothing fired "
              "outside temp trees). D-11 drives the ENTIRE operator "
              "CLI through real child processes over real pinned "
              "geometry with one real point; D-10b/D-10c/D-11b/D-11c "
              "prove the identity join and the boundary are "
              "falsifiable by measured forgeries, shadows and admitted "
              "mutants; D-12 proves every create-once publisher exits "
              "on its postcondition.")
        return 0
    finally:
        pass   # the launcher owns and removes the fixture tree


if __name__ == "__main__":
    sys.exit(main(sys.argv))
