#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TARGET IDENTITY guard (cayley) -- codex cycle-6 review item 3
reproduction B (MAJOR).

The defect: the seed and geometry generators reopened DATA at a named
commit but executed IMPORTED MODULES from the current process and
worktree. With the target argument held fixed, mutating the live
`w2_power_harness_cayley.GRAPH` changed the seed build from four
families to three, and mutating `w2_selection.CAPS["cascadia"]`
changed the geometry build from 16 stations to 15. A named-target
rebuild was therefore only reproducible while the worktree happened
to agree with the target -- which is not what "rebuilds byte-for-byte
at the named target" claims.

This module makes a named-target build hermetic BY VERIFICATION,
which is codex's second option ("load/verify every consumed
implementation from the target pin rather than importing live
modules"):

  `verify_consumed_implementations` refuses unless every consumed
  module's LIVE source bytes equal that module's bytes AT THE TARGET,
  and unless every registered steering CONSTANT, re-derived from the
  target source by AST (no execution, no import side effects), equals
  the live module's runtime value.

The two checks are complementary and both are needed: the file digest
catches a drifted or dirty worktree, and the AST cross-check catches
an in-process mutation that leaves the file untouched -- which is
exactly how codex demonstrated the defect.

A fixed target therefore either reproduces byte-for-byte or REFUSES;
it can no longer quietly change its output. Read-only.
"""
import ast
import hashlib
import os
import subprocess


class TargetIdentityRefusal(ValueError):
    pass


def _refuse(detail):
    raise TargetIdentityRefusal(f"TARGET_IDENTITY_REFUSED: {detail}")


def _lf_sha(b):
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _blob(repo, commit, rel):
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        _refuse(f"consumed implementation unreadable at "
                f"{str(commit)[:12]}: {rel}")
    return p.stdout


def module_rel(module):
    """Repo-relative path of a live module, as the manifest names it."""
    return "monitoring/src/" + os.path.basename(module.__file__)


def literal_at_target(repo, commit, rel, name):
    """Re-derive a module-level literal assignment from the TARGET
    source by AST. Never executes the target module, so it cannot be
    steered by import side effects, and never consults the live
    module's runtime value."""
    src = _blob(repo, commit, rel).decode("utf-8")
    try:
        tree = ast.parse(src, filename=rel)
    except SyntaxError as e:
        _refuse(f"{rel} at {str(commit)[:12]} does not parse: {e}")
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    _refuse(f"{name} in {rel} is not a literal at the "
                            "target, so it cannot be re-derived "
                            "without executing target code")
    _refuse(f"{name} is not assigned at module level in {rel} at "
            f"{str(commit)[:12]}")


def verify_consumed_implementations(repo, commit, *, modules,
                                    constants=()):
    """`modules`: live module objects whose SOURCE must equal the
    target's. `constants`: (module, name, normalizer) triples whose
    live runtime value must equal the target-source literal.

    Returns the bound identity record for the caller to embed, so the
    artifact states exactly which implementations it was built from.
    """
    bound = {}
    for m in modules:
        rel = module_rel(m)
        with open(m.__file__, "rb") as f:
            live = _lf_sha(f.read())
        target = _lf_sha(_blob(repo, commit, rel))
        if live != target:
            _refuse(
                f"consumed implementation {rel} differs from the named "
                f"target ({live[:12]} != {target[:12]}) -- a "
                "named-target build must execute the target's code, "
                "not the worktree's")
        bound[rel] = target
    checked = {}
    for m, name, norm in constants:
        rel = module_rel(m)
        want = literal_at_target(repo, commit, rel, name)
        got = getattr(m, name)
        if norm is not None:
            want, got = norm(want), norm(got)
        if want != got:
            _refuse(
                f"the live value of {rel}:{name} does not equal the "
                f"target's ({got!r} != {want!r}) -- an in-process "
                "mutation cannot steer a named-target build")
        checked[f"{rel}:{name}"] = got
    return {"consumed_implementations": bound,
            "steering_constants_verified": {
                k: (sorted(v) if isinstance(v, (set, frozenset))
                    else v)
                for k, v in sorted(checked.items())},
            "rule": "every consumed implementation's source must "
                    "equal its bytes AT THE NAMED TARGET, and every "
                    "registered steering constant re-derived from the "
                    "target source by AST must equal the live runtime "
                    "value; otherwise the build REFUSES rather than "
                    "producing target-labelled output built from "
                    "other code"}


def regenerate_in_isolated_worker(repo, commit, module, entry,
                                  *, timeout=1800):
    """PRODUCTION target-labelled generation (codex cycle-6b item 2).

    `verify_consumed_implementations` closed the file and
    module-global surfaces, but not the CALLABLE graph: codex
    replaced only the live in-memory `w2_selection.select` callable,
    left every file and both registered constants untouched, and a
    fixed-target build went 16 -> 15 stations while the emitted
    `target_identity` stayed byte-identical. A hand-maintained list
    of function fingerprints would still leave transitive helpers
    mutable, so the durable answer is process isolation.

    This materializes a clean detached worktree AT THE NAMED TARGET
    and runs `entry` there in a fresh interpreter, so every module,
    helper and callable comes from the target's own bytes. Only
    serialized JSON crosses back. Nothing in the caller's process can
    steer it.
    """
    import json
    import shutil
    import sys
    import tempfile

    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", f"{commit}^{{commit}}"],
        capture_output=True, text=True).stdout.strip()
    if not commit:
        _refuse("unresolvable target for isolated regeneration")
    td = tempfile.mkdtemp(prefix="w2-isolated-target-")
    wt = os.path.join(td, "t")
    try:
        add = subprocess.run(
            ["git", "-C", repo, "worktree", "add", "--detach",
             "--no-checkout", wt, commit], capture_output=True)
        if add.returncode:
            _refuse("could not materialize an isolated target "
                    f"worktree: {add.stderr.decode(errors='replace')[:160]}")
        co = subprocess.run(["git", "-C", wt, "checkout", "--", "."],
                            capture_output=True)
        if co.returncode:
            _refuse("isolated target worktree would not check out")
        src = os.path.join(wt, "monitoring", "src")
        code = (
            "import json,sys\n"
            f"sys.path.insert(0, {src!r})\n"
            f"import {module} as M\n"
            f"print('<<<BEGIN>>>' + json.dumps(M.{entry}("
            f"{wt!r}, commit={commit!r}), sort_keys=True))\n")
        run = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, cwd=src,
                             timeout=timeout)
        out = run.stdout.decode("utf-8", errors="replace")
        if run.returncode != 0 or "<<<BEGIN>>>" not in out:
            _refuse(
                "isolated target regeneration failed: "
                + (run.stderr.decode("utf-8", errors="replace")[-300:]
                   or out[-300:]))
        return json.loads(out.split("<<<BEGIN>>>", 1)[1].strip())
    finally:
        subprocess.run(["git", "-C", repo, "worktree", "remove",
                        "--force", wt], capture_output=True)
        shutil.rmtree(td, ignore_errors=True)


def _selftest():
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    repo = os.path.abspath(os.path.join(_here, "..", ".."))
    import w2_power_harness_cayley as PH
    import w2_selection as WSEL

    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    # (+) at HEAD with a clean worktree, the guard passes
    rec = verify_consumed_implementations(
        repo, head, modules=[PH, WSEL],
        constants=[(PH, "GRAPH", tuple), (WSEL, "CAPS", dict)])
    assert rec["consumed_implementations"], rec
    print(f"  T0 PASS  clean worktree at HEAD: "
          f"{len(rec['consumed_implementations'])} implementations "
          f"and {len(rec['steering_constants_verified'])} steering "
          "constants verified against the target")

    # (-) codex's exact reproduction B: mutate the LIVE constant only
    saved = PH.GRAPH
    try:
        PH.GRAPH = ("B1B", "B2A", "B2B")
        try:
            verify_consumed_implementations(
                repo, head, modules=[PH, WSEL],
                constants=[(PH, "GRAPH", tuple)])
            raise SystemExit(
                "T1 IN_PROCESS_MUTATION_ADMITTED: a live GRAPH "
                "mutation was not caught")
        except TargetIdentityRefusal as e:
            assert "does not equal the target" in str(e), str(e)
    finally:
        PH.GRAPH = saved
    print("  T1 PASS  an in-process GRAPH mutation REFUSES (codex's "
          "exact reproduction B, with the file untouched)")

    saved_caps = dict(WSEL.CAPS)
    try:
        WSEL.CAPS["cascadia"] = 15
        try:
            verify_consumed_implementations(
                repo, head, modules=[WSEL],
                constants=[(WSEL, "CAPS", dict)])
            raise SystemExit(
                "T2 CAPS_MUTATION_ADMITTED")
        except TargetIdentityRefusal as e:
            assert "does not equal the target" in str(e), str(e)
    finally:
        WSEL.CAPS.clear()
        WSEL.CAPS.update(saved_caps)
    print("  T2 PASS  an in-process CAPS mutation REFUSES (16 -> 15 "
          "stations can no longer ride a fixed target)")

    # (-) a DIFFERENT target whose bytes differ from the worktree
    older = subprocess.run(
        ["git", "-C", repo, "log", "-2", "--format=%H", "--",
         module_rel(PH)], capture_output=True,
        text=True).stdout.split()
    if len(older) > 1:
        try:
            verify_consumed_implementations(repo, older[1],
                                            modules=[PH])
            raise SystemExit(
                "T3 WORKTREE_DRIFT_ADMITTED: a target whose harness "
                "bytes differ from the worktree was accepted")
        except TargetIdentityRefusal as e:
            assert "differs from the named target" in str(e), str(e)
        print("  T3 PASS  a named target whose consumed bytes differ "
              "from the worktree REFUSES (the drift case)")
    else:
        raise SystemExit(
            "T3 NO_PRIOR_REVISION: cannot construct the drift control")
    print("w2_target_identity selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
