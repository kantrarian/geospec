#!/usr/bin/env python3
"""Red KAT: Tier-S must never treat a pin in an OPEN slot as admitted.

AUTHORED BY CODEX (0531Z review of `021b0ac3`), installed here verbatim
apart from argument handling. It reproduces the CRITICAL they found:
every pin lookup in the Tier-S chain scanned all slots without checking
`status`, so flipping `power_harness` from BOUND to OPEN -- changing no
pin bytes at all -- let `fire_pre` accept the grids, geometry, harness
and FIRING DRIVER pins from a slot the manifest declares unbound. That
reacquires the exact "firing artifact outside the admitted set" shape
Design A exists to close, while the error text still says BOUND.

Kept as a bar rather than a one-off because the defect was invisible to
a 91/91 pin audit and to nine green bars: every one of them resolved
the same pins through the same unchecked lookup.

Run against a checkout containing the candidate objects:
    python tier_s_open_slot_redkat.py <repo> <candidate-commit>

Exit 0 means the control fired and the OPEN-slot mutation refused.
Exit 1 means the production pre-fire path accepted the invalid manifest.
Nothing beyond the create-once pre-invocation is run.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"


def git_blob(repo, commit, path):
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "blob", f"{commit}:{path}"],
        capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(
            f"cannot reopen {path} at {commit}: "
            f"{p.stderr.decode(errors='replace')}"
        )
    return p.stdout


def main():
    # Pinned-bar convention: runnable with NO arguments, judged by exit
    # code, defaulting to this checkout at HEAD. codex's routed
    # two-argument form is preserved exactly, so their own invocation
    # keeps reproducing against any named commit.
    if len(sys.argv) == 1:
        here = os.path.dirname(os.path.abspath(__file__))
        argv = [sys.argv[0],
                os.path.dirname(os.path.dirname(here)), "HEAD"]
    elif len(sys.argv) == 3:
        argv = list(sys.argv)
    else:
        raise SystemExit(
            "usage: test_w2_tier_s_open_slot_redkats_codex.py "
            "[<repo> <commit>]")
    repo = os.path.abspath(argv[1])
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", f"{argv[2]}^{{commit}}"],
        text=True,
    ).strip()
    sys.path.insert(0, str(Path(repo) / "monitoring" / "src"))
    import w2_tier_s_driver_cayley as driver
    import w2_tier_s_runner_cayley as runner

    manifest = json.loads(git_blob(repo, commit, MANIFEST_REL))
    slot = manifest["slots"]["power_harness"]
    if slot.get("status") != "BOUND" or not slot.get("pins"):
        raise AssertionError("anti-vacuity control needs populated BOUND power_harness")

    def reader_for(obj):
        def read(c, path):
            if c == commit and path == MANIFEST_REL:
                return json.dumps(obj, sort_keys=True).encode("utf-8")
            return git_blob(repo, c, path)
        return read

    with tempfile.TemporaryDirectory(prefix="tier-s-bound-control-") as out:
        pre, points = runner.fire_pre(
            repo,
            commit,
            driver.GRIDS_REL,
            driver.GEOMETRY_REL,
            driver.IMPL_REL,
            out,
            blob_reader=reader_for(manifest),
            argv=["open-slot-redkat", "control"],
            resolver=lambda _repo, _commit: commit,
        )
        assert pre["schema"] == runner.PRE_SCHEMA and len(points) == 80
    print("CONTROL PASS: unchanged BOUND slot fires an 80-point pre only")

    mutated = copy.deepcopy(manifest)
    mutated["slots"]["power_harness"]["status"] = "OPEN"
    with tempfile.TemporaryDirectory(prefix="tier-s-open-mutation-") as out:
        try:
            runner.fire_pre(
                repo,
                commit,
                driver.GRIDS_REL,
                driver.GEOMETRY_REL,
                driver.IMPL_REL,
                out,
                blob_reader=reader_for(mutated),
                argv=["open-slot-redkat", "mutation"],
                resolver=lambda _repo, _commit: commit,
            )
        except runner.RunnerRefusal as exc:
            print(f"PASS: OPEN-slot pins refused: {exc}")
            return 0
    print(
        "DEFECT REPRODUCED: fire_pre accepted grids, geometry, harness, "
        "and firing-driver pins from an OPEN slot"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
