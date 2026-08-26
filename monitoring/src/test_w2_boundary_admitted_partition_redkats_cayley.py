#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOUNDARY / ADMITTED-PARTITION RED-KATs (cayley) -- successor v4
part 7, the BOUNDARY side of codex 0527Z finding 5.

THE DIVISION OF LABOUR
----------------------
grassmann landed the capture side: one closed archive partitioning
EVERY authority key exactly into `ADMITTED` | `REFUSED`, with
`admitted_keys()` as the only door. Their headline KAT already proves
that reclassifying a key as REFUSED preserves its evidence and leaves
the admitted census SHORT.

This module locks the **consumer** half, which is mine: the boundary
verifier must take its key set from the archive's ADMITTED partition,
and an expected key that is REFUSED must make the bind **refuse** --
never be silently tolerated, and never be quietly dropped from the
requirement.

WHY A REFUSED KEY CANNOT JUST BE SKIPPED
----------------------------------------
"Honest" means faithfully recorded, not automatically admissible
(codex). A REFUSED key is real evidence about a real exchange, and it
belongs in the archive -- but the scientific census still expects that
key. There are exactly three lawful resolutions, and silence is not
among them:

  1. the key is re-admitted (e.g. the Kp offline replay);
  2. the key resolves to a registered `ADMITTED_ABSENCE` -- the
     provider genuinely published nothing, which SATISFIES the key
     while carrying no value;
  3. the key is removed from the authority by a REGISTERED amendment,
     decided BEFORE seeing which days failed.

Dropping it after the fact is option 4, and option 4 makes the
authority data-dependent -- the failure the whole 2,056-key census
design exists to prevent.

STATUS: red-first where the consumer seam is not yet wired.
Opens no window-2 value; no network; admits nothing.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


class PartitionRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


def _selftest():
    import inspect
    import w2_acquisition_capture_grassmann as CAP
    import w2_accrual_instrument_cayley as AI

    # ---- BP-0: the archive's door exists and is closed ------------
    assert hasattr(CAP, "admitted_keys"), \
        "BP-0: the archive exposes no admitted_keys() door"
    try:
        CAP.admitted_keys({"schema": "not-the-archive"})
        raise PartitionRefusal(
            "BP-0: a non-archive object must not yield keys")
    except PartitionRefusal:
        raise
    except Exception:
        pass
    print("  BP-0 PASS  admitted_keys() is the only door and is "
          "closed against non-archives")

    # ---- BP-1: the BOUNDARY verifier must consume the partition ---
    # The boundary's key set may not come from the authority alone --
    # the authority says what is EXPECTED, the archive says what was
    # ADMITTED, and the bind needs both.
    sig = set(inspect.signature(AI.verify_staged_boundary)
              .parameters)
    if not ({"capture_archive", "archive", "admitted_keys"} & sig):
        raise PartitionRefusal(
            "BP-1 BOUNDARY_IGNORES_PARTITION: verify_staged_boundary "
            f"takes {sorted(sig)} -- none of which is the capture-run "
            "archive. The boundary currently derives its key set from "
            "the authority alone, so a key that was REFUSED at "
            "capture is indistinguishable at bind time from one that "
            "was never attempted. grassmann's archive is the "
            "registered partition; the boundary must consume its "
            "ADMITTED half and refuse on any expected key that sits "
            "in the REFUSED half without a registered resolution "
            "(re-admission, ADMITTED_ABSENCE, or a pre-declared "
            "authority amendment). codex 0527Z finding 5, boundary "
            "side.")
    # HONESTY BOUND on what BP-1 establishes (cayley, self-found
    # 2026-08-26): this is an `inspect.signature` check, so it proves
    # the boundary ACCEPTS the archive -- NOT that it consumes it. A
    # boundary that took `capture_archive` and ignored it would pass
    # this line. That is the same name-not-property proxy that let me
    # call the predecessor bridge bypass-proof when it was not, and I
    # am not going to let it read as behavioural coverage in my own
    # lock. Necessary, not sufficient; the behavioural half is BP-2.
    print("  BP-1 PASS  (STRUCTURAL ONLY) the boundary ACCEPTS the "
          "archive partition -- signature-level, not behaviour")

    # ---- BP-2: a REFUSED expected key must make the bind refuse ---
    # OPEN. Previously this printed "PENDING ... testable once BP-1 is
    # wired" and the file then declared "ALL PASS" anyway. Both halves
    # of that were wrong: BP-1 is wired now, so the stated precondition
    # has expired, and a file may not claim ALL PASS while one of its
    # own named requirements is untested. That is exactly the shape
    # this program keeps paying for -- a summary line asserting more
    # than the checks underneath it establish.
    #
    # The property (an expected key sitting in the archive's REFUSED
    # half must refuse the bind, absent a registered resolution) needs
    # a populated boundary invocation, which currently cannot run here:
    # the boundary requires a disposition capsule and
    # verify_lineage_registry is un-satisfiable over a fixture BY
    # DESIGN (fails closed without the registered archive + source
    # bodies -- probed 2026-08-25T23:57Z). So BP-2 is blocked on the
    # same seam routed to codex, and it is reported as OPEN rather
    # than dressed up as passing.
    open_reqs = ["BP-2 refused-key-refuses-the-bind"]
    print("  BP-2 OPEN  refused-key-refuses-the-bind is NOT tested "
          "here -- blocked on the capsule/fixture seam, not passing")
    print(f"w2 boundary admitted-partition red-KATs: 2 structural "
          f"PASS, {len(open_reqs)} OPEN (NOT 'all pass') -- open: "
          f"{open_reqs}")


if __name__ == "__main__":
    try:
        _selftest()
    except PartitionRefusal as e:
        print(f"RED (expected until the successor lands): {e}")
        raise SystemExit(1)
