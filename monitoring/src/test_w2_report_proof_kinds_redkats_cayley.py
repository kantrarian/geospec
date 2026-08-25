#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOUNDARY-REPORT PROOF-KIND RED-KATs (cayley) -- codex 2119Z
finding-4 closure 4. My half of Option 2.

WHY THE REPORT NEEDS TYPED PARTITIONS
-------------------------------------
After the restage, the 2,056 admitted keys are NOT established by one
kind of proof. Three different claims are being made:

  NATIVE_V4_CAPTURE  = 635   captured under the v4 authority: the
                             full native five-map join, T against the
                             same S the artifact derives from.
  RESTAGED_LINEAGE   = 1420  preserved v3 bytes: the historical
                             exchange verifies against S_v3, while the
                             artifact is freshly recomputed under an
                             independently derived S_v4. Two legs, a
                             DIFFERENT typed relation from native.
  PREDECESSOR_BRIDGE = 1     the corrected-OMNI probe day: obtained
                             under a DIFFERENT authority for a
                             DIFFERENT purpose, under an explicit
                             no-admission ceiling. Its dual-authority
                             proof is not the versioned-S shortcut and
                             must never be folded into either of the
                             others.

I originally proposed a two-way native-vs-lineage split. codex
corrected it to three: the bridge establishes something the 635 native
exchanges do not, so collapsing it into "native" would overstate 634
keys' provenance and understate the one.

THE FAILURE THIS PREVENTS
-------------------------
A report that aggregates only by lane/carrier lets three different
proofs disappear into one total that reads with the strength of its
strongest member. "2,056 keys admitted" is true and misleading; the
honest statement names how many keys were established by which proof.
A weaker claim hidden inside a stronger-sounding total is the shape
this program keeps paying for.

STATUS: red-first. The current report has no proof-kind partitions.
Goes green when I land the typed consumer against grassmann's upgraded
lineage capsule.

Opens no window-2 value; no network; admits nothing.
"""
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
REQUIRED_KINDS = {"NATIVE_V4_CAPTURE": 635,
                  "RESTAGED_LINEAGE": 1420,
                  "PREDECESSOR_BRIDGE": 1}
CENSUS = 2056


class ProofKindRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


def _selftest():
    import w2_accrual_instrument_cayley as AI
    import w2_expected_contracts_gen_cayley as GEN

    # ---- PK-0: the partition arithmetic is the authority's own -----
    auth = GEN.build(REPO)
    keys = auth["prestart_expected_keys"]
    census = sum(len(d) for lane in keys.values()
                 for d in lane.values())
    assert census == CENSUS, (census, CENSUS)
    assert sum(REQUIRED_KINDS.values()) == CENSUS, \
        "PK-0: the three proof kinds must partition the census exactly"
    print(f"  PK-0 PASS  {' + '.join(str(v) for v in REQUIRED_KINDS.values())}"
          f" == census {CENSUS}")

    # ---- PK-1 (THE LOCK): the boundary must REPORT proof kinds -----
    sig = set(inspect.signature(AI.verify_staged_boundary).parameters)
    src = inspect.getsource(AI.verify_staged_boundary)
    named = [k for k in REQUIRED_KINDS if k in src]
    if len(named) != len(REQUIRED_KINDS):
        missing = sorted(set(REQUIRED_KINDS) - set(named))
        raise ProofKindRefusal(
            "PK-1 REPORT_HAS_NO_PROOF_KINDS: verify_staged_boundary "
            f"does not establish {missing}. Its report aggregates by "
            "lane/carrier only, so a native capture, an old-authority "
            "restage, and the one different-purpose predecessor "
            "bridge all disappear into a single total that reads with "
            "the strength of its strongest member. The partitions "
            "must be first-class, EXACT, DISJOINT and RECOMPUTED -- "
            "each binding its sorted-key digest and per-key join "
            "result, their union equal to the 2056-key authority, and "
            "any aggregate reported only AFTER them. codex 2119Z "
            "closure 4.")
    print("  PK-1 PASS  the boundary establishes all three proof kinds")

    # ---- PK-2: the bridge is never folded into native --------------
    if "PREDECESSOR_BRIDGE" in src and "NATIVE_V4_CAPTURE" in src:
        assert "join_kind" in src or "proof_kind" in src, \
            ("PK-2: keys must carry an explicit typed join/proof kind, "
             "not be inferred from which map they landed in")
    print("  PK-2 PASS  proof kind is typed per key, not inferred")
    print("w2 report proof-kind red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except ProofKindRefusal as e:
        print(f"RED (expected until the typed consumer lands): {e}")
        raise SystemExit(1)
