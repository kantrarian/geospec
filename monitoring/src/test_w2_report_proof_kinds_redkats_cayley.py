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

STATUS: red-first at authorship (PK-1 RED: the boundary established
no proof kinds); GREEN from the typed consumer, exercised against the
REAL committed capsule and the REAL authority. PK-1 is deliberately
NOT a source-text scan for the kind names -- that would check a NAME,
not a PROPERTY, which is exactly the weak proxy that let me call the
bridge bypass-proof when it was not.

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

    # ---- PK-1 (THE LOCK): exercised on the REAL capsule + REAL
    # authority. Deliberately NOT a source-text scan for the kind
    # names: that would check a NAME, not a PROPERTY -- the exact
    # weak proxy that let me claim the bridge was bypass-proof.
    import copy
    import json as _json
    cap = _json.load(open(os.path.join(
        REPO, "docs", "f2g_window2_execution",
        "key_disposition_capsule_v4.json"), encoding="utf-8"))
    akeys = {f"{ln}/{ck}/{d}"
             for ln, cs in keys.items()
             for ck, ds in cs.items() for d in ds}
    if not hasattr(AI, "compute_proof_kind_partitions"):
        raise ProofKindRefusal(
            "PK-1 REPORT_HAS_NO_PROOF_KINDS: the boundary exposes no "
            "proof-kind partitioning, so a native capture, an "
            "old-authority restage and the one different-purpose "
            "predecessor bridge disappear into a lane/carrier total "
            "that reads with the strength of its strongest member. "
            "codex 2119Z closure 4.")
    parts = AI.compute_proof_kind_partitions(akeys, cap)
    for kind, want in REQUIRED_KINDS.items():
        got = parts.get(kind, {}).get("count")
        if got != want:
            raise ProofKindRefusal(
                f"PK-1: {kind} count {got} != {want}")
        assert len(parts[kind]["keys_sha256"]) == 64, kind
    assert sum(p["count"] for p in parts.values()) == CENSUS
    print(f"  PK-1 PASS  {parts['NATIVE_V4_CAPTURE']['count']} native "
          f"/ {parts['RESTAGED_LINEAGE']['count']} lineage / "
          f"{parts['PREDECESSOR_BRIDGE']['count']} bridge, union "
          f"{CENSUS}, each key-digest bound")

    # ---- PK-2: partitions must be DISJOINT -----------------------
    dbl = copy.deepcopy(cap)
    pk = sorted(dbl["predecessor"])[0]
    dbl["http_capture"] = list(dbl["http_capture"]) + [pk]
    try:
        AI.compute_proof_kind_partitions(akeys, dbl)
        raise ProofKindRefusal(
            "PK-2: a key disposed twice must REFUSE -- otherwise the "
            "bridge could be silently folded into native")
    except AI.InstrumentRefusal as e:
        assert "DISJOINT" in str(e), str(e)[:90]
    print("  PK-2 PASS  double disposition refuses (the bridge cannot "
          "be folded into native)")

    # ---- PK-3: partitions must cover the authority EXACTLY -------
    thin = copy.deepcopy(cap)
    drop = sorted(thin["reuse_or_bridge"])[0]
    del thin["reuse_or_bridge"][drop]
    try:
        AI.compute_proof_kind_partitions(akeys, thin)
        raise ProofKindRefusal(
            "PK-3: an undisposed authority key must REFUSE")
    except AI.InstrumentRefusal as e:
        assert "EXACTLY" in str(e), str(e)[:90]
    print("  PK-3 PASS  an undisposed authority key refuses")

    # ---- PK-4: anti-vacuity -- the kinds are genuinely distinct ---
    moved = copy.deepcopy(cap)
    del moved["predecessor"][pk]
    moved["http_capture"] = list(moved["http_capture"]) + [pk]
    p2 = AI.compute_proof_kind_partitions(akeys, moved)
    assert p2["PREDECESSOR_BRIDGE"]["count"] == 0
    assert p2["NATIVE_V4_CAPTURE"]["count"] == 636
    assert (p2["NATIVE_V4_CAPTURE"]["keys_sha256"]
            != parts["NATIVE_V4_CAPTURE"]["keys_sha256"])
    print("  PK-4 PASS  anti-vacuity: moving the bridge key changes "
          "both the counts AND the key digest (kinds are real, not "
          "labels)")
    print("w2 report proof-kind red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except ProofKindRefusal as e:
        print(f"RED (expected until the typed consumer lands): {e}")
        raise SystemExit(1)
