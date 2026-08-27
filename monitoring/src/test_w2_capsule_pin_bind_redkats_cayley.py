#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAPSULE PIN-BIND + STRUCTURAL-KAT RED-KATs (cayley).

codex 0057Z P0, and my own 0110Z finding.

THE FINDING THIS LOCKS
----------------------
grassmann's `build_fixture_capsule` (the supported fixture path, and
the right answer to my testability ask) can also be called with the
REAL authority and every real key placed in `http_capture`. The result
passes the SAME strict `verify_lineage_registry`:

    {"census": 2056, "REUSE_OR_BRIDGE": 0, "PREDECESSOR": 0,
     "HTTP_CAPTURE": 2056, "bodies_recomputed": 0,
     "lineage_evidence_verified": true}

and my closure-4 partitioner then reports 2056 native / 0 lineage /
0 bridge, against a truth of 635 / 1420 / 1. That erases the 1,420
restaged keys and the one different-purpose bridge key, reporting all
2,056 with the strength of a native v4 capture -- exactly the failure
closure 4 exists to prevent.

WHY THE CAPSULE'S OWN VERIFIER IS NOT THE DEFENCE
-------------------------------------------------
It cannot be. `verify_lineage_registry` can only check INTERNAL
consistency against the authority key set; a capsule that is
internally true and externally false passes it honestly. A capsule's
authority comes from its PROVENANCE -- derived by build() from the
real archive and store, then PINNED. So the boundary must RESOLVE it
from the registered pin and refuse any substitute. This is the same
repair codex required on the predecessor bridge (resolve from a pin,
keep injection behind a fixture-only entry) which I applied to the
transform dispatcher and failed to apply to the capsule.

AND WHY THAT FORCES THE STRUCTURAL KERNEL
-----------------------------------------
Once production hard-binds the pin, a fixture capsule can no longer be
fed to it -- so portable structural testing must live in a separately
named kernel whose result is structurally incapable of satisfying
admission. That is codex's P0, and this finding is the argument for
it, not against it.

Opens no window-2 value; no network; admits nothing.
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
CAPSULE_REL = ("docs/f2g_window2_execution/"
               "key_disposition_capsule_v4.json")
TRUTH = {"NATIVE_V4_CAPTURE": 635, "RESTAGED_LINEAGE": 1420,
         "PREDECESSOR_BRIDGE": 1}


class PinBindRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


def _manifest_pinning(raw_bytes, path=CAPSULE_REL, commit="ab" * 20):
    """A minimal manifest that pins the capsule, plus a reader."""
    import hashlib
    man = {"slots": {"accrual_impl": {"status": "BOUND", "pins": [
        {"path": path, "commit": commit,
         "blob_sha256": hashlib.sha256(raw_bytes).hexdigest()}]}}}

    def reader(c, p):
        if (c, p) != (commit, path):
            raise KeyError(f"unpinned read attempted: {c}:{p}")
        return raw_bytes
    return man, reader


def _selftest():
    import w2_accrual_instrument_cayley as AI
    import w2_disposition_capsule_grassmann as DISP
    import w2_expected_contracts_gen_cayley as GEN

    real_raw = open(os.path.join(REPO, *CAPSULE_REL.split("/")),
                    "rb").read()
    real = json.loads(real_raw.decode("utf-8"))
    man, reader = _manifest_pinning(real_raw)

    # ---- PB-0: anti-vacuity -- the pin path actually resolves -----
    got = AI.bind_registered_capsule(man, reader)
    assert json.dumps(got, sort_keys=True) == \
        json.dumps(real, sort_keys=True), "PB-0: pin did not resolve"
    print(f"  PB-0 PASS  the capsule resolves FROM THE PIN "
          f"({len(real['http_capture'])} http / "
          f"{len(real['reuse_or_bridge'])} reuse / "
          f"{len(real['predecessor'])} pred)")

    # ---- PB-1 (THE LOCK): the forged SHAPE is REFUSED --------------
    # Originally this minted the forgery through grassmann's
    # build_fixture_capsule. Their d4740ea closed that door -- the
    # constructor now refuses the real registered authority -- which
    # is the stronger fix and broke this test, exactly the mirror of
    # what my pin-bind did to their bar.
    #
    # The repair makes the lock BETTER, not merely green again: my
    # layer must refuse the forged SHAPE regardless of where it came
    # from. Depending on a peer's constructor still being able to
    # produce the bad shape would make my defence untested the moment
    # they hardened theirs -- and a hand-written capsule, an older
    # revision of their module, or any other constructor can still
    # present these bytes. Two doors, tested independently: they stop
    # it being MINTED (PB-5), I stop it being SUPPLIED (here).
    all_keys = sorted(
        set(real["http_capture"])
        | set(real["reuse_or_bridge"])
        | set(real["predecessor"]))
    forged = dict(real)
    forged["http_capture"] = list(all_keys)   # every key "native"
    forged["reuse_or_bridge"] = {}            # 1420 restages erased
    forged["predecessor"] = {}                # the bridge erased
    # the hazard is real: it would report a FALSE provenance
    parts = AI.compute_proof_kind_partitions(set(all_keys), forged)
    assert parts["NATIVE_V4_CAPTURE"]["count"] == len(all_keys)
    assert parts["RESTAGED_LINEAGE"]["count"] == 0
    assert parts["PREDECESSOR_BRIDGE"]["count"] == 0
    try:
        AI.bind_registered_capsule(man, reader, supplied=forged)
        raise PinBindRefusal(
            "PB-1 CAPSULE_SUBSTITUTION_ADMITTED: a capsule claiming "
            "every key is a native capture was accepted in place of "
            f"the pinned one -- {len(all_keys)} native / 0 lineage / "
            f"0 bridge against a truth of {TRUTH['NATIVE_V4_CAPTURE']}"
            f" / {TRUTH['RESTAGED_LINEAGE']} / "
            f"{TRUTH['PREDECESSOR_BRIDGE']}")
    except AI.InstrumentRefusal as e:
        assert "NOT_THE_REGISTERED_CAPSULE" in str(e), str(e)[:120]
    print(f"  PB-1 PASS  the forged shape (claims {len(all_keys)} "
          f"native vs truth {TRUTH['NATIVE_V4_CAPTURE']}) is REFUSED "
          "as a substitute, whatever constructed it")

    # ---- PB-5: the MINT door, verified from my side ----------------
    # grassmann's half. Not my defence, but I depend on it, so I check
    # it rather than assume it.
    if not hasattr(DISP, "build_fixture_capsule"):
        print("  PB-5 SKIP  fixture path absent on this revision")
    else:
        auth = GEN.build(REPO)
        with tempfile.TemporaryDirectory() as td:
            try:
                DISP.build_fixture_capsule(
                    auth, all_keys, os.path.join(td, "store"),
                    os.path.join(td, "arch.json"))
                raise PinBindRefusal(
                    "PB-5 FORGERY_STILL_MINTABLE: the fixture "
                    "constructor accepted the REAL registered "
                    "authority")
            except DISP.DispositionRefusal:
                pass
        print("  PB-5 PASS  the fixture constructor REFUSES the real "
              "registered authority (grassmann d4740ea: the forgery "
              "cannot be minted either)")

    # ---- PB-6 (codex 0151Z P0-3): the slot must be BOUND ----------
    # The distinguishing fixture: accrual_impl OPEN while carrying
    # exactly ONE VALID capsule pin. Right path, resolvable commit,
    # blob recomputes -- every property the object check inspects is
    # correct, and only the slot's own declared status is wrong.
    #
    # The old resolver accepted this. It was protected only
    # INCIDENTALLY, because the generator emits accrual_impl in
    # BOUND_SLOTS and OPEN slots happen to carry zero pins. Neither
    # fact is asserted anywhere, and the slot map is data that gets
    # edited by people who never read the resolver.
    open_man = {"slots": {"accrual_impl": dict(
        man["slots"]["accrual_impl"], status="OPEN")}}
    try:
        AI.bind_registered_capsule(open_man, reader)
        raise PinBindRefusal(
            "PB-6 OPEN_SLOT_ADMITTED: a valid capsule pin was accepted "
            "from a slot the manifest itself declares OPEN -- a pin in "
            "an unbound slot may not authorize an admission")
    except AI.InstrumentRefusal as e:
        assert "not BOUND" in str(e), str(e)[:130]
    # and the SAME fixture with status BOUND must still resolve, so
    # PB-6 refuses the STATUS and not the fixture
    bound_man = {"slots": {"accrual_impl": dict(
        man["slots"]["accrual_impl"], status="BOUND")}}
    AI.bind_registered_capsule(bound_man, reader)
    print("  PB-6 PASS  a valid pin in an OPEN slot REFUSES; the same "
          "pin with status BOUND resolves (the status is what is "
          "checked, not the fixture)")

    # ---- PB-2: a tampered PIN refuses (bytes must match) -----------
    bad_man, bad_reader = _manifest_pinning(real_raw)
    bad_man["slots"]["accrual_impl"]["pins"][0]["blob_sha256"] = "9" * 64
    try:
        AI.bind_registered_capsule(bad_man, bad_reader)
        raise PinBindRefusal(
            "PB-2: capsule bytes diverging from the pin must refuse")
    except AI.InstrumentRefusal as e:
        assert "diverge from the manifest pin" in str(e), str(e)[:120]
    print("  PB-2 PASS  capsule bytes diverging from the pin refuse")

    # ---- PB-3: no pin at all fails CLOSED --------------------------
    for empty in ({"slots": {}},
                  {"slots": {"accrual_impl": {"pins": []}}}):
        try:
            AI.bind_registered_capsule(empty, reader)
            raise PinBindRefusal(
                "PB-3: an absent capsule pin must fail CLOSED, never "
                "fall back to a caller-supplied capsule")
        except AI.InstrumentRefusal:
            pass
    print("  PB-3 PASS  an absent pin fails CLOSED (no caller "
          "fallback)")

    # ---- PB-4: the identical pinned capsule is ACCEPTED ------------
    # anti-vacuity for PB-1: the refusal must be about SUBSTITUTION,
    # not about supplying a capsule at all.
    same = json.loads(real_raw.decode("utf-8"))
    AI.bind_registered_capsule(man, reader, supplied=same)
    print("  PB-4 PASS  anti-vacuity: an identical capsule is "
          "accepted (PB-1 refuses substitution, not supply)")

    # ---- SK-1: the structural kernel exists and is closed ----------
    if not hasattr(AI, "verify_staged_boundary_structure_kat"):
        raise PinBindRefusal(
            "SK-1 NO_STRUCTURAL_KERNEL: portable structural testing "
            "has nowhere to run that cannot satisfy admission "
            "(codex 0057Z P0)")
    import inspect
    ksig = set(inspect.signature(
        AI.verify_staged_boundary_structure_kat).parameters)
    assert "disposition_capsule" not in ksig, \
        "SK-1: the structural kernel must not take a capsule at all"
    print("  SK-1 PASS  a separately named structural kernel exists "
          "and takes no capsule")

    # ---- SK-2: a structural stamp CANNOT be consumed as admission --
    stamp = {"claim_scope": "STRUCTURAL_KAT_ONLY",
             "admission_eligible": False,
             "proof_kind_status": "NOT_EVALUATED",
             "authorizes": "NOTHING",
             "structural_kat_sha256": "0" * 64,
             "structure": {"lanes": {}, "authority_keys": 6}}
    try:
        AI.consume_as_admission(stamp)
        raise PinBindRefusal(
            "SK-2 STRUCTURAL_CONSUMED_AS_ADMISSION: a "
            "STRUCTURAL_KAT_ONLY result was accepted as an admission "
            "fact -- the stamp must make that structurally impossible")
    except AI.InstrumentRefusal as e:
        assert "STRUCTURAL_KAT_ONLY" in str(e), str(e)[:120]
    assert "staged_boundary_sha256" not in stamp
    assert "proof_kinds" not in stamp
    print("  SK-2 PASS  a structural stamp is refused as an admission "
          "fact and carries neither digest nor proof kinds")

    # ---- SK-3: anti-vacuity -- a production-shaped result passes ---
    prod = {"report": {}, "staged_boundary_sha256": "a" * 64,
            "proof_kinds": {k: {"count": v} for k, v in TRUTH.items()}}
    AI.consume_as_admission(prod)
    print("  SK-3 PASS  anti-vacuity: a production-shaped result IS "
          "consumable (SK-2 refuses the stamp, not every dict)")

    print("w2 capsule pin-bind + structural-kat red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except PinBindRefusal as e:
        print(f"RED: {e}")
        raise SystemExit(1)
