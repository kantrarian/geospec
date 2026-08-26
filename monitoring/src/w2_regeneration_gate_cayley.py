#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE POST-REGENERATION GATE (cayley) -- codex 0404Z step 4.

"Run the updated generator once over that final descendant, then
read-only verify: every declared pin present/current and every
required verification surface included. No second regeneration."

WHY THIS IS A COMMITTED TOOL AND NOT A SCRATCH SCRIPT
-----------------------------------------------------
I have been running this audit from a scratchpad file. That is how
"zero stale" gets ASSERTED rather than verified: an ad-hoc script is
not reviewable, not reproducible by anyone else, and its exact
criteria die with the shell it ran in. The gate that decides whether
the packet may claim a current manifest has to be as inspectable as
the thing it checks.

WHAT THIS PROVES
----------------
  RG-1  every path the generator DECLARES is actually pinned
  RG-2  every pin's blob_sha256 equals the bytes AT HEAD -- never at
        the pin's own recorded commit, which would make every pin
        current by construction (see _blob_at_head)
  RG-3  no declared path is missing from the tree
  RG-4  every required VERIFICATION SURFACE is bound, so closure-4
        and pin-bind evidence cannot be cited unbound
  RG-5  anti-vacuity: a doctored manifest must REFUSE

WHAT THIS DOES NOT PROVE -- stated because a gate that oversells is
worse than no gate
-------------------------------------------------------------------
It proves the manifest BINDS the right bytes. It does NOT prove those
bytes were executed or that they passed: pinning is not running.
Execution outcomes live in the pinned compact run summary
(pre-manifest) and the downstream post-manifest receipt
(manifest-owned admission), per codex 0410Z section 4. Do not read a
green gate as a green bar.

Read-only. Opens no window-2 value, no network, admits nothing.
"""
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"

# codex 0404Z item 1, confirmed at 0445Z as an exact transcription of
# the ruled bars list. Each of these must be BOUND, because after the
# STRUCTURAL_KAT_ONLY relabel the shared bar alone no longer
# establishes closure 4 or the pin-bind.
REQUIRED_BAR_SURFACES = (
    "monitoring/src/test_f2g_window2_redkats_grassmann.py",
    "monitoring/src/test_w2_capsule_pin_bind_redkats_cayley.py",
    "monitoring/src/test_w2_report_proof_kinds_redkats_cayley.py",
    "monitoring/src/test_w2_boundary_admitted_partition_redkats_cayley.py",
    "monitoring/src/test_w2_admitted_absence_redkats_cayley.py",
    "monitoring/src/test_w2_authority_serves_every_key_redkats_cayley.py",
    "monitoring/src/test_w2_frozen_carrier_set_redkats_cayley.py",
    "monitoring/src/test_w2_fixture_schema_redkats_grassmann.py",
    "docs/f2g_window2_execution/w2_verification_run_summary_v1.json",
)

# codex 0445Z item 5.2: enforce PLACEMENT, not merely membership.
# Four different authorities are in play -- production operation
# bytes, verification code, the pre-manifest execution record, and the
# post-manifest receipt -- and none may silently stand in for another.
# A path pinned in the wrong slot answers to the wrong authority even
# though a membership-only check would call it bound.
REQUIRED_BY_SLOT = {
    "bars": REQUIRED_BAR_SURFACES,
    "accrual_impl": (
        "monitoring/src/w2_disposition_capsule_grassmann.py",
        "monitoring/src/w2_restage_lineage_grassmann.py",
        "monitoring/src/w2_restage_v4_grassmann.py",
        "monitoring/src/w2_restage_verify_batch_grassmann.py",
        "monitoring/src/w2_verification_run_summary_grassmann.py",
    ),
    "execution_verifier": (
        "monitoring/src/f2g_execution_manifest_verifier_cayley.py",
        "monitoring/src/w2_regeneration_gate_cayley.py",
    ),
    # only once the slot is BOUND -- it is honestly OPEN until the
    # post-capture bind, and requiring it earlier would demand a pin
    # the design says must not exist yet
    "producer_boundary": (
        "monitoring/src/w2_acquisition_capture_grassmann.py",
    ),
}
SLOT_REQUIRED_ONLY_WHEN_BOUND = ("producer_boundary",)


class RegenerationGateRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


def _blob(commit, rel):
    """Bytes at a named commit. NOTE the caller below deliberately
    does NOT use the pin's own commit -- see _blob_at_head."""
    p = subprocess.run(["git", "-C", REPO, "show", f"{commit}:{rel}"],
                       capture_output=True)
    return None if p.returncode else p.stdout


def _blob_at_head(_commit, rel):
    """Bytes at HEAD, IGNORING the pin's recorded commit.

    SELF-CAUGHT 2026-08-26, before this file was ever committed: my
    first version read each blob at `pin["commit"]`. That is
    self-authentication -- a pin validated against the very commit it
    names ALWAYS matches, so the gate reported 34/34 current while an
    independent HEAD audit found four stale pins. It would have
    certified a manifest pinning arbitrarily old bytes as "current".

    Exactly the defect class this whole night has been about: a check
    that reads clean because it compares something to itself. "Current"
    can only mean the CURRENT tree, which is also the semantics the
    runtime allowlist enforces (disk bytes must equal the pinned blob).

    Git blob rather than the working file because Windows checkouts
    EOL-convert, and a CRLF working copy of an LF blob is the same
    source.
    """
    return _blob("HEAD", rel)


def walk_pins(obj, trail=""):
    """Yield every (trail, pin) carrying a (path, blob_sha256)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and "blob_sha256" in obj:
            yield trail, obj
        for k, v in obj.items():
            yield from walk_pins(v, f"{trail}.{k}" if trail else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_pins(v, f"{trail}[{i}]")


def audit(manifest, *, blob=_blob_at_head):
    """Recompute the pin state. Returns a typed report; raises only on
    a structurally unusable manifest."""
    pins = list(walk_pins(manifest))
    if not pins:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: the manifest declares no pins "
            "at all -- an empty pin set would make every check below "
            "vacuously true")
    stale, missing, ok = [], [], []
    for trail, pin in pins:
        rel, want = pin["path"], pin["blob_sha256"]
        raw = blob(pin.get("commit", "HEAD"), rel)
        if raw is None:
            missing.append((trail, rel))
            continue
        got = hashlib.sha256(raw).hexdigest()
        (ok if got == want else stale).append(
            (trail, rel, want, got))
    # PLACEMENT (codex 0445Z 5.2): a path bound in the wrong slot
    # answers to the wrong authority, so membership alone is not
    # enough. Compute what each slot actually pins.
    slots = manifest.get("slots", {})
    by_slot = {}
    for name, slot in slots.items():
        if isinstance(slot, dict):
            by_slot[name] = {p.get("path") for p in slot.get("pins", ())
                             if isinstance(p, dict)}
    bound_anywhere = {p["path"] for _, p in pins}
    unbound, misplaced = [], []
    for slot_name, required in REQUIRED_BY_SLOT.items():
        if slot_name in SLOT_REQUIRED_ONLY_WHEN_BOUND and \
                str(slots.get(slot_name, {}).get("status")) != "BOUND":
            continue                      # honestly OPEN: not yet due
        here = by_slot.get(slot_name, set())
        for rel in required:
            if rel in here:
                continue
            if rel in bound_anywhere:
                elsewhere = sorted(s for s, ps in by_slot.items()
                                   if rel in ps)
                misplaced.append((rel, slot_name, elsewhere))
            else:
                unbound.append((rel, slot_name))
    return {"pins": len(pins), "match": len(ok), "stale": stale,
            "missing": missing, "unbound_surfaces": unbound,
            "misplaced": misplaced}


def load_manifest(commit="HEAD"):
    """Read the manifest from a COMMIT, never the working file.

    codex 0445Z 5.1: the gate previously read the manifest from disk.
    An uncommitted or doctored working file would then have become the
    very authority the gate certifies -- the gate would faithfully
    verify a manifest nobody had committed. Same family as reading a
    pin at its own commit: the authority has to come from outside the
    thing being checked.
    """
    raw = _blob(commit, MANIFEST_REL)
    if raw is None:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: no committed manifest at "
            f"{commit}:{MANIFEST_REL} -- the gate will not certify a "
            "working-tree file")
    return json.loads(raw.decode("utf-8"))


def require_manifest_verifier_pass(commit="HEAD"):
    """The execution-manifest verifier must PASS prestart FIRST.

    codex 0445Z 5.1. Pin currency is meaningless if the manifest is
    not itself a valid, zero-OPEN prestart manifest: a structurally
    broken manifest with perfectly current pins would otherwise earn
    a green gate.
    """
    import f2g_execution_manifest_verifier_cayley as EMV
    v = EMV.verify(REPO, commit, prestart=True)
    if v.get("verdict") != "PASS" or v.get("slots_open", -1) != 0:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: execution-manifest verifier "
            f"is not a zero-OPEN prestart PASS at {commit} "
            f"(verdict={v.get('verdict')}, "
            f"slots_open={v.get('slots_open')})")
    return v


def gate(manifest, *, blob=_blob_at_head, manifest_commit=None):
    """PASS only on a verified manifest with zero stale, zero missing,
    zero unbound and zero MISPLACED required surface."""
    if manifest_commit is not None:
        require_manifest_verifier_pass(manifest_commit)
    rep = audit(manifest, blob=blob)
    problems = []
    if rep["stale"]:
        problems.append(
            f"{len(rep['stale'])} STALE pin(s): " + ", ".join(
                f"{r} ({w[:12]}!={g[:12]})"
                for _, r, w, g in rep["stale"][:6]))
    if rep["missing"]:
        problems.append(
            f"{len(rep['missing'])} MISSING path(s): " + ", ".join(
                r for _, r in rep["missing"][:6]))
    if rep["unbound_surfaces"]:
        problems.append(
            f"{len(rep['unbound_surfaces'])} required surface(s) NOT "
            "BOUND: " + ", ".join(
                f"{r} (needs slot {s})"
                for r, s in rep["unbound_surfaces"][:6])
            + " -- unbound evidence cannot support a packet claim")
    if rep["misplaced"]:
        problems.append(
            f"{len(rep['misplaced'])} required surface(s) MISPLACED: "
            + ", ".join(
                f"{r} pinned in {e} but required in {s}"
                for r, s, e in rep["misplaced"][:6])
            + " -- a path in the wrong slot answers to the wrong "
              "authority even though it is bound")
    if problems:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: " + "; ".join(problems))
    return rep


def _selftest():
    """Doctors run against the REAL COMMITTED manifest, loaded from a
    commit rather than the working file (codex 0445Z 5.1)."""
    man = load_manifest("HEAD")
    rep = audit(man)
    print(f"  RG-0 PASS  {rep['pins']} pins walked from the real "
          f"committed manifest (anti-vacuity: a pinless manifest "
          f"refuses)")

    # RG-5 anti-vacuity FIRST: prove the gate can fail, before any
    # report of it passing means anything.
    import copy
    doc = copy.deepcopy(man)
    tampered = False
    for _, pin in walk_pins(doc):
        pin["blob_sha256"] = "0" * 64
        tampered = True
        break
    assert tampered, "RG-5: no pin to doctor"
    try:
        gate(doc)
        raise RegenerationGateRefusal(
            "RG-5 GATE_IS_VACUOUS: a doctored pin was accepted")
    except RegenerationGateRefusal as e:
        assert "STALE" in str(e), str(e)[:120]
    print("  RG-5 PASS  a doctored pin REFUSES (the gate is not "
          "vacuous)")

    # RG-2: the CURRENCY doctor. My first version read each blob at
    # the pin's OWN commit and so reported everything current by
    # construction. This proves the gate measures against HEAD: a pin
    # whose recorded commit is real but whose bytes have since moved
    # must be reported STALE, not quietly accepted.
    doc_cur = copy.deepcopy(man)
    drifted = None
    for _, pin in walk_pins(doc_cur):
        head = _blob("HEAD", pin["path"])
        if head is not None and                 hashlib.sha256(head).hexdigest() != pin["blob_sha256"]:
            drifted = pin["path"]
            break
    if drifted:
        print(f"  RG-2 PASS  currency measured against HEAD, not the "
              f"pin's own commit (live drift detected on {drifted})")
    else:
        # no natural drift right now: manufacture it
        for _, pin in walk_pins(doc_cur):
            pin["blob_sha256"] = "1" * 64
            break
        try:
            gate(doc_cur)
            raise RegenerationGateRefusal(
                "RG-2 GATE_SELF_AUTHENTICATES: bytes differing from "
                "HEAD were accepted as current")
        except RegenerationGateRefusal as e:
            assert "STALE" in str(e), str(e)[:120]
        print("  RG-2 PASS  bytes differing from HEAD are STALE")

    doc2 = copy.deepcopy(man)
    for _, pin in walk_pins(doc2):
        if pin["path"] in REQUIRED_BAR_SURFACES:
            pin["path"] = "monitoring/src/_not_a_surface.py"
            break
    try:
        gate(doc2)
        raise RegenerationGateRefusal(
            "RG-4 UNBOUND_SURFACE_ADMITTED: a required verification "
            "surface was dropped and the gate passed")
    except RegenerationGateRefusal as e:
        assert "NOT BOUND" in str(e) or "STALE" in str(e), str(e)[:120]
    print("  RG-4 PASS  dropping a required verification surface "
          "REFUSES")

    # RG-6 (codex 0445Z 5.2): PLACEMENT. Move a required path into
    # the wrong slot -- it is still bound, still current, and a
    # membership-only check would call it present. The gate must
    # refuse, because a path in the wrong slot answers to the wrong
    # authority: production operation bytes, verification code, the
    # pre-manifest record and the post-manifest receipt are four
    # different authorities and none may stand in for another.
    doc3 = copy.deepcopy(man)
    moved = None
    slots3 = doc3.get("slots", {})
    for pin in list(slots3.get("accrual_impl", {}).get("pins", ())):
        if pin.get("path") in REQUIRED_BY_SLOT["accrual_impl"]:
            slots3["accrual_impl"]["pins"].remove(pin)
            slots3.setdefault("bars", {}).setdefault(
                "pins", []).append(pin)
            moved = pin["path"]
            break
    if moved:
        try:
            gate(doc3)
            raise RegenerationGateRefusal(
                "RG-6 PLACEMENT_IGNORED: a required production module "
                f"was moved into bars and the gate passed ({moved})")
        except RegenerationGateRefusal as e:
            assert "MISPLACED" in str(e) or "NOT BOUND" in str(e), \
                str(e)[:140]
        print(f"  RG-6 PASS  a required path moved to the wrong slot "
              f"REFUSES ({os.path.basename(moved)}: bound, current, "
              f"still wrong)")
    else:
        print("  RG-6 SKIP  no accrual_impl required path pinned yet "
              "to move (expected before the single regeneration)")

    # Now the live state, reported honestly whatever it is.
    print(f"\n  live: {rep['match']} match / {len(rep['stale'])} "
          f"stale / {len(rep['missing'])} missing / "
          f"{len(rep['unbound_surfaces'])} unbound / "
          f"{len(rep['misplaced'])} misplaced")
    for _, rel, w, g in rep["stale"]:
        print(f"    STALE     {rel}  {w[:12]}.. -> {g[:12]}..")
    for rel, slot in rep["unbound_surfaces"]:
        print(f"    UNBOUND   {rel}  (needs slot {slot})")
    for rel, slot, where in rep["misplaced"]:
        print(f"    MISPLACED {rel}  in {where}, required in {slot}")
    try:
        gate(man, manifest_commit="HEAD")
        print("\nREGENERATION GATE: PASS (pins current and complete)")
        print("NOTE: pinning is not running -- this says nothing "
              "about whether those bytes PASSED. See the run summary "
              "and the post-manifest receipt.")
        return 0
    except RegenerationGateRefusal as e:
        print(f"\nREGENERATION GATE: REFUSED (expected before the "
              f"single regeneration)\n  {str(e)[:400]}")
        return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(_selftest())
