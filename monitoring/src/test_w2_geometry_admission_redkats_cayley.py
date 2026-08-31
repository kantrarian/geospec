#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEOMETRY ADMISSION RED-KATs (cayley) -- codex 1507Z item 5 point 2:

  "arbitrary seed, unpinned input bundle, altered registry/segment/
   mask, or caller-built capsule refuses BEFORE a replicate or side
   effect"

Every control below fires on the LIVE committed capsule and the LIVE
committed manifest -- never a hand-built fixture standing in for them
-- and every refusal is asserted to happen before any replicate is
drawn. The positive control runs first: if the real pinned capsule did
not resolve, every refusal below would be vacuous.

Read-only. Draws no replicate, fires nothing, admits nothing.
Lambda_geo INCONCLUSIVE.
"""
import copy
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
CAPSULE_REL = ("docs/f2g_window2_execution/"
               "bound_geometry_capsule_v2.json")
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"


class GeometryAdmissionRefusal(AssertionError):
    pass


def _blob(commit, rel):
    p = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if p.returncode:
        raise GeometryAdmissionRefusal(f"unreadable {commit}:{rel}")
    return p.stdout


def _selftest():
    import w2_power_harness_cayley as PH
    import w2_geometry_capsule_gen_cayley as GEN

    man = json.loads(_blob("HEAD", MANIFEST_REL).decode("utf-8"))
    ref = {"manifest_commit": "HEAD", "path": CAPSULE_REL}

    # ---- GA-0 POSITIVE: the real pinned capsule resolves ----------
    live = PH._load_bound_geometry(REPO, ref)
    fam = "B1B"
    point = live["effect_grids"][fam][0]
    PH._validate_geometry_capsule(live, fam, point)
    print(f"  GA-0 PASS  the LIVE pinned capsule resolves and "
          f"validates ({sum(len(v) for v in live['registries'].values())}"
          " stations, seed "
          f"{live['seed_authority_sha256'][:12]}..) -- every refusal "
          "below is therefore non-vacuous")

    def refuses(mut, code, why, *, capsule=None):
        """Mutate the CAPSULE and require a typed refusal from the
        validator, which runs before any replicate."""
        bad = copy.deepcopy(capsule if capsule is not None else live)
        mut(bad)
        bad["capsule_digest"] = PH._geometry_capsule_digest(bad)
        try:
            PH._validate_geometry_capsule(bad, fam, point)
        except PH.PowerHarnessError as e:
            if code not in str(e):
                raise GeometryAdmissionRefusal(
                    f"{why}: refused with {str(e)[:90]}, expected "
                    f"{code}")
            return
        raise GeometryAdmissionRefusal(
            f"{why}: ACCEPTED -- no replicate may ever run behind "
            "this")

    # ---- GA-1 ARBITRARY SEED -------------------------------------
    # the capsule-level shape check accepts any 64-hex; the LOADER is
    # where an unregistered root dies, so it is proven there against
    # the live manifest rather than in the shape validator.
    fake = copy.deepcopy(live)
    fake["seed_authority_sha256"] = "d" * 64
    fake["capsule_digest"] = PH._geometry_capsule_digest(fake)
    try:
        PH._resolve_capsule_input_refs(REPO, "HEAD", man, fake)
        raise GeometryAdmissionRefusal(
            "GA-1 ARBITRARY_SEED_ADMITTED: a capsule carrying an "
            "unregistered 64-hex seed authority resolved")
    except PH.PowerHarnessError as e:
        assert "POWER_SEED_AUTHORITY_UNREGISTERED" in str(e), str(e)
    print("  GA-1 PASS  an ARBITRARY 64-hex seed authority refuses "
          "against the manifest-pinned registered root")

    # ---- GA-2 UNPINNED INPUT BUNDLE ------------------------------
    unp = copy.deepcopy(live)
    unp["input_refs"]["inputs_bundle"]["path"] = \
        "docs/f2g_window2_execution/not_a_pinned_bundle.json"
    unp["capsule_digest"] = PH._geometry_capsule_digest(unp)
    try:
        PH._resolve_capsule_input_refs(REPO, "HEAD", man, unp)
        raise GeometryAdmissionRefusal(
            "GA-2 UNPINNED_BUNDLE_ADMITTED")
    except PH.PowerHarnessError as e:
        assert "POWER_GEOMETRY_INPUT_NOT_PINNED" in str(e), str(e)
    print("  GA-2 PASS  an input bundle that is not an ADMITTED pin "
          "refuses (well-formed reference, unadmitted bytes)")

    # a reference whose digest disagrees with the pin also refuses
    div = copy.deepcopy(live)
    div["input_refs"]["inputs_bundle"]["blob_sha256"] = "0" * 64
    div["capsule_digest"] = PH._geometry_capsule_digest(div)
    try:
        PH._resolve_capsule_input_refs(REPO, "HEAD", man, div)
        raise GeometryAdmissionRefusal("GA-2b DIVERGENT_REF_ADMITTED")
    except PH.PowerHarnessError as e:
        assert "POWER_GEOMETRY_INPUT_DIVERGENT" in str(e), str(e)
    print("  GA-2b PASS  an input reference whose digest disagrees "
          "with the admitted pin refuses")

    # ---- GA-2c SAME BLOB, DIFFERENT COMMIT (codex item 3A) -------
    # codex's exact reproduction: point a reference at a commit where
    # the SAME path carries IDENTICAL bytes. Path and blob both match;
    # only the commit differs. That is not the admitted provenance.
    head_full = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    seed_rel = live["input_refs"]["seed_authority"]["path"]
    pin_commit = live["input_refs"]["seed_authority"]["commit"]
    if head_full == pin_commit:
        raise GeometryAdmissionRefusal(
            "GA-2c CONTROL_INERT: HEAD is the pin commit, so this "
            "control cannot distinguish commit from bytes")
    if _blob(head_full, seed_rel) != _blob(pin_commit, seed_rel):
        raise GeometryAdmissionRefusal(
            "GA-2c CONTROL_INERT: the two commits do not carry "
            "identical bytes, so a refusal would not isolate the "
            "commit field")
    same_bytes = copy.deepcopy(live)
    same_bytes["input_refs"]["seed_authority"]["commit"] = head_full
    same_bytes["capsule_digest"] = PH._geometry_capsule_digest(
        same_bytes)
    try:
        PH._resolve_capsule_input_refs(REPO, "HEAD", man, same_bytes)
        raise GeometryAdmissionRefusal(
            "GA-2c SAME_BLOB_OTHER_COMMIT_ADMITTED: a reference "
            "naming a different commit with identical bytes resolved")
    except PH.PowerHarnessError as e:
        assert "POWER_GEOMETRY_INPUT_COMMIT_DIVERGENT" in str(e), \
            str(e)
    print("  GA-2c PASS  a reference with the RIGHT path and the "
          "RIGHT bytes but a DIFFERENT commit refuses (exact "
          "three-field provenance, not two)")

    # ---- GA-3 ALTERED REGISTRY / SEGMENT / MASK ------------------
    refuses(lambda c: c["registries"]["cascadia"].append("CC.EVIL"),
            "POWER_LOCO_GEOMETRY_INVALID",
            "GA-3a added station without a segment")

    def _seg(c):
        s = c["registries"]["cascadia"][0]
        c["segments"]["cascadia"][s] = "not_a_registered_segment"
    # a relabelled segment keeps the station set, so the closed
    # station-set check passes and the capsule digest moves: the
    # defence that matters is that the DIGEST is recomputed and the
    # loader re-resolves, so prove the digest is sensitive
    moved = copy.deepcopy(live)
    _seg(moved)
    if PH._geometry_capsule_digest(moved) == live["capsule_digest"]:
        raise GeometryAdmissionRefusal(
            "GA-3b SEGMENT_MOVE_INVISIBLE: relabelling a station's "
            "segment did not move the capsule digest")
    print("  GA-3b PASS  relabelling one station's segment MOVES the "
          "capsule digest (so the pinned-byte check catches it)")

    refuses(lambda c: c["carrier_masks"]["cascadia"][
        "registered_days"].pop(),
        "CALENDAR_MASK_COMPRESSION",
        "GA-3c compacted registered_days")
    refuses(lambda c: c["carrier_masks"]["cascadia"][
        "available_days"].append("2026-09-03"),
        "CALENDAR_", "GA-3d availability on the PRESTART day")
    print("  GA-3 PASS  altered registry / compacted grid / "
          "PRESTART-day availability all refuse in the validator, "
          "before any replicate")

    # ---- GA-4 CALLER-BUILT CAPSULE -------------------------------
    # the fixture builder can still MINT a structurally valid capsule;
    # what it cannot do is get one admitted, because its input_refs
    # resolve to no pin.
    fixture = GEN.build_capsule(
        calendar_authority=json.loads(_blob(
            "HEAD", "docs/f2g_window2_execution/"
                    "calendar_authority_w2_v4.json").decode("utf-8")),
        effect_grids_artifact=json.loads(_blob(
            "HEAD", "docs/f2g_window2_execution/"
                    "effect_grids_w2_v1.json").decode("utf-8")),
        registries=live["registries"],
        segments=live["segments"],
        available_days_by_carrier={
            ck: v["available_days"]
            for ck, v in live["carrier_masks"].items()},
        calendar_authority_sha256="a" * 64,
        calendar_authority_ref={"commit": "b" * 40,
                                "path": "docs/x.json"},
        seed_authority_sha256=live["seed_authority_sha256"],
        loco_registry_carrier="cascadia",
        input_refs={n: {"commit": "e" * 40,
                        "path": f"docs/caller/{n}.json",
                        "blob_sha256": "f" * 64}
                    for n in PH.GEOMETRY_INPUT_REFS},
        mode="bound")
    PH._validate_geometry_capsule(fixture, fam, point)   # structurally fine
    try:
        PH._resolve_capsule_input_refs(REPO, "HEAD", man, fixture)
        raise GeometryAdmissionRefusal(
            "GA-4 CALLER_CAPSULE_ADMITTED: a caller-built capsule "
            "resolved against the live manifest")
    except PH.PowerHarnessError as e:
        assert "POWER_GEOMETRY_INPUT_NOT_PINNED" in str(e), str(e)
    print("  GA-4 PASS  a CALLER-BUILT capsule is structurally valid "
          "yet REFUSES admission (its input_refs resolve to no "
          "admitted pin) -- structural validity is not admission")

    # ---- GA-6 (codex cycle-6b item 3): the SEED-RECORD locks ----
    # Codex named eight records that must each refuse. They were
    # measured ad hoc when the repair was written; a lock is not a
    # lock until it is a registered control, so they run here.
    seed_rel = live["input_refs"]["seed_authority"]["path"]
    seed_pin = live["input_refs"]["seed_authority"]
    rec = json.loads(_blob(seed_pin["commit"], seed_rel).decode())
    idx = PH._manifest_pin_index(man)
    if PH._verify_seed_authority_record(REPO, man, idx, rec) is not True:
        raise GeometryAdmissionRefusal(
            "GA-6 POSITIVE_FAILED: the live pinned seed record does "
            "not verify, so every probe below would be vacuous")

    def _seed_refuses(mut, label):
        bad = copy.deepcopy(rec)
        mut(bad)
        try:
            PH._verify_seed_authority_record(REPO, man, idx, bad)
        except PH.PowerHarnessError as e:
            if "POWER_SEED_RECORD_STALE" not in str(e):
                raise GeometryAdmissionRefusal(
                    f"GA-6 {label}: wrong refusal {str(e)[:80]}")
            return
        raise GeometryAdmissionRefusal(
            f"GA-6 {label}: ADMITTED -- no replicate may run behind "
            "this record")

    fams = list(rec["families"])
    _seed_refuses(lambda b: [b["grammar_evidence"]["by_family"][f]
                             .__setitem__("replicate_seeds", [])
                             for f in fams], "empty evidence")
    _seed_refuses(lambda b: [b["grammar_evidence"]["by_family"][f]
                             .__setitem__(
                                 "replicate_seeds",
                                 b["grammar_evidence"]["by_family"][f]
                                 ["replicate_seeds"][:1])
                             for f in fams], "short evidence")
    _seed_refuses(lambda b: [b["grammar_evidence"]["by_family"][f]
                             ["replicate_seeds"].append(9)
                             for f in fams], "long evidence")
    _seed_refuses(lambda b: b["grammar_evidence"]["by_family"]
                  .pop(fams[0]), "missing family")
    _seed_refuses(lambda b: b["grammar_evidence"]["by_family"]
                  .__setitem__("BXX", {"master_substream_seed": 1,
                                       "replicate_seeds": [1, 2, 3]}),
                  "extra family")
    _seed_refuses(lambda b: b["grammar_evidence"]["by_family"][fams[0]]
                  .__setitem__("master_substream_seed", 123),
                  "wrong master")
    _seed_refuses(lambda b: b["grammar"]["engine_module"]
                  .__setitem__("function", "evil"),
                  "wrong function name")
    _seed_refuses(lambda b: b["target_identity"]
                  ["consumed_implementations"]
                  .__setitem__("monitoring/src/"
                               "w2_power_harness_cayley.py", "0" * 64),
                  "doctored target identity")
    _seed_refuses(lambda b: b["target_identity"]
                  .__setitem__("execution", "IN_PROCESS_FIXTURE_ONLY"),
                  "in-process execution claim")
    print("  GA-6 PASS  9 seed-record probes each refuse "
          "POWER_SEED_RECORD_STALE (empty / short / long / missing / "
          "extra family, wrong master, wrong function name, doctored "
          "target identity, in-process execution) with the live "
          "record as the positive -- no loop bound comes from "
          "submitted evidence")

    # ---- GA-5 the loader takes no capsule dict -------------------
    try:
        PH._load_bound_geometry(REPO, dict(live))
        raise GeometryAdmissionRefusal("GA-5 CAPSULE_DICT_ACCEPTED")
    except PH.PowerHarnessError as e:
        assert "POWER_GEOMETRY_REF_INVALID" in str(e), str(e)
    print("  GA-5 PASS  certification takes a {manifest_commit, path} "
          "reference; a capsule dict is structurally impossible")
    print("w2 geometry admission red-KATs: ALL PASS")


if __name__ == "__main__":
    _selftest()
