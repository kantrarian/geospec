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
# codex 2328Z: the inverse control must show the bypass was OPEN.
# This is the cycle-6d candidate -- the last commit before the
# capsule-content binding existed -- and it is named, not derived, so
# the control cannot drift onto a commit that already carries the
# repair.
PRE_REPAIR_COMMIT = "bd4caf42f84b8d2189754b9217ecc51d35e62ecb"


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

    # ---- GA-7 (codex cycle-6c): PROVENANCE BY REGENERATION ------
    # The execution label was caller-settable, the child inherited a
    # steerable environment, and admission checked the label on the
    # seed record only. All four of codex's locks run here. These
    # spawn isolated worktrees and are the slowest controls in the
    # suite; that cost is the point -- provenance is proved by
    # rebuilding bytes, never by reading a self-reported string.
    import os as _os
    import subprocess as _sp
    import tempfile as _tf
    import w2_target_identity_cayley as TID
    import w2_selection as WSEL

    head_c = _sp.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
    SEED_REL = "docs/f2g_window2_execution/power_seed_authority_w2_v1.json"
    BUN_REL = "docs/f2g_window2_execution/power_geometry_inputs_w2_v1.json"
    seed_ok, bun_ok = _blob("HEAD", SEED_REL), _blob("HEAD", BUN_REL)
    pair = {"seed_authority": (seed_ok,
                               "w2_power_seed_authority_gen_cayley",
                               True),
            "inputs_bundle": (bun_ok,
                              "w2_power_geometry_inputs_gen_cayley",
                              True)}
    if PH._verify_regenerates(REPO, head_c, pair) is not True:
        raise GeometryAdmissionRefusal(
            "GA-7 POSITIVE_FAILED: the live pinned seed and bundle do "
            "not regenerate, so every probe below is vacuous")
    print("  GA-7a PASS  both live pinned artifacts REGENERATE "
          "byte-identically in the hardened worker (the positive)")

    def _adm_refuses(raw, mod, name, label, *, labelled=True):
        try:
            PH._verify_regenerates(REPO, head_c,
                                   {name: (raw, mod, labelled)})
        except PH.PowerHarnessError:
            return
        raise GeometryAdmissionRefusal(
            f"GA-7 {label}: ADMITTED -- bytes that do not reproduce "
            "must never be admissible")

    # lock 1: parent sets the marker AND monkeypatches a callable
    _os.environ["W2_ISOLATED_TARGET"] = "1"
    _orig_sel = WSEL.select
    try:
        WSEL.select = (lambda ck, dr, c, **k: (
            lambda r: dict(r, selected=sorted(r["selected"])[:-1])
        )(_orig_sel(ck, dr, c, **k)))
        import w2_power_geometry_inputs_gen_cayley as _G
        forged = _G.build(REPO, commit=head_c)
    finally:
        WSEL.select = _orig_sel
        _os.environ.pop("W2_ISOLATED_TARGET", None)
    if forged["target_identity"]["execution"] != \
            "ISOLATED_TARGET_WORKER" or \
            len(forged["registries"]["cascadia"]) != \
            len(live["registries"]["cascadia"]) - 1:
        raise GeometryAdmissionRefusal(
            "GA-7 CONTROL_INERT: the forging probe did not construct "
            "a mislabelled, altered artifact")
    _adm_refuses(json.dumps(forged, indent=1,
                            sort_keys=True).encode() + b"\n",
                 "w2_power_geometry_inputs_gen_cayley",
                 "inputs_bundle", "FORGED_MARKER_ADMITTED")
    print("  GA-7b PASS  a parent-set marker + callable monkeypatch "
          "CAN mint a mislabelled 15-station artifact, and it is NOT "
          "ADMISSIBLE -- the label is a hint, the bytes are the proof")

    # lock 2: PYTHONPATH/sitecustomize steering leaves the hardened
    # worker byte-identical
    probe = _tf.mkdtemp(prefix="ga7-sitecustomize-")
    try:
        with open(os.path.join(probe, "sitecustomize.py"), "w",
                  encoding="utf-8") as f:
            f.write("try:\n"
                    "    import w2_selection as _W\n"
                    "    _o = _W.select\n"
                    "    _W.select = lambda ck, dr, c, **k: (lambda r:"
                    " dict(r, selected=sorted(r['selected'])[:-1]))"
                    "(_o(ck, dr, c, **k))\n"
                    "except Exception:\n    pass\n")
        clean = TID.regenerate_in_isolated_worker(
            REPO, head_c, "w2_power_geometry_inputs_gen_cayley",
            "build")
        _os.environ["PYTHONPATH"] = probe
        try:
            steered = TID.regenerate_in_isolated_worker(
                REPO, head_c, "w2_power_geometry_inputs_gen_cayley",
                "build")
        finally:
            _os.environ.pop("PYTHONPATH", None)
    finally:
        import shutil as _sh
        _sh.rmtree(probe, ignore_errors=True)
    if json.dumps(clean, sort_keys=True) != \
            json.dumps(steered, sort_keys=True):
        raise GeometryAdmissionRefusal(
            "GA-7 CHILD_ENVIRONMENT_STEERED: a PYTHONPATH "
            "sitecustomize probe changed hardened-worker output")
    print("  GA-7c PASS  a PYTHONPATH/sitecustomize steering probe "
          "leaves hardened-worker output BYTE-IDENTICAL (isolated "
          "mode + sanitized environment)")

    # lock 3: a bundle self-declaring IN_PROCESS_FIXTURE_ONLY refuses
    fixture_claim = copy.deepcopy(json.loads(bun_ok.decode()))
    fixture_claim["target_identity"]["execution"] = \
        "IN_PROCESS_FIXTURE_ONLY"
    _adm_refuses(json.dumps(fixture_claim, indent=1,
                            sort_keys=True).encode() + b"\n",
                 "w2_power_geometry_inputs_gen_cayley",
                 "inputs_bundle", "FIXTURE_ONLY_BUNDLE_ADMITTED")
    # lock 4: change ONE byte of either artifact and admission refuses
    tampered = copy.deepcopy(json.loads(seed_ok.decode()))
    tampered["state"] = "TAMPERED"
    _adm_refuses(json.dumps(tampered, indent=1,
                            sort_keys=True).encode() + b"\n",
                 "w2_power_seed_authority_gen_cayley",
                 "seed_authority", "TAMPERED_SEED_ADMITTED")
    print("  GA-7d PASS  a bundle claiming IN_PROCESS_FIXTURE_ONLY "
          "and a one-byte-changed seed record BOTH refuse at "
          "admission, before any capsule or replicate")

    # ---- GA-8 (cayley cycle-6e, self-reported): the capsule's own
    # CONTENT is bound to the inputs it references -----------------
    # Measured at bd4caf42: authenticating the capsule's REFERENCES
    # while never checking its registries/segments/masks let a
    # 50-station capsule pass admission against its own referenced
    # 51-station bundle. Both new bindings are locked here -- the
    # regeneration proof (worker) and the direct cross-check (no
    # worker) -- and the anti-vacuity control shows the pre-existing
    # schema validator still ACCEPTS the same forgery, so these locks
    # are the only thing standing between it and admission.
    cap_raw = json.dumps(live, sort_keys=True,
                         separators=(",", ":")).encode()
    if PH._verify_regenerates(
            REPO, head_c,
            {"geometry_capsule": (cap_raw,
                                  "w2_geometry_capsule_gen_cayley",
                                  False)}) is not True:
        raise GeometryAdmissionRefusal(
            "GA-8 POSITIVE_FAILED: the live pinned capsule does not "
            "regenerate at its own manifest target, so every capsule "
            "probe below is vacuous")
    resolved_inputs = {n: _blob(r["commit"], r["path"])
                       for n, r in live["input_refs"].items()}
    if PH._verify_capsule_matches_inputs(
            live, resolved_inputs) is not True:
        raise GeometryAdmissionRefusal(
            "GA-8 POSITIVE_FAILED: the live capsule does not match "
            "its own referenced inputs")
    n_live = sum(len(v) for v in live["registries"].values())
    print(f"  GA-8a PASS  the live pinned capsule REGENERATES "
          f"byte-identically at its manifest target AND carries "
          f"exactly its referenced inputs' geometry ({n_live} "
          "stations) -- the positive")

    # the ORIGINAL forgery: drop one station, recompute the digest
    forged_cap = copy.deepcopy(live)
    _victim = forged_cap["registries"]["cascadia"][-1]
    forged_cap["registries"]["cascadia"] = [
        st for st in forged_cap["registries"]["cascadia"]
        if st != _victim]
    del forged_cap["segments"]["cascadia"][_victim]
    forged_cap["capsule_digest"] = \
        PH._geometry_capsule_digest(forged_cap)
    if sum(len(v) for v in forged_cap["registries"].values()) != \
            n_live - 1:
        raise GeometryAdmissionRefusal(
            "GA-8 CONTROL_INERT: the forging probe did not construct "
            "a capsule that diverges from its bundle")
    # ANTI-VACUITY: the pre-existing schema validator accepts it
    PH._validate_geometry_capsule(forged_cap, fam, point)
    try:
        PH._resolve_capsule_input_refs(REPO, "HEAD", man, forged_cap)
    except PH.PowerHarnessError as e:
        if not ("POWER_GEOMETRY_CONTENT_UNBOUND" in str(e) or
                "POWER_TARGET_ARTIFACT_UNREPRODUCIBLE" in str(e)):
            raise GeometryAdmissionRefusal(
                f"GA-8 wrong refusal for the forged capsule: "
                f"{str(e)[:110]}")
    else:
        raise GeometryAdmissionRefusal(
            "GA-8 FORGED_CAPSULE_ADMITTED: a capsule carrying "
            "geometry its own referenced bundle does not describe "
            "must never resolve")
    print("  GA-8b PASS  a station-dropped capsule still passes the "
          "SCHEMA validator (anti-vacuity) and is REFUSED by the "
          "content binding -- reference authenticity was never "
          "content authenticity")

    def _content_refuses(label, mut):
        bad = copy.deepcopy(live)
        mut(bad)
        bad["capsule_digest"] = PH._geometry_capsule_digest(bad)
        try:
            PH._verify_capsule_matches_inputs(bad, resolved_inputs)
        except PH.PowerHarnessError as e:
            if "POWER_GEOMETRY_CONTENT_UNBOUND" not in str(e):
                raise GeometryAdmissionRefusal(
                    f"GA-8 {label}: wrong refusal {str(e)[:90]}")
            return
        raise GeometryAdmissionRefusal(
            f"GA-8 {label}: ACCEPTED -- the capsule may not carry "
            "geometry its referenced inputs do not describe")

    def _mut_add(c):
        c["registries"]["cascadia"] = \
            list(c["registries"]["cascadia"]) + ["ZZ.FAKE"]
        c["segments"]["cascadia"]["ZZ.FAKE"] = \
            sorted(set(c["segments"]["cascadia"].values()))[0]

    def _mut_move(c):
        segs = c["segments"]["cascadia"]
        st = sorted(segs)[0]
        other = sorted({v for v in segs.values() if v != segs[st]})
        if not other:
            raise GeometryAdmissionRefusal(
                "GA-8 CONTROL_INERT: cascadia carries a single "
                "segment, so the moved-segment probe cannot move one")
        segs[st] = other[0]

    def _mut_mask(c):
        m = c["carrier_masks"]["cascadia"]["available_days"]
        c["carrier_masks"]["cascadia"]["available_days"] = list(m[:-1])

    def _mut_frame(c):
        c["calendar_frame"]["engine_days"] = \
            list(c["calendar_frame"]["engine_days"])[:-1]

    def _mut_grids(c):
        f0 = sorted(c["effect_grids"])[0]
        c["effect_grids"][f0] = list(c["effect_grids"][f0]) + [999.0]

    def _mut_drop_carrier(c):
        victim = sorted(k for k in c["registries"]
                        if k != c["loco_registry_carrier"])[0]
        for k in ("registries", "segments", "carrier_masks"):
            del c[k][victim]

    def _mut_add_carrier(c):
        src = c["loco_registry_carrier"]
        c["registries"]["ghost"] = ["GH.0001"]
        c["segments"]["ghost"] = {"GH.0001": "sA"}
        c["carrier_masks"]["ghost"] = \
            copy.deepcopy(c["carrier_masks"][src])

    def _mut_registered_days(c):
        rd = c["carrier_masks"]["cascadia"]["registered_days"]
        c["carrier_masks"]["cascadia"]["registered_days"] = list(rd[:-1])

    _content_refuses("added station", _mut_add)
    _content_refuses("moved segment", _mut_move)
    _content_refuses("changed availability mask", _mut_mask)
    _content_refuses("dropped whole carrier", _mut_drop_carrier)
    _content_refuses("added whole carrier", _mut_add_carrier)
    _content_refuses("truncated registered days", _mut_registered_days)
    _content_refuses("changed calendar frame", _mut_frame)
    _content_refuses("changed effect grids", _mut_grids)
    print("  GA-8c PASS  added station, moved segment, changed "
          "availability day, dropped carrier, added carrier, "
          "truncated registered days, changed frame and changed "
          "effect grids EACH refuse POWER_GEOMETRY_CONTENT_UNBOUND "
          "with no worker in the loop")

    # canonical capsule divergence on a field the CONTENT cross-check
    # does not compare: only the regeneration proof can catch this,
    # so it separates the two new bindings rather than letting one
    # stand in for the other.
    off_ref = copy.deepcopy(live)
    off_ref["calendar_authority_ref"] = dict(
        off_ref["calendar_authority_ref"],
        path="docs/f2g_window2_execution/not_the_authority.json")
    off_ref["capsule_digest"] = PH._geometry_capsule_digest(off_ref)
    PH._validate_geometry_capsule(off_ref, fam, point)      # accepted
    if PH._verify_capsule_matches_inputs(
            off_ref, resolved_inputs) is not True:
        raise GeometryAdmissionRefusal(
            "GA-8 CONTROL_INERT: the divergence probe was supposed to "
            "leave the content cross-check satisfied")
    try:
        PH._verify_regenerates(
            REPO, head_c,
            {"geometry_capsule": (json.dumps(
                off_ref, sort_keys=True,
                separators=(",", ":")).encode(),
                "w2_geometry_capsule_gen_cayley", False)})
    except PH.PowerHarnessError as e:
        if "POWER_TARGET_ARTIFACT_UNREPRODUCIBLE" not in str(e):
            raise GeometryAdmissionRefusal(
                f"GA-8 wrong refusal for canonical divergence: "
                f"{str(e)[:100]}")
    else:
        raise GeometryAdmissionRefusal(
            "GA-8 CANONICAL_DIVERGENCE_ADMITTED: a capsule that does "
            "not reproduce at its own manifest target must refuse")
    print("  GA-8d PASS  a capsule diverging on a field the content "
          "cross-check does NOT compare passes the schema validator "
          "AND the cross-check, and is caught only by the "
          "regeneration proof -- the two bindings are independent")

    # an artifact registered as UNLABELLED may not smuggle a label
    smuggled = dict(json.loads(cap_raw.decode()),
                    target_identity={"execution":
                                     "ISOLATED_TARGET_WORKER"})
    _adm_refuses(json.dumps(smuggled, sort_keys=True,
                            separators=(",", ":")).encode(),
                 "w2_geometry_capsule_gen_cayley", "geometry_capsule",
                 "SMUGGLED_LABEL_ADMITTED", labelled=False)
    # and the artifact spec itself is a closed triple
    try:
        PH._verify_regenerates(REPO, head_c,
                               {"geometry_capsule": (cap_raw, "m")})
    except PH.PowerHarnessError as e:
        if "closed" not in str(e):
            raise GeometryAdmissionRefusal(
                f"GA-8 wrong refusal for a 2-tuple spec: {str(e)[:90]}")
    else:
        raise GeometryAdmissionRefusal(
            "GA-8 OPEN_SPEC_ACCEPTED: the regeneration contract must "
            "be a closed (bytes, module, label_required) triple")
    print("  GA-8e PASS  an artifact registered as carrying no target "
          "identity may not smuggle one in, and the regeneration "
          "contract itself refuses a non-closed spec")

    # ---- GA-8f (codex 2328Z): the PRE-REPAIR inverse control ----
    # A lock is only worth its refusal if the thing it refuses was
    # once accepted. This runs the ACTUAL pre-repair code -- the
    # named commit, materialized detached, in a fresh interpreter, so
    # no module in this process can stand in for it -- and requires
    # that it ADMITS the same forged capsule this bar refuses. If the
    # named commit cannot be resolved the control REFUSES; it never
    # degrades to a silent pass.
    pre = _sp.run(["git", "-C", REPO, "rev-parse",
                   PRE_REPAIR_COMMIT + "^{commit}"],
                  capture_output=True, text=True).stdout.strip()
    if not pre:
        raise GeometryAdmissionRefusal(
            "GA-8f UNRESOLVABLE_PRE_REPAIR: the pre-repair commit "
            f"{PRE_REPAIR_COMMIT[:12]} is not in this repository, so "
            "the inverse control cannot run. It is not skippable -- "
            "a clone without that object cannot second-source this "
            "particular lock")
    _td = _tf.mkdtemp(prefix="ga8f-pre-repair-")
    _wt = _os.path.join(_td, "t")
    try:
        add = _sp.run(["git", "-C", REPO, "worktree", "add",
                       "--detach", _wt, pre], capture_output=True)
        if add.returncode:
            raise GeometryAdmissionRefusal(
                "GA-8f could not materialize the pre-repair worktree: "
                + add.stderr.decode(errors="replace")[:160])
        child = (
            "import json,subprocess,sys\n"
            f"sys.path.insert(0, {_os.path.join(_wt, 'monitoring', 'src')!r})\n"
            "import w2_power_harness_cayley as OLD\n"
            f"REPO={REPO!r}\n"
            f"HEAD={head_c!r}\n"
            "def blob(c,r):\n"
            "    return subprocess.run(['git','-C',REPO,'cat-file',"
            "'blob',c+':'+r],capture_output=True).stdout\n"
            f"man=json.loads(blob(HEAD,{MANIFEST_REL!r}).decode())\n"
            f"cap=json.loads(blob(HEAD,{CAPSULE_REL!r}).decode())\n"
            "v=cap['registries']['cascadia'][-1]\n"
            "cap['registries']['cascadia']=[s for s in "
            "cap['registries']['cascadia'] if s!=v]\n"
            "del cap['segments']['cascadia'][v]\n"
            "cap['capsule_digest']=OLD._geometry_capsule_digest(cap)\n"
            "try:\n"
            "    OLD._resolve_capsule_input_refs(REPO,HEAD,man,cap)\n"
            "    print('<<<PRE>>>ACCEPTED')\n"
            "except Exception as e:\n"
            "    print('<<<PRE>>>REFUSED '+str(e)[:160])\n")
        run = _sp.run([sys.executable, "-I", "-E", "-s", "-c", child],
                      capture_output=True,
                      cwd=_os.path.join(_wt, "monitoring", "src"),
                      timeout=1800)
        out = run.stdout.decode("utf-8", errors="replace")
        if "<<<PRE>>>" not in out:
            raise GeometryAdmissionRefusal(
                "GA-8f the pre-repair control did not run: "
                + (run.stderr.decode(errors="replace")[-300:]
                   or out[-300:]))
        verdict = out.split("<<<PRE>>>", 1)[1].strip()
        if not verdict.startswith("ACCEPTED"):
            raise GeometryAdmissionRefusal(
                "GA-8f CONTROL_INERT: the pre-repair code did NOT "
                "admit the forged capsule, so the repair's refusal "
                f"proves nothing here -- {verdict[:140]}")
    finally:
        _sp.run(["git", "-C", REPO, "worktree", "remove", "--force",
                 _wt], capture_output=True)
        import shutil as _sh2
        _sh2.rmtree(_td, ignore_errors=True)
    print(f"  GA-8f PASS  the PRE-REPAIR code at {pre[:12]} ADMITS "
          "the same forged capsule this bar refuses -- the lock "
          "closes a bypass that was demonstrably open")

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
