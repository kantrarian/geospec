#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 BOUND GEOMETRY CAPSULE builder (cayley) -- assembles the
production `f2g-w2-bound-geometry-v2` capsule for certification from
its REGISTERED inputs, so 08-25 capsule production is mechanical.

Inputs (every one a registered artifact or an explicit argument --
nothing is synthesized here):
- the COMMITTED calendar authority (calendar_authority_w2_v4.json)
  supplies `calendar_frame` verbatim;
- the COMMITTED effect grids (effect_grids_w2_v1.json) supply
  `effect_grids` verbatim;
- `registries` / `segments` (frozen carrier registries; production
  values land with selection/staging);
- per-carrier `available_days` (the ANTICIPATED-mask envelope,
  drafted 08-24 from live telemetry) -- masks, never deletions;
- `calendar_authority_sha256` + `calendar_authority_ref`
  (the manifest pin of the authority artifact);
- `seed_authority_sha256` (REGISTERED choice -- routed to codex; the
  builder takes it as an argument and never defaults it);
- `loco_registry_carrier` (cascadia, the NEW registry).

The built capsule is validated through the REAL harness validator
(`w2_power_harness_cayley._validate_geometry_capsule`) for every
registered family/point -- never a private re-implementation -- and
the whole-capsule digest is computed by the harness's own function.
Opens no window-2 value; building a capsule authorizes nothing.

REV v4-cal: rebound to calendar_authority_w2_v4.json (codex Gate-2
2026-08-30T1844Z repair 2); no other behavior change.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_power_harness_cayley as PH


class CapsuleBuildRefusal(ValueError):
    pass


def build_capsule(*, calendar_authority, effect_grids_artifact,
                  registries, segments, available_days_by_carrier,
                  calendar_authority_sha256, calendar_authority_ref,
                  seed_authority_sha256, loco_registry_carrier,
                  input_refs, mode="bound"):
    """FIXTURE-ONLY assembler (codex 1507Z item 4: "keep any
    raw-file/caller-dict builder fixture-only").

    It takes caller dicts, so nothing it produces is admissible on
    its own: PRODUCTION capsules are built by
    `build_capsule_from_manifest`, which reopens every input from the
    admitted pins of a resolved manifest commit. This entry remains
    for KATs and doctors, and the loader still refuses any capsule
    whose input_refs are not admitted pins, so a capsule minted here
    cannot certify even if it is committed.

    Every carrier's engine-facing registered_days = the authority
    engine grid byte-for-byte (the non-compression contract);
    availability is the separate mask argument."""
    if calendar_authority.get("schema") != \
            "f2g-w2-calendar-authority-v4":
        raise CapsuleBuildRefusal(
            "CAPSULE_INPUT_INVALID: calendar authority schema")
    if effect_grids_artifact.get("schema") != \
            "f2g-w2-effect-grids-v1":
        raise CapsuleBuildRefusal(
            "CAPSULE_INPUT_INVALID: effect grids schema")
    frame = json.loads(json.dumps(calendar_authority["frame"]))
    grids = json.loads(json.dumps(
        effect_grids_artifact["grids"]))
    if sorted(registries) != sorted(segments) or \
            sorted(registries) != sorted(available_days_by_carrier):
        raise CapsuleBuildRefusal(
            "CAPSULE_INPUT_INVALID: carrier set mismatch across "
            "registries/segments/masks")
    eng = list(frame["engine_days"])
    masks = {ck: {"registered_days": list(eng),
                  "available_days":
                      [str(d) for d in available_days_by_carrier[ck]]}
             for ck in sorted(available_days_by_carrier)}
    cap = {"schema": PH.BOUND_GEOMETRY_SCHEMA, "bound": True,
           "calendar_authority_mode": str(mode),
           "calendar_authority_sha256":
               str(calendar_authority_sha256),
           "calendar_authority_ref": dict(calendar_authority_ref),
           "seed_authority_sha256": str(seed_authority_sha256),
           "calendar_frame": frame,
           "carrier_masks": masks,
           "registries": {ck: list(registries[ck])
                          for ck in sorted(registries)},
           "segments": {ck: dict(segments[ck])
                        for ck in sorted(segments)},
           "effect_grids": grids,
           "loco_registry_carrier": str(loco_registry_carrier),
           "input_refs": {k: dict(v)
                          for k, v in sorted(input_refs.items())}}
    cap["capsule_digest"] = PH._geometry_capsule_digest(cap)
    # validate through the REAL harness validator at EVERY registered
    # family/point (the closed schema, calendar frame, masks, LOCO
    # geometry, and grid membership all fire here). A fixture-mode
    # build validates a bound-mode STRUCTURAL COPY -- the emitted
    # capsule keeps its honest fixture label and can never certify.
    vcap = cap
    if cap["calendar_authority_mode"] != "bound":
        vcap = dict(cap, calendar_authority_mode="bound")
        vcap["capsule_digest"] = PH._geometry_capsule_digest(vcap)
    for fam, grid in grids.items():
        for point in grid:
            PH._validate_geometry_capsule(vcap, fam, point)
    return cap


# ------------------------------------------------------------------
# PRODUCTION path (codex 1507Z item 4)
# ------------------------------------------------------------------
INPUTS_BUNDLE_REL = ("docs/f2g_window2_execution/"
                     "power_geometry_inputs_w2_v1.json")
SEED_AUTHORITY_REL = ("docs/f2g_window2_execution/"
                      "power_seed_authority_w2_v1.json")
CALENDAR_AUTHORITY_REL = ("docs/f2g_window2_execution/"
                          "calendar_authority_w2_v4.json")
EFFECT_GRIDS_REL = ("docs/f2g_window2_execution/"
                    "effect_grids_w2_v1.json")
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"
REF_PATHS = {"inputs_bundle": INPUTS_BUNDLE_REL,
             "seed_authority": SEED_AUTHORITY_REL,
             "calendar_authority": CALENDAR_AUTHORITY_REL,
             "effect_grids": EFFECT_GRIDS_REL}


def _git_blob(repo, commit, rel):
    import subprocess
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise CapsuleBuildRefusal(
            f"CAPSULE_INPUT_UNREADABLE: {rel} at {str(commit)[:12]}")
    return p.stdout


def build_capsule_from_manifest(repo, manifest_commit):
    """THE production builder. Takes ONE resolved manifest commit and
    reopens every registered input from that manifest's ADMITTED pins
    -- no caller paths, no caller seed string, no free-form CLI.

    Each input's {commit, path, blob_sha256} is taken from the pin
    itself and carried in the capsule, so a named-target rebuild
    reproduces the committed capsule byte-for-byte and the loader can
    re-resolve every reference independently.
    """
    import hashlib as _h
    man = json.loads(
        _git_blob(repo, manifest_commit, MANIFEST_REL).decode("utf-8"))
    idx = {}
    for slot in man["slots"].values():
        if slot["status"] != "BOUND":
            continue
        for pin in slot["pins"]:
            idx[pin["path"]] = pin
    refs, raw = {}, {}
    for name, rel in sorted(REF_PATHS.items()):
        pin = idx.get(rel)
        if pin is None:
            raise CapsuleBuildRefusal(
                f"CAPSULE_INPUT_NOT_PINNED: {name} ({rel}) is not a "
                f"BOUND pin at {str(manifest_commit)[:12]} -- the "
                "production builder never reads an unadmitted input")
        body = _git_blob(repo, pin["commit"], pin["path"])
        got = _h.sha256(body).hexdigest()
        if got != pin["blob_sha256"]:
            raise CapsuleBuildRefusal(
                f"CAPSULE_INPUT_DIVERGENT: {name} bytes {got[:12]} "
                f"!= pin {pin['blob_sha256'][:12]}")
        refs[name] = {"commit": pin["commit"], "path": pin["path"],
                      "blob_sha256": got}
        raw[name] = body

    bundle = json.loads(raw["inputs_bundle"].decode("utf-8"))
    if bundle.get("schema") != "f2g-w2-power-geometry-inputs-v1":
        raise CapsuleBuildRefusal(
            "CAPSULE_INPUT_INVALID: geometry-inputs bundle schema")
    seed = json.loads(raw["seed_authority"].decode("utf-8"))
    if seed.get("schema") != "f2g-w2-power-seed-authority-v1":
        raise CapsuleBuildRefusal(
            "CAPSULE_INPUT_INVALID: seed-authority record schema")
    cal = json.loads(raw["calendar_authority"].decode("utf-8"))
    grids = json.loads(raw["effect_grids"].decode("utf-8"))
    return build_capsule(
        calendar_authority=cal,
        effect_grids_artifact=grids,
        registries=bundle["registries"],
        segments=bundle["segments"],
        available_days_by_carrier=bundle["carrier_masks"],
        calendar_authority_sha256=refs["calendar_authority"][
            "blob_sha256"],
        calendar_authority_ref={
            "commit": refs["calendar_authority"]["commit"],
            "path": refs["calendar_authority"]["path"]},
        seed_authority_sha256=seed["seed_authority_sha256"],
        loco_registry_carrier="cascadia",
        input_refs=refs,
        mode="bound")


def main():
    """PRODUCTION entry (codex 1507Z item 4): ONE resolved manifest
    commit. The former free-form CLI -- caller mask/registry/segment
    paths plus a caller seed string -- is REMOVED: it was exactly the
    unadmitted production entry the ruling closes.

    Usage: gen.py <repo> <manifest_commit> <out_path>
    """
    if len(sys.argv) != 4:
        raise SystemExit(
            "CAPSULE_PRODUCTION_USAGE: gen.py <repo> "
            "<manifest_commit> <out_path> -- geometry inputs are "
            "reopened from the admitted pins of that manifest, never "
            "supplied by the caller")
    repo, manifest_commit, out_p = sys.argv[1:4]
    cap = build_capsule_from_manifest(os.path.abspath(repo),
                                      manifest_commit)
    body = json.dumps(cap, indent=1, sort_keys=True) + "\n"
    with open(out_p, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    import hashlib as _h
    print(f"wrote {out_p}")
    print("capsule sha256:", _h.sha256(body.encode()).hexdigest())
    print("capsule_digest:", cap["capsule_digest"])


def _selftest():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    auth = json.load(open(os.path.join(
        repo, "docs", "f2g_window2_execution",
        "calendar_authority_w2_v4.json"), encoding="utf-8"))
    grids_art = json.load(open(os.path.join(
        repo, "docs", "f2g_window2_execution",
        "effect_grids_w2_v1.json"), encoding="utf-8"))
    eng = auth["frame"]["engine_days"]
    regs = {"istanbul_marmara": [f"IS{i}" for i in range(4)],
            "socal_coachella": [f"SC{i}" for i in range(4)],
            "turkey_kahramanmaras": [f"TK{i}" for i in range(4)],
            "cascadia": [f"CA{i}" for i in range(4)]}
    segs = {ck: {s: ("sA" if i < 2 else "sB")
                 for i, s in enumerate(regs[ck])} for ck in regs}
    masks = {ck: list(eng) for ck in regs}
    masks["cascadia"] = [d for d in eng if d not in set(eng[100:110])]

    kw = dict(calendar_authority=auth,
              effect_grids_artifact=grids_art,
              registries=regs, segments=segs,
              available_days_by_carrier=masks,
              calendar_authority_sha256="a" * 64,
              calendar_authority_ref={"commit": "x" * 40,
                                      "path": "docs/x.json"},
              seed_authority_sha256="b" * 64,
              loco_registry_carrier="cascadia",
              # fixture input_refs: shape-valid so the closed schema
              # is exercised, but they resolve to no admitted pin, so
              # the loader refuses this capsule for certification
              input_refs={n: {"commit": "f" * 40,
                              "path": f"docs/fixture/{n}.json",
                              "blob_sha256": "c" * 64}
                          for n in PH.GEOMETRY_INPUT_REFS},
              mode="fixture")
    cap = build_capsule(**kw)
    # the REAL validator accepted every registered family/point
    # (82 memberships); structural spot checks:
    assert len(cap["calendar_frame"]["engine_days"]) == 192
    assert cap["carrier_masks"]["cascadia"]["registered_days"] == \
        list(eng)                       # grid never compacted
    assert len(cap["carrier_masks"]["cascadia"]["available_days"]) \
        == 182
    assert sum(len(g) for g in cap["effect_grids"].values()) == 82
    # determinism
    assert build_capsule(**kw)["capsule_digest"] == \
        cap["capsule_digest"]
    # A stale predecessor authority must not be silently accepted.
    stale = json.loads(json.dumps(kw))
    stale["calendar_authority"]["schema"] = \
        "f2g-w2-calendar-authority-v2"
    try:
        build_capsule(**stale)
        raise AssertionError("stale v2 calendar authority must refuse")
    except CapsuleBuildRefusal:
        pass
    # doctors: carrier-set mismatch; availability on the PRESTART day
    import copy
    bad = copy.deepcopy(kw)
    del bad["available_days_by_carrier"]["cascadia"]
    try:
        build_capsule(**bad)
        raise AssertionError("carrier mismatch must refuse")
    except CapsuleBuildRefusal:
        pass
    bad2 = copy.deepcopy(kw)
    bad2["available_days_by_carrier"]["cascadia"] = \
        auth["frame"]["baseline_days"] + ["2026-09-03"] \
        + auth["frame"]["evaluation_days"]
    try:
        build_capsule(**bad2)
        raise AssertionError("PRESTART-day availability must refuse")
    except PH.PowerHarnessError as e:
        assert "CALENDAR_EXCLUDED_DAY" in str(e)
    # a doctored authority frame refuses through the REAL validator
    bad3 = copy.deepcopy(kw)
    bad3["calendar_authority"]["frame"]["engine_days"] = eng[:-1]
    try:
        build_capsule(**bad3)
        raise AssertionError("truncated frame must refuse")
    except PH.PowerHarnessError as e:
        assert "CALENDAR_AUTHORITY_MISMATCH" in str(e)
    print("w2_geometry_capsule_gen selftest: ALL PASS (fixture "
          "registries; the real harness validator is the gate)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
