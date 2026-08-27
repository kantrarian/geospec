#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRE-FIRE FRAME-READINESS DOCTOR (cayley).

WHAT THIS EXISTS FOR
--------------------
On 2026-08-27 the entire MAG_FEED/vic lane -- 212 authorized keys --
fired and was refused 212 times at the transform:

    ADMISSION_TRANSFORM_REFUSED: GIN MAG: source orientation 'XYZS' is
    not a registered REPORTED convention and no committed frame capsule
    exists at docs/f2g_window2_execution/mag_capsules/mag_capsule_vic.json

The refusal was correct. Gate 1 was also correct on what it ruled. The
gap between them is this doctor's whole reason to exist, and grassmann
named it exactly: gate 1 verified that the plan derives from a pinned
capsule and that the runner is bound -- it NEVER verified that every
lane in the plan can actually be TRANSFORMED. `may_fire` gates on
HTTP_CAPTURE membership, not on frame readiness.

So a key can be perfectly authorized to fire and impossible to admit,
and nothing in the pre-fire chain says so. This doctor closes that: for
every carrier in a plan, the frame authority must be resolvable FROM
COMMITTED BYTES BEFORE any request is made, or the plan refuses typed.

THE ONE DESIGN CONSTRAINT THAT MATTERS (grassmann, conceding their own
root-cause error and turning it into this requirement)
-----------------------------------------------------------------
Assert on the RESOLVED PATH READ -- never on "a capsule of that basename
exists somewhere in the repo".

That distinction is not pedantic; it is the entire failure. A committed
`mag_capsule_vic.json` DOES exist, at `docs/f2g_window2_freeze/`, and it
is the right object -- same schema, iaga_code VIC, sensor_orientation
XYZS, component_map present. It satisfies nothing, because
`_mag_frame_authority` reads exactly

    f"{EXEC_CAPSULE_DIR}/mag_capsule_{carrier}.json"

and that file is absent. A basename-anywhere check would have returned
READY and let all 212 keys fire -- i.e. it would have reproduced the bug
while reporting green. Control C2 below exists to make that permanently
impossible, and it is a live control, not a hypothetical: the real tree
is in exactly that state right now.

RESOLUTION ORDER
----------------
Mirrors `_mag_frame_authority`'s registered order, evaluated pre-fire:

  1. the carrier's pinned CHARACTERIZATION capsule reports an
     orientation in w2_mag1.REPORTED_CONVENTIONS   -> READY_BODY_REPORTED
  2. else a committed capsule exists AT THE RESOLVED EXECUTION PATH
                                                   -> READY_EXECUTION_CAPSULE
  3. else                                          -> FRAME_UNREADY

CHARACTERIZATION vs SATISFACTION -- these are different roles and the
code keeps them apart on purpose. The characterization spaces are
searched only to LEARN what the source reports. Finding a capsule there
NEVER satisfies readiness; only rule 1 (a registered reported
convention) or rule 2 (the resolved execution path) does.

CLAIM CEILING
-------------
This is a PRE-FIRE readiness check over committed bytes. It does not
fetch, does not transform, and does not certify that admission will
succeed -- only that the frame authority the transform will look for can
be resolved. A live body whose reported orientation diverges from its
pinned characterization is a separate divergence class this doctor
cannot see pre-fire, and it is NOT claimed to be covered.
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# the lane whose transform routes through _mag_frame_authority. Derived
# from admission_transform's dispatch, not guessed: MAG_WEATHER_FEED and
# the retired MF4_FEED take other routes and are out of scope here.
MAG_FRAME_LANE = "MAG_FEED"

# searched ONLY to learn the reported orientation -- see the note above.
CHARACTERIZATION_DIRS = (
    "docs/f2g_window2_execution/mag_capsules",
    "docs/f2g_window2_freeze",
)


class FrameReadinessRefusal(Exception):
    """Typed. The code leads the message."""


def _git_blob(commit, path):
    p = subprocess.run(
        ["git", "-C", REPO, "cat-file", "blob", f"{commit}:{path}"],
        capture_output=True)
    return None if p.returncode else p.stdout


def resolved_execution_capsule_path(carrier):
    """THE path the production transform reads.

    Imported from the production module rather than retyped, so this
    doctor cannot drift away from the read site it is defending. This is
    not vacuous self-agreement: the doctor tests whether a FILE EXISTS at
    that path, which the constant cannot make true.
    """
    import w2_acquisition_capture_grassmann as CAP
    return f"{CAP.EXEC_CAPSULE_DIR}/mag_capsule_{carrier}.json"


def _load_capsule(blob, path):
    raw = blob(path)
    if raw is None:
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


# the schema every committed mag input capsule carries (frn/izn/tuc at
# the execution tree and new/vic at the freeze tree, verified by read)
MAG_CAPSULE_SCHEMA = "f2g-mag-input-capsule-v1"


def _exec_capsule_transformable(cap, carrier):
    """codex 0551Z repair 5: parseable-dict presence is NOT frame
    readiness -- committed bytes `{}` at the resolved path previously
    reported READY_EXECUTION_CAPSULE while the production transform
    would refuse them, recreating the exact spend-without-admission
    defect this doctor exists to prevent.

    So readiness of an execution capsule is decided by THE TRANSFORM'S
    OWN CONTRACT: the same `w2_mag1.convert_frame` the production path
    calls, driven with deterministic one-sample arrays derived from
    the capsule's characterized source orientation. No parallel
    validator to drift from production.

    Returns None if transformable, else the refusal string."""
    import w2_mag1 as MAG1
    if not isinstance(cap, dict):
        return "capsule is not an object"
    if cap.get("schema") != MAG_CAPSULE_SCHEMA:
        return (f"schema {cap.get('schema')!r} is not "
                f"{MAG_CAPSULE_SCHEMA!r}")
    iaga = str(cap.get("iaga_code") or "")
    if iaga.lower() != carrier.lower():
        return (f"capsule iaga_code {iaga!r} does not name carrier "
                f"{carrier!r} -- a misplaced sibling capsule satisfies "
                "nothing")
    orient = (cap.get("sensor_orientation") if cap.get("component_map")
              else cap.get("reported_orientation"))
    # one deterministic sample per plausibly-consumed element;
    # convert_frame reads only what the registered conversion needs,
    # so extra keys are inert
    arrays = {e: [1.0] for e in (cap.get("recorded_elements") or ())}
    for e in ("X", "Y", "Z", "H", "D", "S", "F"):
        arrays.setdefault(e, [1.0])
    try:
        MAG1.convert_frame(cap, arrays, orient)
    except MAG1.Mag1Refusal as exc:
        return str(exc)
    except Exception as exc:  # malformed capsule shapes land here
        return f"{type(exc).__name__}: {exc}"
    return None


def characterization_for(blob, carrier):
    """The carrier's pinned characterization + where it was found.

    Returns (capsule, path) or (None, None). Finding one here does NOT
    make the carrier ready.
    """
    for d in CHARACTERIZATION_DIRS:
        path = f"{d}/mag_capsule_{carrier}.json"
        cap = _load_capsule(blob, path)
        if cap is None:
            continue
        # C6c finding (self-caught): a SIBLING capsule misplaced at
        # this carrier's path must not characterize it -- TUC bytes at
        # the vic path carry registered XYZF and would satisfy rule 1
        # for the wrong observatory. Production is protected by its
        # source-vs-capsule orientation cross-check at transform time;
        # this pre-fire doctor has no body to cross-check against, so
        # carrier identity is checked here instead.
        if str(cap.get("iaga_code") or "").lower() != carrier.lower():
            continue
        return cap, path
    return None, None


def frame_readiness(blob, carrier):
    """(state, detail) for one carrier. Fail-closed."""
    import w2_mag1 as MAG1

    exec_path = resolved_execution_capsule_path(carrier)
    char, char_path = characterization_for(blob, carrier)

    if char is not None:
        rep = char.get("reported_orientation")
        if rep in MAG1.REPORTED_CONVENTIONS:
            return ("READY_BODY_REPORTED",
                    f"{carrier}: reported_orientation {rep!r} is a "
                    f"registered convention (per {char_path})")

    exec_cap = _load_capsule(blob, exec_path)
    if exec_cap is not None:
        why_not = _exec_capsule_transformable(exec_cap, carrier)
        if why_not is None:
            return ("READY_EXECUTION_CAPSULE",
                    f"{carrier}: committed capsule at {exec_path} is "
                    "TRANSFORMABLE under the production contract")
        return ("FRAME_UNREADY",
                f"{carrier}: capsule at {exec_path} is present but NOT "
                f"transformable -- {why_not}")

    rep = char.get("reported_orientation") if char else None
    sen = char.get("sensor_orientation") if char else None
    why = (f"reported_orientation {rep!r} is not a registered "
           f"convention (sensor_orientation {sen!r})" if char
           else "no pinned characterization capsule found")
    elsewhere = ""
    if char_path and char_path != exec_path:
        elsewhere = (f"; a capsule for this carrier DOES exist at "
                     f"{char_path}, which satisfies NOTHING -- the "
                     f"transform reads {exec_path}")
    return ("FRAME_UNREADY",
            f"{carrier}: {why}; no committed capsule at "
            f"{exec_path}{elsewhere}")


def audit_plan(keys, *, blob=None, commit="HEAD"):
    """Classify every MAG_FEED carrier in a plan. Refuses typed if any
    carrier is unready. `keys` are 'LANE/carrier/day' strings."""
    if blob is None:
        def blob(path):
            return _git_blob(commit, path)
    carriers = sorted({k.split("/")[1] for k in keys
                       if k.split("/")[0] == MAG_FRAME_LANE})
    results, unready = {}, []
    for c in carriers:
        state, detail = frame_readiness(blob, c)
        results[c] = (state, detail)
        if state == "FRAME_UNREADY":
            n = sum(1 for k in keys
                    if k.startswith(f"{MAG_FRAME_LANE}/{c}/"))
            unready.append((c, n, detail))
    if unready:
        total = sum(n for _c, n, _d in unready)
        raise FrameReadinessRefusal(
            f"FRAME_UNREADY: {len(unready)} carrier(s) covering {total} "
            "authorized key(s) cannot resolve a frame authority; firing "
            "them would spend the ceiling on keys that CANNOT be "
            "admitted. "
            + " | ".join(d for _c, _n, d in unready))
    return results


# ------------------------------------------------------------------ #
# CONTROLS. Every one is mutation-checked or retrodictive; a doctor
# that only ever refuses would pass C1-C3, which is what C4 is for.
# ------------------------------------------------------------------ #
def _selftest(commit="803c931de01071add597588f269bb46d0dafc6a2"):
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    def real(path):
        return _git_blob(commit, path)

    # ---- C1 RETRODICTIVE: reproduce the real 2026-08-27 outcome from
    # committed bytes ALONE, before any request. This is the control the
    # absence of which cost 212 requests.
    plan = ([f"MAG_FEED/new/2026-01-{d:02d}" for d in range(1, 13)]
            + [f"MAG_FEED/vic/2026-01-{d:02d}" for d in range(1, 13)]
            + [f"MAG_WEATHER_FEED/omni/2026-01-{d:02d}"
               for d in range(1, 13)])
    s_new, _ = frame_readiness(real, "new")
    s_vic, _ = frame_readiness(real, "vic")
    check("C1 retrodictive: new READY_BODY_REPORTED, vic FRAME_UNREADY "
          "-- the doctor calls the real outcome pre-fire",
          s_new == "READY_BODY_REPORTED" and s_vic == "FRAME_UNREADY",
          f"new={s_new} vic={s_vic}")

    refused = None
    try:
        audit_plan(plan, blob=real)
    except FrameReadinessRefusal as exc:
        refused = str(exc)
    check("C1b the plan REFUSES and counts the doomed keys",
          refused is not None and "12 authorized key" in (refused or ""),
          f"refusal={str(refused)[:150]}")

    # ---- C2 PATH SENSITIVITY (grassmann's requirement). LIVE, not
    # hypothetical: mag_capsule_vic.json really does exist at the freeze
    # path right now. A basename-anywhere check returns READY here and
    # reproduces the bug while reporting green.
    char, char_path = characterization_for(real, "vic")
    exec_path = resolved_execution_capsule_path("vic")
    check("C2 path sensitivity: a real committed capsule at a NON-"
          "resolved path does NOT satisfy readiness",
          char is not None and char_path != exec_path
          and s_vic == "FRAME_UNREADY",
          f"found_at={char_path} resolved={exec_path} state={s_vic}")
    check("C2b the refusal NAMES the misplaced capsule rather than "
          "reporting it absent",
          "satisfies NOTHING" in frame_readiness(real, "vic")[1])

    # ---- C3 MUTATION on the READY path: doctor new's characterization
    # so its orientation is no longer registered; readiness must flip.
    def mutated(path):
        raw = real(path)
        if raw is not None and path.endswith("mag_capsule_new.json"):
            obj = json.loads(raw.decode("utf-8"))
            obj["reported_orientation"] = "NOT_A_REGISTERED_CONVENTION"
            return json.dumps(obj).encode("utf-8")
        return raw
    m_new, _ = frame_readiness(mutated, "new")
    check("C3 mutation: an unregistered reported_orientation flips new "
          "to FRAME_UNREADY (the check reads orientation, not presence)",
          m_new == "FRAME_UNREADY", f"new={m_new}")

    # ---- C4 ANTI-ALWAYS-REFUSE: place the vic capsule at the RESOLVED
    # path in a constructed tree; readiness must become READY. Without
    # this, a doctor hardcoded to refuse would pass C1-C3.
    vic_bytes = real("docs/f2g_window2_freeze/mag_capsule_vic.json")

    def repaired(path):
        if path == exec_path:
            return vic_bytes
        return real(path)
    r_vic, _ = frame_readiness(repaired, "vic")
    check("C4 anti-always-refuse: the SAME capsule at the resolved path "
          "yields READY_EXECUTION_CAPSULE (doctor accepts the real fix)",
          r_vic == "READY_EXECUTION_CAPSULE", f"vic={r_vic}")

    try:
        audit_plan(plan, blob=repaired)
        ok_pass = True
    except FrameReadinessRefusal:
        ok_pass = False
    check("C4b the repaired plan PASSES end to end", ok_pass)

    # ---- C6 TRANSFORMABILITY (codex 0551Z repair 5): presence is not
    # readiness. Every case places bytes at the RESOLVED path, so each
    # refusal is attributable to content, never to path resolution.
    def _at_exec(payload):
        def blob6(path):
            if path == exec_path:
                return payload
            return real(path)
        return blob6
    # (a) the `{}` mutation, FIRST -- codex's exact reproduction
    s6a, d6a = frame_readiness(_at_exec(b"{}"), "vic")
    check("C6a committed bytes {} at the resolved path are FRAME_"
          "UNREADY, not READY (presence is not transformability)",
          s6a == "FRAME_UNREADY" and "NOT transformable" in d6a,
          f"{s6a}: {d6a[:120]}")
    # (b) wrong schema
    _wrong_schema = json.loads(vic_bytes.decode("utf-8"))
    _wrong_schema["schema"] = "not-a-mag-capsule"
    s6b, _ = frame_readiness(_at_exec(
        json.dumps(_wrong_schema).encode()), "vic")
    check("C6b wrong-schema capsule refuses", s6b == "FRAME_UNREADY")
    # (c) wrong carrier: the REAL committed TUC capsule at the VIC path
    tuc_bytes = real("docs/f2g_window2_execution/mag_capsules/"
                     "mag_capsule_tuc.json")
    s6c, d6c = frame_readiness(_at_exec(tuc_bytes), "vic")
    check("C6c a real sibling capsule (TUC) at the VIC path refuses "
          "on carrier identity",
          s6c == "FRAME_UNREADY" and "iaga_code" in d6c,
          f"{s6c}: {d6c[:120]}")
    # (d) incomplete component map
    _inc = json.loads(vic_bytes.decode("utf-8"))
    _inc["component_map"] = {
        k: v for k, v in _inc["component_map"].items()
        if k != "geographic_X_north"}
    s6d, _ = frame_readiness(_at_exec(json.dumps(_inc).encode()),
                             "vic")
    check("C6d incomplete component_map refuses via the production "
          "contract", s6d == "FRAME_UNREADY")
    # (e) orientation mismatch / unregistered orientation
    _bad_o = json.loads(vic_bytes.decode("utf-8"))
    _bad_o["sensor_orientation"] = "QQQQ"
    s6e, _ = frame_readiness(_at_exec(json.dumps(_bad_o).encode()),
                             "vic")
    check("C6e unregistered sensor_orientation refuses",
          s6e == "FRAME_UNREADY")
    # (f) and the EXACT committed VIC capsule still passes (C4 already
    # binds this; restated here so the C6 family carries its own
    # positive and cannot be satisfied by an always-refuse validator)
    s6f, _ = frame_readiness(_at_exec(vic_bytes), "vic")
    check("C6f the exact frozen VIC capsule is TRANSFORMABLE and READY",
          s6f == "READY_EXECUTION_CAPSULE")

    # ---- C5 SCOPE: MAG_WEATHER_FEED is not a MAG-frame lane and must
    # never be audited into a false red.
    check("C5 scope: only MAG_FEED carriers are audited",
          "omni" not in audit_plan(
              [k for k in plan if not k.startswith("MAG_FEED/vic/")],
              blob=real))

    print()
    if fails:
        print(f"FRAME-READINESS DOCTOR FAILURES ({len(fails)}): {fails}")
        return 1
    print("FRAME-READINESS DOCTOR: ALL CONTROLS PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(_selftest())
    raise SystemExit(_selftest(sys.argv[1]))
