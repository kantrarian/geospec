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
import re
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
        # codex 1716Z P0-2 -- runtime dependencies of the admission
        # path, pinned nowhere until now. The sentinel is what makes
        # http_requests=0 a MEASURED claim; unpinned, that measurement
        # came from code the manifest did not bind.
        "monitoring/src/w2_no_network_grassmann.py",
        "monitoring/src/w2_producer_grassmann.py",
        # codex 0151Z P0-1: the capture executable itself.
        "monitoring/src/w2_capture_run_v4_grassmann.py",
        # codex 0413Z circular-pin repair: the predecessor plan uses
        # the registered transform before producer_boundary can bind.
        # Its operation role must therefore be placed in the always-
        # BOUND accrual authority; membership in some other slot is
        # insufficient.
        "monitoring/src/w2_acquisition_capture_grassmann.py",
        # codex 0532Z: two PRODUCTION-OPERATION surfaces. The VIC
        # repair module performs a zero-HTTP production operation and
        # emits an operation record, so it is operation bytes and NOT
        # producer_boundary. The predecessor bridge is already
        # committed production-operation code that was pinned NOWHERE
        # -- it stages the one non-capture key, so an unbound bridge
        # would let the 2,056th key arrive through code the manifest
        # never bound.
        "monitoring/src/w2_capture_repair_v4_vic_cayley.py",
        "monitoring/src/w2_predecessor_bridge_cayley.py",
        # codex 1345Z: the one-shot 404 retry -- network-spending, so
        # it must be a required surface the closure walks, or an
        # edited copy could fire outside the pin discipline.
        "monitoring/src/w2_capture_retry_404_v4_cayley.py",
        # codex 2240Z P0-2: the zero-HTTP finalizer joins the
        # required surfaces and the entrypoint closure
        "monitoring/src/w2_zero_http_finalizer_cayley.py",
    ),
    # codex 1758Z P0-1: both power engines together in power_harness.
    # They are fixture-only power-estimation engines behind the power
    # machinery; `_cal_` is the CALENDAR lane (not the
    # calibration-ledger runner), and it IMPORTS the non-calendar
    # engine -- so calibration_runner would be the wrong authority and
    # a split would separate a dependency from its dependent.
    "power_harness": (
        "monitoring/src/f2g_phase_b_power_estimation_cayley.py",
        "monitoring/src/f2g_phase_b_power_estimation_cal_cayley.py",
    ),
    "execution_verifier": (
        "monitoring/src/f2g_execution_manifest_verifier_cayley.py",
        "monitoring/src/w2_regeneration_gate_cayley.py",
        # codex 0532Z: the frame-readiness doctor is the COMMON
        # verification/enforcement surface behind the Gate-1 readiness
        # claim -- the runner and RG-10 import the same one pinned
        # verifier. It is deliberately NOT accrual_impl: it verifies,
        # it does not perform a production operation, and collapsing
        # the two would let verification authority answer to operation
        # bytes.
        "monitoring/src/w2_frame_readiness_doctor_cayley.py",
    ),
    # codex 0532Z: an EXACT four-capsule lock. The 2026-08-27 VIC
    # failure was a carrier-class omission -- IZN/FRN/TUC were
    # relocated into the execution tree and VIC was not, so a 212-key
    # lane was authorized to fire and impossible to admit. A membership
    # check over "some capsules" cannot catch a missing carrier; only
    # an exact required set can.
    "mag_capsules": (
        "docs/f2g_window2_execution/mag_capsules/mag_capsule_frn.json",
        "docs/f2g_window2_execution/mag_capsules/mag_capsule_izn.json",
        "docs/f2g_window2_execution/mag_capsules/mag_capsule_tuc.json",
        "docs/f2g_window2_execution/mag_capsules/mag_capsule_vic.json",
    ),
    "calibration_runner": (
        "monitoring/src/w2_calibration_runner_cayley.py",
        "monitoring/src/w2_calibration_feed_producer_cayley.py",
        "monitoring/src/w2_mf4_catalog_adapter_grassmann.py",
        "monitoring/src/w2_mf4_archive_capsule_gen_grassmann.py",
        "docs/f2g_window2_execution/mf4_archive_capsule_v1.json",
        "docs/f2g_window2_execution/mf4_archive/"
        "daily_risk_rows_v1.jsonl",
        "docs/f2g_window2_execution/mf4_catalog_snapshot/"
        "catalog_snapshot_v1.json",
        "docs/f2g_window2_execution/mf4_catalog_snapshot/"
        "acquisition_receipt_v1.json",
    ),
    # only once the slot is BOUND -- it is honestly OPEN until the
    # post-capture bind, and requiring it earlier would demand a pin
    # the design says must not exist yet
    "producer_boundary": (
        "monitoring/src/w2_acquisition_capture_grassmann.py",
    ),
}
SLOT_REQUIRED_ONLY_WHEN_BOUND = ("producer_boundary",)


# codex 1716Z P0-2: the admission/verification ENTRYPOINTS whose
# transitive local imports must all be bound somewhere. Binding two
# named modules fixes two instances; this closes the CLASS, which is
# what stops the next helper arriving unbound and unnoticed.
ADMISSION_ENTRYPOINTS = (
    "w2_accrual_instrument_cayley.py",
    "w2_restage_verify_batch_grassmann.py",
    "w2_verification_run_summary_grassmann.py",
    "w2_disposition_capsule_grassmann.py",
    "w2_restage_lineage_grassmann.py",
    "w2_acquisition_capture_grassmann.py",
    "w2_regeneration_gate_cayley.py",
    "f2g_execution_manifest_verifier_cayley.py",
    # codex 0151Z P0-1: the runner is an admission-path root,
    # so its own imports join the dependency closure too.
    "w2_capture_run_v4_grassmann.py",
    # P0-4: the calibration lane's admission-path roots -- their
    # transitive local imports must arrive bound, and the ComCat
    # acquisition is a network-spending root like the retry
    "w2_calibration_runner_cayley.py",
    "w2_calibration_feed_producer_cayley.py",
    "w2_mf4_catalog_adapter_grassmann.py",
    "w2_mf4_catalog_acquire_grassmann.py",
    # codex 0532Z: both production-operation roots join the closure.
    # The VIC repair driver reaches the store, the transform and the
    # producer gate; the predecessor bridge stages the one key capture
    # never fires. Rooting them here is what makes their own helpers
    # arrive bound rather than unnoticed.
    "w2_capture_repair_v4_vic_cayley.py",
    "w2_predecessor_bridge_cayley.py",
    # codex 1345Z: the retry one-shot is an admission-path root -- it
    # publishes ordinary staged classes on success -- so its imports
    # join the dependency closure like the runner's.
    "w2_capture_retry_404_v4_cayley.py",
    "w2_zero_http_finalizer_cayley.py",
)

# The linked DESIGN-PIN set. codex's wording is "escape both an
# execution slot AND the linked design-pin set" -- a module carried by
# the byte-pin manifest is bound, just by the other registry, and
# reporting it as unbound would be a false finding. Checking only
# execution slots would have over-reported by two.
DESIGN_PIN_REGISTRY = "docs/f2g_window2_freeze/byte_pin_manifest.json"


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


def local_import_closure(src_dir, entrypoints=ADMISSION_ENTRYPOINTS):
    """Transitive closure of LOCAL module imports from the entrypoints.

    HONEST BOUND on what this sees: static `import`/`from` statements
    resolving to a .py beside the entrypoint. It does NOT see a module
    imported under a computed name, nor one invoked purely as a
    subprocess argv built at runtime. So it is a floor on the
    dependency set, not a proof of completeness -- said plainly here
    because an unbounded claim is the defect this gate exists to
    catch.
    """
    import ast
    seen, stack = set(), [e for e in entrypoints]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        path = os.path.join(src_dir, f)
        if not os.path.isfile(path):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:                                 # noqa: BLE001
            continue
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                names = [n.module]
            for nm in names:
                cand = nm + ".py"
                if os.path.isfile(os.path.join(src_dir, cand)) and \
                        cand not in seen:
                    stack.append(cand)
        # codex 1716Z said "imported OR SUBPROCESS-INVOKED". No
        # subprocess-launched .py helper exists in this tree today, so
        # scanning imports alone happens to be adequate -- which is
        # exactly why this needed writing. A check that is correct
        # because the tree is currently simple stops being correct
        # silently, and the requirement names the MECHANISM, not
        # today's instances.
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fname = getattr(n.func, "attr",
                            getattr(n.func, "id", ""))
            if fname not in ("run", "Popen", "check_output",
                             "call", "check_call"):
                continue
            for a in ast.walk(n):
                if isinstance(a, ast.Constant) and \
                        isinstance(a.value, str) and \
                        a.value.endswith(".py"):
                    cand = os.path.basename(a.value)
                    if os.path.isfile(os.path.join(src_dir, cand)) \
                            and cand not in seen:
                        stack.append(cand)
    return seen


def design_pinned_paths(manifest, *, commit=None):
    """FULL repo-relative paths carried by the linked design registry.

    codex 1758Z P1. My first version was a NOMINAL binding twice over
    and both halves were real defects:

    1. it regexed every `.py` token anywhere in the JSON, so
       `{"note": "monitoring/src/ghost_unbound.py", "pins": {}}`
       reported ghost_unbound.py as design-pinned when the registry
       pins nothing at all -- prose in a note counted as a pin;
    2. it compared BASENAMES, so a same-named file in another
       directory would inherit an unrelated pin.

    Both are the defect class I have spent this session finding in
    other people's checks: a check that matches the NAME of a thing
    instead of binding the thing. Now: parse strictly, take `path`
    only from `pins` entries, compare full paths, and REFUSE a
    malformed registry rather than returning a partial authority --
    an empty set read as "nothing is design-pinned" would silently
    convert a broken registry into a flood of false findings.

    Resolved at the manifest's own `design_manifest_commit`, not at
    symbolic HEAD: the linked registry is whichever one the manifest
    names, and reading HEAD would let a later edit satisfy an earlier
    manifest.
    """
    ref = commit or manifest.get("design_manifest_commit")
    if not isinstance(ref, str) or not ref:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: the manifest names no "
            "design_manifest_commit, so the linked design registry "
            "cannot be resolved -- refusing rather than reading HEAD")
    raw = _blob(ref, DESIGN_PIN_REGISTRY)
    if raw is None:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: the linked design registry "
            f"{DESIGN_PIN_REGISTRY} is not readable at "
            f"{ref[:12]}")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: the linked design registry is "
            f"not parseable JSON ({type(e).__name__})")
    pins = doc.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: the linked design registry "
            "has no `pins` mapping; a malformed registry must refuse, "
            "never be read as 'nothing is design-pinned'")
    out = set()
    for name, entry in pins.items():
        if not isinstance(entry, dict):
            raise RegenerationGateRefusal(
                "REGENERATION_GATE_REFUSED: design pin "
                f"{name!r} is not a record")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise RegenerationGateRefusal(
                "REGENERATION_GATE_REFUSED: design pin "
                f"{name!r} carries no path")
        out.add(path)
    return out


def _generator_declared_paths():
    """Repo-relative paths the generator DECLARES -- i.e. what the
    manifest will bind at the next regeneration. Full paths, not
    basenames (codex 1758Z P1)."""
    try:
        import f2g_execution_manifest_gen_cayley as GEN
    except Exception:                                     # noqa: BLE001
        return set()
    return {p
            for d in getattr(GEN, "BOUND_SLOTS", {}).values()
            for p in d.get("paths", [])}


def unbound_closure_members(manifest, *, src_dir=_HERE,
                            design_commit=None):
    """Split closure members by WHY they are not pinned.

    Compares FULL repo-relative paths (codex 1758Z P1): a basename
    comparison would let a same-named file in another directory
    inherit an unrelated pin.

    Distinguishing never_declared from pending_regeneration is the
    whole point, and the distinction is PHASE-NEUTRAL:
    generator-declared but absent from the CURRENT manifest is a
    pending manifest refresh, not a no-registry gap. Conflating them
    cries wolf on healthy modules and buries the ones that are
    genuinely declared NOWHERE.
    """
    closure = local_import_closure(src_dir)
    rel_of = {f: "monitoring/src/" + f for f in closure}
    pinned = {p["path"] for _, p in walk_pins(manifest)}
    pinned |= design_pinned_paths(manifest, commit=design_commit)
    for slot_name in SLOT_REQUIRED_ONLY_WHEN_BOUND:
        for rel in REQUIRED_BY_SLOT.get(slot_name, ()):
            pinned.add(rel)
    declared = _generator_declared_paths()
    never, pending = [], []
    for f in sorted(closure):
        rel = rel_of[f]
        if rel in pinned:
            continue
        (pending if rel in declared else never).append(rel)
    return {"never_declared": never,
            "pending_regeneration": pending}


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


def dual_result_gate(manifest, *, manifest_commit="HEAD",
                     blob=_blob_at_head):
    """THE THREE-RESULT INSTRUMENT (codex 1758Z, adopting my 0458Z
    option 2).

    The tension this resolves: requiring a zero-OPEN prestart PASS
    before judging pins made pin-currency unreportable at step 4,
    because producer_boundary and calibration_ledgers are HONESTLY
    OPEN until the post-capture bind. Dropping the precondition would
    have let perfectly current pins on a structurally broken manifest
    earn a green gate. Neither collapse is acceptable, so the
    instrument reports three explicit results and short-circuits none
    of them.

      manifest_default_contract  PASS|REFUSE
      pin_currency               PASS|REFUSE|NOT_EVALUATED_INVALID_MANIFEST
      prestart_overall           PASS|REFUSE

    Pin currency may read PASS only after the DEFAULT-mode verifier
    PASS has established schema, linkage, slot coherence and every
    existing BOUND pin -- otherwise currency would be measured against
    a manifest not known to be well formed, and it is reported
    NOT_EVALUATED_INVALID_MANIFEST rather than REFUSE, because "we did
    not look" is not "we looked and it was wrong".

    `prestart_overall` stays REFUSE while any slot is honestly OPEN.
    **No caller may read pin_currency=PASS as an overall PASS**; that
    collapse is the whole failure this design exists to prevent, and
    the returned record keeps them as separate typed fields rather
    than one verdict.
    """
    import f2g_execution_manifest_verifier_cayley as EMV
    out = {"claim_scope": "PIN_BINDING_ONLY",
           "authorizes": "NOTHING",
           "manifest_commit": manifest_commit}

    dflt = EMV.verify(REPO, manifest_commit)
    out["manifest_default_contract"] = (
        "PASS" if dflt.get("verdict") == "PASS" else "REFUSE")
    out["manifest_default_detail"] = dflt.get("verdict")

    if out["manifest_default_contract"] != "PASS":
        out["pin_currency"] = "NOT_EVALUATED_INVALID_MANIFEST"
        out["pin_currency_detail"] = (
            "the default-mode manifest contract did not PASS, so pin "
            "currency was never measured")
    else:
        try:
            rep = audit(manifest, blob=blob)
            clo = unbound_closure_members(manifest)
            problems = []
            if rep["stale"]:
                problems.append(f"{len(rep['stale'])} stale")
            if rep["missing"]:
                problems.append(f"{len(rep['missing'])} missing")
            if rep["unbound_surfaces"]:
                problems.append(
                    f"{len(rep['unbound_surfaces'])} unbound required")
            if rep["misplaced"]:
                problems.append(f"{len(rep['misplaced'])} misplaced")
            if clo["never_declared"]:
                problems.append(
                    f"{len(clo['never_declared'])} closure "
                    "dependenc(ies) in no registry")
            out["pin_currency"] = "REFUSE" if problems else "PASS"
            out["pin_currency_detail"] = "; ".join(problems) or "clean"
            out["pin_audit"] = {
                "match": rep["match"], "stale": len(rep["stale"]),
                "missing": len(rep["missing"]),
                "unbound_required": len(rep["unbound_surfaces"]),
                "misplaced": len(rep["misplaced"]),
                "closure_never_declared": clo["never_declared"],
                "closure_pending": len(clo["pending_regeneration"])}
        except RegenerationGateRefusal as e:
            out["pin_currency"] = "REFUSE"
            out["pin_currency_detail"] = str(e)[:200]

    pre = EMV.verify(REPO, manifest_commit, prestart=True)
    zero_open = (pre.get("verdict") == "PASS"
                 and pre.get("slots_open", -1) == 0)
    out["prestart_overall"] = "PASS" if zero_open else "REFUSE"
    out["prestart_detail"] = (
        f"verdict={pre.get('verdict')}, "
        f"slots_open={pre.get('slots_open')}")
    if out["prestart_overall"] == "PASS" and             out["pin_currency"] != "PASS":
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: prestart_overall PASS with "
            f"pin_currency {out['pin_currency']} -- an overall pass "
            "may never outrank the pin audit beneath it")
    return out


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
    clo = unbound_closure_members(manifest)
    if clo["never_declared"]:
        problems.append(
            f"{len(clo['never_declared'])} admission-path "
            "dependenc(ies) declared in NO registry: "
            + ", ".join(clo["never_declared"][:6])
            + " -- imported by the admission/verification entrypoints "
              "yet bound by neither an execution slot nor the design "
              "pins, so a governed claim would rest on code the "
              "manifest never bound")
    if problems:
        raise RegenerationGateRefusal(
            "REGENERATION_GATE_REFUSED: " + "; ".join(problems))
    rep["closure"] = clo
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
        print("  RG-6 SKIP  no currently pinned accrual_impl "
              "required path exists to move")

    # RG-7 (codex 1716Z P0-2): DEPENDENCY CLOSURE. Binding two named
    # modules fixes two instances; this closes the class.
    #
    # SENSITIVITY DOCTOR, because a classifier that never moves proves
    # nothing: drop one DECLARED module from the generator's view and
    # the same module must flip pending -> never_declared. (My first
    # comment here claimed an injected fake entrypoint, which I had
    # not written. Correcting the claim rather than leaving prose
    # ahead of the code.)
    # codex 2118Z P2: this must ALWAYS run. It previously fired only
    # while `pending_regeneration` was non-empty, so a clean
    # post-regeneration tree SKIPPED it -- a sensitivity doctor that
    # stops exercising exactly when the system is healthy proves
    # nothing about the state it is meant to police. Precondition is
    # now CONSTRUCTED from a deep copy rather than borrowed.
    _base = unbound_closure_members(man)
    _declared = _generator_declared_paths()
    _closure_rel = {"monitoring/src/" + f
                    for f in local_import_closure(_HERE)}
    _exempt = set(design_pinned_paths(man))
    for _sn in SLOT_REQUIRED_ONLY_WHEN_BOUND:
        _exempt |= set(REQUIRED_BY_SLOT.get(_sn, ()))
    _cands = sorted((_closure_rel & _declared) - _exempt)
    if not _cands:
        raise RegenerationGateRefusal(
            "RG-7a NO_CANDIDATE: no closure path is generator-declared "
            "and unexempt, so the sensitivity doctor cannot construct "
            "its own precondition")
    _victim = _cands[0]
    _doc7 = copy.deepcopy(man)
    for _sl in _doc7.get("slots", {}).values():
        if isinstance(_sl, dict):
            _sl["pins"] = [p for p in _sl.get("pins", ())
                           if not (isinstance(p, dict)
                                   and p.get("path") == _victim)]
    _step1 = unbound_closure_members(_doc7)
    if _victim not in _step1["pending_regeneration"]:
        raise RegenerationGateRefusal(
            f"RG-7a UNPINNED_NOT_PENDING: {_victim} was unpinned but "
            "did not appear as pending_regeneration")
    _real = globals()["_generator_declared_paths"]
    try:
        globals()["_generator_declared_paths"] = (
            lambda _v=_victim, _r=_real: {x for x in _r() if x != _v})
        _step2 = unbound_closure_members(_doc7)
    finally:
        globals()["_generator_declared_paths"] = _real
    if _victim not in _step2["never_declared"] or \
            _victim in _step2["pending_regeneration"]:
        raise RegenerationGateRefusal(
            "RG-7a CLASSIFIER_INSENSITIVE: undeclaring "
            f"{_victim} did not move it pending -> never_declared, so "
            "the split does not track declaration at all")
    print(f"  RG-7a PASS  sensitivity CONSTRUCTED on a copy: "
          f"{os.path.basename(_victim)} unpinned -> pending, then "
          f"undeclared -> NO-REGISTRY")
    _clo = _base
    print(f"  RG-7 PASS  closure walked: "
          f"{len(_clo['never_declared'])} declared in NO registry, "
          f"{len(_clo['pending_regeneration'])} generator-declared "
          f"but absent from the current manifest (the two are NOT the "
          f"same defect)")
    for _f in _clo["never_declared"]:
        print(f"    NO-REGISTRY {_f}")

    # ---- codex 1758Z P1 doctors: the linked design registry ------
    # Each of these PASSED under my old regex-any-.py-token version,
    # which is why they exist.
    import copy as _copy

    def _reg_refuses(doc, label, want="REFUSE"):
        _real = globals()["_blob"]
        try:
            globals()["_blob"] = (
                lambda c, r, _d=doc, _r=_real:
                json.dumps(_d).encode() if r == DESIGN_PIN_REGISTRY
                else _r(c, r))
            try:
                got = design_pinned_paths(
                    {"design_manifest_commit": "a" * 40})
                return ("ACCEPTED", got)
            except RegenerationGateRefusal as e:
                return ("REFUSED", str(e)[:70])
        finally:
            globals()["_blob"] = _real

    # (a) a path mentioned only in PROSE is not a pin
    st, got = _reg_refuses(
        {"note": "monitoring/src/ghost_unbound.py", "pins": {}}, "note")
    if st == "ACCEPTED" and got:
        raise RegenerationGateRefusal(
            "RG-8a NOTE_READ_AS_PIN: prose in the registry counted as "
            f"a design pin ({sorted(got)[:2]})")
    print("  RG-8a PASS  a path named only in a note is NOT a pin "
          f"({st.lower()})")

    # (b) a real pin exposes its FULL path, so a same-named file in
    #     another directory cannot inherit it
    st, got = _reg_refuses(
        {"pins": {"x": {"path": "somewhere/else/w2_producer_grassmann.py"}}},
        "basename")
    same_basename_leaked = (
        st == "ACCEPTED"
        and "monitoring/src/w2_producer_grassmann.py" in got)
    if same_basename_leaked:
        raise RegenerationGateRefusal(
            "RG-8b BASENAME_COLLISION: a same-named file in another "
            "directory inherited an unrelated design pin")
    print("  RG-8b PASS  pins compare as FULL paths; a same basename "
          "elsewhere inherits nothing")

    # (c) a malformed registry REFUSES rather than reading as
    #     "nothing is design-pinned" (which would flood false findings)
    for bad, lab in (({"pins": {}}, "empty pins"),
                     ({"pins": "not-a-map"}, "pins not a map"),
                     ({"pins": {"x": {"no_path": 1}}}, "pin without path"),
                     ({}, "no pins key")):
        st, _d = _reg_refuses(bad, lab)
        if st != "REFUSED":
            raise RegenerationGateRefusal(
                f"RG-8c MALFORMED_REGISTRY_ACCEPTED: {lab}")
    print("  RG-8c PASS  a malformed registry REFUSES (never reads as "
          "'nothing is design-pinned')")

    # RG-7b (codex 1953Z): the SUBPROCESS EDGE, committed as a
    # doctor rather than verified once by hand.
    #
    # I checked this branch in a temp tree and reported the
    # result. That is execution EVIDENCE, not a LOCK. codex's
    # reproduction is the whole point: the OLD import-only
    # implementation at 65988bcd passes the identical committed
    # RG-0..RG-9 suite, so nothing in the suite bites if
    # subprocess discovery is deleted again. A branch I tested
    # once is not a branch the bar defends.
    #
    # The entrypoint deliberately carries NO import edge to the
    # helper -- only a literal subprocess argv -- so this can
    # only pass through the subprocess detector.
    import tempfile as _tempfile
    from pathlib import Path as _Path

    with _tempfile.TemporaryDirectory() as _td:
        _root = _Path(_td)
        (_root / "entry.py").write_text(
            "import subprocess\n"
            "subprocess.run(['python', 'spawned_helper.py'])\n",
            encoding="utf-8")
        (_root / "spawned_helper.py").write_text(
            "VALUE = 1\n", encoding="utf-8")
        _got = local_import_closure(
            str(_root), entrypoints=("entry.py",))
        if "spawned_helper.py" not in _got:
            raise RegenerationGateRefusal(
                "RG-7b SUBPROCESS_EDGE_ESCAPED: a subprocess-only "
                "local helper was absent from dependency closure")
    print("  RG-7b PASS  subprocess-only local helper joins closure")

    # (d) HEAD-DRIFT: a pin added to the registry at HEAD, AFTER the
    #     commit the manifest links, must NOT count. This is the
    #     doctor for resolving at design_manifest_commit rather than
    #     symbolic HEAD -- without it, a later edit to the registry
    #     could satisfy an earlier manifest retroactively, which is
    #     provenance running backwards.
    _LINKED = "b" * 40
    _real_blob = globals()["_blob"]
    try:
        def _drifting(c, r, _r=_real_blob):
            if r != DESIGN_PIN_REGISTRY:
                return _r(c, r)
            at_linked = {"pins": {"real": {
                "path": "docs/f2g_window2_freeze/annex_b1b.md"}}}
            at_head = {"pins": {
                "real": {"path": "docs/f2g_window2_freeze/annex_b1b.md"},
                "added_later": {
                    "path": "monitoring/src/drifted_in_at_head.py"}}}
            return json.dumps(
                at_linked if c == _LINKED else at_head).encode()
        globals()["_blob"] = _drifting
        _paths = design_pinned_paths(
            {"design_manifest_commit": _LINKED})
        if "monitoring/src/drifted_in_at_head.py" in _paths:
            raise RegenerationGateRefusal(
                "RG-8d HEAD_DRIFT_ADMITTED: a pin added to the "
                "registry at HEAD satisfied a manifest linked to an "
                "EARLIER commit -- provenance running backwards")
        if "docs/f2g_window2_freeze/annex_b1b.md" not in _paths:
            raise RegenerationGateRefusal(
                "RG-8d resolved the wrong commit entirely")
    finally:
        globals()["_blob"] = _real_blob
    print("  RG-8d PASS  a pin added at HEAD after the linked design "
          "commit does NOT count (resolution is at "
          "design_manifest_commit, not HEAD)")

    # ---- RG-9: the COLLAPSE GUARD, with BOTH controls -------------
    # codex 2118Z P1. My previous version mocked a zero-OPEN prestart
    # PASS and then unconditionally demanded a refusal. That was only
    # ever right while pins happened to be stale: once the manifest
    # regeneration made them current, the mocked PASS produced a
    # LEGITIMATE all-PASS and my doctor raised a false red on correct
    # state. It borrowed ambient staleness instead of constructing the
    # contradiction, so it decayed the moment the system got healthy.
    #
    # Now both controls run under the SAME installed mock:
    #   positive -- current pins must be accepted as a real all-PASS;
    #   negative -- a doctored pin must force pin_currency REFUSE and
    #               trip the typed guard.
    import f2g_execution_manifest_verifier_cayley as _EMV
    _rv = _EMV.verify
    try:
        _EMV.verify = (lambda *a, **k: {"verdict": "PASS",
                                        "slots_open": 0,
                                        "mode": "prestart",
                                        "pins_checked": 1})
        # (+) legitimate all-PASS is ACCEPTED, not mistaken for a
        #     collapse. codex 2145Z: this must CONSTRUCT its currency
        #     rather than read ambient `man`. Reading ambient pins made
        #     the control circular -- it could only pass when the tree
        #     already happened to be current, and the replacement that
        #     would make it current is blocked until this very test
        #     passes.
        _posdoc = copy.deepcopy(man)
        _refreshed = 0
        for _t, _pin in walk_pins(_posdoc):
            _cur = _blob_at_head(None, _pin["path"])
            if _cur is None:
                raise RegenerationGateRefusal(
                    "RG-9 POSITIVE_UNREADABLE: cannot read "
                    f"{_pin['path']} at HEAD to construct the positive "
                    "control")
            _pin["blob_sha256"] = hashlib.sha256(_cur).hexdigest()
            _refreshed += 1
        # COMPLETENESS, not only currency. Refreshing existing pins
        # cannot satisfy a surface that is REQUIRED but not yet
        # pinned -- which is exactly the state right after a new
        # required surface is declared and before the manifest
        # replacement binds it. Without this the positive control
        # constructs half its precondition and inherits the other
        # half from whatever the tree happens to contain, which is
        # the defect this doctor was rewritten to stop having.
        _added = 0
        _have = {p["path"] for _t, p in walk_pins(_posdoc)}
        for _slot_name, _required in REQUIRED_BY_SLOT.items():
            if _slot_name in SLOT_REQUIRED_ONLY_WHEN_BOUND:
                continue
            _slot = _posdoc.setdefault("slots", {}).setdefault(
                _slot_name, {"status": "BOUND", "pins": []})
            for _rel in _required:
                if _rel in _have:
                    continue
                _cur = _blob_at_head(None, _rel)
                if _cur is None:
                    raise RegenerationGateRefusal(
                        "RG-9 POSITIVE_UNREADABLE: required surface "
                        f"{_rel} is not readable at HEAD")
                _slot.setdefault("pins", []).append(
                    {"path": _rel, "commit": "HEAD",
                     "blob_sha256": hashlib.sha256(_cur).hexdigest()})
                _have.add(_rel)
                _added += 1
        if _refreshed < 1:
            raise RegenerationGateRefusal(
                "RG-9 POSITIVE_NO_PINS: no pin was refreshed, so the "
                "positive control would be vacuous")
        _pos = dual_result_gate(_posdoc, manifest_commit="HEAD")
        if not all(_pos[k] == "PASS" for k in
                   ("manifest_default_contract", "pin_currency",
                    "prestart_overall")):
            raise RegenerationGateRefusal(
                "RG-9 POSITIVE_CONTROL_FAILED: a CONSTRUCTED "
                f"all-current manifest ({_refreshed} pins refreshed) "
                "was not accepted as all-PASS "
                f"({_pos.get('pin_currency')}/"
                f"{_pos.get('prestart_overall')})")
        # (-) doctor ONE real pin so pin_currency must REFUSE while
        #     the mock still forces a zero-OPEN prestart PASS
        _doc9 = copy.deepcopy(_posdoc)
        _tampered = False
        for _t, _pin in walk_pins(_doc9):
            _pin["blob_sha256"] = "0" * 64
            _tampered = True
            break
        if not _tampered:
            raise RegenerationGateRefusal(
                "RG-9 NO_PIN_TO_DOCTOR: cannot construct the negative "
                "control")
        try:
            _bad = dual_result_gate(_doc9, manifest_commit="HEAD")
            raise RegenerationGateRefusal(
                "RG-9 COLLAPSE_ADMITTED: prestart_overall="
                f"{_bad.get('prestart_overall')} was returned with "
                f"pin_currency={_bad.get('pin_currency')} -- an "
                "overall pass outranked the pin audit beneath it")
        except RegenerationGateRefusal as e:
            if "may never outrank" not in str(e):
                raise
    finally:
        _EMV.verify = _rv
    print(f"  RG-9 PASS  collapse guard, BOTH controls CONSTRUCTED "
          f"({_refreshed} refreshed + {_added} required pins added): "
          "an all-current, all-complete manifest is "
          "accepted as all-PASS, and doctoring one of those pins under "
          "a forced zero-OPEN prestart PASS is refused")

    # ---- RG-10 (codex 0510Z P1): frame readiness is a GATE concern,
    # not only a runner concern. On 2026-08-27 a 212-key lane was
    # perfectly authorized to fire and impossible to admit, because
    # may_fire gates on HTTP_CAPTURE membership and nothing checked that
    # the transform could resolve a frame. Runner-only enforcement would
    # leave the gate's readiness claim blind to exactly that.
    #
    # BOTH controls are CONSTRUCTED here rather than read from ambient
    # state -- a doctor that only fires while the tree happens to be
    # broken stops proving anything the moment it is repaired.
    import w2_frame_readiness_doctor_cayley as _FRD
    _vic_src = "docs/f2g_window2_freeze/mag_capsule_vic.json"
    _vic_bytes = _blob_at_head(None, _vic_src)
    if not _vic_bytes:
        raise RegenerationGateRefusal(
            f"FRAME_DOCTOR_FIXTURE_ABSENT: {_vic_src} unreadable at "
            "HEAD; RG-10 cannot construct its controls")
    _exec_rel = _FRD.resolved_execution_capsule_path("vic")
    _fplan = [f"MAG_FEED/vic/2026-01-{_d:02d}" for _d in (1, 2, 3)]

    def _blob_without(path):
        return None if path == _exec_rel else _blob_at_head(None, path)

    def _blob_with(path):
        return (_vic_bytes if path == _exec_rel
                else _blob_at_head(None, path))

    _fired = False
    try:
        _FRD.audit_plan(_fplan, blob=_blob_without)
    except _FRD.FrameReadinessRefusal:
        _fired = True
    if not _fired:
        raise RegenerationGateRefusal(
            "FRAME_DOCTOR_INSENSITIVE: with no capsule at the "
            f"production-resolved path {_exec_rel}, the frame doctor "
            "accepted a plan it must refuse")
    # positive control: the SAME bytes at the resolved path must ACCEPT,
    # or the doctor is merely an always-refuse and proves nothing
    _FRD.audit_plan(_fplan, blob=_blob_with)
    print("  RG-10 PASS  frame readiness, BOTH controls CONSTRUCTED: a "
          "plan whose carrier resolves no frame capsule at "
          "the production-resolved path is REFUSED, and the same bytes "
          "placed at that exact path are ACCEPTED (a capsule elsewhere "
          "never satisfies)")

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
        print(f"\nREGENERATION GATE: REFUSED (full gate not "
              f"satisfied)\n  {str(e)[:400]}")
        return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(_selftest())
