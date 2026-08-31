#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WINDOW-2 FROZEN-CARRIER-SET RED-KATs (cayley) -- the executable lock
for codex 0527Z postflight finding 1.

THE DEFECT THIS LOCKS
---------------------
`w2_expected_contracts_gen_cayley.MAG_OBSERVATORIES` was authored from
the EXECUTION capsule directory (izn/frn/tuc) rather than the FROZEN
MAG-1 admitted carrier set, which also admits VIC and NEW under the
2026-08-22 pre-freeze cascadia amendment. The v3 authority therefore
registered three observatories, the 1,794-key capture never requested
VIC/NEW, and every green check still passed -- because each one
verified the authority against ITSELF. An authority that defines its
own carrier set cannot detect a carrier the freeze admitted and the
authority omitted.

THE DERIVATION RULE (why this is not vacuous)
---------------------------------------------
The admitted observatory set is DERIVED, never typed here: it is the
`iaga_code` of every `mag_capsule_*.json` pinned by EITHER registered
manifest --

  * design/byte-pin manifest  -> VIC, NEW  (the amendment capsules)
  * execution manifest        -> IZN, FRN, TUC

-- so the set comes from the pinned bytes of the two authorities that
already govern this program. A future frozen-carrier addition that the
authority fails to register trips FC-1 automatically. Typing the five
names into this file would have reproduced the very defect it locks.

STATUS: red-first at authorship (geospec 2e1b2f2, FC-1 RED: 3
registered != 5 admitted -- the executable proof of finding 1), GREEN
from the successor v4 generator, which replaced the typed tuple with
the derivation above. It is now the REGRESSION lock: any future frozen
carrier the authority fails to register trips FC-1 again.

This module opens no window-2 value, makes no network call, and
admits nothing.
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
DESIGN_MANIFEST = "docs/f2g_window2_freeze/byte_pin_manifest.json"
EXEC_MANIFEST = ("docs/f2g_window2_execution/"
                 "execution_manifest.json")
CAPSULE_RE = re.compile(r"mag_capsule_[a-z0-9_]+\.json$")


class CarrierSetRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


# KAT-ONLY seam for the alias control below; production reads bytes
_READ_OVERRIDE = {}


def _read(rel):
    if rel in _READ_OVERRIDE:
        return _READ_OVERRIDE[rel]
    with open(os.path.join(REPO, rel.replace("/", os.sep)),
              encoding="utf-8") as f:
        return json.load(f)


def _design_capsule_paths(design):
    """Every mag capsule PIN of the FROZEN byte-pin manifest, as
    {path: {commit, path, blob_sha256}} -- the full pin, because a
    path alone is not pin resolution (codex cycle-6b item 5)."""
    out = {}
    pins = design.get("pins")
    entries = (pins.values() if isinstance(pins, dict)
               else (pins or []))
    for e in entries:
        p = e.get("path") if isinstance(e, dict) else None
        if p and CAPSULE_RE.search(str(p)):
            out[str(p)] = e
    return out


def _exec_capsule_paths(execm):
    """Every mag capsule PIN of the EXECUTION manifest, same shape."""
    out = {}
    for slot in (execm.get("slots") or {}).values():
        if not isinstance(slot, dict):
            continue
        for pin in slot.get("pins") or []:
            p = pin.get("path") if isinstance(pin, dict) else pin
            if p and CAPSULE_RE.search(str(p)) and \
                    isinstance(pin, dict):
                out[str(p)] = pin
    return out


# KAT-ONLY seams: {path: raw bytes} overriding the pin-resolved read,
# and {path: pin} overriding the resolved pin itself
_RAW_OVERRIDE = {}
_PIN_OVERRIDE = {}


def _pinned_raw(pin):
    """Reopen a capsule's RAW COMMITTED BYTES from its pin and verify
    the raw digest against that pin.

    codex cycle-6b item 5: the routed rule is "collapse only when the
    committed BYTES agree", and my first implementation compared
    canonicalized PARSED objects read from the WORKTREE. Two bodies
    differing only in field order or whitespace have different
    SHA-256 and identical parsed objects, so that version admitted a
    real byte divergence as an alias -- and let dirty worktree state
    steer a bar described as deriving from pins.
    """
    path = pin["path"]
    if path in _RAW_OVERRIDE:
        raw = _RAW_OVERRIDE[path]
    else:
        p = subprocess.run(
            ["git", "-C", REPO, "cat-file", "blob",
             f"{pin['commit']}:{path}"], capture_output=True)
        if p.returncode != 0 or not p.stdout:
            raise CarrierSetRefusal(
                f"CARRIER_CAPSULE_UNREADABLE: {path} at "
                f"{str(pin['commit'])[:12]}")
        raw = p.stdout
    got = hashlib.sha256(raw).hexdigest()
    if got != pin.get("blob_sha256"):
        raise CarrierSetRefusal(
            f"CARRIER_CAPSULE_PIN_DIVERGENT: {path} bytes {got[:12]} "
            f"!= pin {str(pin.get('blob_sha256'))[:12]}")
    return raw, got


def admitted_observatories(repo=REPO):
    """THE derivation: iaga_code of every capsule pinned by either
    registered manifest. Returns {iaga: capsule_path}."""
    design = _read(DESIGN_MANIFEST)
    execm = _read(EXEC_MANIFEST)
    pins = dict(_design_capsule_paths(design))
    pins.update(_exec_capsule_paths(execm))
    # KAT-ONLY seam: override a resolved pin so a control can supply
    # bytes together with their HONEST digest, putting the ALIAS rule
    # under test rather than the pin check
    for _p, _pin in _PIN_OVERRIDE.items():
        if _p in pins:
            pins[_p] = _pin
    if not pins:
        raise CarrierSetRefusal(
            "CARRIER_SET_UNDERIVABLE: no mag capsule is pinned by "
            "either manifest -- the derivation has no authority")
    admitted, bodies = {}, {}
    for p in sorted(pins):
        # RAW committed bytes, verified against the pin, parsed only
        # AFTER that: the digest that decides an alias is the pin's
        # own, never a canonicalization of a worktree read.
        raw, raw_sha = _pinned_raw(pins[p])
        try:
            cap = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise CarrierSetRefusal(
                f"CARRIER_CAPSULE_UNPARSABLE: {p} at "
                f"{str(pins[p]['commit'])[:12]}")
        iaga = cap.get("iaga_code")
        if not isinstance(iaga, str) or not iaga:
            raise CarrierSetRefusal(
                f"CARRIER_CAPSULE_UNTYPED: {p} carries no iaga_code")
        # The same IAGA legitimately appears at BOTH the design-freeze
        # path and the execution path -- VIC does, and those two blobs
        # are byte-identical. An ALIAS is not a duplicate. Collapse it
        # only when the RAW COMMITTED BYTES agree: a body differing
        # only in field order or whitespace parses identically and is
        # a genuine byte divergence, so it must still refuse.
        if iaga in admitted:
            if raw_sha == bodies[iaga]:
                continue
            raise CarrierSetRefusal(
                f"CARRIER_CAPSULE_DUPLICATE: {iaga} pinned twice with "
                f"DIVERGENT committed bytes ({admitted[iaga]} "
                f"{bodies[iaga][:12]}, {p} {raw_sha[:12]}) -- an "
                "alias may share an IAGA only when its bytes agree")
        admitted[iaga] = p
        bodies[iaga] = raw_sha
    return admitted


def _selftest():
    import w2_expected_contracts_gen_cayley as GEN

    admitted = admitted_observatories()
    derived = {k.upper() for k in admitted}
    # the REGISTERED set is what the authority actually publishes --
    # read from the built artifact's MAG_FEED carriers, not from any
    # module constant (successor v4 removed the typed tuple that was
    # the defect; this now checks the real end-to-end output)
    built = GEN.build(REPO)
    registered = {str(c).upper() for c in
                  built["static_layer"]["MAG_FEED"]["carriers"]}
    print(f"  derived (pinned capsules) : {sorted(derived)}")
    print(f"  registered (authority)    : {sorted(registered)}")

    # ---- FC-0: the derivation actually reads pins (anti-vacuity).
    # Remove the design-pinned capsules and the derived set MUST
    # shrink; a checker that ignored the manifests would not move.
    design = _read(DESIGN_MANIFEST)
    d_paths = _design_capsule_paths(design)
    assert d_paths, ("FC-0: the design manifest pins no mag capsule "
                     "-- the amendment capsules are unreachable")
    e_paths = _exec_capsule_paths(_read(EXEC_MANIFEST))
    assert e_paths, "FC-0: the execution manifest pins no mag capsule"
    assert not (set(d_paths) & set(e_paths)), \
        "FC-0: design and execution pin the same capsule path"
    print(f"  FC-0 PASS  design-pinned={len(d_paths)} "
          f"execution-pinned={len(e_paths)} (disjoint, both read)")

    # ---- FC-2: every pinned capsule is typed and unique (proved by
    # admitted_observatories() having returned at all)
    assert len(admitted) == len(derived), "FC-2: iaga collision"
    print(f"  FC-2 PASS  {len(admitted)} pinned capsules, all typed "
          "and unique")

    # ---- FC-1 (THE LOCK): the registered authority set must EQUAL
    # the frozen admitted set. RED until the successor registers all.
    missing = sorted(derived - registered)
    extra = sorted(registered - derived)
    if missing or extra:
        raise CarrierSetRefusal(
            "FC-1 CARRIER_SET_DIVERGENCE: the execution authority "
            f"omits {missing or 'nothing'} and invents "
            f"{extra or 'nothing'} relative to the frozen admitted "
            f"set derived from the pinned capsules "
            f"({sorted(derived)}). codex 0527Z finding 1.")
    print(f"  FC-1 PASS  authority == frozen admitted set "
          f"{sorted(derived)}")
    # ---- FC-alias (codex item 5.1): the alias collapse is not a
    # blanket dedup. Same IAGA + IDENTICAL bytes collapses (the live
    # VIC case, proven above by this function returning at all);
    # same IAGA + DIVERGENT bytes must still refuse.
    _pins = dict(_design_capsule_paths(_read(DESIGN_MANIFEST)))
    _pins.update(_exec_capsule_paths(_read(EXEC_MANIFEST)))
    _vic = sorted(p for p in _pins
                  if p.endswith("mag_capsule_vic.json"))
    if len(_vic) < 2:
        raise CarrierSetRefusal(
            "FC-alias CONTROL_INERT: VIC is no longer pinned at two "
            "paths, so the alias case cannot be constructed")
    _a, _b = _vic[:2]
    _raw_a, _sha_a = _pinned_raw(_pins[_a])
    _raw_b, _sha_b = _pinned_raw(_pins[_b])
    if _sha_a != _sha_b:
        raise CarrierSetRefusal(
            "FC-alias CONTROL_INERT: the two VIC capsules already "
            "diverge, so the identical-bytes collapse is untested")

    def _alias_refuses(raw_bytes, why, needle="DIVERGENT committed"):
        saved_raw, saved_pin = _RAW_OVERRIDE.copy(), _PIN_OVERRIDE.copy()
        try:
            # supply the bytes AND their honest pin digest, so the pin
            # check passes and the ALIAS rule is what is under test
            _RAW_OVERRIDE[_b] = raw_bytes
            _PIN_OVERRIDE[_b] = dict(
                _pins[_b],
                blob_sha256=hashlib.sha256(raw_bytes).hexdigest())
            try:
                admitted_observatories()
                raise CarrierSetRefusal(
                    f"FC-alias {why}: admitted as an alias")
            except CarrierSetRefusal as e:
                if needle not in str(e):
                    raise
        finally:
            _RAW_OVERRIDE.clear(); _RAW_OVERRIDE.update(saved_raw)
            _PIN_OVERRIDE.clear(); _PIN_OVERRIDE.update(saved_pin)

    # codex's exact probe: FORMATTING-ONLY raw divergence. Different
    # SHA-256, IDENTICAL parsed object. My previous implementation
    # compared canonicalized parsed objects and admitted this.
    _reformatted = json.dumps(json.loads(_raw_b.decode("utf-8")),
                              indent=3, sort_keys=True).encode()
    if _reformatted == _raw_b:
        raise CarrierSetRefusal(
            "FC-alias CONTROL_INERT: the reformat did not change the "
            "bytes")
    if json.loads(_reformatted.decode("utf-8")) != \
            json.loads(_raw_b.decode("utf-8")):
        raise CarrierSetRefusal(
            "FC-alias CONTROL_INERT: the reformat changed the parsed "
            "object, so it does not isolate BYTES from SEMANTICS")
    _alias_refuses(_reformatted, "FORMATTING_ONLY_DIVERGENCE_ADMITTED")
    # semantic divergence must refuse too
    _semantic = json.dumps(
        dict(json.loads(_raw_b.decode("utf-8")),
             _alias_divergence_probe=True)).encode()
    _alias_refuses(_semantic, "SEMANTIC_DIVERGENCE_ADMITTED")
    # and a body whose digest does NOT match its pin refuses at the
    # pin check, before the alias rule is even reached
    _saved2 = _RAW_OVERRIDE.copy()
    try:
        _RAW_OVERRIDE[_b] = _raw_b + b"\n"
        try:
            admitted_observatories()
            raise CarrierSetRefusal(
                "FC-alias WRONG_PIN_DIGEST_ADMITTED")
        except CarrierSetRefusal as e:
            if "PIN_DIVERGENT" not in str(e):
                raise
    finally:
        _RAW_OVERRIDE.clear()
        _RAW_OVERRIDE.update(_saved2)
    print("  FC-alias PASS  the alias rule is decided by RAW "
          f"COMMITTED BYTES ({_sha_a[:12]}): formatting-only "
          "divergence with an identical parsed object REFUSES (the "
          "case my parsed-JSON version admitted), semantic "
          "divergence REFUSES, and bytes that do not match their pin "
          "refuse at the pin check first")
    print("w2 frozen-carrier-set red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except CarrierSetRefusal as e:
        print(f"RED (expected until the successor lands): {e}")
        raise SystemExit(1)
