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


def _read(rel):
    with open(os.path.join(REPO, rel.replace("/", os.sep)),
              encoding="utf-8") as f:
        return json.load(f)


def _design_capsule_paths(design):
    """Every mag capsule path pinned by the FROZEN byte-pin manifest."""
    out = set()
    pins = design.get("pins")
    entries = (pins.values() if isinstance(pins, dict)
               else (pins or []))
    for e in entries:
        p = e.get("path") if isinstance(e, dict) else None
        if p and CAPSULE_RE.search(str(p)):
            out.add(str(p))
    return out


def _exec_capsule_paths(execm):
    """Every mag capsule path pinned by the EXECUTION manifest."""
    out = set()
    for slot in (execm.get("slots") or {}).values():
        if not isinstance(slot, dict):
            continue
        for pin in slot.get("pins") or []:
            p = pin.get("path") if isinstance(pin, dict) else pin
            if p and CAPSULE_RE.search(str(p)):
                out.add(str(p))
    return out


def admitted_observatories(repo=REPO):
    """THE derivation: iaga_code of every capsule pinned by either
    registered manifest. Returns {iaga: capsule_path}."""
    design = _read(DESIGN_MANIFEST)
    execm = _read(EXEC_MANIFEST)
    paths = _design_capsule_paths(design) | _exec_capsule_paths(execm)
    if not paths:
        raise CarrierSetRefusal(
            "CARRIER_SET_UNDERIVABLE: no mag capsule is pinned by "
            "either manifest -- the derivation has no authority")
    admitted = {}
    for p in sorted(paths):
        cap = _read(p)
        iaga = cap.get("iaga_code")
        if not isinstance(iaga, str) or not iaga:
            raise CarrierSetRefusal(
                f"CARRIER_CAPSULE_UNTYPED: {p} carries no iaga_code")
        if iaga in admitted:
            raise CarrierSetRefusal(
                f"CARRIER_CAPSULE_DUPLICATE: {iaga} pinned twice "
                f"({admitted[iaga]}, {p})")
        admitted[iaga] = p
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
    assert not (d_paths & e_paths), \
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
    print("w2 frozen-carrier-set red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except CarrierSetRefusal as e:
        print(f"RED (expected until the successor lands): {e}")
        raise SystemExit(1)
