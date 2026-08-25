#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AUTHORITY-SERVES-EVERY-KEY RED-KATs (cayley) -- codex 1746Z gate-1
finding 1.

THE DEFECT THIS LOCKS
---------------------
The v4 authority passed `_validate_expected_keys_authority`, both
generator and instrument selftests were green, and I routed a
capture-readiness packet describing the authority as byte-identical
and closed. It was neither: `MAG_WEATHER_FEED/omni`'s
`static_contract_template` still carried `OPEN_REVIEW_ROUND`, so

    authoritative_static_contract(auth, "MAG_WEATHER_FEED", "omni", d)
      -> PRESTART_ADMISSION_REFUSED: ... carries OPEN tokens

for EVERY day. Neither the 211 remaining OMNI captures nor the
predecessor bridge could have derived their S. Worse, the corrected
probe had already fired and produced closed contract/transcript/body
evidence -- the authority simply never consumed it, and "byte-identical
re-pin" was a symptom of that, which I reported as a virtue.

WHY EVERY EXISTING CHECK MISSED IT
----------------------------------
They all verified that the authority was *well-formed*: closed schema,
census, key digest, generator reproduction. None verified that it could
actually *serve* -- that each registered key yields a usable static
contract. A schema-valid authority with an unfilled template is exactly
that gap, and it survives every structural check by construction.

THE RULE
--------
An authority is production-ready only if EVERY registered
`(lane, carrier, day)` derives a static contract that
  * succeeds (no typed refusal),
  * carries no `OPEN_REVIEW_ROUND` anywhere,
  * leaves no unresolved `{token}` after substitution,
  * and names the key it was derived for.

Serving is checked over the WHOLE key set, not a sample: a per-carrier
sample would have passed here too, since only one carrier was unfilled.

Opens no window-2 value; no network; admits nothing.
"""
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
TOKEN_RE = re.compile(r"\{[A-Za-z][A-Za-z0-9_]*\}")


class ServesRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


def _selftest():
    import w2_accrual_instrument_cayley as AI
    import w2_expected_contracts_gen_cayley as GEN

    auth = GEN.build(REPO)
    keys = auth["prestart_expected_keys"]
    total = sum(len(d) for lane in keys.values()
                for d in lane.values())
    print(f"  enumerating ALL {total} registered keys "
          "(no sampling -- one unfilled carrier is invisible to a "
          "sample)")

    failures, unresolved, mismatched = [], [], []
    served = 0
    for lane in sorted(keys):
        for ck in sorted(keys[lane]):
            for day in keys[lane][ck]:
                try:
                    sc = AI.authoritative_static_contract(
                        auth, lane, ck, day)
                except Exception as e:                   # noqa: BLE001
                    failures.append(
                        (f"{lane}/{ck}/{day}",
                         f"{type(e).__name__}: {str(e)[:80]}"))
                    continue
                blob = json.dumps(sc, sort_keys=True)
                if "OPEN_REVIEW_ROUND" in blob:
                    failures.append((f"{lane}/{ck}/{day}",
                                     "OPEN_REVIEW_ROUND survived"))
                    continue
                left = TOKEN_RE.findall(blob)
                if left:
                    unresolved.append((f"{lane}/{ck}/{day}",
                                       sorted(set(left))))
                    continue
                if (sc.get("lane"), sc.get("carrier"),
                        sc.get("utc_day")) != (lane, ck, day):
                    mismatched.append(f"{lane}/{ck}/{day}")
                    continue
                served += 1

    if failures:
        by_carrier = {}
        for k, why in failures:
            by_carrier.setdefault("/".join(k.split("/")[:2]),
                                  [0, why])[0] += 1
        raise ServesRefusal(
            f"SK-1 AUTHORITY_CANNOT_SERVE: {len(failures)} of {total} "
            "registered keys do not derive a usable static contract. "
            f"By carrier: { {c: f'{n} keys -- {w}' for c, (n, w) in by_carrier.items()} }. "
            "A schema-valid authority that cannot serve a key is not "
            "production-ready: neither a capture nor the predecessor "
            "bridge can derive that key's S. codex 1746Z finding 1.")
    if unresolved:
        raise ServesRefusal(
            f"SK-2 UNRESOLVED_TOKENS: {len(unresolved)} keys carry "
            f"tokens after substitution, e.g. {unresolved[:3]}")
    if mismatched:
        raise ServesRefusal(
            f"SK-3 KEY_IDENTITY_DIVERGENT: {len(mismatched)} contracts "
            f"do not name their own key, e.g. {mismatched[:3]}")

    assert served == total, (served, total)

    # ---- SK-0 anti-vacuity: the lock must FIRE on the exact defect
    # it was written for. Reinstate an OPEN token in ONE carrier and
    # require SK-1 to catch it -- a lock that cannot fail proves
    # nothing, and a per-carrier sample would have missed this.
    import copy
    hurt = copy.deepcopy(auth)
    hurt["static_layer"]["MAG_WEATHER_FEED"]["carriers"]["omni"][
        "static_contract_template"]["request_params"] =         "OPEN_REVIEW_ROUND"
    caught = 0
    for day in keys["MAG_WEATHER_FEED"]["omni"][:5]:
        try:
            AI.authoritative_static_contract(
                hurt, "MAG_WEATHER_FEED", "omni", day)
        except Exception:                                # noqa: BLE001
            caught += 1
    assert caught == 5, (
        "SK-0: reinstating an OPEN token must make those keys "
        f"unservable, but {5 - caught} of 5 still derived")
    print("  SK-0 PASS  anti-vacuity: an OPEN token in one carrier "
          "makes its keys unservable (the lock can fail)")
    print(f"  SK-1 PASS  all {served} keys derive a usable contract")
    print("  SK-2 PASS  no unresolved template tokens anywhere")
    print("  SK-3 PASS  every contract names its own key")
    print("w2 authority-serves-every-key red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except ServesRefusal as e:
        print(f"RED: {e}")
        raise SystemExit(1)
