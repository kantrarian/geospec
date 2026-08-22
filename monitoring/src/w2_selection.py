#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 SELECTION ENGINE (cayley) -- the frozen cutoff-stable
selection algorithm of docs/f2g_window2_freeze/selection_constants.md
(design freeze CLOSED @ 12161f6/5fba544), implemented against
grassmann's bar seam pin (test_f2g_window2_redkats_grassmann.py @
8a78d5f). This commit FIXES the R1.2-open seam name as `w2_selection`.

Frozen algorithm (selection_constants.md line 16, verbatim semantics):
greedy by (presence DESC, station_id ASC) into the cap over stations
meeting the presence floor; then drop-worst by presence until the
carrier-set churn >= floor or the minimum is reached; floor unreachable
at the minimum -> carrier admitted with churn disclosed BELOW_FLOOR,
typed. Pool minimum is tested BEFORE the cap (INSUFFICIENT_POOL).

Interpretation pins (disclosed; sources cited):
- drop-worst tie rule: among lowest-presence ties drop the
  lexicographically LAST station id -- the unique reading consistent
  with the frozen preference order (presence DESC, id ASC): the
  least-preferred member is the lex-last of the worst presence tier.
  (grassmann's stated bar pin, ratified; implemented as the TRUE
  lexicographic-last, i.e. max by (-presence, id) -- see the R1.2 note
  on the oracle's [-ord] key prefix-corner.)
- adjacent-day Jaccard with both restricted sets empty: J(0,0) := 1.0
  (identical sets), per the bar oracle's pinned convention; 0 or 1
  day_sets entries -> churn := 1.0 (no adjacent pairs to disagree).

Executes ONCE at the availability cutoff on pre-evaluation telemetry
only; no evaluation-window value may reach any argument
(HEALTH_ADMISSION_VIOLATION is the caller-side typed refusal per
annex_b1b.md). This module opens no window-2 value.
"""

# module literals REQUIRED equal to the frozen table (bar W-SEL contract)
PRESENCE_FLOOR = 0.85
CHURN_FLOOR = 0.80
CAPS = {"istanbul_marmara": 16, "socal_coachella": 20,
        "turkey_kahramanmaras": 14, "cascadia": 16}
MINIMUM = 8


class InsufficientPool(ValueError):
    """Typed INSUFFICIENT_POOL: eligible pool below the carrier minimum
    (tested BEFORE the cap)."""


class SelectionInputInvalid(ValueError):
    """Typed refusal for malformed inputs (unknown carrier without an
    explicit cap; presence outside [0, 1] or non-finite)."""


def carrier_churn(selected, day_sets):
    """Mean over adjacent lookback day-pairs of the Jaccard SIMILARITY
    between the selected-set restrictions of the measured station sets
    (selection_constants.md line 12). J(empty, empty) := 1.0; fewer
    than 2 days -> 1.0."""
    sel = set(selected)
    sims = []
    for a, b in zip(day_sets, day_sets[1:]):
        aa, bb = set(a) & sel, set(b) & sel
        union = aa | bb
        sims.append(1.0 if not union else len(aa & bb) / len(union))
    return sum(sims) / len(sims) if sims else 1.0


def select(carrier_key, presence_by_station, day_sets, cap=None):
    """Returns {"selected": [ids ASC], "churn": float,
    "typing": None | "BELOW_FLOOR"}; raises InsufficientPool (typed
    INSUFFICIENT_POOL) when the eligible pool is below MINIMUM.

    cap=None resolves the frozen per-carrier cap from CAPS; an explicit
    cap (bar fixtures) is used verbatim."""
    if cap is None:
        if carrier_key not in CAPS:
            raise SelectionInputInvalid(
                f"UNKNOWN_CARRIER: {carrier_key!r} and no explicit cap")
        cap = CAPS[carrier_key]
    for s, p in presence_by_station.items():
        if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
            raise SelectionInputInvalid(
                f"PRESENCE_INVALID: {s!r} -> {p!r}")

    eligible = sorted(
        (s for s, p in presence_by_station.items()
         if p >= PRESENCE_FLOOR),
        key=lambda s: (-presence_by_station[s], s))
    if len(eligible) < MINIMUM:  # pool test precedes the cap
        raise InsufficientPool(
            f"INSUFFICIENT_POOL: carrier={carrier_key} "
            f"eligible={len(eligible)} < minimum={MINIMUM}")

    sel = list(eligible[:cap])
    while carrier_churn(sel, day_sets) < CHURN_FLOOR \
            and len(sel) > MINIMUM:
        # worst = least-preferred under the frozen (presence DESC,
        # id ASC) order: lowest presence, then TRUE lexicographic-last
        worst = max(sel, key=lambda s: (-presence_by_station[s], s))
        sel.remove(worst)
    churn = carrier_churn(sel, day_sets)
    return {"selected": sorted(sel), "churn": churn,
            "typing": None if churn >= CHURN_FLOOR else "BELOW_FLOOR"}


def _selftest():
    # nominal greedy: the bar's W-SEL-a fixture SHAPE with valid
    # presence values (the bar's 0.80+i*0.02 yields 1.02 at i=11,
    # outside [0,1] -- R1.2 flagged to grassmann; engine validation is
    # deliberately strict)
    pres = {f"S{i:02d}": 0.78 + i * 0.02 for i in range(12)}
    days = [[f"S{i:02d}" for i in range(12)]] * 5
    r = select("fixture", pres, days, cap=8)
    elig = {s: p for s, p in pres.items() if p >= PRESENCE_FLOOR}
    assert r["selected"] == sorted(
        sorted(elig, key=lambda s: (-elig[s], s))[:8])
    assert r["typing"] is None and abs(r["churn"] - 1.0) < 1e-12

    # presence ties select lexicographic ASC
    pres2 = {s: 0.90 for s in "BACEDFGHI"}
    r2 = select("fixture", pres2, [list(pres2)] * 3, cap=8)
    assert r2["selected"] == sorted(pres2)[:8]

    # INSUFFICIENT_POOL at pool==7, before the cap
    try:
        select("fixture", {f"T{i}": 0.9 for i in range(7)},
               [["T0"]] * 3, cap=16)
        raise AssertionError("pool=7 must refuse")
    except InsufficientPool as exc:
        assert "INSUFFICIENT_POOL" in str(exc)

    # drop-worst path: 8 stable + 3 flappy (even-days-only) gives
    # J = 8/11 < floor; ONE drop of the lowest-presence flapper (Z9)
    # restores J = 8/10 = 0.80 >= floor and the loop stops
    stable = {f"A{i}": 0.99 for i in range(8)}
    flappy = {"Z7": 0.87, "Z8": 0.86, "Z9": 0.85}
    pres3 = dict(stable, **flappy)
    d_even = sorted(stable) + sorted(flappy)
    d_odd = sorted(stable)
    days3 = [d_even, d_odd] * 5
    r3 = select("fixture", pres3, days3, cap=16)
    assert r3["selected"] == sorted(list(stable) + ["Z7", "Z8"]), r3
    assert r3["typing"] is None and abs(r3["churn"] - 0.80) < 1e-12

    # presence-tie drop through the ACTUAL drop path, prefix corner:
    # 7 stable + {A, AB} tied at 0.90 flapping -> J = 7/9 < floor;
    # the stated pin drops the TRUE lexicographic LAST ("AB"), after
    # which J = 7/8 >= floor and "A" survives
    pres4 = {"A": 0.90, "AB": 0.90}
    pres4.update({f"K{i}": 0.99 for i in range(7)})
    kd = sorted(f"K{i}" for i in range(7))
    days4 = [kd + ["A"], kd + ["AB"]] * 4
    r4 = select("fixture", pres4, days4, cap=16)
    assert "AB" not in r4["selected"] and "A" in r4["selected"], r4

    # BELOW_FLOOR: floor unreachable at the minimum -> typed, admitted
    pres5 = {f"M{i}": 0.90 for i in range(8)}
    ms = sorted(pres5)
    days5 = [ms[:4], ms[4:]] * 4   # alternating halves: churn 0 at min
    r5 = select("fixture", pres5, days5, cap=16)
    assert r5["typing"] == "BELOW_FLOOR" and len(r5["selected"]) == 8

    # frozen constants as module literals
    assert (PRESENCE_FLOOR, CHURN_FLOOR, MINIMUM) == (0.85, 0.80, 8)
    assert CAPS == {"istanbul_marmara": 16, "socal_coachella": 20,
                    "turkey_kahramanmaras": 14, "cascadia": 16}
    print("w2_selection selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
