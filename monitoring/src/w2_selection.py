#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 SELECTION ENGINE (cayley) -- the frozen cutoff-stable
selection algorithm of docs/f2g_window2_freeze/selection_constants.md
(design freeze CLOSED @ 12161f6/5fba544). REV 2: codex bounded-review
repairs (0231Z note, all three P1s):

P1-1 EXACT ARITHMETIC: every daily Jaccard is an integer ratio, so the
churn objective and both floors are computed in fractions.Fraction and
compared EXACTLY; floats appear only in returned artifacts. (The REV-1
fixture's pass was host-dependent: py3.12+ compensated float summation
vs py3.11 plain summation -- the no-bit-equality-on-floats lesson.)

P1-2 LOOKBACK CROSS-BINDING: the production seam carries DATED day
records plus the cutoff. The frame must be exactly the 90 unique
consecutive UTC days [cutoff-89, cutoff] (typed LOOKBACK_FRAME_INVALID);
presence numerators are DERIVED from those records; a supplied
declaration is verified against the derived count (typed
PRESENCE_LOOKBACK_MISMATCH). Churn uses the same records. J(0,0)=1 can
no longer manufacture lookback evidence.

P1-3 CAP POLICY: production always resolves the frozen CAPS entry; an
explicit production cap must equal it exactly (typed
CAP_OVERRIDE_REFUSED). Arbitrary geometry lives ONLY in the clearly
fixture-only path `select_fixture`, which still validates
`type(cap) is int` and `cap >= MINIMUM`.

Frozen algorithm (selection_constants.md line 16): greedy by
(presence DESC, station_id ASC) into the cap over stations meeting the
presence floor; then drop-worst by presence (ties: TRUE lexicographic
LAST, grassmann-ratified) until carrier-set churn >= floor or the
minimum is reached; floor unreachable at minimum -> BELOW_FLOOR typed.
Pool minimum tested BEFORE the cap (INSUFFICIENT_POOL).
J(empty, empty) := 1 (ratified pin). Executes ONCE at the availability
cutoff on pre-evaluation telemetry only.
"""
from datetime import date, timedelta
from fractions import Fraction

# module literals REQUIRED equal to the frozen table (bar W-SEL contract)
PRESENCE_FLOOR = 0.85
CHURN_FLOOR = 0.80
CAPS = {"istanbul_marmara": 16, "socal_coachella": 20,
        "turkey_kahramanmaras": 14, "cascadia": 16}
MINIMUM = 8
LOOKBACK_DAYS = 90

_PRESENCE_FLOOR_F = Fraction(17, 20)   # == 0.85 exactly
_CHURN_FLOOR_F = Fraction(4, 5)        # == 0.80 exactly


class InsufficientPool(ValueError):
    """Typed INSUFFICIENT_POOL: eligible pool below the carrier minimum
    (tested BEFORE the cap)."""


class SelectionInputInvalid(ValueError):
    """Typed refusals: UNKNOWN_CARRIER, PRESENCE_INVALID,
    LOOKBACK_FRAME_INVALID, PRESENCE_LOOKBACK_MISMATCH,
    CAP_OVERRIDE_REFUSED, CAP_INVALID."""


def churn_exact(selected, day_sets):
    """Mean adjacent-day Jaccard similarity restricted to `selected`,
    as an EXACT Fraction. J(empty, empty) := 1; < 2 days -> 1."""
    sel = set(selected)
    sims = []
    for a, b in zip(day_sets, day_sets[1:]):
        aa, bb = set(a) & sel, set(b) & sel
        union = aa | bb
        sims.append(Fraction(1) if not union
                    else Fraction(len(aa & bb), len(union)))
    if not sims:
        return Fraction(1)
    return sum(sims, Fraction(0)) / len(sims)


def carrier_churn(selected, day_sets):
    """Float view of churn_exact (artifact use only -- never compared
    against the floor)."""
    return float(churn_exact(selected, day_sets))


def _greedy_drop(presence, day_sets, cap):
    """The frozen core over an already-validated frame. `presence`
    values may be Fraction (production) or float (fixture); floors are
    compared exactly against the Fraction constants when possible."""
    def meets_floor(p):
        return (p >= _PRESENCE_FLOOR_F if isinstance(p, Fraction)
                else p >= PRESENCE_FLOOR)

    eligible = sorted((s for s, p in presence.items()
                       if meets_floor(p)),
                      key=lambda s: (-presence[s], s))
    if len(eligible) < MINIMUM:  # pool test precedes the cap
        raise InsufficientPool(
            f"INSUFFICIENT_POOL: eligible={len(eligible)} < "
            f"minimum={MINIMUM}")
    sel = list(eligible[:cap])
    while churn_exact(sel, day_sets) < _CHURN_FLOOR_F \
            and len(sel) > MINIMUM:
        # worst = lowest presence, then TRUE lexicographic-last
        worst = max(sel, key=lambda s: (-presence[s], s))
        sel.remove(worst)
    c = churn_exact(sel, day_sets)
    return {"selected": sorted(sel), "churn": float(c),
            "typing": None if c >= _CHURN_FLOOR_F else "BELOW_FLOOR"}


def _validate_frame(day_records, cutoff):
    try:
        cut = date.fromisoformat(str(cutoff))
    except ValueError:
        raise SelectionInputInvalid(
            f"LOOKBACK_FRAME_INVALID: bad cutoff {cutoff!r}")
    days = sorted(day_records)
    if len(days) != LOOKBACK_DAYS or len(set(days)) != len(days):
        raise SelectionInputInvalid(
            f"LOOKBACK_FRAME_INVALID: {len(days)} records, need "
            f"{LOOKBACK_DAYS} unique")
    expect = [(cut - timedelta(days=LOOKBACK_DAYS - 1 - i)).isoformat()
              for i in range(LOOKBACK_DAYS)]
    if days != expect:
        raise SelectionInputInvalid(
            "LOOKBACK_FRAME_INVALID: records are not exactly the "
            f"consecutive UTC days [{expect[0]}, {expect[-1]}]")
    return days


def select(carrier_key, day_records, cutoff, cap=None,
           presence_declared=None):
    """PRODUCTION seam (codex P1-2/P1-3 shape). `day_records` =
    {iso_day: iterable of station ids with >= 1 admissible sample that
    day}; the frame must be exactly [cutoff-89, cutoff]. Presence is
    DERIVED from the records; `presence_declared` (station -> int count
    or exact fraction float) is VERIFIED, never trusted. The frozen cap
    always applies; an explicit cap must equal it exactly."""
    if carrier_key not in CAPS:
        raise SelectionInputInvalid(f"UNKNOWN_CARRIER: {carrier_key!r}")
    frozen_cap = CAPS[carrier_key]
    if cap is not None:
        if type(cap) is not int or cap != frozen_cap:
            raise SelectionInputInvalid(
                f"CAP_OVERRIDE_REFUSED: carrier={carrier_key} "
                f"frozen={frozen_cap} got={cap!r}")
    days = _validate_frame(day_records, cutoff)
    day_sets = [frozenset(day_records[d]) for d in days]

    stations = sorted({s for ds in day_sets for s in ds})
    count = {s: sum(1 for ds in day_sets if s in ds) for s in stations}
    if presence_declared is not None:
        for s, decl in presence_declared.items():
            got = count.get(s, 0)
            if isinstance(decl, int):
                ok = decl == got
            else:
                ok = Fraction(str(decl)) == Fraction(got, LOOKBACK_DAYS)
            if not ok:
                raise SelectionInputInvalid(
                    f"PRESENCE_LOOKBACK_MISMATCH: {s!r} declared "
                    f"{decl!r} vs derived {got}/{LOOKBACK_DAYS}")
    presence = {s: Fraction(count[s], LOOKBACK_DAYS) for s in stations}
    return _greedy_drop(presence, day_sets, frozen_cap)


def select_fixture(presence_by_station, day_sets, cap):
    """FIXTURE-ONLY path (codex P1-3): arbitrary geometry for bar
    fixtures. Never valid for a production carrier decision."""
    if type(cap) is not int or cap < MINIMUM:
        raise SelectionInputInvalid(
            f"CAP_INVALID: fixture cap must be int >= {MINIMUM}, "
            f"got {cap!r}")
    for s, p in presence_by_station.items():
        if isinstance(p, Fraction):
            ok = 0 <= p <= 1
        else:
            ok = isinstance(p, (int, float)) and 0.0 <= p <= 1.0
        if not ok:
            raise SelectionInputInvalid(
                f"PRESENCE_INVALID: {s!r} -> {p!r}")
    return _greedy_drop(presence_by_station, day_sets, cap)


def _selftest():
    # --- codex P1-1 locking KATs: exact churn arithmetic -------------
    stable = {f"A{i}": 0.99 for i in range(8)}
    flappy = {"Z7": 0.87, "Z8": 0.86, "Z9": 0.85}
    pres = dict(stable, **flappy)
    d_even = sorted(stable) + sorted(flappy)
    d_odd = sorted(stable)
    days3 = [d_even, d_odd] * 5
    # first score exactly 8/11; ONE drop only; internal score exactly
    # 4/5; ten survivors; typing None
    assert churn_exact(list(pres), days3) == Fraction(8, 11)
    r = select_fixture(pres, days3, cap=16)
    assert r["selected"] == sorted(list(stable) + ["Z7", "Z8"]), r
    assert churn_exact(r["selected"], days3) == Fraction(4, 5)
    assert r["typing"] is None

    # mirror no-tolerance case: day 5 additionally missing A7 ->
    # post-first-drop churn 7/9 < 4/5 EXACTLY -> a FURTHER drop must
    # occur (no epsilon credit); final 9 stations at exact 70/81
    days_m = [list(d) for d in days3]
    days_m[5] = sorted(set(d_odd) - {"A7"})
    two_flappy = dict(stable, Z8=0.86, Z9=0.85)
    ds_m = [sorted(set(d_even) - {"Z7"}) if i % 2 == 0 else days_m[i]
            for i, _ in enumerate(days_m)]
    assert churn_exact(list(two_flappy), ds_m) == \
        Fraction(7 * 8 + 2 * 7, 9 * 10)   # 70/90 = 7/9 < 4/5
    r = select_fixture(two_flappy, ds_m, cap=16)
    assert len(r["selected"]) == 9 and "Z9" not in r["selected"], r
    assert churn_exact(r["selected"], ds_m) == Fraction(70, 81)
    assert r["typing"] is None

    # --- codex P1-2 locking KATs: lookback cross-binding -------------
    cut = "2026-08-20"
    full_days = [(date(2026, 8, 20) - timedelta(days=89 - i))
                 .isoformat() for i in range(90)]
    sts = [f"S{i:02d}" for i in range(20)]
    rec_full = {d: list(sts) for d in full_days}
    r = select("cascadia", rec_full, cut)
    assert len(r["selected"]) == 16 and r["typing"] is None

    for bad in ({}, {full_days[0]: sts},
                {d: sts for d in full_days[:89]},
                dict(rec_full, **{"2026-08-21": sts})):
        try:
            select("cascadia", bad, cut)
            raise AssertionError(f"frame {len(bad)} must refuse")
        except SelectionInputInvalid as e:
            assert "LOOKBACK_FRAME_INVALID" in str(e)
    gap = {d: sts for d in full_days if d != full_days[40]}
    gap["2026-08-21"] = sts       # 90 records, not consecutive
    try:
        select("cascadia", gap, cut)
        raise AssertionError("nonconsecutive must refuse")
    except SelectionInputInvalid as e:
        assert "LOOKBACK_FRAME_INVALID" in str(e)

    # declared 81/90 must match the derived count exactly (14-station
    # pool so the 0.9-presence station fits inside cascadia's cap 16)
    sts14 = [f"S{i:02d}" for i in range(14)]
    rec81 = {d: (list(sts14) if i < 81 else [s for s in sts14
                                             if s != "S00"])
             for i, d in enumerate(full_days)}
    r = select("cascadia", rec81, cut,
               presence_declared={"S00": 81})
    assert "S00" in r["selected"]
    r = select("cascadia", rec81, cut,
               presence_declared={"S00": 0.9})   # 81/90 == 9/10
    for wrong in (80, 82, 0.85):
        try:
            select("cascadia", rec81, cut,
                   presence_declared={"S00": wrong})
            raise AssertionError(f"declared {wrong} must refuse")
        except SelectionInputInvalid as e:
            assert "PRESENCE_LOOKBACK_MISMATCH" in str(e)

    # codex's 90-empty-days probe: derived presence is 0 everywhere ->
    # INSUFFICIENT_POOL, never an admitted churn of 1.0
    try:
        select("cascadia", {d: [] for d in full_days}, cut)
        raise AssertionError("empty lookback must refuse")
    except InsufficientPool as e:
        assert "INSUFFICIENT_POOL" in str(e)

    # --- codex P1-3 locking KATs: cap policy -------------------------
    for bad_cap in (0, 8, 15, 17):
        try:
            select("cascadia", rec_full, cut, cap=bad_cap)
            raise AssertionError(f"cap={bad_cap} must refuse")
        except SelectionInputInvalid as e:
            assert "CAP_OVERRIDE_REFUSED" in str(e)
    assert select("cascadia", rec_full, cut, cap=16) == \
        select("cascadia", rec_full, cut)
    try:
        select_fixture({f"T{i}": 0.9 for i in range(9)},
                       [[f"T{i}" for i in range(9)]] * 3, cap=7)
        raise AssertionError("fixture cap < MINIMUM must refuse")
    except SelectionInputInvalid as e:
        assert "CAP_INVALID" in str(e)

    # --- retained REV-1 fixtures (fixture path) ----------------------
    pres_n = {f"S{i:02d}": 0.78 + i * 0.02 for i in range(12)}
    days_n = [[f"S{i:02d}" for i in range(12)]] * 5
    r = select_fixture(pres_n, days_n, cap=8)
    elig = {s: p for s, p in pres_n.items() if p >= PRESENCE_FLOOR}
    assert r["selected"] == sorted(
        sorted(elig, key=lambda s: (-elig[s], s))[:8])
    pres_t = {s: 0.90 for s in "BACEDFGHI"}
    assert select_fixture(pres_t, [list(pres_t)] * 3,
                          cap=8)["selected"] == sorted(pres_t)[:8]
    try:
        select_fixture({f"T{i}": 0.9 for i in range(7)},
                       [["T0"]] * 3, cap=16)
        raise AssertionError("pool=7 must refuse")
    except InsufficientPool:
        pass
    pres4 = {"A": 0.90, "AB": 0.90}
    pres4.update({f"K{i}": 0.99 for i in range(7)})
    kd = sorted(f"K{i}" for i in range(7))
    r = select_fixture(pres4, [kd + ["A"], kd + ["AB"]] * 4, cap=16)
    assert "AB" not in r["selected"] and "A" in r["selected"], r
    pres5 = {f"M{i}": 0.90 for i in range(8)}
    ms = sorted(pres5)
    r = select_fixture(pres5, [ms[:4], ms[4:]] * 4, cap=16)
    assert r["typing"] == "BELOW_FLOOR" and len(r["selected"]) == 8

    # frozen constants as module literals
    assert (PRESENCE_FLOOR, CHURN_FLOOR, MINIMUM) == (0.85, 0.80, 8)
    assert CAPS == {"istanbul_marmara": 16, "socal_coachella": 20,
                    "turkey_kahramanmaras": 14, "cascadia": 16}
    assert _CHURN_FLOOR_F == Fraction(str(CHURN_FLOOR))
    assert _PRESENCE_FLOOR_F == Fraction(str(PRESENCE_FLOOR))
    print("w2_selection selftest (REV 2): ALL PASS")


if __name__ == "__main__":
    _selftest()
