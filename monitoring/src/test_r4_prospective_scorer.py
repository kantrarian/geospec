#!/usr/bin/env python3
"""KAT battery for the Amendment R4 prospective scorer. Offline, stdlib-only.

Run:  python monitoring/src/test_r4_prospective_scorer.py   -> N/N PASS expected
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r4_prospective_scorer import (Event, Exclusion, R4_CONFIG, build_episodes,
                                   build_causal_windows, day_excluded, decluster,
                                   gk_distance_km, gk_time_days,
                                   haversine_km, hit_eligible, molchan, score)

PASS = []


def kat(name, ok, detail=""):
    PASS.append(bool(ok))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def days_range(start, n, tier_fn):
    d0 = date(*map(int, start.split("-")))
    return [((d0 + timedelta(days=i)).isoformat(), tier_fn(i)) for i in range(n)]


print("=== R4 scorer KATs ===")

# K1 -- G-K windows + cap
L71, T71 = gk_distance_km(7.1), gk_time_days(7.1)
L55, T55 = gk_distance_km(5.5), gk_time_days(5.5)
kat("K1 G-K windows: M7.1 -> L~73km, T CAPPED at 365d; M5.5 -> L~46km, T~264d uncapped",
    abs(L71 - 72.9) < 1.5 and T71 == 365.0 and abs(L55 - 46.2) < 1.5 and 250 < T55 < 280,
    f"L71={L71:.1f} T71={T71:.0f} L55={L55:.1f} T55={T55:.0f}")

# K2 -- declustering: aftershock+foreshock inside window removed; distant + late kept
main = Event("m", "2026-08-10", 32.8, 130.7, 7.0, "kumamoto")
aft = Event("a", "2026-08-15", 32.9, 130.8, 5.8, "kumamoto")     # inside L,T of m
fore = Event("f", "2026-08-05", 32.8, 130.6, 5.6, "kumamoto")    # before m, inside window (symmetric)
far = Event("x", "2026-08-15", 35.7, 139.7, 5.8, "tokyo_kanto")  # ~880 km away
late = Event("l", "2027-09-01", 32.8, 130.7, 5.9, "kumamoto")    # beyond capped 365d
ms, cl = decluster([main, aft, fore, far, late])
kat("K2 declustering: {m, far, late} = mainshocks; {aftershock, foreshock} removed",
    {e.event_id for e in ms} == {"m", "x", "l"} and {e.event_id for e in cl} == {"a", "f"},
    f"mainshocks={sorted(e.event_id for e in ms)}")

# K3 -- symmetric exclusion: post-event days vanish from BOTH sides of tau
excl = build_causal_windows([main])
kat("K3a exclusion window: opens at mainshock, capped end",
    excl[0].start == "2026-08-10" and excl[0].end == "2027-08-10")
series = {"kumamoto": days_range("2026-08-01", 30, lambda i: 2 if i >= 9 else 0)}  # alarm from 08-10 on
eps = build_episodes(series["kumamoto"], "kumamoto", excl)
mol = molchan(series, eps, [], excl)
pr = mol["per_region"]["kumamoto"]
kat("K3b symmetric: 21 post-event alarm days excluded from numerator AND time base",
    pr["scoreable_days"] == 9 and pr["alarm_days"] == 0 and pr["excluded_days"] == 21,
    f"{pr}")
kat("K3c region-wide temporal exclusion captures a same-region day inside the window",
    day_excluded("kumamoto", "2026-09-01", excl) is True)

# K4 -- episode grouping: gap<=3 merges, gap>3 splits; tier>=2 only
tiers = {0: 2, 1: 2, 5: 2, 6: 2, 12: 2}   # days 0,1 + 5,6 (gap 4 -> split) + 12 (gap 6 -> split)
series4 = days_range("2026-09-01", 15, lambda i: tiers.get(i, 0))
eps4 = build_episodes(series4, "r", [])
kat("K4 episode grouping: gap 4 and 6 split -> 3 episodes; tier-1 days never group",
    len(eps4) == 3 and [(e.onset, e.end) for e in eps4] ==
    [("2026-09-01", "2026-09-02"), ("2026-09-06", "2026-09-07"), ("2026-09-13", "2026-09-13")])

# K5 -- one hit per episode; one crediting episode per mainshock (nearest-preceding)
seriesK5 = days_range("2026-09-01", 40, lambda i: 2 if i in (0, 1, 20, 21) else 0)
epsK5 = build_episodes(seriesK5, "r", [])
ev1 = Event("E1", "2026-09-25", 0.0, 0.0, 6.0, "r")   # within 14d of episode-2 (ends 09-22)
out = score(epsK5, [ev1], [], today="2026-12-31")
hit_eps = [e for e in epsK5 if e.status == "hit"]
fa_eps = [e for e in epsK5 if e.status == "false_alarm"]
kat("K5a nearest-preceding episode credited once; earlier episode becomes FA after window closes",
    len(hit_eps) == 1 and hit_eps[0].onset == "2026-09-21" and len(fa_eps) == 1,
    f"outcomes={[o['outcome'] for o in out['outcomes']]}")
ev2 = Event("E2", "2026-09-26", 0.0, 0.0, 5.7, "r")   # second event, episode already credited
epsK5b = build_episodes(seriesK5, "r", [])
out2 = score(epsK5b, [ev1, ev2], [], today="2026-12-31")
oc = {o["event"]: o["outcome"] for o in out2["outcomes"]}
# R4 v2: E1 (a mainshock) opens its own exclusion window; E2 one day later lands INSIDE it,
# so E2 is excluded_unscored, not a miss -- either way it cannot re-credit E1's episode.
kat("K5b second mainshock cannot re-credit the same episode (excluded by E1's causal window)",
    oc["E1"] == "hit" and oc["E2"] in ("miss", "excluded_unscored"), f"{oc}")

# K6 -- supersession + freshness (14 tier-0 days) inside an exclusion window
m0 = Event("M0", "2026-09-01", 32.8, 130.7, 6.0, "kumamoto")
excl6 = build_causal_windows([m0])
# alarm standing from BEFORE m0 (never reset) -> stale; then reset 14d, fresh episode, bigger event
def tier6(i):
    d0 = date(2026, 8, 20) + timedelta(days=i)
    if d0 <= date(2026, 9, 5):  return 2          # standing alarm through the mainshock
    if d0 <= date(2026, 9, 19): return 0          # 14 tier-0 days (reset)
    if d0 <= date(2026, 9, 25): return 2          # FRESH episode
    return 0
series6 = {"kumamoto": days_range("2026-08-20", 45, lambda i: tier6(i))}
eps6 = build_episodes(series6["kumamoto"], "kumamoto", excl6)
big = Event("BIG", "2026-09-28", 32.82, 130.72, 6.5, "kumamoto")   # larger, inside M0's window
out6 = score(eps6, [m0, big], excl6, today="2026-12-31")
oc6 = {o["event"]: o["outcome"] for o in out6["outcomes"]}
fresh_eps = [e for e in eps6 if e.fresh]
kat("K6a freshness: only the post-reset episode is fresh",
    len(fresh_eps) == 1 and fresh_eps[0].onset == "2026-09-20",
    f"episodes={[(e.onset, e.fresh) for e in eps6]}")
kat("K6b supersession: larger in-window event scored as hit via the FRESH episode only",
    oc6["BIG"] == "hit_supersession", f"{oc6}")
stale = Event("BIG2", "2026-09-04", 32.82, 130.72, 6.5, "kumamoto")  # larger but only stale alarm up
eps6b = build_episodes(series6["kumamoto"], "kumamoto", excl6)
out6b = score(eps6b, [m0, stale], excl6, today="2026-12-31")
oc6b = {o["event"]: o["outcome"] for o in out6b["outcomes"]}
kat("K6c supersession without a fresh episode -> unscored (standing alarm cannot claim it)",
    oc6b["BIG2"] == "supersession_no_fresh_episode_unscored", f"{oc6b}")

# K7 -- Molchan accounting on a constructed scenario with known tau, nu
seriesK7 = {"A": days_range("2026-08-01", 20, lambda i: 2 if i < 5 else 0),   # 5/20 alarm
            "B": days_range("2026-08-01", 20, lambda i: 0)}                    # 0/20
epsK7 = build_episodes(seriesK7["A"], "A", []) + build_episodes(seriesK7["B"], "B", [])
evh = Event("H", "2026-08-10", 0.0, 0.0, 6.0, "A")    # within 14d of episode end 08-05 -> hit
evm = Event("Miss", "2026-08-15", 0.0, 0.0, 6.0, "B") # region B never alarmed -> miss
outK7 = score(epsK7, [evh, evm], [], today="2026-12-31")
molK7 = molchan(seriesK7, epsK7, outK7["outcomes"], [])
p = molK7["pooled"]
kat("K7 Molchan: tau=5/40, nu=1/2, hits=1, misses=1",
    p["scoreable_days"] == 40 and p["alarm_days"] == 5 and abs(p["tau"] - 0.125) < 1e-12
    and p["hits"] == 1 and p["misses"] == 1 and abs(p["nu"] - 0.5) < 1e-12, f"{p}")

# K8 -- descriptive-only banner below the pre-registered minimum sample
kat("K8 descriptive_only=True below 10 pooled mainshocks", molK7["descriptive_only"] is True)

# K9 -- pending episodes stay pending while their window is open
epsK9 = build_episodes(days_range("2026-12-20", 5, lambda i: 2), "r", [])
score(epsK9, [], [], today="2026-12-26")   # window (14d past 12-24) NOT yet elapsed
kat("K9 open-window episode remains pending (not prematurely a false alarm)",
    epsK9[0].status == "pending")

# K10 -- pre-start mainshock opens the exclusion, but is never itself scored
m_prestart = Event("PRE", "2026-07-28", 32.75, 130.65, 7.1, "kumamoto")
excl10 = build_causal_windows([m_prestart])
kat("K10a pre-start M7.1 opens Kumamoto exclusion through 2027-07-28",
    excl10[0].end == "2027-07-28")
series10 = {"kumamoto": days_range("2026-07-29", 10, lambda i: 2)}
eps10 = build_episodes(series10["kumamoto"], "kumamoto", excl10)
out10 = score(eps10, [], excl10, today="2026-12-31")   # PRE filtered out before score()
mol10 = molchan(series10, eps10, out10["outcomes"], excl10)
kat("K10b post-event alarm days all excluded from tau; no outcomes scored",
    mol10["per_region"]["kumamoto"]["scoreable_days"] == 0
    and mol10["pooled"]["admissible_mainshocks"] == 0)

# K11 -- exclusion-contained episode with no supersession closes as 'excluded', never FA
kat("K11 exclusion-contained episode closes as 'excluded' (symmetric guarantee)",
    len(eps10) == 1 and eps10[0].excluded and eps10[0].status == "excluded",
    f"status={eps10[0].status}")

# ============================================================================
# R4 v3 COMPOSED RED-KATs (codex CE1-CE5 + recheck R4-R1..R4-R4) via run()'s ACTUAL
# composition: one causal ledger over ALL raw events -> episodes -> score. No KAT may
# hand-feed a different ledger than run() uses (the flaw codex caught in v2).
# ============================================================================
print("--- composed red-KATs via run()-faithful pipeline ---")

def run_like(events, series_by_region, start, today):
    """Mirror run() exactly: causal ledger over ALL events; episodes+score+molchan from it."""
    windows = build_causal_windows(events)
    eps = []
    for region, days in series_by_region.items():
        eps.extend(build_episodes(days, region, windows, start_date=start))
    sc = score(eps, [e for e in events if e.t >= start], windows, today)
    oc = {}
    for o in sc["outcomes"]:
        oc.setdefault(o["event"], o["outcome"])
    return oc, windows, eps

# CE1 / R4-R1 -- batch-removed earlier event still seeds supersession guard
m0c = Event("M0", "2026-09-01", 32.8, 130.7, 6.0, "kumamoto")
bigc = Event("BIG", "2026-09-28", 32.81, 130.71, 6.5, "kumamoto", origin_utc="2026-09-28T12:00:00Z")
oc1, _, _ = run_like([m0c, bigc], {"kumamoto": days_range("2026-08-20", 45, lambda i: 2)},
                     "2026-07-29", "2026-12-31")
kat("CE1/R4-R1 supersession-reset bypass CLOSED in run()-composition (BIG not an ordinary hit)",
    oc1.get("BIG") == "supersession_no_fresh_episode_unscored", f"{oc1}")

# R4-R1b -- pre-start region-wide guard suppresses a spatially distant same-region event
pre = Event("PRE", "2026-07-28", 0.0, 0.0, 7.1, "r", origin_utc="2026-07-28T00:00:00Z")
far = Event("FAR", "2026-08-15", 0.80, 0.0, 5.6, "r", origin_utc="2026-08-15T12:00:00Z")  # ~89km, same region
oc1b, _, _ = run_like([pre, far], {"r": days_range("2026-07-29", 30, lambda i: 0)},
                      "2026-07-29", "2026-12-31")
kat("R4-R1b pre-start region-wide guard CLOSED: distant same-region event is excluded, not a miss",
    oc1b.get("FAR") == "excluded_unscored", f"{oc1b}")

# CE2 / R4-R2 -- excluded smaller event cannot extend the guard
root = Event("ROOT", "2026-08-01", 0.0, 0.0, 7.1, "r", origin_utc="2026-08-01T00:00:00Z")
inside = Event("INSIDE", "2026-12-20", 0.0, 0.0, 6.0, "r", origin_utc="2026-12-20T00:00:00Z")
after = Event("AFTER", "2027-09-01", 0.0, 0.0, 5.6, "r", origin_utc="2027-09-01T12:00:00Z")
win = build_causal_windows([root, inside, after])
# ROOT window ends ~2027-08-01 (365 cap); INSIDE must NOT open a window extending past it;
# AFTER (2027-09-01) is past ROOT's window and INSIDE opened none -> AFTER is scoreable.
r_windows = [w for w in win if w.mainshock_id in ("ROOT", "INSIDE")]
inside_opened = any(w.mainshock_id == "INSIDE" for w in win)
oc2, _, _ = run_like([root, inside, after], {"r": days_range("2026-07-29", 5, lambda i: 0)},
                     "2026-07-29", "2027-12-31")
kat("CE2/R4-R2 excluded smaller event opens NO window; later event past ROOT is scoreable",
    not inside_opened and oc2.get("AFTER") == "miss", f"inside_opened={inside_opened} {oc2}")

# CE3 / R4-R3 -- merged memberships share identical exclusion accounting
ev_ab = Event("AB", "2026-08-10", 0.0, 0.0, 6.5, region="A", origin_utc="2026-08-10T00:00:00Z",
              regions=("A", "B"))
wins_ab = build_causal_windows([ev_ab])
regions_excluded = sorted({w.region for w in wins_ab})
kat("CE3/R4-R3 merged memberships CLOSED: event excludes BOTH A and B, not just the first",
    regions_excluded == ["A", "B"], f"excluded={regions_excluded}")

# CE4 -- same-date post-event alarm rejected (UTC eligibility)
oc4, _, _ = run_like([Event("E", "2026-10-01", 0.0, 0.0, 6.0, "r", origin_utc="2026-10-01T00:01:00Z")],
                     {"r": days_range("2026-10-01", 1, lambda i: 2)}, "2026-07-29", "2026-12-31")
kat("CE4 same-date post-event CLOSED: 00:01Z event vs 23:59Z alarm -> miss", oc4.get("E") == "miss",
    f"{oc4}")

# CE5 / R4-R4 -- n>=10 WITHOUT a registered stats plan stays descriptive_only
seriesN = {}
evs = []
for j in range(10):
    reg = f"z{j}"
    seriesN[reg] = days_range(f"2026-08-0{1 if j<9 else 1}", 3, lambda i: 0)  # no alarms -> misses
    evs.append(Event(f"EV{j}", "2026-08-20", float(j), 0.0, 6.0, reg,
                     origin_utc="2026-08-20T00:00:00Z"))
windowsN = build_causal_windows(evs)
epsN = []
for reg, days in seriesN.items():
    epsN.extend(build_episodes(days, reg, windowsN, start_date="2026-07-29"))
scN = score(epsN, evs, windowsN, "2026-12-31")
molN = molchan({r: [(d, ti) for d, ti in days] for r, days in seriesN.items()},
               epsN, scN["outcomes"], windowsN)
kat("CE5/R4-R4 n>=10 without a registered stats plan STAYS descriptive_only (over-claim guard)",
    molN["pooled"]["admissible_mainshocks"] >= 10 and molN["descriptive_only"] is True
    and molN["eligible_for_plan"] is True, f"n={molN['pooled']['admissible_mainshocks']} "
    f"descriptive={molN['descriptive_only']} eligible={molN['eligible_for_plan']}")

n = sum(PASS)
print(f"=== R4 scorer KATs: {n}/{len(PASS)} PASS ===")
raise SystemExit(0 if n == len(PASS) else 1)
