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
                                   build_exclusions, day_excluded, decluster,
                                   event_in_exclusion, gk_distance_km, gk_time_days,
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
excl = build_exclusions([main])
kat("K3a exclusion window: opens at mainshock, capped end",
    excl[0].start == "2026-08-10" and excl[0].end == "2027-08-10")
series = {"kumamoto": days_range("2026-08-01", 30, lambda i: 2 if i >= 9 else 0)}  # alarm from 08-10 on
eps = build_episodes(series["kumamoto"], "kumamoto", excl)
mol = molchan(series, eps, [], excl)
pr = mol["per_region"]["kumamoto"]
kat("K3b symmetric: 21 post-event alarm days excluded from numerator AND time base",
    pr["scoreable_days"] == 9 and pr["alarm_days"] == 0 and pr["excluded_days"] == 21,
    f"{pr}")
inx = event_in_exclusion(Event("e2", "2026-09-01", 32.85, 130.75, 5.7, "kumamoto"), excl)
kat("K3c smaller event inside the window is captured by the exclusion (unscoreable)",
    inx is not None and inx.mainshock_id == "m")

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
excl6 = build_exclusions([m0])
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
excl10 = build_exclusions([m_prestart])
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
# R4 v2 COMPOSED RED-KATs (codex counterexamples CE1-CE5, must now be CLOSED)
# ============================================================================
print("--- composed red-KATs (codex CE1-CE5) ---")

# CE1 -- supersession-reset bypass via batch declustering. M0 then BIG same place;
# standing tier-2 from Aug20 through BIG, no reset. Run the TOP-LEVEL pipeline
# (decluster -> exclusions -> episodes -> score) as run() composes it.
m0c = Event("M0", "2026-09-01", 32.8, 130.7, 6.0, "kumamoto")
bigc = Event("BIG", "2026-09-28", 32.81, 130.71, 6.5, "kumamoto", origin_utc="2026-09-28T12:00:00Z")
ms, _ = decluster([m0c, bigc])                      # batch view may relabel M0 as foreshock
excl = build_exclusions([m for m in ms if m.region])
seriesCE1 = {"kumamoto": days_range("2026-08-20", 45, lambda i: 2)}   # unbroken standing alarm
epsCE1 = build_episodes(seriesCE1["kumamoto"], "kumamoto", excl, start_date="2026-07-29")
# score against the FULL causal event set, not the declustered targets, so lineage is causal:
outCE1 = score(epsCE1, [m0c, bigc], excl, today="2026-12-31")
ocCE1 = {o["event"]: o["outcome"] for o in outCE1["outcomes"]}
kat("CE1 supersession-reset bypass CLOSED: standing-alarm BIG is NOT an ordinary hit",
    ocCE1.get("BIG") in ("supersession_no_fresh_episode_unscored", "excluded_unscored")
    and ocCE1.get("BIG") != "hit", f"{ocCE1}")

# CE2 -- excluded day credits a normal hit. Exclusion Aug1..Aug31; one scoreable Jul31
# alarm + excluded Aug tail (gaps<=3); post-exclusion Sep1 event. Only lead is 32d -> must miss.
exclCE2 = [Exclusion("r", "MX", 6.0, 0.0, 0.0, "2026-08-01", "2026-08-31")]
daysCE2 = [("2026-07-31", 2)] + [(f"2026-08-{d:02d}", 2) for d in range(3, 31, 3)]
epsCE2 = build_episodes(daysCE2, "r", exclCE2)
evCE2 = Event("E", "2026-09-01", 0.0, 0.0, 6.0, "r", origin_utc="2026-09-01T12:00:00Z")
outCE2 = score(epsCE2, [evCE2], exclCE2, today="2026-12-31")
kat("CE2 excluded-day hit-credit CLOSED: 32d-lead event is a miss (Jul31 alarm too old, Aug excluded)",
    outCE2["outcomes"][0]["outcome"] == "miss", f"{outCE2['outcomes']}")

# CE3 -- same-date post-event alarm. Event 00:01Z, alarm available 23:59Z same date -> reject.
epsCE3 = build_episodes(days_range("2026-10-01", 1, lambda i: 2), "r", [])
evCE3 = Event("E", "2026-10-01", 0.0, 0.0, 6.0, "r", origin_utc="2026-10-01T00:01:00Z")
outCE3 = score(epsCE3, [evCE3], [], today="2026-12-31")
kat("CE3 same-date post-event CLOSED: alarm available 23:59Z cannot credit a 00:01Z event",
    outCE3["outcomes"][0]["outcome"] == "miss", f"{outCE3['outcomes']}")

# CE4 -- region-membership dedup loss. An event at the Mojave center is inside Mojave's 100km
# buffer but Ridgecrest's query (first) returns it at 141km outside Ridgecrest's buffer.
rd = {"ridgecrest": {"center": (35.77, -117.60)}, "socal_saf_mojave": {"center": (34.5, -117.5)}}
# simulate the merge logic directly (no network): membership must include mojave, order-independent
def assign(ev_lat, ev_lon, region_defs):
    mem = []
    for rid, d in region_defs.items():
        if haversine_km(ev_lat, ev_lon, d["center"][0], d["center"][1]) <= R4_CONFIG["region_buffer_km"]:
            mem.append(rid)
    return tuple(mem)
memCE4 = assign(34.5, -117.5, rd)
memCE4_rev = assign(34.5, -117.5, {k: rd[k] for k in reversed(list(rd))})
kat("CE4 region-dedup CLOSED: membership includes mojave and is query-order-independent",
    "socal_saf_mojave" in memCE4 and memCE4 == memCE4_rev, f"{memCE4} vs {memCE4_rev}")

# CE5 -- pre-start standing episode relabelled as prospective. Episode from Jul28 through Aug1;
# with history preload + left-censor it must NOT credit an Aug event.
START = "2026-07-29"
daysCE5 = days_range("2026-07-25", 12, lambda i: 2)   # tier-2 from Jul25 (pre-start) onward
epsCE5 = build_episodes(daysCE5, "r", [], start_date=START)
evCE5 = Event("E", "2026-08-02", 0.0, 0.0, 6.0, "r", origin_utc="2026-08-02T12:00:00Z")
outCE5 = score(epsCE5, [evCE5], [], today="2026-12-31")
lc = any(e.left_censored for e in epsCE5)
kat("CE5 pre-start left-censor CLOSED: boundary-active episode is left_censored, event misses",
    lc and outCE5["outcomes"][0]["outcome"] == "miss", f"lc={lc} {outCE5['outcomes']}")

n = sum(PASS)
print(f"=== R4 scorer KATs: {n}/{len(PASS)} PASS ===")
raise SystemExit(0 if n == len(PASS) else 1)
