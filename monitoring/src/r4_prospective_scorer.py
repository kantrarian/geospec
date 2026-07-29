#!/usr/bin/env python3
"""r4_prospective_scorer.py - Amendment R4 prospective-arm scorer.

Implements docs/AMENDMENT_2026-07-29_prospective_arm.md (registered + owner-signed
2026-07-29) EXACTLY:

  - Gardner-Knopoff declustering to mainshocks-only (365-day time cap, disclosed).
  - SYMMETRIC region exclusion after any admissible mainshock: excluded region-days
    vanish from BOTH the alarm-time numerator and the time base; events inside an
    exclusion window are neither hits nor misses (supersession excepted).
  - Alarm EPISODES (consecutive tier>=2 scoreable days, gap tolerance 3 d);
    one episode <= one hit; one mainshock <= one crediting episode.
  - Supersession: a larger event re-opens the window and is scoreable only if a
    FRESH episode preceded it (onset after >=14 consecutive tier-0 days).
  - Molchan accumulation (tau, nu) from 2026-07-29; per-region + pooled.
  - Descriptive-only until >=10 pooled admissible mainshocks (pre-registered).

Pure standard library. The scoring core is pure functions over plain data so the
KAT battery (test_r4_prospective_scorer.py) runs offline; network I/O (USGS) and
file I/O live at the edges only.

Usage (on the runner, after the daily ensemble):
    python -m src.r4_prospective_scorer                    # writes docs/r4_prospective_record.json
    python -m src.r4_prospective_scorer --end-date YYYY-MM-DD
"""
from __future__ import annotations

import csv
import json
import math
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# ============================================================================
# R4 REGISTERED PARAMETERS (docs/AMENDMENT_2026-07-29_prospective_arm.md)
# ============================================================================

R4_CONFIG = {
    "spec": "AMENDMENT_2026-07-29_prospective_arm (R4)",
    "start_date": "2026-07-29",        # accumulation start (R2 effective date)
    "min_magnitude": 5.5,              # admissible events (R2)
    "hit_window_days": 14,             # R2
    "alarm_min_tier": 2,               # registered alarm level
    "episode_gap_days": 3,             # episode grouping tolerance
    "alarm_reset_days": 14,            # consecutive tier-0 days for a "fresh" episode
    "exclusion_cap_days": 365,         # disclosed G-K deviation
    "region_buffer_km": 100.0,         # admissibility buffer (current config)
    "min_mainshocks_for_skill": 10,    # pre-registered minimum sample
}


# ============================================================================
# GARDNER-KNOPOFF WINDOWS
# ============================================================================

def gk_distance_km(mag: float) -> float:
    """G-K (1974) spatial window: L = 10^(0.1238*M + 0.983) km."""
    return 10.0 ** (0.1238 * mag + 0.983)


def gk_time_days(mag: float, cap_days: int = R4_CONFIG["exclusion_cap_days"]) -> float:
    """G-K (1974) temporal window, capped (R4 disclosed deviation).

    T = 10^(0.032*M + 2.7389) d for M >= 6.5, else 10^(0.5409*M - 0.547) d.
    """
    t = 10.0 ** (0.032 * mag + 2.7389) if mag >= 6.5 else 10.0 ** (0.5409 * mag - 0.547)
    return min(t, float(cap_days))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ============================================================================
# DATA TYPES (plain, serializable)
# ============================================================================

@dataclass
class Event:
    event_id: str
    t: str            # ISO date (day resolution is sufficient for 14-d windows)
    lat: float
    lon: float
    mag: float
    region: str = ""  # assigned admissibility region ("" = none)


def _d(s: str) -> date:
    return date(*map(int, s[:10].split("-")))


# ============================================================================
# STEP 1 - DECLUSTERING (targets = mainshocks only)
# ============================================================================

def decluster(events: list[Event]) -> tuple[list[Event], list[Event]]:
    """G-K declustering: an event is CLUSTERED (removed) if a LARGER event exists
    with distance <= L(M_larger) and |dt| <= T(M_larger) (symmetric in time, so
    foreshocks and aftershocks are both removed; ties broken by time, earlier wins).
    Returns (mainshocks, clustered)."""
    order = sorted(events, key=lambda e: (-e.mag, e.t))
    mainshocks: list[Event] = []
    clustered: list[Event] = []
    for e in order:
        parent = None
        for m in mainshocks:
            if m.mag < e.mag or (m.mag == e.mag and m.t == e.t and m is e):
                continue
            dt = abs((_d(e.t) - _d(m.t)).days)
            if dt <= gk_time_days(m.mag) and haversine_km(e.lat, e.lon, m.lat, m.lon) <= gk_distance_km(m.mag):
                parent = m
                break
        (clustered if parent else mainshocks).append(e)
    mainshocks.sort(key=lambda e: e.t)
    clustered.sort(key=lambda e: e.t)
    return mainshocks, clustered


# ============================================================================
# STEP 2 - REGION EXCLUSION WINDOWS (symmetric unscoreability)
# ============================================================================

@dataclass
class Exclusion:
    region: str
    mainshock_id: str
    mag: float
    lat: float
    lon: float
    start: str          # mainshock date
    end: str            # start + min(gk_time, cap)


def build_exclusions(mainshocks: list[Event]) -> list[Exclusion]:
    out = []
    for m in mainshocks:
        if not m.region:
            continue
        end = _d(m.t) + timedelta(days=int(round(gk_time_days(m.mag))))
        out.append(Exclusion(m.region, m.event_id, m.mag, m.lat, m.lon, m.t, end.isoformat()))
    return out


def day_excluded(region: str, day: str, exclusions: list[Exclusion]) -> bool:
    return any(x.region == region and x.start <= day <= x.end for x in exclusions)


def event_in_exclusion(e: Event, exclusions: list[Exclusion]) -> Exclusion | None:
    """The exclusion window (if any) containing event e spatially+temporally,
    opened by an EARLIER mainshock. Same-day parent does not exclude itself."""
    for x in exclusions:
        if x.mainshock_id == e.event_id:
            continue
        if x.start <= e.t <= x.end and haversine_km(e.lat, e.lon, x.lat, x.lon) <= gk_distance_km(x.mag):
            return x
    return None


# ============================================================================
# STEP 3 - ALARM EPISODES on the scoreable day-series
# ============================================================================

@dataclass
class Episode:
    region: str
    onset: str
    end: str
    n_days: int
    fresh: bool         # onset preceded by >= alarm_reset_days consecutive tier-0 days
    excluded: bool = False      # ALL days inside an exclusion window -> supersession-only
    status: str = "pending"     # pending | hit | false_alarm | excluded
    credited_event: str = ""


def build_episodes(days: list[tuple[str, int]], region: str,
                   exclusions: list[Exclusion],
                   gap: int = R4_CONFIG["episode_gap_days"],
                   min_tier: int = R4_CONFIG["alarm_min_tier"],
                   reset: int = R4_CONFIG["alarm_reset_days"]) -> list[Episode]:
    """days: sorted (date, tier) for one region, restricted to the accumulation
    period. Episodes are grouped on the FULL series (exclusion windows included):
    an episode wholly inside an exclusion is tagged `excluded` and participates in
    scoring ONLY via supersession (the signed R4 exception) -- it can never become
    a hit or a false alarm through the normal path, and its days never enter the
    Molchan accounting (that filter lives in molchan()). Gap tolerance is
    calendar-day."""
    scoreable = list(days)
    alarm_days = [d0 for d0, t in scoreable if t >= min_tier]
    eps: list[Episode] = []
    if alarm_days:
        start = prev = alarm_days[0]
        for d0 in alarm_days[1:]:
            if (_d(d0) - _d(prev)).days > gap:
                eps.append(Episode(region, start, prev, (_d(prev) - _d(start)).days + 1, False))
                start = d0
            prev = d0
        eps.append(Episode(region, start, prev, (_d(prev) - _d(start)).days + 1, False))
    # freshness: >= reset consecutive tier-0 scoreable days immediately before onset
    tier_by_day = dict(scoreable)
    for ep in eps:
        run = 0
        d0 = _d(ep.onset) - timedelta(days=1)
        while True:
            key = d0.isoformat()
            if key not in tier_by_day:
                break               # data gap / excluded day interrupts the run
            if tier_by_day[key] == 0:
                run += 1
                d0 -= timedelta(days=1)
            else:
                break
        ep.fresh = run >= reset
        ep.excluded = all(day_excluded(region, (_d(ep.onset) + timedelta(days=i)).isoformat(),
                                       exclusions)
                          for i in range((_d(ep.end) - _d(ep.onset)).days + 1))
    return eps


# ============================================================================
# STEP 4 - SCORING (hits / false alarms / misses; one-to-one crediting)
# ============================================================================

def score(episodes: list[Episode], mainshocks: list[Event],
          exclusions: list[Exclusion], today: str,
          window: int = R4_CONFIG["hit_window_days"]) -> dict:
    """Mutates episode statuses; returns per-mainshock outcomes."""
    outcomes = []
    credited: set[int] = set()          # episode indices already credited
    for m in mainshocks:
        if not m.region:
            continue
        x = event_in_exclusion(m, exclusions)
        if x is not None:
            # supersession: larger than the window's mainshock AND a fresh episode precedes it
            if m.mag > x.mag:
                cands = [(i, e) for i, e in enumerate(episodes)
                         if e.region == m.region and e.fresh and i not in credited
                         and e.onset <= m.t and (_d(m.t) - _d(e.end)).days <= window
                         and (_d(m.t) - _d(e.onset)).days >= 0]
                if cands:
                    i, e = max(cands, key=lambda ie: ie[1].onset)
                    e.status = "hit"; e.credited_event = m.event_id; credited.add(i)
                    outcomes.append({"event": m.event_id, "mag": m.mag, "region": m.region,
                                     "outcome": "hit_supersession", "episode_onset": e.onset})
                else:
                    outcomes.append({"event": m.event_id, "mag": m.mag, "region": m.region,
                                     "outcome": "supersession_no_fresh_episode_unscored"})
            else:
                outcomes.append({"event": m.event_id, "mag": m.mag, "region": m.region,
                                 "outcome": "excluded_unscored"})
            continue
        # normal scoring: nearest-preceding uncredited NON-EXCLUDED episode within the window
        cands = [(i, e) for i, e in enumerate(episodes)
                 if e.region == m.region and i not in credited and not e.excluded
                 and e.onset <= m.t and (_d(m.t) - _d(e.end)).days <= window]
        if cands:
            i, e = max(cands, key=lambda ie: ie[1].onset)
            e.status = "hit"; e.credited_event = m.event_id; credited.add(i)
            outcomes.append({"event": m.event_id, "mag": m.mag, "region": m.region,
                             "outcome": "hit", "episode_onset": e.onset})
        else:
            outcomes.append({"event": m.event_id, "mag": m.mag, "region": m.region,
                             "outcome": "miss"})
    # close remaining episodes whose windows have fully elapsed; exclusion-contained
    # episodes close as "excluded" (symmetric unscoreability), never as false alarms
    for e in episodes:
        if e.status == "pending" and (_d(today) - _d(e.end)).days > window:
            e.status = "excluded" if e.excluded else "false_alarm"
    return {"outcomes": outcomes}


# ============================================================================
# STEP 5 - MOLCHAN ACCOUNTING
# ============================================================================

def molchan(day_series: dict[str, list[tuple[str, int]]], episodes: list[Episode],
            outcomes: list[dict], exclusions: list[Exclusion],
            min_tier: int = R4_CONFIG["alarm_min_tier"]) -> dict:
    per_region = {}
    tot_days = tot_alarm = 0
    for region, days in day_series.items():
        sc = [(d0, t) for d0, t in days if not day_excluded(region, d0, exclusions)]
        n = len(sc)
        a = sum(1 for _, t in sc if t >= min_tier)
        excl = len(days) - n
        per_region[region] = {"scoreable_days": n, "alarm_days": a,
                              "excluded_days": excl,
                              "tau": (a / n) if n else None}
        tot_days += n; tot_alarm += a
    scored = [o for o in outcomes if o["outcome"] in ("hit", "hit_supersession", "miss")]
    hits = sum(1 for o in scored if o["outcome"].startswith("hit"))
    misses = sum(1 for o in scored if o["outcome"] == "miss")
    n_main = hits + misses
    return {
        "per_region": per_region,
        "pooled": {
            "scoreable_days": tot_days, "alarm_days": tot_alarm,
            "tau": (tot_alarm / tot_days) if tot_days else None,
            "admissible_mainshocks": n_main, "hits": hits, "misses": misses,
            "nu": (misses / n_main) if n_main else None,
        },
        "descriptive_only": n_main < R4_CONFIG["min_mainshocks_for_skill"],
    }


# ============================================================================
# I/O EDGES (not exercised by the KATs)
# ============================================================================

REPO = Path(__file__).resolve().parents[2]


def load_day_series(csv_path: Path, start: str, end: str) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d0 = row["date"][:10]
            if start <= d0 <= end:
                out.setdefault(row["region"], []).append((d0, int(row["tier"])))
    for v in out.values():
        v.sort()
    return out


def fetch_usgs_events(start: str, end: str, region_defs: dict) -> list[Event]:
    """Admissible-candidate events near any region center (assignment via buffer)."""
    events: list[Event] = []
    for rid, rd in region_defs.items():
        lat, lon = rd["center"]
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
               f"&starttime={start}&endtime={end}&latitude={lat}&longitude={lon}"
               f"&maxradiuskm=500&minmagnitude={R4_CONFIG['min_magnitude']}")
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        for feat in data.get("features", []):
            p = feat["properties"]; c = feat["geometry"]["coordinates"]
            t = datetime.fromtimestamp(p["time"] / 1000, timezone.utc).date().isoformat()
            e = Event(feat["id"], t, c[1], c[0], p["mag"])
            if haversine_km(e.lat, e.lon, lat, lon) <= R4_CONFIG["region_buffer_km"]:
                e.region = rid
            if all(e.event_id != x.event_id for x in events):
                events.append(e)
    return events


def run(end_date: str | None = None) -> dict:
    from src.validate_predictions import REGION_DEFINITIONS   # single source of truth
    today = end_date or date.today().isoformat()
    start = R4_CONFIG["start_date"]
    csv_path = REPO / "monitoring" / "dashboard" / "data.csv"
    if not csv_path.exists():
        csv_path = REPO / "docs" / "data.csv"
    series = load_day_series(csv_path, start, today)
    # Fetch from one exclusion-cap BEFORE the accumulation start: mainshocks that
    # precede the start (e.g. the 2026-07-28 Kumamoto M7.1, one day before R4's
    # start) must still OPEN exclusion windows ("Kumamoto unscoreable until
    # 2027-07-28"), even though pre-start events are never themselves scored.
    fetch_start = (_d(start) - timedelta(days=R4_CONFIG["exclusion_cap_days"])).isoformat()
    events = fetch_usgs_events(fetch_start, today, REGION_DEFINITIONS)
    mainshocks, clustered = decluster(events)
    exclusions = build_exclusions([m for m in mainshocks if m.region])
    scoreable_mainshocks = [m for m in mainshocks if m.t >= start]
    episodes: list[Episode] = []
    for region, days in series.items():
        episodes.extend(build_episodes(days, region, exclusions))
    sc = score(episodes, scoreable_mainshocks, exclusions, today)
    mol = molchan(series, episodes, sc["outcomes"], exclusions)
    record = {
        "generated": datetime.now().isoformat(),
        "config": R4_CONFIG,
        "period": {"start": start, "end": today},
        "molchan": mol,
        "episodes": [asdict(e) for e in sorted(episodes, key=lambda e: (e.region, e.onset))],
        "event_outcomes": sc["outcomes"],
        "exclusions": [asdict(x) for x in exclusions],
        "n_events_fetched": len(events), "n_clustered_removed": len(clustered),
        "note": ("DESCRIPTIVE ONLY - below the pre-registered minimum sample "
                 f"({R4_CONFIG['min_mainshocks_for_skill']} pooled mainshocks); no skill claims."
                 if mol["descriptive_only"] else
                 "Minimum sample reached; skill assessment per R4 statistical plan."),
    }
    out = REPO / "docs" / "r4_prospective_record.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"R4 record written: {out} (mainshocks={mol['pooled']['admissible_mainshocks']}, "
          f"tau={mol['pooled']['tau']}, descriptive_only={mol['descriptive_only']})")
    return record


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Amendment R4 prospective-arm scorer")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()
    run(a.end_date)
