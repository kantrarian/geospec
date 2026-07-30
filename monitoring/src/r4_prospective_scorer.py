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
    "min_mainshocks_for_skill": 10,    # pre-registered minimum sample (eligibility only)
    "stats_plan_amendment": None,      # R4-R4/R6 §5: descriptive_only lifts ONLY when a
                                       # statistical-plan amendment id is registered here.
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
    t: str                     # ISO date (declustering/exclusion day-resolution)
    lat: float
    lon: float
    mag: float
    region: str = ""           # legacy single-region field (kept for decluster/KAT compat)
    origin_utc: str = ""       # R4-3: full UTC origin timestamp (ISO); default = t+"T00:00:00Z"
    regions: tuple = ()        # R4-4: ALL region memberships (buffers may overlap)

    def __post_init__(self):
        if not self.origin_utc:
            self.origin_utc = self.t[:10] + "T00:00:00Z"
        if not self.regions and self.region:
            self.regions = (self.region,)


def _d(s: str) -> date:
    return date(*map(int, s[:10].split("-")))


def _utc(s: str) -> datetime:
    """Parse an ISO UTC timestamp (trailing Z or offset) to aware datetime."""
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _alarm_available_at(day: str) -> datetime:
    """R4-3/R6: an alarm for local date D is available at 23:59:59Z of D (conservative;
    the real publish-commit time is later, so this can only make hits HARDER to earn)."""
    return _utc(day[:10] + "T23:59:59+00:00")


def hit_eligible(alarm_day: str, event_origin_utc: str,
                 window_days: int = R4_CONFIG["hit_window_days"]) -> bool:
    """R4-3/R6 eligibility: 0 < (event_origin - alarm_available_at) <= window."""
    delta = _utc(event_origin_utc) - _alarm_available_at(alarm_day)
    return timedelta(0) < delta <= timedelta(days=window_days)


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
    origin_utc: str = ""  # opener's full UTC origin (causal ordering for scoring)


# NOTE: build_exclusions() + event_in_exclusion() were REMOVED (R4 v3, codex recheck):
# they were spatial/first-region footguns that composed the pre-v3 bypass. The single
# causal ledger below is the only exclusion authority.
def build_causal_windows(events: list[Event]) -> list[Exclusion]:
    """The single causal exclusion ledger (R4 v3, codex recheck R4-R1/R2/R3). Chronological
    over ALL admissible events INCLUDING pre-start (which seed guard state); per region
    MEMBERSHIP; a window opens ONLY on the no-live-window path or a strictly-larger
    supersession -- a smaller/equal event inside a live window causes NO transition.
    Region-wide + temporal (R6 §2); spatial G-K is for target declustering, not scoreability."""
    windows: list[Exclusion] = []
    for m in sorted(events, key=lambda e: e.origin_utc):
        for region in (m.regions or ((m.region,) if m.region else ())):
            live = [w for w in windows if w.region == region
                    and w.start <= m.t <= w.end and w.origin_utc < m.origin_utc]
            gov = max(live, key=lambda w: w.mag) if live else None
            if gov is None or m.mag > gov.mag:
                end = (_d(m.t) + timedelta(days=int(round(gk_time_days(m.mag))))).isoformat()
                windows.append(Exclusion(region, m.event_id, m.mag, m.lat, m.lon,
                                         m.t, end, m.origin_utc))
    return windows


def day_excluded(region: str, day: str, exclusions: list[Exclusion]) -> bool:
    return any(x.region == region and x.start <= day <= x.end for x in exclusions)


# ============================================================================
# STEP 3 - ALARM EPISODES on the scoreable day-series
# ============================================================================

@dataclass
class Episode:
    region: str
    onset: str
    end: str
    n_days: int
    fresh: bool                 # onset preceded by >= alarm_reset_days consecutive tier-0 days
    alarm_dates: tuple = ()     # R4-2: the explicit SCOREABLE tier>=2 dates in this episode
    excluded: bool = False      # sits entirely inside an exclusion window -> supersession-only
    left_censored: bool = False # R4-5: active at the accumulation boundary -> no hit credit
    status: str = "pending"     # pending | hit | false_alarm | excluded | left_censored
    credited_event: str = ""


def build_episodes(days: list[tuple[str, int]], region: str,
                   exclusions: list[Exclusion],
                   start_date: str = None,
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
    # R4-2: episodes are runs of SCOREABLE tier>=2 dates. An excluded day breaks a run
    # (no bridging across an exclusion boundary); a >gap calendar jump also breaks it.
    scoreable = list(days)
    tier_by_day = dict(scoreable)
    alarm_days = [d0 for d0, ti in scoreable
                  if ti >= min_tier and not day_excluded(region, d0, exclusions)]
    # ALSO track fully-inside-exclusion alarm runs separately (supersession-only episodes).
    excl_alarm_days = [d0 for d0, ti in scoreable
                       if ti >= min_tier and day_excluded(region, d0, exclusions)]

    def group(dates):
        out = []
        if not dates:
            return out
        s = pv = dates[0]; run = [pv]
        for d0 in dates[1:]:
            if (_d(d0) - _d(pv)).days > gap:
                out.append((s, pv, tuple(run))); s = d0; run = []
            run.append(d0); pv = d0
        out.append((s, pv, tuple(run)))
        return out

    eps: list[Episode] = []
    for s, e, dts in group(alarm_days):
        eps.append(Episode(region, s, e, (_d(e) - _d(s)).days + 1, False, alarm_dates=dts))
    for s, e, dts in group(excl_alarm_days):
        eps.append(Episode(region, s, e, (_d(e) - _d(s)).days + 1, False,
                           alarm_dates=dts, excluded=True))

    for ep in eps:
        # freshness: >= reset consecutive tier-0 scoreable days immediately before onset
        run = 0; d0 = _d(ep.onset) - timedelta(days=1)
        while True:
            key = d0.isoformat()
            if key not in tier_by_day:
                break
            if tier_by_day[key] == 0:
                run += 1; d0 -= timedelta(days=1)
            else:
                break
        ep.fresh = run >= reset
        # R4-5: left-censored if this episode was active at/before the accumulation start
        # (its onset is not itself a fresh post-start onset). start_date passed by run().
        if start_date is not None and ep.onset <= start_date:
            ep.left_censored = True
    return eps


# ============================================================================
# STEP 4 - SCORING (hits / false alarms / misses; one-to-one crediting)
# ============================================================================

def score(episodes: list[Episode], mainshocks: list[Event],
          exclusions: list[Exclusion] = None, today: str = None,
          window: int = R4_CONFIG["hit_window_days"]) -> dict:
    """R4 v2 chronological guard-state scorer. Events are processed in UTC-origin order;
    each event's exclusion window is opened AFTER it is scored, so the exclusion state an
    event sees is exactly what earlier events created (causal lineage, R4-1) -- never a
    batch relabelling. Hit eligibility is by UTC alarm-availability vs event origin (R4-3)
    against SCOREABLE alarm dates only (R4-2). Left-censored episodes cannot credit (R4-5).
    The `exclusions` arg is accepted for API compat but the authoritative windows are
    rebuilt causally here from the mainshocks themselves."""
    outcomes = []
    credited: set[int] = set()
    # R4 v3: the causal ledger is PRECOMPUTED (build_causal_windows over all events) and
    # passed in. If omitted, build it from the given events (back-compat for stage KATs).
    ledger = exclusions if exclusions is not None else build_causal_windows(mainshocks)

    def eligible_episode(region, origin_utc, require_fresh=False, allow_excluded=False):
        # allow_excluded=True ONLY on the supersession path: a superseding event's fresh
        # episode necessarily lies inside the earlier exclusion window (R4/R6), so there
        # the excluded flag must not disqualify it.
        best = None
        for i, e in enumerate(episodes):
            if e.region != region or i in credited or e.left_censored:
                continue
            if e.excluded and not allow_excluded:
                continue
            if require_fresh and not e.fresh:
                continue
            if any(hit_eligible(d, origin_utc, window) for d in e.alarm_dates):
                if best is None or e.onset > episodes[best].onset:
                    best = i
        return best

    for m in sorted(mainshocks, key=lambda e: e.origin_utc):
        mem = m.regions or ((m.region,) if m.region else ())
        for region in mem:
            live = [w for w in ledger if w.region == region
                    and w.start <= m.t <= w.end and w.mainshock_id != m.event_id
                    and (not w.origin_utc or w.origin_utc < m.origin_utc)]
            if live:
                w = max(live, key=lambda w: w.mag)     # the governing (largest-mag) window
                if m.mag > w.mag:
                    i = eligible_episode(region, m.origin_utc, require_fresh=True,
                                         allow_excluded=True)
                    if i is not None:
                        episodes[i].status = "hit"; episodes[i].credited_event = m.event_id
                        credited.add(i)
                        outcomes.append({"event": m.event_id, "mag": m.mag, "region": region,
                                         "outcome": "hit_supersession",
                                         "episode_onset": episodes[i].onset})
                    else:
                        outcomes.append({"event": m.event_id, "mag": m.mag, "region": region,
                                         "outcome": "supersession_no_fresh_episode_unscored"})
                else:
                    outcomes.append({"event": m.event_id, "mag": m.mag, "region": region,
                                     "outcome": "excluded_unscored"})
            else:
                i = eligible_episode(region, m.origin_utc)
                if i is not None:
                    episodes[i].status = "hit"; episodes[i].credited_event = m.event_id
                    credited.add(i)
                    outcomes.append({"event": m.event_id, "mag": m.mag, "region": region,
                                     "outcome": "hit", "episode_onset": episodes[i].onset})
                else:
                    outcomes.append({"event": m.event_id, "mag": m.mag, "region": region,
                                     "outcome": "miss"})
            # R4-R2: score() opens NO windows; the ledger is authoritative and precomputed.

    # terminal states for episodes never credited
    for e in episodes:
        if e.status != "pending":
            continue
        last_alarm = e.alarm_dates[-1] if e.alarm_dates else e.end
        if (_d(today) - _d(last_alarm)).days > window:
            e.status = ("excluded" if e.excluded else
                        "left_censored" if e.left_censored else "false_alarm")
    return {"outcomes": outcomes}


def _score_legacy_unused():  # (previous batch scorer removed in R4 v2)
    pass

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
        # R4-R4/R6 §5: sample count NEVER lifts descriptive_only; only a registered
        # statistical-plan amendment does. n>=min is mere eligibility to run it.
        "eligible_for_plan": n_main >= R4_CONFIG["min_mainshocks_for_skill"],
        "descriptive_only": R4_CONFIG.get("stats_plan_amendment") is None,
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
    """Admissible-candidate events. R4-3: keep the full UTC origin timestamp. R4-4:
    deduplicate by catalog id and MERGE all region memberships (buffers may overlap), so
    membership never depends on query order."""
    by_id: dict[str, Event] = {}
    for rid, rd in region_defs.items():
        lat, lon = rd["center"]
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
               f"&starttime={start}&endtime={end}&latitude={lat}&longitude={lon}"
               f"&maxradiuskm=500&minmagnitude={R4_CONFIG['min_magnitude']}")
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        for feat in data.get("features", []):
            pr = feat["properties"]; c = feat["geometry"]["coordinates"]
            origin = datetime.fromtimestamp(pr["time"] / 1000, timezone.utc).isoformat()
            day = origin[:10]
            e = by_id.get(feat["id"])
            if e is None:
                e = Event(feat["id"], day, c[1], c[0], pr["mag"], origin_utc=origin)
                by_id[feat["id"]] = e
            if haversine_km(e.lat, e.lon, lat, lon) <= R4_CONFIG["region_buffer_km"]:
                if rid not in e.regions:
                    e.regions = e.regions + (rid,)
                    e.region = e.regions[0]
    return list(by_id.values())


def run(end_date: str | None = None) -> dict:
    from src.validate_predictions import REGION_DEFINITIONS   # single source of truth
    today = end_date or date.today().isoformat()
    start = R4_CONFIG["start_date"]
    csv_path = REPO / "monitoring" / "dashboard" / "data.csv"
    if not csv_path.exists():
        csv_path = REPO / "docs" / "data.csv"
    # R4-5: load alarm history from before the accumulation start so episode/reset state
    # exists at the boundary and pre-start standing episodes are left-censored, not relabelled.
    hist_start = (_d(start) - timedelta(days=R4_CONFIG["alarm_reset_days"]
                                        + R4_CONFIG["episode_gap_days"] + 5)).isoformat()
    series = load_day_series(csv_path, hist_start, today)
    # Fetch from one exclusion-cap BEFORE the accumulation start: mainshocks that
    # precede the start (e.g. the 2026-07-28 Kumamoto M7.1, one day before R4's
    # start) must still OPEN exclusion windows ("Kumamoto unscoreable until
    # 2027-07-28"), even though pre-start events are never themselves scored.
    fetch_start = (_d(start) - timedelta(days=R4_CONFIG["exclusion_cap_days"])).isoformat()
    events = fetch_usgs_events(fetch_start, today, REGION_DEFINITIONS)
    # R4 v3: ONE causal ledger over ALL admissible events (pre-start seeds guard state);
    # batch decluster is diagnostic-only (R6). Episodes, scoring, and Molchan all read the
    # SAME ledger, so no composition can bypass the guard.
    exclusions = build_causal_windows(events)
    _, clustered = decluster(events)        # diagnostic cluster count only
    scoreable_events = [m for m in events if m.t >= start]
    episodes: list[Episode] = []
    for region, days in series.items():
        episodes.extend(build_episodes(days, region, exclusions, start_date=start))
    sc = score(episodes, scoreable_events, exclusions, today)
    accum = {r: [(d0, ti) for d0, ti in days if d0 >= start] for r, days in series.items()}
    mol = molchan(accum, episodes, sc["outcomes"], exclusions)
    record = {
        "generated": datetime.now().isoformat(),
        "config": R4_CONFIG,
        "period": {"start": start, "end": today},
        "molchan": mol,
        "episodes": [asdict(e) for e in sorted(episodes, key=lambda e: (e.region, e.onset))],
        "event_outcomes": sc["outcomes"],
        "exclusions": [asdict(x) for x in exclusions],
        "n_events_fetched": len(events), "n_clustered_removed": len(clustered),
        "note": ("DESCRIPTIVE ONLY - no registered statistical-plan amendment "
                 f"(R6 §5); eligible_for_plan={mol['pooled']['admissible_mainshocks']}>="
                 f"{R4_CONFIG['min_mainshocks_for_skill']}={mol.get('eligible_for_plan')}. "
                 "No skill claim is made."),
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
