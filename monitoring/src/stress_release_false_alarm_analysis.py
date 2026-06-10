#!/usr/bin/env python3
"""
stress_release_false_alarm_analysis.py
False-alarm-rate analysis for the stress-release drop detector.

Runs the *actual* detector (stress_release_detector.detect_stress_release_drops)
over the full committed tier history, then scores every firing against the USGS
event catalog using a tectonic-domain association rule (grassmann 2026-06-09):
a drop in a correlation group is a HIT if a qualifying event occurred anywhere
on that group's plate-boundary domain within the domain window; solo monitors
fall back to region-local scoring (an event within the dropping region's bounds
+ --buffer-deg at >= --min-magnitude within --forward-days). Otherwise a FALSE
ALARM.

This answers the grassmann 2026-06-08 handoff item:
  "Run the detector across the full daily_states.csv history to estimate how
   often it fires without a subsequent event."

--------------------------------------------------------------------------------
FIDELITY CAVEATS (read before quoting the number):

1. TIER-PROXY delta_z. The committed history (docs/data.csv) carries tier + risk
   but NOT per-region z-scores. The production detector keys delta_z off z-scores
   and only falls back to the prior tier when z is absent -- and in that fallback
   it SKIPS the min_delta_z gate (see stress_release_detector.py line ~157). So
   this run is strictly MORE permissive than production and the reported firing
   count / false-alarm rate is an UPPER BOUND on the z-score-driven detector.
   To get the faithful number, re-run with --ensemble-dir pointed at a directory
   of real ensemble_<date>.json files (with z-scores) -- those live in
   monitoring/data/ensemble_results/ on the production box, not in the repo.

2. WINDOW. The committed history starts 2025-10-18, so the pre-2025 validated
   events (Tohoku 2011, Kaikoura 2016, Kumamoto 2016, Turkey 2023) are OUTSIDE
   the scored window. Only Philippines 2026 (Jun 7) falls inside it. The hit/FA
   split therefore reflects the monitoring period, not the full validation set.

3. The multi-region sync filter IS faithful here (all regions are present in the
   CSV), as is the elevated-days logic. Only delta_z is proxied.
--------------------------------------------------------------------------------

Usage:
    python stress_release_false_alarm_analysis.py
    python stress_release_false_alarm_analysis.py --min-magnitude 6.5 --forward-days 7
    python stress_release_false_alarm_analysis.py --ensemble-dir ../data/ensemble_results
    python stress_release_false_alarm_analysis.py --offline   # firing rate only, no scoring

Author: cayley (geomen), per grassmann handoff 2026-06-08
Date: 2026-06-09
"""

import argparse
import csv
import json
import logging
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stress_release_detector import detect_stress_release_drops, StressReleaseDrop

try:
    from earthquake_events import REGION_BOUNDS  # (minlat, maxlat, minlon, maxlon)
except ImportError:
    REGION_BOUNDS = {}

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

TIER_NAMES = {0: 'NORMAL', 1: 'WATCH', 2: 'ELEVATED', 3: 'CRITICAL'}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / 'docs' / 'data.csv'

# --- Tectonic-domain association (grassmann 2026-06-09 domain-mapping note) ----
# A stress-release drop is a plate-boundary signal, not a point source: a
# correlation group sits on one plate-boundary system and should be scored
# against events anywhere on that boundary, not just inside the monitor's own
# region bounds. Drops whose region is in a domain use the domain's bounds /
# min_magnitude / forward window; solo monitors keep region-local + buffer.
# bounds = (minlat, maxlat, minlon, maxlon).
TECTONIC_DOMAINS = {
    'western_pacific': {
        'regions': {'hualien', 'tokyo_kanto', 'kumamoto'},
        'bounds': (5.0, 50.0, 120.0, 155.0),
        'min_magnitude': 6.5,
        'forward_days': 7,
    },
    'cascadia_norcal': {
        'regions': {'cascadia', 'norcal_hayward'},
        'bounds': (35.0, 52.0, -132.0, -118.0),
        'min_magnitude': 6.0,
        'forward_days': 7,
    },
    'socal': {
        'regions': {'ridgecrest', 'socal_saf_mojave', 'socal_saf_coachella'},
        'bounds': (30.0, 37.0, -122.0, -114.0),
        'min_magnitude': 6.0,
        'forward_days': 7,
    },
}

# region -> (domain_name, domain_spec), built once from TECTONIC_DOMAINS.
_REGION_TO_DOMAIN = {
    region: (name, spec)
    for name, spec in TECTONIC_DOMAINS.items()
    for region in spec['regions']
}


def domain_for_region(region):
    """Return (domain_name, domain_spec) for a region, or (None, None) if solo."""
    return _REGION_TO_DOMAIN.get(region, (None, None))


def event_in_bounds(ev, bounds):
    """True if event lat/lon falls within bounds = (minlat, maxlat, minlon, maxlon)."""
    minlat, maxlat, minlon, maxlon = bounds
    return minlat <= ev['lat'] <= maxlat and minlon <= ev['lon'] <= maxlon


def load_tier_history(csv_path: Path):
    """Read the dashboard CSV into {date_str: {region: tier}}. Returns sorted dates too."""
    by_date = defaultdict(dict)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_s = row.get('date', '').strip()
            region = row.get('region', '').strip()
            if not date_s or not region or region == 'region':
                continue
            try:
                tier = int(row.get('tier', 0))
            except (TypeError, ValueError):
                continue
            by_date[date_s][region] = tier
    dates = sorted(by_date.keys())
    return by_date, dates


def materialize_ensemble_dir(by_date) -> Path:
    """
    Write synthetic ensemble_<date>.json files (tier only, no z-score) so the
    REAL detector code can scan them. z_score is omitted on purpose -> the
    detector takes its tier-fallback delta_z path (see caveat 1).
    """
    tmp = Path(tempfile.mkdtemp(prefix='srfa_'))
    for date_s, regions in by_date.items():
        payload = {'date': date_s, 'regions': {}}
        for region, tier in regions.items():
            payload['regions'][region] = {
                'tier': tier,
                'tier_name': TIER_NAMES.get(tier, f'TIER_{tier}'),
                'combined_risk': 0.0,
                'components': {},  # no z-score -> tier fallback
            }
        (tmp / f'ensemble_{date_s}.json').write_text(json.dumps(payload))
    return tmp


def fetch_usgs_events(start: datetime, end: datetime, min_magnitude: float):
    """Fetch the USGS catalog for [start, end] at min_magnitude. Returns list of dicts."""
    url = (
        'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson'
        f'&starttime={start.strftime("%Y-%m-%d")}'
        f'&endtime={end.strftime("%Y-%m-%d")}'
        f'&minmagnitude={min_magnitude}'
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    events = []
    for feat in data.get('features', []):
        p = feat.get('properties', {})
        g = feat.get('geometry', {})
        coords = g.get('coordinates') or [None, None]
        if p.get('mag') is None or coords[0] is None:
            continue
        events.append({
            'mag': p['mag'],
            'place': p.get('place', ''),
            'time': datetime.utcfromtimestamp(p['time'] / 1000),
            'lon': coords[0],
            'lat': coords[1],
        })
    return events


def assign_region(lat: float, lon: float, buffer_deg: float):
    """Return the first region whose bounds (+ buffer) contain (lat, lon), else None."""
    for region, b in REGION_BOUNDS.items():
        minlat, maxlat, minlon, maxlon = b
        if (minlat - buffer_deg <= lat <= maxlat + buffer_deg and
                minlon - buffer_deg <= lon <= maxlon + buffer_deg):
            return region
    return None


def score_drops(drops, events, forward_days: int, buffer_deg: float, min_magnitude: float):
    """
    Classify each drop as hit/false_alarm against the event catalog.

    Domain-aware (grassmann 2026-06-09): if the dropping region belongs to a
    tectonic domain, the drop is scored against ANY event on that plate-boundary
    domain, using the domain's bounds / min_magnitude / forward window. Solo
    monitors keep the region-local + buffer behaviour: an event assigned to the
    drop's own region within `forward_days` at >= `min_magnitude`.
    """
    # Index events by assigned region for the solo / fallback path.
    events_by_region = defaultdict(list)
    for ev in events:
        region = assign_region(ev['lat'], ev['lon'], buffer_deg)
        if region:
            events_by_region[region].append(ev)

    results = []
    for d in drops:
        drop_dt = datetime.strptime(d.drop_date, '%Y-%m-%d')
        domain_name, spec = domain_for_region(d.region)

        if spec is not None:
            # Domain path: any event on the plate boundary, domain thresholds.
            window_end = drop_dt + timedelta(days=spec['forward_days'])
            candidates = [
                ev for ev in events
                if ev['mag'] >= spec['min_magnitude'] and event_in_bounds(ev, spec['bounds'])
            ]
            assoc = domain_name
        else:
            # Solo path: region-local + buffer, CLI thresholds (existing behaviour).
            window_end = drop_dt + timedelta(days=forward_days)
            candidates = [
                ev for ev in events_by_region.get(d.region, [])
                if ev['mag'] >= min_magnitude
            ]
            assoc = 'solo'

        matched = None
        for ev in candidates:
            if drop_dt <= ev['time'] <= window_end:
                if matched is None or ev['mag'] > matched['mag']:
                    matched = ev
        results.append({
            'drop': d,
            'classification': 'hit' if matched else 'false_alarm',
            'event': matched,
            'association': assoc,
        })
    return results


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # report contains →, ≥, °, ⚠ etc.
    except Exception:
        pass
    ap = argparse.ArgumentParser(description='Stress-release drop detector false-alarm analysis')
    ap.add_argument('--csv', type=str, default=str(DEFAULT_CSV), help='Tier-history CSV (default: docs/data.csv)')
    ap.add_argument('--ensemble-dir', type=str, default=None,
                    help='Use a real ensemble_<date>.json dir instead of the CSV proxy (faithful z-score run)')
    ap.add_argument('--min-magnitude', type=float, default=6.0, help='Min event magnitude to count as a hit')
    ap.add_argument('--forward-days', type=int, default=14, help='Days after a drop in which an event counts as a hit')
    ap.add_argument('--buffer-deg', type=float, default=1.0, help='Degrees of buffer around region bounds for event assignment')
    ap.add_argument('--min-delta-z', type=float, default=1.5)
    ap.add_argument('--min-elevated-days', type=int, default=3)
    ap.add_argument('--no-sync-filter', action='store_true', help='Disable multi-region sync filter')
    ap.add_argument('--offline', action='store_true', help='Skip USGS scoring; report firing rate only')
    ap.add_argument('--report', type=str, default=None, help='Write a markdown report to this path')
    args = ap.parse_args()

    # --- 1. detector run over the history --------------------------------------
    proxy = args.ensemble_dir is None
    if args.ensemble_dir:
        ensemble_dir = Path(args.ensemble_dir)
        import re
        files = sorted(f for f in ensemble_dir.glob('ensemble_*.json')
                       if re.match(r'ensemble_\d{4}-\d{2}-\d{2}\.json$', f.name))
        if not files:
            print(f'No ensemble_YYYY-MM-DD.json in {ensemble_dir}', file=sys.stderr)
            return 1
        dates = [f.stem.replace('ensemble_', '') for f in files]
        n_region_days = '(from real ensemble files)'
    else:
        csv_path = Path(args.csv)
        by_date, dates = load_tier_history(csv_path)
        if not dates:
            print(f'No usable rows in {csv_path}', file=sys.stderr)
            return 1
        ensemble_dir = materialize_ensemble_dir(by_date)
        n_region_days = sum(len(v) for v in by_date.values())

    first_date = datetime.strptime(dates[0], '%Y-%m-%d')
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    span_days = (last_date - first_date).days + 1
    n_regions = len({r for d in (by_date.values() if proxy else []) for r in d}) if proxy else 0

    drops = detect_stress_release_drops(
        ensemble_dir=ensemble_dir,
        target_date=last_date,
        lookback_days=span_days,
        min_delta_z=args.min_delta_z,
        min_elevated_days=args.min_elevated_days,
        require_multi_region=not args.no_sync_filter,
    )

    by_conf = Counter(d.confidence for d in drops)
    region_days = n_region_days if isinstance(n_region_days, int) else span_days * max(n_regions, 1)
    firing_rate_per_region_year = (len(drops) / region_days * 365) if region_days else 0.0

    # --- 2. score against USGS -------------------------------------------------
    scored = None
    if not args.offline and drops:
        try:
            # Widen the fetch to cover every domain's window/magnitude as well as
            # the CLI solo-path values, so both scoring paths see all candidates.
            fetch_min_mag = min([s['min_magnitude'] for s in TECTONIC_DOMAINS.values()]
                                + [args.min_magnitude])
            fetch_fwd_days = max([s['forward_days'] for s in TECTONIC_DOMAINS.values()]
                                 + [args.forward_days])
            events = fetch_usgs_events(first_date - timedelta(days=1),
                                       last_date + timedelta(days=fetch_fwd_days),
                                       fetch_min_mag)
            scored = score_drops(drops, events, args.forward_days, args.buffer_deg, args.min_magnitude)
        except Exception as e:
            print(f'WARNING: USGS scoring failed ({type(e).__name__}: {e}); reporting firing rate only',
                  file=sys.stderr)

    # --- 3. report -------------------------------------------------------------
    lines = []
    lines.append('# Stress-Release Drop Detector — False-Alarm Analysis')
    lines.append('')
    lines.append(f'- **Source:** {"CSV tier-proxy (" + str(args.csv) + ")" if proxy else "real ensemble dir " + str(args.ensemble_dir)}')
    lines.append(f'- **Window:** {dates[0]} → {dates[-1]} ({span_days} days, {region_days} region-days)')
    lines.append(f'- **Detector params:** min_elevated_days={args.min_elevated_days}, '
                 f'min_delta_z={args.min_delta_z}, sync_filter={not args.no_sync_filter}')
    lines.append('')
    if proxy:
        lines.append('> ⚠️ **Tier-proxy run.** docs/data.csv has no z-scores, so the detector uses its '
                     'tier-fallback delta_z path (which skips the delta_z gate). This is an **UPPER BOUND** '
                     'on the z-score-driven production detector. Re-run with `--ensemble-dir` on the '
                     'production box for the faithful number. Pre-2025 validated events are outside this window.')
        lines.append('')
    lines.append('## Firing rate')
    lines.append('')
    lines.append(f'- **Total firings:** {len(drops)}')
    lines.append(f'- **By confidence:** ' + (', '.join(f'{k}={v}' for k, v in sorted(by_conf.items())) or 'none'))
    lines.append(f'- **Firing rate:** {firing_rate_per_region_year:.2f} per region-year '
                 f'(target ≈ 1/region/year)')
    lines.append('')

    if scored is not None:
        n_hit = sum(1 for s in scored if s['classification'] == 'hit')
        n_fa = sum(1 for s in scored if s['classification'] == 'false_alarm')
        total = n_hit + n_fa
        fa_rate = n_fa / total if total else 0.0
        lines.append('## Hit / false-alarm split (domain-aware association)')
        lines.append('')
        lines.append('Drops are scored with the tectonic-domain association rule (grassmann '
                     '2026-06-09): a drop in a correlation group is a HIT if a qualifying event '
                     'occurred **anywhere on that group\'s plate-boundary domain** within the domain '
                     'window. This captures the teleseismic case — the western-Pacific monitors '
                     '(Hualien, Tokyo Kanto, Kumamoto) → the M7.8 **Mindanao** rupture ~2000 km away, '
                     'which region-local scoring scored as a false alarm. Solo monitors keep '
                     'region-local + buffer scoring.')
        lines.append('')
        for name, spec in TECTONIC_DOMAINS.items():
            b = spec['bounds']
            lines.append(f'- **{name}** {{{", ".join(sorted(spec["regions"]))}}}: '
                         f'M≥{spec["min_magnitude"]} within {spec["forward_days"]}d, '
                         f'bounds lat[{b[0]}, {b[1]}] lon[{b[2]}, {b[3]}]')
        lines.append(f'- **solo monitors:** M≥{args.min_magnitude} within {args.forward_days}d, '
                     f'region bounds +{args.buffer_deg}° buffer')
        if proxy:
            lines.append('')
            lines.append('> ⚠️ Domain association removes the region-local scoring distortion, but the '
                         'tier-proxy permissiveness (caveat above) still applies on a CSV run — the '
                         'faithful number needs `--ensemble-dir`.')
        lines.append('')
        lines.append(f'- **Hits:** {n_hit}')
        lines.append(f'- **False alarms:** {n_fa}')
        lines.append(f'- **False-alarm rate:** {fa_rate:.1%}  (precision {1 - fa_rate:.1%})')
        lines.append('')
        lines.append('### Per-firing detail')
        lines.append('')
        lines.append('| region | drop_date | conf | assoc | class | matched event |')
        lines.append('|---|---|---|---|---|---|')
        for s in sorted(scored, key=lambda x: x['drop'].drop_date):
            d = s['drop']
            ev = s['event']
            evtxt = f"M{ev['mag']:.1f} {ev['place']} ({ev['time'].date()})" if ev else '—'
            lines.append(f"| {d.region} | {d.drop_date} | {d.confidence} | {s['association']} | "
                         f"{s['classification']} | {evtxt} |")
        lines.append('')

    report = '\n'.join(lines)

    if args.report:
        Path(args.report).write_text(report, encoding='utf-8')
        print(f'[report written to {args.report}]', file=sys.stderr)

    print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
