#!/usr/bin/env python3
"""
run_ensemble_daily.py
Production Daily Ensemble Runner for GeoSpec Monitoring System.

Runs the three-method ensemble assessment for all configured regions:
1. Lambda_geo (GPS) - When available (2-14 day latency)
2. Fault Correlation - Regions with defined fault segments (California, Cascadia, etc.)
3. Seismic THD - All regions via IU/BK/GE global network stations

Seismic data sources:
- California: IU.TUC (Tucson), BK.BKS (Berkeley)
- Cascadia: IU.COR (Corvallis)
- Japan: NIED Hi-net (128 Kanto stations @ 100Hz), IU.MAJO fallback
- Turkey: IU.ANTO (Ankara) - 40Hz for consistent THD baselines

Features:
- Persistence tracking: WATCH requires 2 consecutive days for CONFIRMED status
- Tier gating: ELEVATED/CRITICAL requires >=2 methods available
- Coverage tracking: Logs segment availability for fault correlation

Outputs combined risk assessment to monitoring/data/ensemble_results/

Usage:
    python run_ensemble_daily.py                    # Run all regions
    python run_ensemble_daily.py --region ridgecrest  # Single region
    python run_ensemble_daily.py --date 2024-01-15    # Specific date

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os
import argparse
import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from ensemble import GeoSpecEnsemble, EnsembleResult, RISK_TIERS

# Earthquake event fetching
try:
    from earthquake_events import fetch_region_events, REGION_BOUNDS
    EARTHQUAKE_EVENTS_AVAILABLE = True
except ImportError:
    EARTHQUAKE_EVENTS_AVAILABLE = False
    REGION_BOUNDS = {}

# Lambda_geo pilot integration
try:
    from lambda_geo_pilot import get_lambda_geo_for_ensemble, check_pilot_status, PILOT_REGION
    LAMBDA_GEO_PILOT_AVAILABLE = True
except ImportError:
    LAMBDA_GEO_PILOT_AVAILABLE = False
    PILOT_REGION = None

# NGL-based Lambda_geo for all regions with polygon definitions
try:
    from live_data import NGLLiveAcquisition, acquire_region_data
    from regions import FAULT_POLYGONS
    NGL_LAMBDA_GEO_AVAILABLE = True
except ImportError:
    NGL_LAMBDA_GEO_AVAILABLE = False
    FAULT_POLYGONS = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# REGION CONFIGURATION
# =============================================================================

REGIONS = {
    # California - Using IU global network (SCEDC has gaps, BK works)
    'ridgecrest': {
        'name': 'Ridgecrest/Mojave',
        'thd_station': 'TUC',      # IU.TUC Tucson - reliable via IRIS
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 3,         # SCEDC needs 3+ day latency
    },
    'socal_saf_mojave': {
        'name': 'SoCal SAF Mojave',
        'thd_station': 'TUC',      # IU.TUC Tucson - nearest IU station
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 3,
    },
    'socal_saf_coachella': {
        'name': 'SoCal SAF Coachella',
        'thd_station': 'TUC',      # IU.TUC Tucson
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 3,
    },
    'norcal_hayward': {
        'name': 'NorCal Hayward',
        'thd_station': 'BKS',      # BK.BKS Berkeley - 100% availability!
        'thd_network': 'BK',
        'seismic_available': True,
        'latency_days': 0,         # Real-time data available
    },
    'cascadia': {
        'name': 'Cascadia',
        'thd_station': 'COR',      # IU.COR Corvallis, OR - via IRIS
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 0,
    },

    # International - Japan
    'tokyo_kanto': {
        'name': 'Tokyo Kanto',
        # Primary: NIED Hi-net (Phase 2 complete - January 2026)
        'thd_station': 'N.KI2H',   # Hi-net Kita-Ibaraki - 100Hz, Kanto region
        'thd_network': 'HINET',    # NIED Hi-net via status page polling
        'hinet_enabled': True,     # Hi-net integration active
        'hinet_network': '0101',   # NIED Hi-net network code
        'seismic_available': True,
        'latency_days': 0,
        # Fallback to IU.MAJO when Hi-net unavailable
        'fallback_station': 'MAJO',
        'fallback_network': 'IU',
        'notes': 'Hi-net primary (128 Kanto stations @ 100Hz), IU.MAJO fallback',
    },
    'istanbul_marmara': {
        'name': 'Istanbul Marmara',
        'thd_station': 'ANTO',     # IU.ANTO Ankara - nearest IU
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 0,
    },
    'turkey_kahramanmaras': {
        'name': 'Turkey Kahramanmaras',
        'thd_station': 'ANTO',     # IU.ANTO Ankara - using for consistent 40Hz data
        'thd_network': 'IU',       # Note: GE.ARPR (Arapgir) closer but 20Hz causes THD inflation
        'seismic_available': True,
        'latency_days': 0,
    },

    # Italy - Volcanic caldera pilot (Method 2 sandbox)
    'campi_flegrei': {
        'name': 'Campi Flegrei',
        'thd_station': 'CAFE',     # IV.CAFE - CSFT has connection issues, CAFE reliable 100Hz
        'thd_network': 'IV',       # INGV network - open data via EIDA
        'seismic_available': True,
        'latency_days': 0,
        'notes': 'Volcanic caldera - bradyseismic unrest since 2012, densest FC coverage',
    },
}


# =============================================================================
# DAILY RUNNER
# =============================================================================

def run_region_assessment(
    region: str,
    target_date: datetime,
    lambda_geo_ratio: Optional[float] = None,
    use_seismic: bool = True,
) -> Optional[EnsembleResult]:
    """
    Run ensemble assessment for a single region.

    Args:
        region: Region key from REGIONS dict
        target_date: Date to assess
        lambda_geo_ratio: Optional Lambda_geo ratio (from live data)
        use_seismic: Whether to use seismic methods

    Returns:
        EnsembleResult or None if failed
    """
    config = REGIONS.get(region)
    if not config:
        logger.error(f"Unknown region: {region}")
        return None

    logger.info(f"Assessing {config['name']} for {target_date.date()}")

    try:
        ensemble = GeoSpecEnsemble(region=region)

        # Set Lambda_geo if provided
        if lambda_geo_ratio is not None:
            ensemble.set_lambda_geo(target_date, lambda_geo_ratio)

        # Determine if seismic should be used
        seismic_ok = use_seismic and config['seismic_available']

        if seismic_ok and config['thd_station']:
            # Full ensemble
            result = ensemble.compute_risk(
                target_date,
                thd_station=config['thd_station'],
                thd_network=config.get('thd_network', 'CI')
            )
        else:
            # Lambda_geo only
            lg_result = ensemble.compute_lambda_geo_risk(target_date)
            risk = lg_result.risk_score
            tier, tier_name = ensemble.get_tier(risk)

            result = EnsembleResult(
                region=region,
                date=target_date,
                combined_risk=risk,
                tier=tier,
                tier_name=tier_name,
                components={'lambda_geo': lg_result},
                confidence=0.5 if lg_result.available else 0.0,
                agreement='single_method' if lg_result.available else 'no_data',
                methods_available=1 if lg_result.available else 0,
            )

        return result

    except Exception as e:
        logger.error(f"Assessment failed for {region}: {e}")
        return None


def fetch_earthquake_events(regions: List[str], lookback_days: int = 90) -> Dict:
    """
    Fetch recent earthquake events for all regions.

    Args:
        regions: List of region keys
        lookback_days: Days of history to fetch

    Returns:
        Dict mapping region to event data
    """
    if not EARTHQUAKE_EVENTS_AVAILABLE:
        logger.warning("Earthquake events module not available")
        return {}

    events_data = {}

    for region in regions:
        if region not in REGION_BOUNDS:
            logger.debug(f"No bounds defined for region {region}")
            continue

        try:
            result = fetch_region_events(region, lookback_days, min_magnitude=4.0)
            if result:
                events_data[region] = result.to_dict()
                if result.largest_event:
                    logger.info(f"  {region}: {result.event_count} events, "
                               f"largest M{result.largest_event.magnitude:.1f}")
                else:
                    logger.debug(f"  {region}: No M4+ events in last {lookback_days} days")
        except Exception as e:
            logger.warning(f"Failed to fetch events for {region}: {e}")

    return events_data


def fetch_ngl_lambda_geo(
    regions: List[str],
    target_date: datetime,
    days_back: int = 120,
) -> Dict[str, float]:
    """
    Fetch Lambda_geo ratios from NGL GPS data for all regions with polygon definitions.

    Args:
        regions: List of region keys to process
        target_date: Target date for assessment
        days_back: Number of days of GPS data to use (default 120)

    Returns:
        Dict mapping region to Lambda_geo ratio (baseline multiplier)
    """
    if not NGL_LAMBDA_GEO_AVAILABLE:
        logger.warning("NGL Lambda_geo module not available")
        return {}

    lambda_geo_data = {}

    # Initialize NGL acquisition with cache directory
    cache_dir = Path(__file__).parent.parent / 'data' / 'gps_cache'
    ngl = NGLLiveAcquisition(cache_dir)

    # Load station catalog once
    logger.info("Loading NGL station catalog for Lambda_geo computation...")
    ngl.load_station_catalog()

    for region in regions:
        # Skip regions without polygon definitions
        if region not in FAULT_POLYGONS:
            logger.debug(f"No polygon definition for {region}, skipping Lambda_geo")
            continue

        try:
            logger.info(f"Computing Lambda_geo for {region}...")
            result = acquire_region_data(region, ngl, days_back, target_date)

            if result and result.n_stations >= 3 and result.lambda_geo_max > 0:
                # Convert to baseline ratio
                # Baseline Lambda_geo ~ 0.01 for stable regions (calibrated from observations)
                # Values below baseline = normal, above = elevated
                baseline = 0.01
                ratio = result.lambda_geo_max / baseline

                # Clamp to reasonable range (0.1x to 50x)
                ratio = max(0.1, min(50.0, ratio))

                lambda_geo_data[region] = ratio
                logger.info(f"  {region}: Lambda_geo ratio = {ratio:.1f}x "
                           f"({result.n_stations} stations, {result.data_quality})")
            else:
                quality = result.data_quality if result else 'no_data'
                n_stations = result.n_stations if result else 0
                logger.info(f"  {region}: Lambda_geo unavailable ({n_stations} stations, {quality})")

        except Exception as e:
            logger.warning(f"Failed to compute Lambda_geo for {region}: {e}")

    return lambda_geo_data


def run_all_regions(
    target_date: datetime,
    regions: Optional[List[str]] = None,
    lambda_geo_data: Optional[Dict[str, float]] = None,
    use_seismic: bool = True,
    fetch_events: bool = True,
) -> tuple:
    """
    Run ensemble assessment for multiple regions.

    Args:
        target_date: Date to assess
        regions: List of region keys (default: all)
        lambda_geo_data: Dict mapping region to Lambda_geo ratio
        use_seismic: Whether to use seismic methods
        fetch_events: Whether to fetch recent earthquake events

    Returns:
        Tuple of (Dict mapping region to EnsembleResult, Dict of earthquake events)
    """
    if regions is None:
        regions = list(REGIONS.keys())

    if lambda_geo_data is None:
        lambda_geo_data = {}

    # Check for Lambda_geo pilot data (real-time RTCM)
    if LAMBDA_GEO_PILOT_AVAILABLE:
        pilot_status = check_pilot_status()
        logger.info(f"Lambda_geo pilot status: {pilot_status.message}")

        if pilot_status.ready_for_lambda_geo and PILOT_REGION:
            available, ratio, notes = get_lambda_geo_for_ensemble(PILOT_REGION, target_date)
            if available and ratio is not None:
                lambda_geo_data[PILOT_REGION] = ratio
                logger.info(f"Lambda_geo pilot data for {PILOT_REGION}: {ratio:.1f}x ({notes})")
            else:
                logger.info(f"Lambda_geo pilot not ready: {notes}")
        else:
            logger.info(f"Lambda_geo pilot data accumulating: {pilot_status.days_accumulated} days "
                       f"(need {3 - pilot_status.days_accumulated} more)")

    # Fetch Lambda_geo from NGL for all regions with polygon definitions
    # This supplements pilot data with historical GPS data (2-14 day latency)
    if NGL_LAMBDA_GEO_AVAILABLE:
        ngl_lambda_geo = fetch_ngl_lambda_geo(regions, target_date)
        # Merge NGL data, but don't override pilot data if available
        for region, ratio in ngl_lambda_geo.items():
            if region not in lambda_geo_data:
                lambda_geo_data[region] = ratio
        logger.info(f"Lambda_geo available for {len(lambda_geo_data)} regions via NGL/pilot data")

    results = {}

    for region in regions:
        lg_ratio = lambda_geo_data.get(region)
        result = run_region_assessment(
            region=region,
            target_date=target_date,
            lambda_geo_ratio=lg_ratio,
            use_seismic=use_seismic,
        )
        if result:
            results[region] = result

    # Fetch earthquake events for correlation analysis
    events_data = {}
    if fetch_events:
        logger.info("Fetching recent earthquake events from USGS...")
        events_data = fetch_earthquake_events(regions)

    return results, events_data


def load_previous_results(
    output_dir: Path,
    target_date: datetime,
    days_back: int = 1,
) -> Optional[Dict]:
    """
    Load results from a previous day for persistence checking.

    Args:
        output_dir: Directory containing ensemble results
        target_date: Current target date
        days_back: How many days back to look

    Returns:
        Dict of previous results or None if not found
    """
    previous_date = target_date - timedelta(days=days_back)
    date_str = previous_date.strftime('%Y-%m-%d')
    previous_file = output_dir / f'ensemble_{date_str}.json'

    if not previous_file.exists():
        logger.debug(f"No previous results found at {previous_file}")
        return None

    try:
        with open(previous_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load previous results: {e}")
        return None


def check_persistence(
    current_results: Dict[str, EnsembleResult],
    output_dir: Path,
    target_date: datetime,
    required_consecutive: int = 2,
) -> Dict[str, Dict]:
    """
    Check which regions have persistent elevated status.

    A region is considered "confirmed" at WATCH or higher if it has been
    at that tier for N consecutive days.

    Args:
        current_results: Current assessment results
        output_dir: Directory containing historical results
        target_date: Current target date
        required_consecutive: Days required for confirmation (default 2)

    Returns:
        Dict mapping region to persistence info:
        {
            'current_tier': int,
            'consecutive_days': int,
            'is_confirmed': bool,
            'tier_history': [tier, tier, ...]  # oldest first
        }
    """
    persistence = {}

    for region, result in current_results.items():
        tier_history = [result.tier]
        current_tier = result.tier

        # Look back for consecutive days at same or higher tier
        for days_back in range(1, required_consecutive + 2):
            prev_data = load_previous_results(output_dir, target_date, days_back)
            if prev_data and region in prev_data.get('regions', {}):
                prev_tier = prev_data['regions'][region].get('tier', 0)
                tier_history.insert(0, prev_tier)
            else:
                tier_history.insert(0, None)

        # Count consecutive days at current tier level or above (WATCH = 1)
        consecutive = 1
        for past_tier in reversed(tier_history[:-1]):
            if past_tier is not None and past_tier >= 1 and current_tier >= 1:
                consecutive += 1
            else:
                break

        # Check if confirmed (requires consecutive days at WATCH+)
        is_confirmed = consecutive >= required_consecutive if current_tier >= 1 else False

        persistence[region] = {
            'current_tier': current_tier,
            'consecutive_days': consecutive if current_tier >= 1 else 0,
            'is_confirmed': is_confirmed,
            'tier_history': tier_history,
        }

        if current_tier >= 1:
            status = 'CONFIRMED' if is_confirmed else 'PRELIMINARY'
            logger.info(f"{region}: {result.tier_name} ({status}, {consecutive} consecutive days)")

    return persistence


def save_results(
    results: Dict[str, EnsembleResult],
    output_dir: Path,
    target_date: datetime,
    persistence: Optional[Dict[str, Dict]] = None,
    events_data: Optional[Dict] = None,
) -> Path:
    """Save assessment results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = target_date.strftime('%Y-%m-%d')
    output_file = output_dir / f'ensemble_{date_str}.json'

    output_data = {
        'date': date_str,
        'timestamp': datetime.now().isoformat(),
        'regions': {},
        'summary': {
            'total_regions': len(results),
            'tier_counts': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0},  # -1 = DEGRADED
            'confirmed_watch_count': 0,
            'preliminary_watch_count': 0,
            'degraded_count': 0,
            'max_risk_region': None,
            'max_risk': 0.0,
        },
        'earthquake_events': events_data or {},
    }

    for region, result in results.items():
        region_data = result.to_dict()

        # Add persistence info if available
        if persistence and region in persistence:
            region_data['persistence'] = persistence[region]
            if result.tier >= 1:
                if persistence[region]['is_confirmed']:
                    output_data['summary']['confirmed_watch_count'] += 1
                else:
                    output_data['summary']['preliminary_watch_count'] += 1

        output_data['regions'][region] = region_data
        output_data['summary']['tier_counts'][result.tier] += 1

        # Track DEGRADED separately
        if result.tier == -1:
            output_data['summary']['degraded_count'] += 1

        if result.tier >= 0 and result.combined_risk > output_data['summary']['max_risk']:
            output_data['summary']['max_risk'] = result.combined_risk
            output_data['summary']['max_risk_region'] = region

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {output_file}")
    return output_file


def append_to_daily_csv(
    results: Dict[str, EnsembleResult],
    output_dir: Path,
    target_date: datetime,
    persistence: Optional[Dict[str, Dict]] = None,
) -> Path:
    """
    Append results to daily_state.csv for trending and dashboard.

    Format: date,region,tier,risk,methods,confidence,lg_ratio,thd,fc_l2l1,status,notes
    """
    csv_file = output_dir / 'daily_states.csv'
    date_str = target_date.strftime('%Y-%m-%d')

    # Check if file exists to determine if we need header
    file_exists = csv_file.exists()

    # Check if this date already has entries (avoid duplicates)
    if file_exists:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0 and row[0] == date_str:
                    logger.info(f"Date {date_str} already in CSV, skipping append")
                    return csv_file

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow([
                'date', 'region', 'tier', 'risk', 'methods', 'confidence',
                'lg_ratio', 'thd', 'fc_l2l1', 'status', 'notes'
            ])

        for region, result in results.items():
            # Extract component values
            lg_ratio = ''
            thd_val = ''
            fc_l2l1 = ''

            if 'lambda_geo' in result.components:
                comp = result.components['lambda_geo']
                if comp.available:
                    lg_ratio = f"{comp.raw_value:.1f}"

            if 'seismic_thd' in result.components:
                comp = result.components['seismic_thd']
                if comp.available:
                    thd_val = f"{comp.raw_value:.3f}"

            if 'fault_correlation' in result.components:
                comp = result.components['fault_correlation']
                if comp.available:
                    fc_l2l1 = f"{comp.raw_value:.4f}"

            # Determine status
            status = 'PRELIMINARY'
            if persistence and region in persistence:
                if persistence[region]['is_confirmed']:
                    status = 'CONFIRMED'

            # Build notes
            notes_list = []
            if result.notes and 'capped' in result.notes.lower():
                notes_list.append('tier_capped')
            if result.tier == -1:
                notes_list.append('degraded')
            notes = ','.join(notes_list) if notes_list else ''

            writer.writerow([
                date_str,
                region,
                result.tier,
                f"{result.combined_risk:.3f}",
                result.methods_available,
                f"{result.confidence:.2f}",
                lg_ratio,
                thd_val,
                fc_l2l1,
                status,
                notes
            ])

    logger.info(f"Appended {len(results)} rows to: {csv_file}")
    return csv_file


def print_summary(results: Dict[str, EnsembleResult], persistence: Optional[Dict[str, Dict]] = None):
    """Print summary table to console."""
    print("\n" + "=" * 90)
    print("GEOSPEC ENSEMBLE DAILY ASSESSMENT")
    print("=" * 90)
    print()

    header = f"{'Region':<25} {'Risk':>8} {'Tier':<10} {'Status':<12} {'Conf':>6} {'Methods':>8}"
    print(header)
    print("-" * 90)

    for region, result in sorted(results.items(), key=lambda x: -x[1].combined_risk):
        config = REGIONS.get(region, {})
        name = config.get('name', region)[:24]

        # Determine persistence status
        if persistence and region in persistence:
            p = persistence[region]
            if result.tier >= 1:
                status = f"CONFIRMED({p['consecutive_days']}d)" if p['is_confirmed'] else f"PRELIM({p['consecutive_days']}d)"
            else:
                status = "-"
        else:
            status = "-"

        print(f"{name:<25} {result.combined_risk:>8.3f} {result.tier_name:<10} {status:<12} "
              f"{result.confidence:>6.2f} {result.methods_available:>8}")

    print("-" * 90)

    # Tier summary
    tier_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    confirmed_count = 0
    preliminary_count = 0

    for region, result in results.items():
        tier_counts[result.tier] += 1
        if persistence and region in persistence and result.tier >= 1:
            if persistence[region]['is_confirmed']:
                confirmed_count += 1
            else:
                preliminary_count += 1

    tier_line = f"NORMAL={tier_counts[0]} WATCH={tier_counts[1]} ELEVATED={tier_counts[2]} CRITICAL={tier_counts[3]}"
    if tier_counts[-1] > 0:
        tier_line += f" DEGRADED={tier_counts[-1]}"
    print(f"\nTier Distribution: {tier_line}")

    if confirmed_count > 0 or preliminary_count > 0:
        print(f"Persistence: {confirmed_count} CONFIRMED, {preliminary_count} PRELIMINARY")
        print("(CONFIRMED = 2+ consecutive days at WATCH or higher)")

    # Alert if any confirmed elevated or critical
    confirmed_elevated = []
    preliminary_elevated = []

    for region, result in results.items():
        if result.tier >= 1:
            if persistence and region in persistence and persistence[region]['is_confirmed']:
                confirmed_elevated.append(region)
            else:
                preliminary_elevated.append(region)

    if confirmed_elevated:
        print("\n*** ALERT: CONFIRMED elevated regions (2+ consecutive days) ***")
        for region in confirmed_elevated:
            res = results[region]
            days = persistence[region]['consecutive_days'] if persistence else '?'
            print(f"  - {region}: {res.tier_name} (risk={res.combined_risk:.3f}, {days} days)")

    if preliminary_elevated:
        print("\nNote: Preliminary elevated regions (requires confirmation):")
        for region in preliminary_elevated:
            res = results[region]
            print(f"  - {region}: {res.tier_name} (risk={res.combined_risk:.3f})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='GeoSpec Daily Ensemble Assessment'
    )
    parser.add_argument(
        '--region', type=str, default=None,
        help='Single region to assess (default: all)'
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='Target date YYYY-MM-DD (default: 2 days ago due to data latency)'
    )
    parser.add_argument(
        '--latency', type=int, default=2,
        help='Data latency offset in days (default: 2). Seismic/GPS data has 1-14 day delay.'
    )
    parser.add_argument(
        '--no-seismic', action='store_true',
        help='Skip seismic methods (Lambda_geo only)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: monitoring/data/ensemble_results)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress console output'
    )

    args = parser.parse_args()

    # Parse date (apply latency offset if no explicit date given)
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        # Default: N days ago due to data latency (seismic ~1 day, GPS 2-14 days)
        target_date = datetime.now() - timedelta(days=args.latency)
        logger.info(f"Using {args.latency}-day latency offset (data date: {target_date.date()})")

    # Determine regions
    regions = [args.region] if args.region else None

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'data' / 'ensemble_results'

    # Run assessment
    logger.info(f"Starting ensemble assessment for {target_date.date()}")

    results, events_data = run_all_regions(
        target_date=target_date,
        regions=regions,
        use_seismic=not args.no_seismic,
        fetch_events=True,
    )

    if not results:
        logger.error("No results produced")
        return 1

    # Check persistence (requires 2 consecutive days for confirmed status)
    persistence = check_persistence(
        results,
        output_dir,
        target_date,
        required_consecutive=2,
    )

    # Save results with persistence info and earthquake events
    save_results(results, output_dir, target_date, persistence, events_data)

    # Append to daily CSV for trending
    append_to_daily_csv(results, output_dir, target_date, persistence)

    # Print summary
    if not args.quiet:
        print_summary(results, persistence)

    # Return exit code based on max tier (only count confirmed as real alerts)
    confirmed_max_tier = 0
    for region, result in results.items():
        if result.tier >= 1 and persistence.get(region, {}).get('is_confirmed', False):
            confirmed_max_tier = max(confirmed_max_tier, result.tier)

    # If no confirmed alerts, return 0 (normal)
    # Otherwise return the confirmed max tier
    if confirmed_max_tier > 0:
        logger.info(f"Exit code: {confirmed_max_tier} (confirmed tier)")
        return confirmed_max_tier
    else:
        max_tier = max(r.tier for r in results.values())
        if max_tier > 0:
            logger.info(f"Exit code: 0 (tier {max_tier} is preliminary, not confirmed)")
        return 0  # Preliminary alerts don't trigger exit codes


if __name__ == "__main__":
    sys.exit(main())
