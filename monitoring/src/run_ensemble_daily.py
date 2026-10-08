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
import hashlib
import json
import logging
import subprocess
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from ensemble import GeoSpecEnsemble, EnsembleResult, RISK_TIERS
# method-comparability-v1 (prospective; only rows that carry a method set are affected)
import method_comparability as MC
import calibration_eligibility as CE
import evidence_redaction as ER
# Immutable public revision store (asylum 2026-09-02: "use immutable
# revision"; codex's model). The runner's production path publishes
# every scored day create-once under docs/ensemble/<date>/<run_id>.json
# and derives data.csv / ensemble_latest.json from the append-only index.
import ensemble_revisions_cayley as REV

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
# GeoNet for New Zealand regions (lower latency than NGL)
try:
    from live_data import NGLLiveAcquisition, GeoNetLiveAcquisition, acquire_region_data
    from regions import FAULT_POLYGONS
    NGL_LAMBDA_GEO_AVAILABLE = True
except ImportError:
    NGL_LAMBDA_GEO_AVAILABLE = False
    FAULT_POLYGONS = {}

# Prediction validation (track record building)
try:
    from validate_predictions import run_validation
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# Stress-release drop detector (pre-rupture quiescence signature, v1.6+)
try:
    from dataclasses import asdict
    from stress_release_detector import detect_stress_release_drops
    STRESS_RELEASE_AVAILABLE = True
except ImportError:
    STRESS_RELEASE_AVAILABLE = False

# Trans-Pacific correlation findings (Jan 2026) have been consolidated into
# the stress-release drop detector's multi-region sync filter.
# See: monitoring/src/stress_release_detector.py CORRELATION_GROUPS
TRANS_PACIFIC_AVAILABLE = False

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

    # New Historical Regions (Added Jan 2026)
    'kaikoura': {
        'name': 'New Zealand (Kaikoura)',
        'thd_station': 'SNZO',     # South Karori, Wellington (IU) - IRIS accessible
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 0,
        'notes': 'IU.SNZO (Wellington) - only IRIS-accessible station in NZ',
    },
    'anchorage': {
        'name': 'Alaska (Anchorage)',
        # candidate (cayley a8f99001, codex ab7c2ee1: accepted as an offline proposal): the configured primary is the
        # station that actually supplies every recorded anchorage THD value. AK.SSL is retired: it is absent from the
        # 2026-09-27 EarthScope listing that asked for it and never returned data. IU.COLA is ~444 km from Anchorage,
        # now stated instead of hidden behind a failed primary. AK.SSN is a CANDIDATE only (its own response, tidal-
        # band support and baseline first); no value served from COLA changes.
        'thd_station': 'COLA',     # College, AK (IU global)
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 0,
        'fallback_station': 'BMR',   # Burnt Mountain (AK): no baseline of its own -> its value is ineligible
        'fallback_network': 'AK',
        'notes': 'IU.COLA primary (actual support, ~444 km); AK.BMR fallback (uncalibrated). AK.SSL retired.',
    },
    'kumamoto': {
        'name': 'Japan (Kumamoto)',
        'thd_station': 'MAJO',     # Matsushiro (IU) - Global standard fallback
        'thd_network': 'IU',       # Using IU to avoid complex F-net/Hi-net auth for backtest
        'seismic_available': True,
        'latency_days': 0,
    },
    'hualien': {
        'name': 'Taiwan (Hualien)',
        'thd_station': 'TATO',     # Taipei (IU) - Reliable global station
        'thd_network': 'IU',
        'seismic_available': True,
        'latency_days': 0,
    },
    'mexico_guerrero': {
        'name': 'Mexico (Guerrero)',
        'thd_station': 'TLIG',     # MX.TLIG Tlapa, Guerrero - closest to subduction zone
        'thd_network': 'MX',
        'seismic_available': True,
        'latency_days': 0,
        'fallback_station': 'UNM', # G.UNM UNAM Mexico City - IDA global network
        'fallback_network': 'G',
        'notes': 'Guerrero subduction zone - Jan 2026 M6.5 event region',
    },
}


def configured_thd_station_regions(regions_map: Dict[str, Dict]) -> Dict[str, List[str]]:
    """calibration-eligibility-v3 wiring: 'NET.STA' -> every region CONFIGURED to read that THD station, as
    primary or fallback, with the same network defaults run_region_assessment uses ('CI' primary, 'IU'
    fallbacks). These are CONFIGURED roles, not the station a region actually recorded on a given day: a region
    whose primary answered never read its fallback. The eligibility rule uses the map only to DISCLOSE shared
    support (calibrated -> shared_station, both eligible), so the configured superset can over-disclose sharing
    but cannot change a tier. Read only while the rule is active."""
    served: Dict[str, set] = {}
    for region, config in regions_map.items():
        slots = ((config.get('thd_station'), config.get('thd_network', 'CI')),
                 (config.get('fallback_station'), config.get('fallback_network', 'IU')),
                 (config.get('fallback2_station'), config.get('fallback2_network', 'IU')))
        for station, network in slots:
            if station:
                served.setdefault(f'{network}.{station}', set()).add(region)
    return {key: sorted(regions) for key, regions in sorted(served.items())}


THD_STATION_REGIONS = configured_thd_station_regions(REGIONS)

# thd-station-attempts-v1 (METHOD_QUALIFICATION_DELIVERY_PLAN M4, 2026-10-04): PROSPECTIVE, OFF. When on, each
# region's record carries `thd_attempts`: every CONFIGURED THD station (primary, fallback, fallback2) with what
# happened to it on this run -- the value it supplied, the reason it supplied none, or that it was not attempted
# because an earlier station answered -- and, per station, the provider records of its fetch. Off, every output is
# byte-identical to the issued reports (legacy reports retained no attempt detail: NOT_RETAINED).
RECORD_THD_ATTEMPTS = True
# v2 = v1 provider records plus adapter, routing, typed_outcome, exception_class and http_status
# (thd-provider-routing-v1); same label would carry content the identity omits.
# v3 (candidate, codex 1515 s4) = v2 plus selector_basis on every provider record and, on DATA_RETURNED, the
# pre-merge coverage facts of the trace used; and LOCAL_PROCESSING_ERROR, with the coverage of each returned trace
# id, where a provider returned data the local merge/detrend refused. Values are unchanged; v2 never shipped.
THD_ATTEMPTS_SCHEMA = 'thd-station-attempts-v3'
THD_ROLES = ('CONFIGURED_PRIMARY', 'CONFIGURED_FALLBACK', 'CONFIGURED_FALLBACK2')


def thd_station_outcome(component, provider_records):
    """The outcome of one configured station's attempt, from the THD component and its provider records. The reason
    is redacted (evidence_redaction): an ERROR note carries exception text, which must not reach evidence raw."""
    if component is not None and component.available:
        return 'VALUE', ER.redact(component.notes)
    notes = component.notes if component is not None else 'no THD component'
    if notes.startswith('Insufficient data from'):
        # a BOUND station (thd_bound_station_operator): the provider answered but the operator refused the window by
        # name (gap/overlap, rate, coverage ...) -- not "no data"; the operator's own code is carried, redacted.
        refused = [r for r in provider_records or () if r.get('outcome') == 'OPERATOR_REFUSED']
        if refused and not any(r.get('outcome') == 'DATA_RETURNED' for r in provider_records or ()):
            return 'OPERATOR_REFUSED', ER.redact('%s (bound operator refused: %s)' % (
                notes, '; '.join(str(r.get('reason')) for r in refused)))
        if any(r.get('outcome') == 'DATA_RETURNED' for r in provider_records or ()):
            return 'INSUFFICIENT_SAMPLES', ER.redact(notes + ' (data returned, shorter than the 12 h minimum)')
        local = [r for r in provider_records or () if r.get('outcome') == 'LOCAL_PROCESSING_ERROR']
        if local:
            return 'LOCAL_PROCESSING_ERROR', ER.redact('%s (data returned, local processing refused it: %s)' % (
                notes, '; '.join(str(r.get('reason')) for r in local)))
        return 'NO_DATA', ER.redact(notes)
    return 'ERROR', ER.redact(notes)


# =============================================================================
# DAILY RUNNER
# =============================================================================

# codex 1246 replay-purity contract (cayley bar ac05448): set True by main() when an
# explicit --date was supplied. Historical replays fit the R5 model deterministically
# as-of the target date and never read/write the persistent live model store.
_HISTORICAL_REPLAY = False


def run_region_assessment(
    region: str,
    target_date: datetime,
    lambda_geo_ratio: Optional[float] = None,
    use_seismic: bool = True,
    lambda_geo_provenance: Optional[dict] = None,
    station_regions: Optional[Dict[str, List[str]]] = None,
) -> Optional[EnsembleResult]:
    """
    Run ensemble assessment for a single region.

    Args:
        region: Region key from REGIONS dict
        target_date: Date to assess
        lambda_geo_ratio: Optional Lambda_geo ratio (from live data)
        use_seismic: Whether to use seismic methods
        lambda_geo_provenance: calibration-eligibility-v3 -- the baseline record the ratio was derived under
            (None = unknown); read only while the eligibility rule is active
        station_regions: calibration-eligibility-v3 -- configured 'NET.STA' -> regions map; read only while active

    Returns:
        EnsembleResult or None if failed
    """
    config = REGIONS.get(region)
    if not config:
        logger.error(f"Unknown region: {region}")
        return None

    logger.info(f"Assessing {config['name']} for {target_date.date()}")

    # calibration-eligibility: the rule for THIS scored day (OFF before the amendment's effective boundary). Evaluated
    # OUTSIDE the per-region try: a misconfigured rule (active with no boundary) must stop the run, not turn every
    # region into a logged failure.
    rule_for_day = CE.rule_active_for_scored_day(target_date)
    try:
        ensemble = GeoSpecEnsemble(region=region, station_regions=station_regions,
                                   record_thd_attempts=RECORD_THD_ATTEMPTS,
                                   eligibility_rule_active=rule_for_day)

        # Set Lambda_geo if provided
        if lambda_geo_ratio is not None:
            ensemble.set_lambda_geo(target_date, lambda_geo_ratio, provenance=lambda_geo_provenance)

        # Determine if seismic should be used
        seismic_ok = use_seismic and config['seismic_available']

        if seismic_ok and config['thd_station']:
            # Build list of stations to try (primary + fallbacks)
            stations_to_try = [
                (config['thd_station'], config.get('thd_network', 'CI'))
            ]
            roles = [THD_ROLES[0]]
            # Add fallback stations if defined
            if config.get('fallback_station'):
                stations_to_try.append(
                    (config['fallback_station'], config.get('fallback_network', 'IU'))
                )
                roles.append(THD_ROLES[1])
            if config.get('fallback2_station'):
                stations_to_try.append(
                    (config['fallback2_station'], config.get('fallback2_network', 'IU'))
                )
                roles.append(THD_ROLES[2])
            station_records = []

            # Try each station until we get THD data
            result = None
            for station_code, network_code in stations_to_try:
                logger.debug(f"  Trying THD station {network_code}.{station_code}...")
                result = ensemble.compute_risk(
                    target_date,
                    thd_station=station_code,
                    thd_network=network_code
                )
                # Check if THD was successful
                thd_component = result.components.get('seismic_thd')
                if RECORD_THD_ATTEMPTS:
                    providers = list(ensemble.last_thd_fetch_attempts or [])
                    outcome, reason = thd_station_outcome(thd_component, providers)
                    station_records.append({
                        'role': roles[len(station_records)], 'station': f'{network_code}.{station_code}',
                        'attempted': True, 'outcome': outcome, 'reason': reason,
                        'selected': outcome == 'VALUE', 'providers': providers})
                if thd_component and thd_component.available:
                    logger.info(f"  THD data obtained from {network_code}.{station_code}")
                    break
                else:
                    logger.debug(f"  {network_code}.{station_code} returned no data, trying next...")

            if RECORD_THD_ATTEMPTS and result is not None:
                for (code, net), role in list(zip(stations_to_try, roles))[len(station_records):]:
                    station_records.append({
                        'role': role, 'station': f'{net}.{code}', 'attempted': False,
                        'outcome': 'NOT_ATTEMPTED', 'reason': 'an earlier configured station supplied the value',
                        'selected': False, 'providers': []})
                result.thd_attempts = {'schema': THD_ATTEMPTS_SCHEMA, 'scored_day': target_date.strftime('%Y-%m-%d'),
                                       'stations': station_records}
            # Return best result we got (even if THD failed on all stations)
            return result
        else:
            if ensemble.eligibility_rule_active:
                # method-comparability-v1 (codex db9a28ff finding 2): with the rule on, the Lambda_geo-only path goes
                # through the qualified combiner; the legacy construction below would bypass qualification.
                return ensemble.compute_risk(target_date, methods=('lambda_geo',))
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


def load_lambda_geo_baselines() -> Dict:
    """
    Load region-specific Lambda_geo baselines from calibration file.

    These baselines are computed from 90 days of REAL NGL GPS data using
    calibrate_lambda_geo_baselines.py. Using region-specific baselines
    ensures consistent units between live computation and historical reference.

    Returns:
        Dict mapping region_id to baseline statistics
    """
    baseline_file = Path(__file__).parent.parent / 'data' / 'baselines' / 'lambda_geo_baselines.json'

    if not baseline_file.exists():
        logger.warning(f"Lambda_geo baselines file not found: {baseline_file}")
        logger.warning("Run calibrate_lambda_geo_baselines.py to generate baselines")
        return {}

    try:
        with open(baseline_file) as f:
            data = json.load(f)
        return data.get('regions', {})
    except Exception as e:
        logger.error(f"Failed to load Lambda_geo baselines: {e}")
        return {}


def read_lambda_geo_calibration(baseline_file: Optional[Path] = None) -> Dict:
    """calibration-eligibility-v4 (codex a2a5cfe0 repair 1): ONE immutable read of the Lambda_geo calibration file.

    The ratio denominators AND every region's provenance are derived from the same bytes, and each provenance
    record carries their sha256, so a file replaced between two reads can never pair one version's denominator
    with another version's record (v3 read the file twice). Returns
        {'regions': ..., 'provenance': {region: record}, 'sha256': hex | None, 'status': READ | ABSENT | UNREADABLE}
    `regions` is exactly what load_lambda_geo_baselines() returns for the same bytes (missing or unreadable -> {};
    a JSON object -> its 'regions' value as written; anything else -> {}), so the ratio path is unchanged.
    A provenance record exists only for a region with an available baseline in that same object; its fields are
    passed as written and validated by the eligibility rule, never repaired here. Never raises.
    `baseline_file` overrides the path for tests only; production passes nothing."""
    if baseline_file is None:
        baseline_file = Path(__file__).parent.parent / 'data' / 'baselines' / 'lambda_geo_baselines.json'
    try:
        with open(baseline_file, 'rb') as f:
            raw = f.read()
    except FileNotFoundError:
        logger.warning(f"Lambda_geo baselines file not found: {baseline_file}")
        logger.warning("Run calibrate_lambda_geo_baselines.py to generate baselines")
        return {'regions': {}, 'provenance': {}, 'sha256': None, 'status': 'ABSENT'}
    except Exception as e:
        logger.error(f"Failed to load Lambda_geo baselines: {e}")
        return {'regions': {}, 'provenance': {}, 'sha256': None, 'status': 'UNREADABLE'}
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to load Lambda_geo baselines: {e}")
        return {'regions': {}, 'provenance': {}, 'sha256': digest, 'status': 'UNREADABLE'}
    regions = data.get('regions', {}) if isinstance(data, dict) else {}
    provenance = {}
    if isinstance(regions, dict):
        stamp = data.get('calibration_timestamp')
        calibrated_on = stamp[:10] if isinstance(stamp, str) else None
        for region, info in regions.items():
            if not isinstance(info, dict) or not info.get('available', False):
                continue
            period = info.get('calibration_period')
            start, sep, end = period.partition(' to ') if isinstance(period, str) else ('', '', '')
            provenance[region] = {
                'source': f'lambda_geo_baselines.json:{region}',
                'n_days': info.get('n_samples'),
                'window_start': start if sep else None,
                'window_end': end if sep else None,
                'calibrated_on': calibrated_on,
                'quality': info.get('quality'),
                'calibration_sha256': digest,
            }
    return {'regions': regions, 'provenance': provenance, 'sha256': digest, 'status': 'READ'}


# Global cache for Lambda_geo baselines (loaded once per session)
_LAMBDA_GEO_BASELINES = None


def get_lambda_geo_baseline(region: str) -> Optional[float]:
    """
    Get the calibrated Lambda_geo baseline for a region.

    Returns mean_lambda_geo from the calibration file, or None if unavailable.
    """
    global _LAMBDA_GEO_BASELINES

    if _LAMBDA_GEO_BASELINES is None:
        _LAMBDA_GEO_BASELINES = load_lambda_geo_baselines()

    baseline_info = _LAMBDA_GEO_BASELINES.get(region, {})
    if baseline_info.get('available', False):
        return baseline_info.get('mean_lambda_geo')
    return None


def fetch_ngl_lambda_geo(
    regions: List[str],
    target_date: datetime,
    days_back: int = 120,
    provenance_out: Optional[Dict[str, Optional[dict]]] = None,
) -> Dict[str, float]:
    """
    Fetch Lambda_geo ratios from NGL GPS data for all regions with polygon definitions.

    CRITICAL FIX (January 2026): Now uses region-specific baselines computed from
    real NGL GPS data (lambda_geo_baselines.json) instead of a hardcoded 0.01.
    This ensures consistent units between live computation and baseline reference.

    Args:
        regions: List of region keys to process
        target_date: Target date for assessment
        days_back: Number of days of GPS data to use (default 120)
        provenance_out: calibration-eligibility-v3 -- when a dict is given, filled with region -> the baseline
            provenance of each ratio returned (None when a fallback median or hardcoded baseline was used: that
            is not the region's own calibration). The returned ratios are unchanged.

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

    # Initialize GeoNet acquisition for New Zealand regions (lower latency than NGL)
    geonet = GeoNetLiveAcquisition(cache_dir)

    # Load station catalog once
    logger.info("Loading NGL station catalog for Lambda_geo computation...")
    ngl.load_station_catalog()

    # Load region-specific baselines -- calibration-eligibility-v4: ONE read; the denominators below and the
    # provenance attached to each ratio come from the same bytes (codex a2a5cfe0 repair 1).
    _calibration = read_lambda_geo_calibration()
    baselines = _calibration['regions']
    if not baselines:
        logger.warning("No Lambda_geo baselines available - ratios will be unreliable")

    _r5_dual = {}   # Amendment R5 dual-publication collector
    _own_provenance = _calibration['provenance']

    for region in regions:
        # Skip regions without polygon definitions
        if region not in FAULT_POLYGONS:
            logger.debug(f"No polygon definition for {region}, skipping Lambda_geo")
            continue

        try:
            logger.info(f"Computing Lambda_geo for {region}...")
            # Use GeoNet for NZ regions (kaikoura) - has ~3 day latency vs NGL's 10-14 days
            result = acquire_region_data(region, ngl, days_back, target_date, geonet)

            if result and result.n_stations >= 3 and result.lambda_geo_max > 0:
                # Get region-specific baseline (calibrated from real GPS data)
                baseline_info = baselines.get(region, {})
                if baseline_info.get('available', False):
                    baseline = baseline_info['mean_lambda_geo']
                    baseline_quality = baseline_info.get('quality', 'unknown')
                else:
                    # Fallback: use median across all calibrated regions
                    # This is a temporary fallback - proper baseline should be computed
                    all_means = [b['mean_lambda_geo'] for b in baselines.values()
                                if b.get('available', False)]
                    if all_means:
                        baseline = float(np.median(all_means))
                        baseline_quality = 'fallback_median'
                        logger.warning(f"  {region}: Using fallback baseline ({baseline:.4f})")
                    else:
                        # Last resort: use 0.1 (order of magnitude estimate)
                        baseline = 0.1
                        baseline_quality = 'hardcoded_fallback'
                        logger.warning(f"  {region}: Using hardcoded fallback baseline (0.1)")

                # Compute ratio
                ratio = result.lambda_geo_max / baseline

                # Clamp to reasonable range (0.1x to 50x)
                ratio = max(0.1, min(50.0, ratio))

                # Amendment R5 (registered 2026-07-29, owner-signed): precip-regressed
                # residual, rank-remapped onto the training ratio distribution so the
                # downstream risk mapping is unchanged. FAIL-OPEN: on ANY failure the
                # raw R3 ratio stands. Dual record collected for docs/r5_daily.json.
                r5 = None
                try:
                    try:
                        from src.precip_residual import r5_transform
                        from src.validate_predictions import REGION_DEFINITIONS as _r5_rd
                    except ImportError:
                        from precip_residual import r5_transform
                        from validate_predictions import REGION_DEFINITIONS as _r5_rd
                    _c = _r5_rd.get(region, {}).get('center')
                    if _c:
                        r5 = r5_transform(region, _c[0], _c[1], ratio,
                                          target_date.strftime('%Y-%m-%d'),
                                          historical=_HISTORICAL_REPLAY)
                except Exception as _e:
                    logger.warning(f"  {region}: R5 unavailable ({_e}); using R3 ratio")
                # SHADOW MODE (codex red-team 2026-07-30, findings R5-1..R5-5 verified by
                # executable counterexample): the R5 statistic is COMPUTED and dual-published
                # but does NOT replace the operational ratio until the bounded correction
                # (raw-ratio lineage, stale-model policy, leverage gates, exact rank
                # convention) lands and activation is registered at a pre-fixed timestamp.
                # This is the R5-5/A-5 shadow-first discipline.
                lambda_geo_data[region] = ratio
                if provenance_out is not None:
                    provenance_out[region] = (_own_provenance.get(region)
                                              if baseline_info.get('available', False) else None)
                if r5:
                    logger.info(f"  {region}: R5 SHADOW residual=p{100*r5['residual_percentile']:.0f} "
                                f"stat={r5['stat']:.2f}x (raw {ratio:.2f}x, fit n={r5['n_fit']}) "
                                f"[NOT substituted]")
                _r5_dual[region] = r5
                logger.info(f"  {region}: Lambda_geo ratio = {ratio:.1f}x "
                           f"(baseline={baseline:.4f}, {baseline_quality}, "
                           f"{result.n_stations} stations, {result.data_quality})")
            else:
                quality = result.data_quality if result else 'no_data'
                n_stations = result.n_stations if result else 0
                logger.info(f"  {region}: Lambda_geo unavailable ({n_stations} stations, {quality})")

        except Exception as e:
            logger.warning(f"Failed to compute Lambda_geo for {region}: {e}")

    # Amendment R5 dual publication (fail-open)
    try:
        try:
            from src.precip_residual import publish_r5_daily
        except ImportError:
            from precip_residual import publish_r5_daily
        publish_r5_daily(target_date.strftime('%Y-%m-%d'), _r5_dual)
    except Exception as _e:
        logger.warning(f"R5 dual publication skipped ({_e})")

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
    # calibration-eligibility-v3: provenance of the ratio each region actually uses. Pilot and caller-supplied
    # ratios carry none (unknown); an NGL ratio carries its baseline record. Read only while the rule is active.
    lambda_geo_provenance: Dict[str, Optional[dict]] = {}

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
        ngl_provenance: Dict[str, Optional[dict]] = {}
        ngl_lambda_geo = fetch_ngl_lambda_geo(regions, target_date, provenance_out=ngl_provenance)
        # Merge NGL data, but don't override pilot data if available
        for region, ratio in ngl_lambda_geo.items():
            if region not in lambda_geo_data:
                lambda_geo_data[region] = ratio
                lambda_geo_provenance[region] = ngl_provenance.get(region)
        logger.info(f"Lambda_geo available for {len(lambda_geo_data)} regions via NGL/pilot data")

    results = {}

    for region in regions:
        lg_ratio = lambda_geo_data.get(region)
        result = run_region_assessment(
            region=region,
            target_date=target_date,
            lambda_geo_ratio=lg_ratio,
            use_seismic=use_seismic,
            lambda_geo_provenance=lambda_geo_provenance.get(region),
            station_regions=THD_STATION_REGIONS,
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
    loader=None,
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
        prior_rows = []  # method-comparability-v1: the issued prior-day rows, nearest first (None = hole)

        # Look back for consecutive days at same or higher tier
        for days_back in range(1, required_consecutive + 2):
            # `loader` (production path) serves the PUBLIC revision for the
            # prior day or None for a hole; the default keeps the local-dir
            # replay semantics the corpus bar exercises.
            if loader is not None:
                prev_data = loader(days_back)
            else:
                prev_data = load_previous_results(output_dir, target_date, days_back)
            if prev_data and region in prev_data.get('regions', {}):
                prev_tier = prev_data['regions'][region].get('tier', 0)
                tier_history.insert(0, prev_tier)
                prior_rows.append(prev_data['regions'][region])
            else:
                tier_history.insert(0, None)
                prior_rows.append(None)

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
        # method-comparability-v1 (rule active only): count only prior days issued under the SAME regime; a
        # regime change is recorded, never carried across. Issued tiers in tier_history stay as issued.
        method_set = getattr(result, 'method_set', None)
        if method_set is not None:
            regime = MC.regime_persistence(current_tier, method_set, prior_rows, required_consecutive)
            consecutive = max(1, regime.pop('consecutive_days'))
            is_confirmed = regime.pop('is_confirmed')
            persistence[region]['consecutive_days'] = consecutive if current_tier >= 1 else 0
            persistence[region]['is_confirmed'] = is_confirmed
            persistence[region]['regime'] = regime

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
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'regions': {},
        'summary': {
            'total_regions': 0,
            'tier_counts': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0},
            'confirmed_watch_count': 0,
            'preliminary_watch_count': 0,
            'degraded_count': 0,
            'max_risk_region': None,
            'max_risk': 0.0,
        },
        'earthquake_events': events_data or {},
    }

    # Load existing data if file exists to merge
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                # Preserve existing regions and summary
                if 'regions' in existing_data:
                    output_data['regions'] = existing_data['regions']
                if 'summary' in existing_data:
                    output_data['summary'] = existing_data['summary']
                # Merge earthquake events
                if 'earthquake_events' in existing_data:
                    output_data['earthquake_events'].update(existing_data['earthquake_events'])
            logger.info(f"Merging results into existing file: {output_file}")
        except Exception as e:
            logger.warning(f"Failed to read existing file for merge: {e}")

    # Update with new results
    for region, result in results.items():
        region_data = result.to_dict()

        # Add persistence info if available
        if persistence and region in persistence:
            region_data['persistence'] = persistence[region]
            if result.tier >= 1:
                # Update counters if this is a new entry or status change
                # Note: This simple counter update is imperfect for merges but sufficient for dashboard
                pass 

        output_data['regions'][region] = region_data

    # Re-calculate summary from scratch based on ALL regions (merged)
    tier_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    max_risk = 0.0
    max_risk_region = None
    confirmed_count = 0
    preliminary_count = 0
    degraded_count = 0

    for r_name, r_data in output_data['regions'].items():
        tier = r_data.get('tier', 0)
        risk = r_data.get('combined_risk', 0.0)
        
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        if risk > max_risk:
            max_risk = risk
            max_risk_region = r_name
            
        if tier == -1:
            degraded_count += 1
            
        # Check persistence from stored data
        if 'persistence' in r_data and r_data['persistence'].get('is_confirmed'):
            confirmed_count += 1
        elif tier >= 1:
            preliminary_count += 1

    output_data['summary'] = {
        'total_regions': len(output_data['regions']),
        'tier_counts': tier_counts,
        'confirmed_watch_count': confirmed_count,
        'preliminary_watch_count': preliminary_count,
        'degraded_count': degraded_count,
        'max_risk_region': max_risk_region,
        'max_risk': max_risk,
    }
    # method-comparability-v1 (rule active only): combined_risk is a conditional mean over each region's
    # included methods, so the cross-region maximum is reported per comparability group, and a single
    # maximum across different groups is withheld. Rows without a method set leave the summary as before.
    with_method_set = sorted(r for r, d in output_data['regions'].items() if d.get('method_set'))
    if with_method_set:
        groups = MC.comparison_groups(output_data['regions'])
        incomplete = sorted(r for r, d in output_data['regions'].items()
                            if (d.get('method_set') or {}).get('included')
                            and not d['method_set'].get('comparability_complete'))
        missing = sorted(set(output_data['regions']) - set(with_method_set))
        output_data['summary']['comparison'] = {
            'contract_version': MC.COMPARISON_CONTRACT_VERSION,
            'risk_basis': MC.RISK_BASIS,
            'groups': groups,
            'regions_without_method_set': missing,
            'regions_with_incomplete_support': incomplete,
        }
        if missing or incomplete or not groups:
            output_data['summary']['max_risk_region'] = None
            output_data['summary']['max_risk'] = None
            output_data['summary']['max_risk_withheld'] = (
                'mixed legacy rows, incomplete support, or no qualified comparison group; inspect summary.comparison')
        elif len(groups) > 1:
            output_data['summary']['max_risk_region'] = None
            output_data['summary']['max_risk'] = None
            output_data['summary']['max_risk_withheld'] = (
                f'regions span {len(groups)} comparability groups; the maximum is reported per group '
                f'(summary.comparison.groups)')
        elif len(groups) == 1:
            only = next(iter(groups.values()))
            output_data['summary']['max_risk_region'] = only['max_risk_region']
            output_data['summary']['max_risk'] = only['max_risk']
            if len(only['max_risk_regions']) > 1:
                # an exact tie (e.g. two regions on one shared station) is reported, not broken by order
                output_data['summary']['max_risk_region'] = None
                output_data['summary']['max_risk_tied_regions'] = list(only['max_risk_regions'])

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

    # Check which regions already have entries for this date (avoid duplicates per region)
    existing_regions = set()
    if file_exists:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] == date_str:
                    existing_regions.add(row[1])  # row[1] is region

    # Filter to only regions not already in CSV for this date
    regions_to_add = {r: res for r, res in results.items() if r not in existing_regions}

    if not regions_to_add:
        logger.info(f"All {len(results)} regions already in daily CSV for {date_str}, skipping")
        return csv_file

    if existing_regions:
        logger.info(f"Found {len(existing_regions)} existing regions for {date_str}, adding {len(regions_to_add)} new")

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow([
                'date', 'region', 'tier', 'risk', 'methods', 'confidence',
                'lg_ratio', 'thd', 'fc_l2l1', 'status', 'notes'
            ])

        for region, result in regions_to_add.items():
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

    logger.info(f"Appended {len(regions_to_add)} rows to: {csv_file}")
    return csv_file


def append_to_dashboard_csv(
    results: Dict[str, EnsembleResult],
    target_date: datetime,
) -> Path:
    """
    Append results to monitoring/dashboard/data.csv for the 30-day history chart.

    This is the authoritative source that gets copied to docs/data.csv by run_and_publish.ps1.
    Format: date,region,tier,risk,confidence,methods,agreement

    Args:
        results: Dict mapping region to EnsembleResult
        target_date: Date of the assessment

    Returns:
        Path to the CSV file
    """
    # Path to dashboard CSV (authoritative source for GitHub Pages)
    csv_file = Path(__file__).parent.parent / 'dashboard' / 'data.csv'
    date_str = target_date.strftime('%Y-%m-%d')

    # Check if file exists to determine if we need header
    file_exists = csv_file.exists()

    # Guard: row count must never decrease (prevents silent history truncation)
    pre_append_rows = 0
    existing_regions = set()
    if file_exists:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            pre_append_rows = len(rows)
            for row in rows:
                if len(row) >= 2 and row[0] == date_str:
                    existing_regions.add(row[1])

        # Ensure file ends with newline before appending
        with open(csv_file, 'rb') as f:
            f.seek(-1, 2)  # Go to last byte
            if f.read(1) != b'\n':
                with open(csv_file, 'a') as f2:
                    f2.write('\n')

        # Date-continuity guard: warn if appending date isn't consecutive to last date in CSV
        dates_in_csv = sorted({row[0] for row in rows if len(row) >= 2 and row[0][:4].isdigit()})
        if dates_in_csv:
            last_csv_date = datetime.strptime(dates_in_csv[-1], '%Y-%m-%d')
            expected_next = last_csv_date + timedelta(days=1)
            if target_date.date() > expected_next.date():
                gap_days = (target_date.date() - last_csv_date.date()).days - 1
                logger.warning(
                    f"DATE GAP: {gap_days} day(s) missing between {dates_in_csv[-1]} and {date_str} "
                    f"— dashboard chart will interpolate across the gap"
                )

    # Filter to only regions not already in CSV for this date
    regions_to_add = {r: res for r, res in results.items() if r not in existing_regions}

    if not regions_to_add:
        logger.info(f"All {len(results)} regions already in dashboard CSV for {date_str}, skipping")
        return csv_file

    if existing_regions:
        logger.info(f"Found {len(existing_regions)} existing regions for {date_str}, adding {len(regions_to_add)} new")

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow([
                'date', 'region', 'tier', 'risk', 'confidence', 'methods', 'agreement'
            ])

        for region, result in regions_to_add.items():
            writer.writerow([
                date_str,
                region,
                result.tier,
                f"{result.combined_risk:.4f}",
                f"{result.confidence:.2f}",
                result.methods_available,
                result.agreement or ''
            ])

    # Post-write guard: verify row count never decreased
    with open(csv_file, 'r', newline='') as f:
        post_append_rows = sum(1 for _ in f)
    if post_append_rows < pre_append_rows:
        raise RuntimeError(
            f"Dashboard CSV history shrank from {pre_append_rows} to {post_append_rows} rows — aborting"
        )

    logger.info(f"Appended {len(regions_to_add)} rows to dashboard CSV: {csv_file} ({pre_append_rows} -> {post_append_rows})")
    return csv_file


def run_stress_release_detection(
    output_dir: Path,
    target_date: datetime,
    lookback_days: int = 7,
) -> List[Dict]:
    """
    Run the stress-release drop detector against the just-saved ensemble
    history and persist its verdict into ensemble_<date>.json.

    The GitHub Pages dashboard recomputes drops client-side from data.csv; this
    writes the authoritative *server-side* record so logging, alerting, and the
    audit trail are driven by the Python detector (single source of truth)
    rather than only the JS recompute, which can silently drift from it.

    Non-critical: any failure is logged and swallowed so it never blocks the
    daily run. Returns the detected drops as dicts, newest first.
    """
    if not STRESS_RELEASE_AVAILABLE:
        logger.debug("Stress-release detector unavailable; skipping")
        return []

    try:
        drops = detect_stress_release_drops(
            ensemble_dir=output_dir,
            target_date=target_date,
            lookback_days=lookback_days,
        )
    except Exception as e:
        logger.warning(f"Stress-release detection failed (non-critical): {e}")
        return []

    drop_dicts = [asdict(d) for d in drops]

    if drops:
        logger.info(f"Stress-release detector: {len(drops)} drop(s) flagged")
        for d in drops:
            logger.info(
                f"  {d.region} {d.drop_date}: tier {d.prior_tier}->0, "
                f"dz={d.delta_z:.2f}, {d.consecutive_elevated_days}d elevated "
                f"[{d.confidence}]"
            )
    else:
        logger.info("Stress-release detector: no drops in lookback window")

    # Persist into the authoritative ensemble JSON for this date.
    date_str = target_date.strftime('%Y-%m-%d')
    output_file = output_dir / f'ensemble_{date_str}.json'
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            data['stress_release_drops'] = drop_dicts
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Stress-release verdict written to {output_file.name}")
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not write stress-release verdict to JSON: {e}")

    return drop_dicts


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


def _inputs_capsule(pins: List[Dict], scored_day: str, view: Dict) -> Dict:
    """codex 1831Z F3: the inputs capsule from the COMMITTED view (code and
    calibration reopened from Git at HEAD by REV.committed_inputs_view), one
    prior_revision / legacy_record per non-hole pin with the real reopened
    byte length, and one scored_day entry over its canonical preimage. No
    checkout bytes are read here."""
    cap = REV.load_legacy_baseline(REPO_ROOT)
    pin_entries = [REV.pin_input_entry(REPO_ROOT, cap, pin)
                   for pin in pins if pin['kind'] != 'hole']
    return REV.inputs_from_view(view, pin_entries, scored_day)


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
    parser.add_argument(
        '--cutover', action='store_true',
        help='OWNER-RUN ONCE at landing: write docs/ensemble/legacy_baseline_v1.json '
             '(the immutable cutover capsule enumerating every committed pre-cutover '
             'record and binding the frozen data.csv prefix). Scores nothing; refuses '
             'if the capsule exists.'
    )
    parser.add_argument(
        '--rescore', type=str, default=None, metavar='REASON',
        help='Owner-authorized RE-SCORE of a day that already has a public '
             'revision: appends a NEW immutable revision naming the previous '
             'one and this reason. Without it, a second run of a scored day '
             'REFUSES (exit 12) and writes nothing public.'
    )

    args = parser.parse_args()

    if args.cutover:
        try:
            cap = REV.build_legacy_baseline(REPO_ROOT)
            path = REV.write_legacy_baseline(REPO_ROOT, cap)
        except REV.RevisionRefusal as e:
            logger.error(str(e))
            return 12
        logger.info(f"Cutover capsule written: {path} "
                    f"({len(cap['records'])} committed records, "
                    f"{cap['legacy_csv']['row_count']} frozen csv rows)")
        return 0

    # Parse date (apply latency offset if no explicit date given)
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
        # codex 1246: an explicit --date is a HISTORICAL REPLAY -- R5 fits as-of the
        # target date, store-free (replay-order/store-state invariance).
        global _HISTORICAL_REPLAY
        _HISTORICAL_REPLAY = True
    else:
        # Default: N days ago due to data latency (seismic ~1 day, GPS 2-14 days)
        # B6 (asylum 2026-09-02): the scored day is keyed in UTC. A naive
        # host-local clock moved the day boundary with the runner's zone;
        # the key is now the UTC calendar day minus the latency, held as a
        # naive midnight so every downstream strftime stays byte-identical.
        today_utc = datetime.now(timezone.utc).date()
        target_date = (datetime(today_utc.year, today_utc.month, today_utc.day)
                       - timedelta(days=args.latency))
        logger.info(f"Using {args.latency}-day latency offset (UTC data date: {target_date.date()})")

    # Determine regions
    regions = [args.region] if args.region else None

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'data' / 'ensemble_results'

    # Production path (no --output-dir) publishes an immutable public
    # revision; a replay into a custom --output-dir keeps the local-dir
    # semantics and publishes nothing.
    production = args.output_dir is None

    # ---- Production PREFLIGHT, before any scoring or acquisition (codex
    # 1912Z; the 1831Z F3 repair as accepted). The program that scores must
    # be the COMMITTED program: resolve the Git view of code + calibration at
    # HEAD and refuse typed on any untracked / missing / dirty file; refuse a
    # dirty store; require the cutover capsule; capture the journal bytes and
    # resolve the prior-day pins -- ALL of it before run_all_regions (which
    # owns the provider-fetch seam). A typed refusal here exits 12 with
    # nothing scored, nothing fetched and nothing written. publish_revision
    # re-runs the Git/checkout check after scoring; that second check is the
    # post-scoring TOCTOU guard and stays.
    pins, loader, journal_snapshot, legacy_cap, committed_view = [], None, None, None, None
    if production:
        try:
            committed_view = REV.committed_inputs_view(REPO_ROOT, 'HEAD')
            REV.checkout_matches_committed(REPO_ROOT, committed_view)
            REV.check_store_clean(REPO_ROOT)          # typed recovery refusal
            legacy_cap = REV.load_legacy_baseline(REPO_ROOT)
            if legacy_cap is None:
                raise REV.RevisionRefusal(
                    "LEGACY_CAPSULE_ABSENT: run `--cutover` once (owner) before "
                    "the first production run of the revision model")
            journal_snapshot = REV.journal_bytes(REPO_ROOT)
            view = REV.prior_days_view(REPO_ROOT, journal_snapshot, legacy_cap,
                                       target_date.strftime('%Y-%m-%d'), 3)
        except REV.RevisionRefusal as e:
            logger.error(str(e))
            return 12
        prior_map = {k + 1: rec for k, (_ds, rec, _pin) in enumerate(view)}
        pins = [pin for (_ds, _rec, pin) in view]
        holes = [pin['date'] for pin in pins if pin['kind'] == 'hole']
        if holes:
            logger.warning(f"Persistence holes (no public revision, no legacy record): {holes}")
        loader = lambda days_back: prior_map.get(days_back)  # noqa: E731

    # Run assessment (scoring + provider fetch; only reached once the
    # production preflight above has passed)
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

    # Check persistence (requires 2 consecutive days for confirmed status).
    # Production path: prior days come from the PUBLIC revision store only,
    # through the pins / loader the preflight resolved against the captured
    # journal bytes; the exact (date, run_id, sha256) consumed and any holes
    # are recorded on the published record. Replay: local-dir semantics.
    persistence = check_persistence(
        results,
        output_dir,
        target_date,
        required_consecutive=2,
        loader=loader,
    )

    # Save results with persistence info and earthquake events
    save_results(results, output_dir, target_date, persistence, events_data)

    # Append to daily CSV for trending (detailed log)
    append_to_daily_csv(results, output_dir, target_date, persistence)

    # docs/data.csv is now DERIVED from the revision index by one writer
    # (REV.write_data_csv, inside publish_revision); the old appender is
    # no longer called on the production path.

    # Run the stress-release drop detector and persist its verdict into the
    # local ensemble JSON BEFORE publication (a revision is create-once).
    run_stress_release_detection(output_dir, target_date)

    # ---- Publish the immutable public revision (production path only)
    if production:
        date_str = target_date.strftime('%Y-%m-%d')
        with open(output_dir / f'ensemble_{date_str}.json', 'r', encoding='utf-8') as f:
            record = json.load(f)
        try:
            entry = REV.publish_revision(
                REPO_ROOT, record, _inputs_capsule(pins, date_str, committed_view),
                journal_snapshot, pins, datetime.now(timezone.utc),
                rescore_reason=args.rescore)
        except REV.RevisionRefusal as e:
            logger.error(str(e))
            return 12
        logger.info(f"Published public revision {entry['date']}/{entry['run_id']} "
                    f"sha256 {entry['sha256'][:12]} (supersedes {entry['supersedes']}); "
                    "commit docs/ensemble, docs/ensemble_latest.json and docs/data.csv TOGETHER")

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
        exit_code = confirmed_max_tier
    else:
        max_tier = max(r.tier for r in results.values())
        if max_tier > 0:
            logger.info(f"Exit code: 0 (tier {max_tier} is preliminary, not confirmed)")
        exit_code = 0  # Preliminary alerts don't trigger exit codes

    # Run prediction validation (builds track record)
    # Validates predictions from 7-14 days ago against actual events
    if VALIDATION_AVAILABLE:
        try:
            logger.info("Running prediction validation (7-14 day lookback)...")
            validated, stats = run_validation(
                lookback_start_days=7,
                lookback_end_days=14,
                min_tier=2,  # ELEVATED or higher (WATCH is awareness only, not scored)
                min_magnitude=5.5,  # M5.5+ for operationally significant events
            )
            if stats.get('hits', 0) > 0:
                logger.info(f"Validation: {stats['hits']} hits, {stats['false_alarms']} false alarms")
            else:
                logger.info(f"Validation: No new hits in lookback window")
        except Exception as e:
            logger.warning(f"Prediction validation failed (non-critical): {e}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
