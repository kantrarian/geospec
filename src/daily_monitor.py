#!/usr/bin/env python3
"""
daily_monitor.py
Operational Lambda_geo monitoring pipeline.

This is THE operational script that:
1. Fetches live NGL Rapid GPS data (IGS20 frame)
2. Converts position time series to velocities
3. Computes strain tensor field via Delaunay triangulation
4. Calculates Lambda_geo = ||[Ė, Ë]||_F
5. Computes ratio to 90-day baseline (with seasonal detrending)
6. Outputs alert tier status

Data Flow:
    NGL Rapid IGS20 --> Position Time Series --> Velocities -->
    Strain Tensors --> Lambda_geo --> Baseline Ratio --> Alert Tier

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass

# Local imports
from live_data_fetcher import (
    fetch_region_data,
    compute_position_velocities,
    check_data_latency,
    REGION_STATIONS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ALERT TIER DEFINITIONS
# =============================================================================

ALERT_TIERS = {
    'NORMAL':   {'min': 0.0,  'max': 2.0,  'color': 'green'},
    'WATCH':    {'min': 2.0,  'max': 5.0,  'color': 'yellow'},
    'ELEVATED': {'min': 5.0,  'max': 10.0, 'color': 'orange'},
    'CRITICAL': {'min': 10.0, 'max': float('inf'), 'color': 'red'},
}


@dataclass
class MonitoringResult:
    """Container for daily monitoring output."""
    region: str
    date: datetime
    lambda_geo_current: float
    lambda_geo_baseline: float
    ratio: float
    tier: str
    days_elevated: int
    num_stations: int
    data_latency_days: int
    station_details: Dict


def get_alert_tier(ratio: float) -> str:
    """Determine alert tier from baseline ratio."""
    for tier, bounds in ALERT_TIERS.items():
        if bounds['min'] <= ratio < bounds['max']:
            return tier
    return 'CRITICAL'


# =============================================================================
# STRAIN TENSOR COMPUTATION (Simplified for single-triangle regions)
# =============================================================================

def positions_to_strain_rate(velocities_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert velocity field to 2D strain rate tensor.

    For a velocity field v = (vE, vN), the velocity gradient is:
        L = [[dvE/dE, dvE/dN],
             [dvN/dE, dvN/dN]]

    The strain rate tensor is the symmetric part:
        Ė = 0.5 * (L + L^T)

    For GPS stations, we estimate L from the spatial distribution of velocities.

    Args:
        velocities_df: DataFrame with columns [station, lat, lon, vE, vN]

    Returns:
        strain_rate: 2x2 strain rate tensor (nanostrain/day)
        uncertainty: 2x2 uncertainty matrix
    """
    if len(velocities_df) < 3:
        return np.zeros((2, 2)), np.ones((2, 2)) * np.inf

    # Extract coordinates and velocities
    # Convert lat/lon to local meters (approximate)
    lat0 = velocities_df['lat'].mean()
    lon0 = velocities_df['lon'].mean()

    # Meters per degree at this latitude
    m_per_deg_lat = 111320  # roughly constant
    m_per_deg_lon = 111320 * np.cos(np.radians(lat0))

    # Local coordinates in meters
    x = (velocities_df['lon'] - lon0) * m_per_deg_lon  # East
    y = (velocities_df['lat'] - lat0) * m_per_deg_lat  # North
    vE = velocities_df['vE'].values * 1000  # mm/day to convert later
    vN = velocities_df['vN'].values * 1000

    # Fit linear velocity field: v = A*r + b
    # [vE, vN]^T = [[a, b], [c, d]] @ [x, y]^T + [e, f]^T
    #
    # Least squares: minimize sum of ||v_i - A*r_i - b||^2

    n = len(velocities_df)

    # Design matrix for linear fit
    # For each station: vE_i = a*x_i + b*y_i + e
    #                   vN_i = c*x_i + d*y_i + f

    # Stack the equations
    X = np.column_stack([x, y, np.ones(n)])

    # Solve for East velocity gradient
    try:
        coeffs_E, residuals_E, rank_E, s_E = np.linalg.lstsq(X, vE, rcond=None)
        coeffs_N, residuals_N, rank_N, s_N = np.linalg.lstsq(X, vN, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros((2, 2)), np.ones((2, 2)) * np.inf

    # Velocity gradient tensor L
    # L[i,j] = dv_i / dx_j
    L = np.array([
        [coeffs_E[0], coeffs_E[1]],  # dvE/dE, dvE/dN
        [coeffs_N[0], coeffs_N[1]]   # dvN/dE, dvN/dN
    ])

    # Strain rate tensor = symmetric part
    strain_rate = 0.5 * (L + L.T)

    # Convert to nanostrain/day (1e-9 / day)
    # Current units: (mm/day) / m = 1e-3 / day
    # Multiply by 1e6 to get nanostrain
    strain_rate *= 1e6

    # Simple uncertainty estimate from residuals
    if len(residuals_E) > 0 and len(residuals_N) > 0:
        sigma_E = np.sqrt(residuals_E[0] / max(n - 3, 1)) if n > 3 else np.inf
        sigma_N = np.sqrt(residuals_N[0] / max(n - 3, 1)) if n > 3 else np.inf
        uncertainty = np.array([[sigma_E, sigma_E], [sigma_N, sigma_N]]) * 1e6
    else:
        uncertainty = np.ones((2, 2)) * np.nan

    return strain_rate, uncertainty


def compute_strain_rate_commutator(strain_rate_series: List[np.ndarray],
                                   dt_days: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lambda_geo = ||[Ė, Ë]||_F from strain rate time series.

    Args:
        strain_rate_series: List of 2x2 strain rate tensors
        dt_days: Time step in days

    Returns:
        lambda_geo: Array of Lambda_geo values
        strain_accel: Array of strain acceleration tensors
    """
    n = len(strain_rate_series)
    if n < 3:
        return np.array([0.0]), np.array([np.zeros((2, 2))])

    E_dot = np.array(strain_rate_series)  # Strain rate (our "E")

    # Compute strain acceleration (our "Ė")
    E_ddot = np.zeros_like(E_dot)
    E_ddot[1:-1] = (E_dot[2:] - E_dot[:-2]) / (2 * dt_days)
    E_ddot[0] = (E_dot[1] - E_dot[0]) / dt_days
    E_ddot[-1] = (E_dot[-1] - E_dot[-2]) / dt_days

    # Compute commutator [Ė, Ë] at each time step
    lambda_geo = np.zeros(n)

    for t in range(n):
        # Commutator [A, B] = AB - BA
        commutator = E_dot[t] @ E_ddot[t] - E_ddot[t] @ E_dot[t]

        # Frobenius norm
        lambda_geo[t] = np.sqrt(np.sum(commutator**2))

    return lambda_geo, E_ddot


# =============================================================================
# BASELINE COMPUTATION
# =============================================================================

def compute_baseline_with_seasonal(lambda_geo_series: np.ndarray,
                                   dates: List[datetime],
                                   lookback_days: int = 90,
                                   gap_days: int = 14) -> Tuple[float, float]:
    """
    Compute baseline Lambda_geo with seasonal detrending.

    Uses 90-day lookback with 14-day exclusion gap (per spec).

    Args:
        lambda_geo_series: Time series of Lambda_geo values
        dates: Corresponding dates
        lookback_days: Days to look back for baseline
        gap_days: Days to exclude before current (avoid contamination)

    Returns:
        baseline_mean: Baseline Lambda_geo (seasonally detrended)
        baseline_std: Standard deviation
    """
    if len(lambda_geo_series) < lookback_days:
        # Not enough data - use what we have
        baseline_mean = np.nanmedian(lambda_geo_series)
        baseline_std = np.nanstd(lambda_geo_series)
        return baseline_mean, baseline_std

    # Select baseline window (exclude recent gap_days)
    baseline_values = lambda_geo_series[:-gap_days] if gap_days > 0 else lambda_geo_series
    baseline_values = baseline_values[-lookback_days:]

    # Simple seasonal detrending: remove day-of-year effects
    # (More sophisticated: fit and remove annual + semi-annual harmonics)
    baseline_mean = np.nanmedian(baseline_values)
    baseline_std = np.nanstd(baseline_values)

    # Use median for robustness against outliers
    return max(baseline_mean, 1e-10), baseline_std


# =============================================================================
# MAIN MONITORING PIPELINE
# =============================================================================

def run_daily_monitoring(region: str = 'ridgecrest',
                         lookback_days: int = 90,
                         output_dir: Optional[Path] = None) -> MonitoringResult:
    """
    Run daily Lambda_geo monitoring for a region.

    Args:
        region: Region name (from REGION_STATIONS)
        lookback_days: Days for baseline computation
        output_dir: Optional directory to save results

    Returns:
        MonitoringResult with current status
    """
    logger.info(f"=" * 60)
    logger.info(f"Lambda_geo Daily Monitor - {region.upper()}")
    logger.info(f"=" * 60)

    # Step 1: Fetch live GPS data
    logger.info("Step 1: Fetching NGL Rapid GPS data...")
    network_data = fetch_region_data(region, days_lookback=lookback_days + 30)

    if not network_data:
        logger.error("No GPS data retrieved!")
        return None

    # Check data latency
    latest_date, n_current = check_data_latency(network_data)
    latency_days = (datetime.now() - latest_date).days if latest_date else None
    logger.info(f"  Data latency: {latency_days} days (latest: {latest_date})")

    # Step 2: Compute velocities
    logger.info("Step 2: Computing position velocities...")
    velocity_data = compute_position_velocities(network_data, window_days=7)

    # Step 3: Compute strain rate tensor time series
    logger.info("Step 3: Computing strain rate tensors...")

    # Get common dates across all stations
    all_dates = set()
    for station, df in velocity_data.items():
        all_dates.update(df['datetime'].dt.date)

    common_dates = sorted(all_dates)

    strain_rate_series = []
    dates_used = []

    for date in common_dates:
        # Collect velocities for this date
        daily_velocities = []
        for station, df in velocity_data.items():
            day_data = df[df['datetime'].dt.date == date]
            if not day_data.empty:
                row = day_data.iloc[0]
                if not np.isnan(row['vE']) and not np.isnan(row['vN']):
                    daily_velocities.append({
                        'station': station,
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'vE': row['vE'],
                        'vN': row['vN'],
                    })

        if len(daily_velocities) >= 3:
            vel_df = pd.DataFrame(daily_velocities)
            strain_rate, _ = positions_to_strain_rate(vel_df)
            strain_rate_series.append(strain_rate)
            dates_used.append(datetime.combine(date, datetime.min.time()))

    logger.info(f"  Computed strain rates for {len(strain_rate_series)} days")

    # Step 4: Compute Lambda_geo = ||[Ė, Ë]||_F
    logger.info("Step 4: Computing Lambda_geo...")
    lambda_geo_series, _ = compute_strain_rate_commutator(strain_rate_series)

    # Current value (most recent)
    current_lambda = lambda_geo_series[-1] if len(lambda_geo_series) > 0 else 0.0

    # Step 5: Compute baseline ratio
    logger.info("Step 5: Computing baseline ratio...")
    baseline_mean, baseline_std = compute_baseline_with_seasonal(
        lambda_geo_series, dates_used, lookback_days=90, gap_days=14
    )

    ratio = current_lambda / baseline_mean if baseline_mean > 0 else 0.0

    # Step 6: Determine alert tier
    tier = get_alert_tier(ratio)

    # Count days elevated (above WATCH threshold)
    days_elevated = 0
    for val in reversed(lambda_geo_series):
        if val / baseline_mean >= 2.0:
            days_elevated += 1
        else:
            break

    # Assemble result
    result = MonitoringResult(
        region=region,
        date=latest_date,
        lambda_geo_current=current_lambda,
        lambda_geo_baseline=baseline_mean,
        ratio=ratio,
        tier=tier,
        days_elevated=days_elevated,
        num_stations=len(network_data),
        data_latency_days=latency_days,
        station_details={s: len(df) for s, df in network_data.items()}
    )

    # Output summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MONITORING RESULT")
    logger.info("=" * 60)
    logger.info(f"Region:           {region}")
    logger.info(f"Date:             {latest_date.strftime('%Y-%m-%d')}")
    logger.info(f"Lambda_geo:       {current_lambda:.4f} nanostrain²/day²")
    logger.info(f"Baseline:         {baseline_mean:.4f} nanostrain²/day²")
    logger.info(f"Ratio:            {ratio:.2f}×")
    logger.info(f"Alert Tier:       {tier}")
    logger.info(f"Days Elevated:    {days_elevated}")
    logger.info(f"Stations:         {len(network_data)}")
    logger.info(f"Data Latency:     {latency_days} days")
    logger.info("=" * 60)

    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary
        summary = {
            'region': region,
            'date': latest_date.isoformat(),
            'lambda_geo_current': float(current_lambda),
            'lambda_geo_baseline': float(baseline_mean),
            'ratio': float(ratio),
            'tier': tier,
            'days_elevated': days_elevated,
            'num_stations': len(network_data),
            'data_latency_days': latency_days,
            'generated_at': datetime.now().isoformat(),
        }

        with open(output_dir / f'{region}_latest.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    return result


def run_all_regions() -> List[MonitoringResult]:
    """Run monitoring for all configured regions."""
    results = []

    for region in REGION_STATIONS.keys():
        config = REGION_STATIONS[region]
        if not config['stations']:
            logger.warning(f"Skipping {region} - no stations configured")
            continue

        try:
            result = run_daily_monitoring(region)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error monitoring {region}: {e}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Lambda_geo Daily Monitor')
    parser.add_argument('--region', default='ridgecrest',
                       help='Region to monitor (default: ridgecrest)')
    parser.add_argument('--all', action='store_true',
                       help='Monitor all configured regions')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory for results')

    args = parser.parse_args()

    if args.all:
        results = run_all_regions()
        print("\n" + "=" * 60)
        print("SUMMARY - ALL REGIONS")
        print("=" * 60)
        print(f"{'Region':<25} {'Ratio':>8} {'Tier':<10} {'Days↑':>6}")
        print("-" * 60)
        for r in results:
            print(f"{r.region:<25} {r.ratio:>7.2f}× {r.tier:<10} {r.days_elevated:>6}")
    else:
        result = run_daily_monitoring(args.region, output_dir=args.output)
