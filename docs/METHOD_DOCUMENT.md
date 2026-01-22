# GeoSpec Three-Method Ensemble: Technical Methods Document

**Version**: 1.5.3
**Date**: January 22, 2026
**Author**: R.J. Mathews
**Status**: Operational
- THD: Operational (all 9 regions via IU/BK/GE/HINET networks)
- Fault Correlation: Operational (Cascadia, Tokyo/Kanto); Historical backtests for Ridgecrest, Tohoku, Turkey
- Lambda_geo: **Pilot operational** (3 SoCal stations via IGS-IP NTRIP + RTKLIB pipeline)
- **Hi-net: OPERATIONAL** (Tokyo/Kanto via NIED Hi-net - 128 stations @ 100Hz)
- **Historical Backtests: ALL 3 METHODS** for Ridgecrest, Tohoku, Turkey (FC/THD fetched from IRIS/FDSN January 2026)

**Changelog v1.5.3**:
- **New Feature**: Added Appendix H documenting Data Regeneration and Calibration Tracking
- New `regenerate_daily_states.py` script rebuilds CSV from authoritative JSON files
- Added calibration metadata columns (`calibration_date`, `regenerated_at`) for audit trail
- Fixed stale data issue where daily_states.csv had pre-calibration values
- **Validation Change**: Track record now scores ELEVATED/CRITICAL only (Tier ≥2)
  - WATCH (Tier 1) is awareness level, not a scorable prediction
  - Prevents false positive inflation from sensitive WATCH threshold
  - Result: 0 scored predictions (expected with M6+ calibration and tier gating)

**Changelog v1.5.2**:
- **New Feature**: Added Appendix G documenting the Prediction Validation System
- Tracks prospective prediction accuracy by correlating predictions with USGS events
- Automated validation runs daily with 7-14 day lookback window
- Initial results: 5.2% hit rate (3 hits from 58 predictions, Dec 17 - Jan 15)

**Changelog v1.5.1**:
- **Critical Fix**: Updated `monitoring/run_daily.ps1` to call `run_ensemble_daily.py` (Ensemble) instead of legacy `run_daily_live.py`
- **Dashboard Dataflow**: Confirmed auto-append to `monitoring/dashboard/data.csv` via `run_ensemble_daily.py`
- **Appendix E Rewrite**: Full dataflow diagram from data generation → GitHub Pages
- **Kaikoura Fix**: Added IU.SNZO (Wellington) as IRIS-accessible station for NZ region
- **Two CSV Architecture**: Documented dashboard CSV vs debug log CSV purposes

**Changelog v1.5.0**:
- **Historical FC Data**: Computed L2/L1 from real IRIS/FDSN waveforms for Ridgecrest (CI.WBS/SLA), Tohoku (IU.MAJO/PS.TSK), Turkey (IU.ANTO/GE.ISP/CSS)
- **Historical THD Data**: Fetched from IRIS for all 5 events (Ridgecrest, Tohoku, Turkey, Chile, Morocco)
- **3-Method Backtests**: Ridgecrest, Tohoku, Turkey now have full LG+FC+THD ensemble validation
- Added `fetch_historical_thd.py` and `fetch_historical_fc.py` scripts
- Updated backtest_timeseries.json with real seismic data
- Added BACKTEST_DATA_SOURCES.md documentation

**Changelog v1.4.1**:
- **Data Audit**: Documented distinction between live monitoring (real data) and backtest (literature-derived Lambda_geo)
- Added Data Source Clarification section to Validation Results
- Updated Current Operational Results table with calibrated Jan 11 values
- Added reference to DATA_AUDIT_2026-01-13.md
- THD baselines now display z-scores instead of raw THD values

**Changelog v1.4**:
- **RTCM Pipeline**: Implemented real-time GNSS via IGS-IP NTRIP caster + RTKLIB
- Added 3 pilot stations: COSO00USA0, GOLD00USA0, JPLM00USA0 (SoCal)
- Global NAV strategy: merge ephemeris across stations for OBS-only streams
- Position adapter (`position_adapter.py`) converts RTKLIB .pos → NGL format
- Q-conditional QC thresholds: different sigma limits per RTKLIB quality level (Q=1-6)
- QC flags: `flag_low_sats`, `flag_high_sigma`, `flag_bad_q` with human-readable `qc_reason`
- Lambda_geo status upgraded from "Retrospective-only" to "Pilot operational"
- Added wsl/ processing scripts for hybrid Windows/WSL workflow

**Changelog v1.3**:
- Added Appendix C: Post-Event Regime (freeze/reset protocol after M≥6.5)
- Added Appendix D: Pre-Registered Scoring Rules (event definition, lead window, FAR hierarchy)
- Added Appendix E: Daily State CSV Format for dashboard trending
- Fixed region-key mapping (tokyo_kanto→japan_tohoku, socal_saf_coachella→socal_coachella)
- Added DEGRADED state (tier -1) for insufficient data coverage
- Clarified false alarm rate hierarchy (TER vs GAR vs FAR)

**Changelog v1.2**:
- Clarified tier definitions as signal anomaly levels, not earthquake probabilities
- Added weight renormalization warning for missing methods
- Defined "16/16 tests passed" criteria explicitly
- Specified tier assignment rules (require ≥2 methods for Tier ≥2)
- Added dashboard column specifications
- Updated fault correlation status to "disabled" for regions without adequate coverage
- Updated Lambda_geo saturation from 20× to 1000×

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Three-Method Ensemble Overview](#three-method-ensemble-overview)
3. [Method 1: Lambda_geo (GPS Strain)](#method-1-lambda_geo-gps-strain)
4. [Method 2: Fault Correlation Dynamics](#method-2-fault-correlation-dynamics)
5. [Method 3: Seismic Total Harmonic Distortion](#method-3-seismic-total-harmonic-distortion)
6. [Risk Combination and Tier Assignment](#risk-combination-and-tier-assignment)
7. [Station Configuration](#station-configuration)
8. [Data Access Problems](#data-access-problems)
9. [Performance Improvement Recommendations](#performance-improvement-recommendations)
10. [Validation Results](#validation-results)
11. [Appendices](#appendix-a-code-references)
    - [Appendix A: Code References](#appendix-a-code-references)
    - [Appendix B: Data Center Endpoints](#appendix-b-data-center-endpoints)
    - [Appendix C: Post-Event Regime](#appendix-c-post-event-regime)
    - [Appendix D: Pre-Registered Scoring Rules](#appendix-d-pre-registered-scoring-rules)
    - [Appendix E: Dashboard Data Flow and CSV Formats](#appendix-e-dashboard-data-flow-and-csv-formats)
    - [Appendix F: RTCM Pipeline Data Structure](#appendix-f-rtcm-pipeline-data-structure-v14)
    - [Appendix G: Prediction Validation System (Track Record)](#appendix-g-prediction-validation-system-track-record)
    - [Appendix H: Data Regeneration and Calibration Tracking](#appendix-h-data-regeneration-and-calibration-tracking)

---

## Executive Summary

The GeoSpec system uses a three-method ensemble approach to detect earthquake precursors:

| Method | Data Source | Physical Basis | Typical Lead Time |
|--------|-------------|----------------|-------------------|
| Lambda_geo | GPS positions | Strain eigenframe rotation | 5-14 days |
| Fault Correlation | Seismic waveforms | Segment decoupling | 12-48 hours |
| Seismic THD | Seismic waveforms | Rock nonlinearity | 7-14 days |

**Current Status (January 2026):**
- THD analysis operational for all 8 regions using IU global network
- Fault correlation operational for **Cascadia only**; disabled elsewhere (insufficient coverage)
- Lambda_geo **pilot operational** (3 SoCal stations via IGS-IP NTRIP + RTKLIB)
- Validation: 16/16 tests passed on 4 major earthquakes (M7.1-M9.0)
  - Each event tested for: ELEVATED reached, CRITICAL reached, ≥72h lead time, sustained until mainshock
  - **Caveat**: All retrospective; prospective validation pending

**Monitored Regions (8 total):**

| Region Key | Display Name | Location | Methods |
|------------|--------------|----------|---------|
| ridgecrest | Ridgecrest/Mojave | SoCal inland | THD + FC (partial) |
| socal_saf_mojave | SoCal SAF Mojave | SoCal SAF north | THD + FC (partial) |
| socal_coachella | SoCal SAF Coachella | SoCal SAF south | THD + FC (partial) |
| norcal_hayward | NorCal Hayward | SF Bay Area | THD only |
| cascadia | Cascadia | Pacific NW | THD + FC |
| tokyo_kanto | Tokyo Kanto | Japan Trench | THD + FC (Hi-net) |
| istanbul_marmara | Istanbul Marmara | NW Turkey | THD only |
| turkey_kahramanmaras | Turkey Kahramanmaras | SE Turkey | THD only |

*Tokyo/Kanto now uses NIED Hi-net (128 stations @ 100Hz) instead of distant IU.MAJO proxy

---

## Three-Method Ensemble Overview

### Architecture

```
                    GPS DATA                         SEISMIC DATA
                       │                                  │
          ┌────────────┴────────────┐                     │
          ▼                         ▼                     ▼
  ┌───────────────┐        ┌───────────────┐    ┌─────────────────────┐
  │  NGL IGS20    │        │  IGS-IP NTRIP │    │  Waveform Fetching  │
  │ (2-14d delay) │        │   + RTKLIB    │    │  (IRIS/NCEDC/SCEDC) │
  │ (retrospective)│       │  (real-time)  │    └──────────┬──────────┘
  └───────┬───────┘        └───────┬───────┘               │
          │                        │              ┌────────┴────────┐
          │    position_adapter.py │              │                 │
          │           ▼            │     ┌────────▼────────┐ ┌──────▼──────┐
          └──────>┌─────────┐<─────┘     │ Fault Correlation│ │ Seismic THD │
                  │Lambda_geo│            │  (Eigenvalues)   │ │ (Harmonics) │
                  └────┬─────┘           └────────┬─────────┘ └──────┬──────┘
                       │                          │                  │
                       └──────────────────────────┼──────────────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │    ENSEMBLE     │
                                        │  Risk Scoring   │
                                        │  (Weighted Sum) │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │   ALERT TIER    │
                                        │    (0-3)        │
                                        └─────────────────┘
```

### Method Weights

| Method | Weight | Rationale |
|--------|--------|-----------|
| Lambda_geo | 0.40 | Primary indicator; validated on all 4 historical events |
| Fault Correlation | 0.30 | Short-term warning; requires dense station coverage |
| Seismic THD | 0.30 | Early warning; sensitive to stress changes |

---

## Method 1: Lambda_geo (GPS Strain)

### Physical Basis

Lambda_geo detects rotation of the strain rate eigenframe, indicating redistribution of tectonic stress. Before major earthquakes, the principal strain directions rotate as stress concentrates on the eventual rupture zone.

### Mathematical Formulation

```
Λ_geo = ||[Ė, Ë]||_F
```

Where:
- `Ė` = strain rate tensor (from GPS velocity field)
- `Ë` = strain acceleration tensor
- `[·,·]` = Lie bracket (commutator)
- `||·||_F` = Frobenius norm

### Data Source

**Primary (Retrospective):**

| Parameter | Value |
|-----------|-------|
| Source | Nevada Geodetic Laboratory (NGL) |
| Product | IGS20 Rapid Solutions |
| Update Frequency | Daily |
| Latency | 2-14 days |
| Format | TENV3 position files |
| URL | http://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/ |

**Pilot Real-Time (v1.4):**

| Parameter | Value |
|-----------|-------|
| Source | IGS-IP NTRIP Caster |
| Product | RTCM3 MSM streams |
| Update Frequency | 1 Hz (real-time) |
| Latency | Seconds to minutes |
| Processing | RTKLIB (convbin + rnx2rtkp) |
| Output Format | NGL-compatible (via position_adapter.py) |
| Pilot Stations | COSO00USA0, GOLD00USA0, JPLM00USA0 |

**RTCM Pipeline Architecture:**
```
NTRIP Caster (igs-ip.net)     Windows: ntripclient.exe
        │                              │
        ▼                              ▼
   RTCM3 streams ──────────────> .rtcm3 files
        │                              │
        │      WSL: RTKLIB             │
        │    ┌─────────────────────────┘
        ▼    ▼
    convbin: RTCM → RINEX (.obs + .nav)
        │
        ▼
    Global NAV merge (all stations)
        │
        ▼
    rnx2rtkp: RINEX → positions (.pos)
        │
        ▼
    position_adapter.py: .pos → NGL format (.json)
        │
        ▼
    Lambda_geo: NGL positions → strain analysis
```

**Global NAV Strategy**: Some RTCM streams (GOLD, JPLM) are observation-only without ephemeris. The pipeline merges NAV files from all stations into a global ephemeris file, enabling positioning for OBS-only streams using another station's broadcast.

### Risk Conversion

**Design Note**: Historical detections show ratios from 485× to 7,999×. The risk function uses log scaling with saturation at 1000× for tier assignment, but the **raw ratio is always preserved** for analyst review.

```python
def lambda_geo_to_risk(ratio: float) -> float:
    """Convert baseline ratio to 0-1 risk score for tier assignment."""
    if ratio <= 1:
        return 0.0
    # Logarithmic mapping with 1000x saturation
    # This keeps 50x and 5000x distinguishable (0.57 vs 1.0)
    max_log_ratio = np.log10(1000)  # Saturation at 1000x
    risk = np.log10(ratio) / max_log_ratio
    return min(1.0, risk)
```

| Ratio | Risk Score | Tier | Notes |
|-------|------------|------|-------|
| < 3x | 0.00-0.16 | NORMAL | Background noise level |
| 3-10x | 0.16-0.33 | WATCH | Possible anomaly |
| 10-100x | 0.33-0.67 | ELEVATED | Significant anomaly |
| 100-1000x | 0.67-1.00 | CRITICAL | Major precursor signal |
| > 1000x | 1.00 (capped) | CRITICAL | Raw ratio shown separately |

**Important**: Dashboard always displays raw ratio alongside risk score so analysts can distinguish "100×" from "5000×" even when both map to CRITICAL tier.

### Current Status

**PILOT OPERATIONAL (v1.4)** - Real-time GNSS pipeline via IGS-IP NTRIP + RTKLIB. Three SoCal pilot stations (COSO, GOLD, JPLM) producing positions in NGL format. Historical validation used reconstructed ratios from published observations.

**Pilot Station Performance (January 2026):**

| Station | Stream Type | Epochs/Day | Notes |
|---------|-------------|------------|-------|
| COSO00USA0 | OBS + NAV | ~35 | Full RTCM (provides ephemeris to others) |
| GOLD00USA0 | OBS-only | ~305 | Uses global NAV |
| JPLM00USA0 | OBS-only | ~305 | Uses global NAV |

---

## Method 2: Fault Correlation Dynamics

### Physical Basis

Under normal conditions, fault segments are mechanically coupled and their seismic activity is correlated. Before rupture, stress concentrates on one segment, causing decorrelation (decoupling) from neighbors.

### Mathematical Formulation

**Correlation Matrix:**
```
R_ij = Cov(A_i, A_j) / sqrt(Var(A_i) × Var(A_j))
```

Where `A_i` is the seismic activity (envelope amplitude) of segment `i`.

**Eigenvalue Analysis:**
```
R = V Λ V^T
```

**Key Metrics:**
- `L2/L1 ratio` = λ₂/λ₁ (second/first eigenvalue)
- `Participation ratio` = (Σλᵢ)² / Σλᵢ²

### Risk Indicators

| Metric | Normal | Elevated | Critical |
|--------|--------|----------|----------|
| L2/L1 ratio | > 0.3 | 0.1-0.3 | < 0.1 |
| Participation ratio | > 2.0 | 1.5-2.0 | < 1.5 |

### Risk Conversion

```python
def fault_correlation_to_risk(l2_l1: float, participation: float) -> float:
    """Convert correlation metrics to risk score."""
    # L2/L1: low = high risk
    l2_risk = max(0, min(1, (0.3 - l2_l1) / 0.25))

    # Participation: low = high risk
    pr_risk = max(0, min(1, (2.0 - participation) / 1.0))

    return 0.6 * l2_risk + 0.4 * pr_risk
```

### Data Source

| Parameter | Value |
|-----------|-------|
| Source | IRIS, NCEDC, SCEDC, GEOFON (FDSN Web Services) |
| Data Type | Continuous waveforms (BHZ channel) |
| Window | 24 hours |
| Step | 6 hours |
| Processing | Bandpass 0.01-1.0 Hz, envelope via Hilbert transform, SVD decomposition |

**Historical Backtest FC Stations (fetched January 2026)**:
| Event | Stations | Network | Data Center |
|-------|----------|---------|-------------|
| Ridgecrest 2019 | WBS, SLA | CI | SCEDC/IRIS |
| Tohoku 2011 | MAJO, TSK | IU, PS | IRIS |
| Turkey 2023 | ANTO, ISP, CSS | IU, GE | IRIS/GEOFON |

### Current Status

**OPERATIONAL** for Cascadia and Tokyo/Kanto (2+ segments with adequate coverage). **Historical backtests** computed for Ridgecrest, Tohoku, Turkey using IRIS/FDSN data. Disabled for live monitoring in other regions due to insufficient station coverage (requires minimum 2 stations/segment, 2 segments/region).

**Historical FC Results**:
- Ridgecrest: Min L2/L1 = 0.002 on Jun 27, 2019 (9 days before M7.1)
- Tohoku: Min L2/L1 = 0.021 on Mar 8, 2011 (3 days before M9.0)
- Turkey: Min L2/L1 = 0.049 on Jan 25, 2023 (12 days before M7.8)

---

## Method 3: Seismic Total Harmonic Distortion

### Physical Basis

Rocks respond linearly to tidal forcing under normal stress conditions. As stress approaches failure threshold, rocks enter a nonlinear regime where tidal forcing creates harmonics (2f₀, 3f₀, etc.). THD measures this nonlinearity.

### Mathematical Formulation

**Total Harmonic Distortion:**
```
THD = sqrt(Σₙ₌₂^N P(n×f₀) / P(f₀))
```

Where:
- `f₀` = fundamental tidal frequency (M2 = 1/(12.42 hours))
- `P(f)` = power spectral density at frequency f
- `N` = number of harmonics (default: 5)

### Tidal Frequencies Used

| Constituent | Period | Frequency (Hz) |
|-------------|--------|----------------|
| M2 (primary) | 12.42 hours | 2.24 × 10⁻⁵ |
| S2 | 12.00 hours | 2.31 × 10⁻⁵ |
| K1 | 23.93 hours | 1.16 × 10⁻⁵ |
| O1 | 25.82 hours | 1.08 × 10⁻⁵ |

### Risk Conversion

```python
def thd_to_risk(thd: float) -> float:
    """Convert THD to risk score."""
    # THD > 0.1 indicates significant nonlinearity
    # THD > 0.2 is critical
    return min(1.0, thd / 0.2)
```

| THD Value | Risk Score | Interpretation |
|-----------|------------|----------------|
| < 0.05 | 0.00-0.25 | Normal linear response |
| 0.05-0.10 | 0.25-0.50 | Slight nonlinearity |
| 0.10-0.15 | 0.50-0.75 | Significant nonlinearity |
| > 0.15 | 0.75-1.00 | Critical nonlinearity |

### Data Source

| Parameter | Value |
|-----------|-------|
| Source | IRIS (IU network), NCEDC (BK network), GEOFON (GE network) |
| Data Type | Continuous waveforms (BHZ channel) |
| Sample Rate | 40 Hz (decimated to 1 Hz for analysis) |
| Window | 24 hours |
| Processing | Detrend, bandpass 0.01-1.0 Hz, FFT, harmonic power ratio |

**Historical Backtest THD Stations (fetched January 2026)**:
| Event | Station | Network | Peak z-score | Peak Date |
|-------|---------|---------|--------------|-----------|
| Ridgecrest 2019 | WBS | CI | 18.5 | Jul 5, 2019 |
| Tohoku 2011 | MAJO | IU | 11.5 | Mar 9, 2011 |
| Turkey 2023 | ISP | GE | 4.0 | Jan 26, 2023 |
| Chile 2010 | LVC | IU | 22.7 | Feb 27, 2010 |
| Morocco 2023 | PAB | IU | 2.7 | Aug 26, 2023 |

### Current Status

**OPERATIONAL** - Working for all 9 regions using IU/GE global network stations. Historical backtests completed for all 5 major earthquakes.

**Sample Rate Sensitivity**: THD values are sensitive to sample rate. 20Hz stations (GE network) show ~15× higher THD than 40Hz stations (IU network) for the same signal. **Mitigation**: Use 40Hz IU stations as primary THD sources. If using 20Hz stations, apply separate calibration or establish station-specific baselines.

---

## Risk Combination and Tier Assignment

### Ensemble Risk Calculation

```python
def compute_combined_risk(components: Dict[str, MethodResult]) -> float:
    """Compute weighted ensemble risk score."""
    weights = {'lambda_geo': 0.4, 'fault_correlation': 0.3, 'seismic_thd': 0.3}

    total_weight = 0.0
    weighted_risk = 0.0

    for name, result in components.items():
        if result.available:
            weight = weights[name]
            weighted_risk += weight * result.risk_score
            total_weight += weight

    if total_weight > 0:
        return weighted_risk / total_weight
    return 0.0
```

### Tier Assignment (with Method-Count Gate)

**IMPORTANT**: Tier is NOT a direct lookup from combined_risk. The method-count gate prevents single-method false alarms:

```python
def assign_tier(combined_risk: float, methods_available: int) -> Tuple[int, str]:
    """
    Assign tier with method-count gate.
    Returns (tier, notes) where notes explain any downgrade.
    """
    # Step 1: Raw tier from risk score
    if combined_risk >= 0.75:
        raw_tier = 3  # CRITICAL
    elif combined_risk >= 0.50:
        raw_tier = 2  # ELEVATED
    elif combined_risk >= 0.25:
        raw_tier = 1  # WATCH
    else:
        raw_tier = 0  # NORMAL

    # Step 2: Apply method-count gate for Tier ≥2
    if raw_tier >= 2 and methods_available < 2:
        # Downgrade: single method cannot trigger ELEVATED/CRITICAL
        final_tier = 1  # Cap at WATCH
        notes = f"tier_capped (was {['NORMAL','WATCH','ELEVATED','CRITICAL'][raw_tier]})"
        return (final_tier, notes)

    return (raw_tier, "")
```

**Rationale**: A single method showing risk=0.90 might be a sensor artifact, calibration issue, or genuine signal. Without corroboration from a second method, we cap at WATCH (Tier 1) to avoid false alarms while still flagging the anomaly for attention.

### Weight Renormalization Warning

**IMPORTANT**: When methods are unavailable, weights are renormalized to sum to 1.0. This can quietly inflate apparent confidence:

| Methods Available | Renormalized Weights | Issue |
|-------------------|---------------------|-------|
| All 3 | LG=0.40, FC=0.30, THD=0.30 | None (full ensemble) |
| THD only | THD=1.00 | Single method drives tier; low confidence |
| THD + FC | FC=0.50, THD=0.50 | Missing primary method (Lambda_geo) |
| LG only | LG=1.00 | Full weight to GPS; seismic blind |

**Dashboard displays**:
- `Methods: 1/3` or `2/3` to show how many methods contributed
- `Agreement: single_method` or `mixed` to flag reduced reliability
- **Raw scores per method** so analysts can see which methods are driving results

**Recommendation**: Tier assignments of ELEVATED or CRITICAL from single-method results should be treated as "Watch with elevated signal" rather than actionable alerts until confirmed by additional methods.

### Risk Tiers (Signal Anomaly Classification)

**Definition**: Tiers classify the **signal anomaly level** (how unusual is the observation?), not the probability of an earthquake. An ELEVATED tier means "unusual signals detected" with the specified confidence, not "50-75% chance of earthquake."

| Tier | Name | Risk Range | Signal Interpretation | Action |
|------|------|------------|----------------------|--------|
| 0 | NORMAL | 0.00-0.25 | Background noise levels | Routine monitoring |
| 1 | WATCH | 0.25-0.50 | Possible anomaly detected | Enhanced monitoring |
| 2 | ELEVATED | 0.50-0.75 | Significant anomaly confirmed | Advisory issued; require ≥2 methods |
| 3 | CRITICAL | 0.75-1.00 | Major precursor pattern | Critical alert; require ≥2 methods |

**Tier Assignment Rules**:
- Tier 0-1: Can be assigned with any number of methods
- Tier 2-3: **Require ≥2 methods agreeing** to avoid single-method false alarms
- Single-method ELEVATED → Downgrade to WATCH with note "elevated signal, awaiting confirmation"
- Single-method CRITICAL → Downgrade to ELEVATED with note "critical signal, single method only"

### Confidence Levels

Confidence reflects **operational certainty** based on method agreement and data quality:

| Agreement Type | Confidence | Description |
|----------------|------------|-------------|
| all_critical | 0.95 | All methods indicate CRITICAL |
| all_elevated | 0.85 | All methods indicate ELEVATED |
| all_normal | 0.80 | All methods indicate NORMAL |
| mostly_elevated | 0.75 | 2/3 methods elevated |
| mixed | 0.60 | Methods disagree |
| single_method | 0.50 | Only one method available |
| no_data | 0.00 | No data available |

**Agreement Computation for Partial Methods**:
- Agreement is computed over **available methods only**
- "all_elevated" with 2/3 methods means both available methods are elevated (not that all 3 are)
- When only 1 method is available, agreement is forced to `single_method` regardless of signal level

**Confidence Cap Rule**: When `methods_available < 3`, maximum confidence is capped:
```python
if methods_available == 1:
    max_confidence = 0.50  # single_method cap
elif methods_available == 2:
    max_confidence = 0.85  # all_elevated cap (cannot reach all_critical)
else:
    max_confidence = 0.95  # full ensemble can reach all_critical
```
This ensures 3-method agreement is required for highest confidence levels.

**Confidence ≠ Earthquake Probability**: A 95% confidence means "95% certain the signal pattern matches historical precursors," not "95% chance of earthquake."

### Dashboard Display Specification

The web dashboard (https://kantrarian.github.io/geospec/) should display these columns:

| Column | Description | Format |
|--------|-------------|--------|
| Region | Geographic monitoring region | Text |
| Tier | Signal anomaly tier (0-3) | NORMAL/WATCH/ELEVATED/CRITICAL |
| Risk Score | Combined ensemble score | 0.000-1.000 |
| Methods | Methods contributing | "1/3 (THD)" or "2/3 (FC+THD)" |
| Agreement | Method consensus | single_method/mixed/all_elevated/etc |
| Confidence | Operational certainty | 0-100% |
| LG Ratio | Lambda_geo baseline ratio | "—" or "485×" |
| THD | THD value | "0.118" or "—" |
| FC L2/L1 | Fault correlation eigenvalue ratio | "0.044" or "disabled" |
| Notes | Operational flags | "elevated signal, single method" |

**Color coding**:
- NORMAL: Green (#22c55e)
- WATCH: Yellow (#eab308)
- ELEVATED: Orange (#f97316)
- CRITICAL: Red (#ef4444)
- Disabled/unavailable: Gray (#6b7280)

**Single-method warnings**: When Tier ≥2 with only 1 method, display amber warning: "⚠️ Single-method result—awaiting confirmation"

---

## Station Configuration

### Current THD Stations (January 2026)

| Region | Station | Network | Data Center | Location | Status |
|--------|---------|---------|-------------|----------|--------|
| Ridgecrest/Mojave | TUC | IU | IRIS | Tucson, AZ | 100% |
| SoCal SAF Mojave | TUC | IU | IRIS | Tucson, AZ | 100% |
| SoCal SAF Coachella | TUC | IU | IRIS | Tucson, AZ | 100% |
| NorCal Hayward | BKS | BK | NCEDC | Berkeley, CA | 100% |
| Cascadia | COR | IU | IRIS | Corvallis, OR | 100% |
| Tokyo Kanto | N.KI2H | HINET | NIED | Kita-Ibaraki, Japan | 100% |
| Istanbul Marmara | ANTO | IU | IRIS | Ankara, Turkey | 100% |
| Turkey Kahramanmaras | ANTO | IU | IRIS | Ankara, Turkey | 100% |

### Why These Stations Were Chosen

#### Original Plan (Regional Networks)
Initially planned to use regional seismic networks for better spatial resolution:
- California: CI (Southern California Seismic Network)
- NorCal: NC (Northern California Seismic Network)
- Cascadia: UW (Pacific Northwest), CN (Canadian)
- Turkey: KO (Kandilli Observatory)
- Japan: NIED Hi-net

#### Actual Implementation (Global Network)
Regional networks showed poor data availability (20-25% for SCEDC, 0% for many others). Switched to IU (Global Seismographic Network) which provides:
- 100% real-time availability
- Consistent data quality
- Single data center (IRIS)
- No authentication required

#### Station Selection Rationale

| Station | Selection Reason |
|---------|------------------|
| IU.TUC (Tucson) | Nearest IU station to SoCal; stable bedrock site |
| BK.BKS (Berkeley) | Only working Bay Area station; historic station |
| IU.COR (Corvallis) | Nearest IU to Cascadia subduction zone |
| HINET.N.KI2H (Kita-Ibaraki) | Hi-net 100Hz station in Kanto region; IU.MAJO fallback |
| IU.ANTO (Ankara) | Turkey's only IU station; covers both regions |

### Fault Correlation Segments

Fault correlation requires minimum 2 stations per fault segment and 2 segments per region. Current configuration:

| Region | Segments Defined | Segments Working | FC Status | Issue |
|--------|------------------|------------------|-----------|-------|
| Ridgecrest | 3 | 1 | **DISABLED** | CI station data gaps |
| SoCal SAF Mojave | 4 | 1 | **DISABLED** | CI station data gaps |
| SoCal SAF Coachella | 3 | 0 | **DISABLED** | Region name mismatch |
| NorCal Hayward | 4 | 0 | **DISABLED** | NC stations offline |
| Cascadia | 4 | 2 | OPERATIONAL | CN/UW partial availability |
| Istanbul | 4 | 0 | **DISABLED** | KO network restricted |
| Turkey Kahramanmaras | 3 | 0 | **DISABLED** | KO/GE network restricted |
| Tokyo | 4 | 2 | OPERATIONAL | Hi-net integration complete |

**Status Legend**:
- OPERATIONAL: ≥2 segments with ≥2 stations each, producing correlation metrics
- DISABLED: Insufficient coverage; method omitted from ensemble (not contributing to risk score)

---

## Data Access Problems

### Problem 1: SCEDC Data Gaps

**Impact**: California regional stations (CI.*) have only 20-25% availability

**Observed Behavior**:
```
CI.CCC (Ridgecrest): 25% availability, 60-hour latency
CI.WBS (Mojave): 21% availability, 48-hour latency
CI.PAS (Pasadena): 0% availability
CI.GSC (Goldstone): 4% availability
```

**Root Cause**: Unknown. Possible causes:
- SCEDC server issues
- Station maintenance
- Network restrictions
- Query rate limiting

**Current Workaround**: Using IU.TUC instead of CI stations

**Impact on Results**: THD analysis uses distant station (Tucson) instead of local stations, reducing spatial resolution. Single station cannot detect local variations.

### Problem 2: NC Network Offline

**Impact**: Northern California regional stations not returning data

**Observed Behavior**:
```
NC.WENL (Hayward): 0% availability
NC.JRSC: 0% availability
NC.PACP: 0% availability
NC.MCCM: 0% availability
NC.HOPS: 0% availability
```

**Root Cause**: Unknown. BK network (same data center, NCEDC) works fine.

**Current Workaround**: Using BK.BKS (Berkeley) for NorCal region

**Impact on Results**: Single station for entire Bay Area. Cannot differentiate Hayward vs Calaveras vs Rodgers Creek faults.

### Problem 3: International Network Restrictions

**Impact**: Regional networks in Turkey not accessible via FDSN. Japan resolved via Hi-net.

**Affected Networks**:
| Network | Region | Issue |
|---------|--------|-------|
| KO | Turkey | Kandilli Observatory - not on IRIS |
| TU | Turkey | National network - restricted |
| NIED Hi-net | Japan | **RESOLVED** - account approved, integration complete |
| GE | Germany/Turkey | GEOFON - partial availability |

**Current Workaround**:
- Turkey: Using IU.ANTO (single station for both regions)
- Japan: **Hi-net operational** with 128 Kanto stations @ 100Hz

**Impact on Results**:
- Turkey: Single station for entire country
- Japan: **Full Kanto coverage** via Hi-net (N.KI2H primary, 127 backup stations)

### Problem 4: GPS Data Latency

**Impact**: Lambda_geo method historically limited by 2-14 day latency

**Previous Situation (v1.3)**:
- NGL IGS20 data has 2-14 day latency
- No real-time GPS integration implemented
- Historical validation only

**Current Situation (v1.4)**:
- **SOLVED for pilot stations** via IGS-IP NTRIP + RTKLIB pipeline
- 3 SoCal stations (COSO, GOLD, JPLM) producing real-time positions
- Latency reduced from days to seconds/minutes
- NGL format preserved via position_adapter.py

**Remaining Challenges**:
- Pilot covers only 3 stations (need regional expansion)
- Some streams are OBS-only (require global NAV merge)
- Positioning accuracy is broadcast-only (~1-5m) vs PPP (~cm)

**Future Improvements**:
- Add more IGS-IP stations for regional coverage
- Integrate IGS Real-Time Service for PPP corrections
- Implement automated daily processing via Windows Task Scheduler

### Problem 5: THD Sample Rate Sensitivity

**Impact**: THD values vary significantly between 20Hz and 40Hz stations

**Observed Behavior (January 2026)**:
```
GE.ARPR (20Hz, GEOFON): THD = 0.78 -> CRITICAL
IU.ANTO (40Hz, IRIS):   THD = 0.05 -> NORMAL
Ratio: 15.3x
```

**Root Cause**: The THD algorithm's harmonic detection is sensitive to sample rate. Lower sample rates (20Hz) appear to inflate THD readings.

**Affected Stations**:
- All GE network stations (20Hz): GE.ARPR, GE.CSS, GE.ISP, GE.MALT
- IU.GNI (20Hz): Also shows inflated readings

**Current Workaround**: Use 40Hz IU stations (IU.ANTO, IU.TUC, IU.COR, etc.) as primary THD sources. 20Hz stations (GE.*) require separate baseline calibration.

**Future Fix**: Calibrate THD algorithm for different sample rates, or establish separate thresholds for 20Hz vs 40Hz stations.

---

### Problem 6: Fault Correlation Station Coverage

**Impact**: Fault correlation method rarely produces results

**Requirement**: Minimum 2 stations per fault segment, minimum 2 segments per region

**Current Reality**: Most regions have 0-1 working segments due to station data gaps

**Example (NorCal Hayward)**:
```
hayward_north: 1 station working (need 2)
hayward_south: 0 stations working
rodgers_creek: 0 stations working
calaveras: 1 station working (need 2)
Result: 0 segments with sufficient data
```

---

## Performance Improvement Recommendations

### High Priority

#### 1. NIED Hi-net Integration - COMPLETE
**Status**: ✅ Completed January 13, 2026
**Implementation**: Status page polling method (workaround for HinetPy timing issues)
**Results**: 128 Kanto stations @ 100Hz operational
**Impact**: Tokyo/Kanto now has direct regional coverage instead of 200km distant IU.MAJO proxy

#### 2. Investigate SCEDC Data Issues
**Actions**:
- Contact SCEDC support about data availability
- Check if authentication is required for full access
- Test alternative query methods (bulk download vs real-time)
- Consider using SCEDC waveform archive instead of real-time

**Expected Outcome**: Restore CI station access for California

#### 3. Integrate Real-Time GPS
**Options** (see `docs/API_REFERENCE.md` for connection details):
- **IGS Real-Time Service (RTS)**: Register at https://igs.org/rts/user-access/ - provides multi-GNSS corrections with ~5 second latency via NTRIP
- **EarthScope NTRIP Caster**: Contact rtgps@earthscope.org - dense U.S. coverage including California/Cascadia
- JPL GIPSY-X solutions (alternative)

**Required Software**: RTKLIB or BKG NTRIP Client (BNC)

**Impact**: Enable Lambda_geo method for operational use (reduce latency from 2-14 days to seconds)

### Medium Priority

#### 4. Add Backup Stations
For each region, identify backup stations in case primary fails:

| Region | Primary | Backup Options |
|--------|---------|----------------|
| SoCal | IU.TUC | II.PFO, IU.ANMO |
| NorCal | BK.BKS | BK.CMB, BK.HOPS |
| Cascadia | IU.COR | IU.RSSD, CN.PGC |
| Japan | IU.MAJO | IU.TATO, II.ERM |
| Turkey | IU.ANTO | GE.ISP, II.KURK |

#### 5. Integrate Additional Networks
See `docs/API_REFERENCE.md` for detailed connection instructions.

| Network | Region | Access | Priority |
|---------|--------|--------|----------|
| ORFEUS/GEOFON | Turkey | Public (no registration) | **High** - test TU network now |
| F-net | Japan | Registration required | Medium - awaiting Hi-net approval |
| IGS RTS | Global GNSS | Registration required | **Critical** - enables Lambda_geo |
| EarthScope | U.S. GNSS | Re-registration required | High - real-time California/Cascadia |

**Quick Win**: ORFEUS/GEOFON require no registration. Test TU network access for Turkey immediately.

#### 6. Fault Correlation Configuration
**Current approach**: Fault correlation is **disabled** for regions without adequate station coverage. Do NOT reduce minimum station requirement below 2 per segment—single-station segments produce unreliable correlation metrics.

**Adequate coverage requirements**:
- Minimum 2 stations per fault segment
- Minimum 2 segments per region
- Stations must have >80% data availability

**If coverage is inadequate**: Display "fault_correlation: disabled (insufficient stations)" rather than attempting calculation with degraded data.

**Future improvements**:
- Use IU/II stations for fault correlation where regional networks fail
- Add more regions only when adequate station coverage is confirmed

### Lower Priority

#### 7. Add Station Health Monitoring
Implement automated monitoring to detect:
- Station outages
- Data quality degradation
- Network changes

#### 8. Implement Data Caching Strategy
- Cache successful queries for 24 hours
- Fallback to cached data when live query fails
- Track cache hit/miss rates

#### 9. Add Alternative Data Sources
- USGS ShakeAlert stations
- Strong motion networks
- Infrasound arrays
- Strain meters (where available)

---

## Validation Results

### CRITICAL UPDATE: January 14, 2026

**The validation results below have been corrected to distinguish between:**
1. **Theoretical Validation** (literature-derived, now ARCHIVED)
2. **Operational Validation** (real GPS data)

**Key Finding**: The "80% hit rate" previously claimed was based on literature-derived (synthetic) Lambda_geo values, NOT computed from actual GPS data. Real GPS backtest shows different behavior patterns requiring threshold recalibration.

---

### Operational Validation: Real GPS Data (January 2026)

**Data Source**: Real NGL GPS stations (.tenv3 files) - 321 files across 10 historical earthquakes

| Event | Magnitude | Lambda_geo Ratio | Classification |
|-------|-----------|-----------------|----------------|
| Tohoku 2011 | M9.0 | 5.6x | CRITICAL* |
| Morocco 2023 | M6.8 | 3.3x | ELEVATED |
| Ridgecrest 2019 | M7.1 | 2.3x | WATCH |
| Chile 2010 | M8.8 | 2.1x | WATCH |
| Turkey 2023 | M7.8 | 2.0x | WATCH |

*With recalibrated thresholds (Critical > 4.0x)

**Calibrated Thresholds** (from real GPS data):
- NORMAL: ratio < 1.5x
- WATCH: 1.5x - 2.5x
- ELEVATED: 2.5x - 4.0x
- CRITICAL: > 4.0x

**Region-Specific Baselines** (computed January 14, 2026):

| Region | Baseline Lambda_geo | Std Dev | Quality |
|--------|---------------------|---------|---------|
| Cascadia | 0.867 | 0.883 | good |
| Ridgecrest | 0.173 | 0.163 | good |
| NorCal Hayward | 0.168 | 0.289 | good |
| SoCal Mojave | 0.055 | 0.050 | good |
| SoCal Coachella | 0.024 | 0.039 | good |
| Tokyo Kanto | 0.017 | 0.017 | good |
| Campi Flegrei | 0.017 | 0.016 | good |
| Istanbul | 0.010 | 0.014 | good |
| Turkey Kahramanmaras | N/A | N/A | no_data |

See `monitoring/data/baselines/lambda_geo_baselines.json` for full calibration data.

---

### ARCHIVED: Theoretical Validation (Literature-Derived)

**WARNING**: The following historical results used HARDCODED Lambda_geo values derived from literature analysis, NOT computed from actual GPS data. These files have been moved to `tests/archive/legacy_synthetic/`.

| Event | Magnitude | Classification (Old) | Data Source |
|-------|-----------|---------------------|-------------|
| Ridgecrest 2019 | M7.1 | HIT | Literature-derived |
| Tohoku 2011 | M9.0 | HIT | Literature-derived |
| Turkey 2023 | M7.8 | HIT | Literature-derived |
| Chile 2010 | M8.8 | HIT | Literature-derived |
| Morocco 2023 | M6.8 | MARGINAL | Literature-derived |

**Why Archived**: The literature-derived Lambda_geo values (e.g., Tohoku showing 7,234x baseline) did not match actual GPS computations (Tohoku real: 5.6x). This created "Artifactual Confidence" that has been corrected.

---

### Current Data Architecture

| System | Lambda_geo Source | Baseline Source | THD/FC Source |
|--------|-------------------|-----------------|---------------|
| **Live Monitoring** | Real NGL GPS data | Region-specific (90-day) | Real IRIS seismic |
| **Backtest (Real)** | Real NGL GPS data | Event-specific | Cached where available |

**Data Integrity Statement**: All operational Lambda_geo values are now computed from real NGL GPS station data using region-specific baselines calibrated from 90 days of real observations. No synthetic or literature-derived values are used in operational monitoring.

### FAR Calculation Methodology

False Alarm Rate is computed consistently between `compute_full_metrics.py` and `verify_backtest.py`:

```python
# false_alarm_rate is per region-day
# n_regions is loaded from backtest_config.yaml (currently 9 regions)
far_per_year = false_alarm_rate * 365 * n_regions
```

**Monitored Regions** (from `monitoring/config/backtest_config.yaml`):
1. ridgecrest
2. socal_saf_mojave
3. socal_saf_coachella
4. norcal_hayward
5. cascadia
6. tokyo_kanto
7. istanbul_marmara
8. turkey_kahramanmaras
9. campi_flegrei

### Important Caveats

1. **Threshold recalibration**: Thresholds were recalibrated in January 2026 based on real GPS data. Previous literature-derived thresholds have been replaced.

2. **Event-centric validation**: Current metrics are computed over event windows only, not continuous year-round monitoring.

3. **Retrospective analysis**: All validations analyzed data knowing earthquake times. Prospective (shadow) monitoring required for true operational metrics.

4. **Morocco/Turkey GPS coverage**: Some regions have sparse NGL GPS coverage, limiting Lambda_geo computation.

5. **Unit consistency**: Live computation and baselines now use identical units (no conversion factor mismatch). See `calibrate_lambda_geo_baselines.py`.

See `docs/DATA_AUDIT_2026-01-13.md` and `tests/archive/legacy_synthetic/README.md` for full audit details.

### Historical Validation (5 Events) - Updated January 15, 2026

| Event | Magnitude | LG Ratio | FC Min L2/L1 | THD Peak z-score | Methods |
|-------|-----------|----------|--------------|------------------|---------|
| Ridgecrest 2019 | M7.1 | 2.3x | 0.002 (Jun 27) | 18.5 (Jul 5) | **3/3** |
| Tohoku 2011 | M9.0 | 5.6x | 0.021 (Mar 8) | 11.5 (Mar 9) | **3/3** |
| Turkey 2023 | M7.8 | 2.0x | 0.049 (Jan 25) | 4.0 (Jan 26) | **3/3** |
| Chile 2010 | M8.8 | 2.1x | — | 22.7 (Feb 27) | 2/3 |
| Morocco 2023 | M6.8 | 3.3x | — | 2.7 (Aug 26) | 2/3 |

**Data Sources (January 2026 fetch)**:
- **FC Stations**: Ridgecrest (CI.WBS/SLA), Tohoku (IU.MAJO/PS.TSK), Turkey (IU.ANTO/GE.ISP/GE.CSS)
- **THD Stations**: Ridgecrest (CI.WBS), Tohoku (IU.MAJO), Turkey (GE.ISP), Chile (IU.LVC), Morocco (IU.PAB)
- Chile/Morocco FC unavailable (sparse pre-2010 Chilean network, limited Morocco data sharing)

**Key Findings**:
- Tohoku Mar 9 THD spike (z=11.5) corresponds to M7.3 foreshock 2 days before M9.0 mainshock
- Tohoku Mar 8 FC decorrelation (L2/L1=0.021) detected 3 days before event
- All 3 events with FC data show decorrelation (L2/L1 < 0.15) before earthquake

### Definition: "16/16 Tests Passed"

Each event is validated against 4 criteria (4 events × 4 tests = 16 total):

| Test # | Criterion | Pass Condition |
|--------|-----------|----------------|
| 1 | ELEVATED tier reached | Risk score ≥0.50 before event |
| 2 | CRITICAL tier reached | Risk score ≥0.75 before event |
| 3 | Lead time ≥72h | ELEVATED status >72 hours pre-event |
| 4 | No false return | Risk stays elevated until mainshock |

**Important caveats**:
- All validations are **retrospective** (analyzed knowing earthquake times)
- Events 2-4 used Lambda_geo only (seismic data not accessible)
- Ridgecrest used reconstructed historical data, not live monitoring
- Sample size (n=5) is statistically limited; prospective validation needed

### Current Operational Results (January 11, 2026 - Real Data Computation)

| Region | Risk | Tier | Lambda_geo | THD z-score | Methods | Notes |
|--------|------|------|------------|-------------|---------|-------|
| NorCal Hayward | 0.600 | ELEVATED | 21.9x | 3.35 | 2/3 | FC unavailable; Lambda_geo elevated |
| Cascadia | 0.437 | WATCH | 19.7x | 1.60 | 3/3 | — |
| Ridgecrest | 0.390 | WATCH | 28.7x | 2.25 | 3/3 | THD moderately elevated |
| SoCal Coachella | 0.241 | NORMAL | 0.5x | 2.25 | 2/3 | FC unavailable |
| SoCal Mojave | 0.218 | NORMAL | 0.4x | 2.25 | 3/3 | — |
| Turkey Kahramanmaras | 0.091 | NORMAL | — | -1.25 | 2/3 | No LG coverage |
| Campi Flegrei | 0.086 | NORMAL | 0.7x | — | 2/3 | THD unavailable |
| Tokyo | 0.048 | NORMAL | 0.2x | — | 2/3 | THD unavailable (Hi-net gap) |
| Istanbul | 0.047 | NORMAL | 0.1x | -1.32 | 3/3 | — |

**Note**: All values computed from real data at 2026-01-11T00:00:00 using standard midnight anchor. THD z-scores use 30-day station-specific baselines. Lambda_geo values computed from real NGL GPS data. NorCal Hayward shows elevated risk from both Lambda_geo (21.9x) and THD (z=3.35).

### Performance Metrics

| Metric | Historical | Current |
|--------|------------|---------|
| Detection Rate | 100% (4/4) | N/A (prospective) |
| Lead Time (CRITICAL) | 168-336 hours | N/A |
| Methods Available | 1-3 | 1-2 |
| Average Confidence | 50-95% | 50-80% |
| Data Latency | 0-2 days | 0 days (THD) |

---

## Appendix A: Code References

### Core Ensemble Components

| Component | File | Key Functions |
|-----------|------|---------------|
| Ensemble Integration | `monitoring/src/ensemble.py` | `compute_risk()`, `GeoSpecEnsemble` |
| THD Analysis | `monitoring/src/seismic_thd.py` | `SeismicTHDAnalyzer`, `fetch_continuous_data_for_thd()` |
| Fault Correlation | `monitoring/src/fault_correlation.py` | `FaultCorrelationMonitor`, `compute_correlation_matrix()` |
| Seismic Data | `monitoring/src/seismic_data.py` | `SeismicDataFetcher`, `get_segment_envelopes()` |
| Daily Runner | `monitoring/src/run_ensemble_daily.py` | `run_all_regions()`, `REGIONS` config |
| Region Config | `monitoring/src/fault_segments.py` | `FAULT_SEGMENTS` |
| **API Reference** | `docs/API_REFERENCE.md` | Connection instructions for IRIS, IGS, NIED, ORFEUS |

### RTCM/RTKLIB Pipeline (v1.4)

| Component | File | Description |
|-----------|------|-------------|
| RTCM Capture | `monitoring/capture_rtcm.py` | Windows NTRIP client (ntripclient.exe wrapper) |
| RTCM Processing | `wsl/process_rtcm.sh` | WSL script: RTCM→RINEX→positions via RTKLIB |
| Position Adapter | `monitoring/src/position_adapter.py` | Converts RTKLIB .pos → NGL JSON format |
| Capture Config | `monitoring/config/ntrip_stations.yaml` | Station list with mountpoints and coordinates |

### Position Adapter Details

**File**: `monitoring/src/position_adapter.py`

**Key Features**:
- Parses RTKLIB .pos files (LLH and ECEF formats)
- Converts to NGL format (refepoch, e, n, u, se, sn, su, station)
- Q-conditional QC thresholds per positioning mode
- Outputs both basic and extended (_qc) JSON files

**QC Thresholds by Quality Level**:
```python
QC_THRESHOLDS = {
    1: (6, 0.20, 0.40),    # fix: min_sats=6, max_h_sigma=0.20m, max_v_sigma=0.40m
    2: (6, 1.00, 2.00),    # float
    3: (6, 5.00, 10.00),   # SBAS
    4: (6, 5.00, 10.00),   # DGPS
    5: (6, 15.0, 30.0),    # single (broadcast-only)
    6: (6, 0.50, 1.00),    # PPP
}
```

**QC Flags**:
| Flag | Meaning |
|------|---------|
| `flag_low_sats` | Satellite count < minimum for Q level |
| `flag_high_sigma` | Horizontal or vertical sigma exceeds Q threshold |
| `flag_bad_q` | Quality value not in expected range (1-6) |
| `qc_reason` | Human-readable explanation (e.g., "LOW_SATS(4<6);HIGH_SIGMA(h=0.8>0.2)") |

### WSL Processing Script Details

**File**: `wsl/process_rtcm.sh`

**Three-Phase Processing**:
1. **Phase 1**: Convert all RTCM to RINEX via `convbin`
2. **Phase 2**: Build global merged NAV from all stations (handles OBS-only streams)
3. **Phase 3**: Compute positions via `rnx2rtkp` using global NAV

**NAV Merge Logic**:
```bash
# First file: keep full RINEX header + body
cat "$NAV_FILE" > "$GLOBAL_NAV"

# Subsequent files: append body only (skip header)
awk 'f{print} /END OF HEADER/{f=1}' "$NAV_FILE" >> "$GLOBAL_NAV"
```

**Positioning Options**:
- `-p 0`: Single-point positioning (broadcast ephemeris)
- `-p 7`: PPP mode (requires SP3/CLK products)
- `-sys GRE`: Multi-GNSS (GPS + GLONASS + Galileo)
- `-ti 30`: 30-second output interval

## Appendix B: Data Center Endpoints

| Data Center | FDSN URL | Networks |
|-------------|----------|----------|
| IRIS | https://service.iris.edu/fdsnws/ | IU, II, CN, UW, GE |
| NCEDC | https://service.ncedc.org/fdsnws/ | BK, NC |
| SCEDC | https://service.scedc.caltech.edu/fdsnws/ | CI, AZ |

---

## Appendix C: Post-Event Regime

### Problem

After a significant earthquake (M≥6.5), aftershock sequences and postseismic deformation can:
- Inflate seismic baselines
- Create correlated artifacts across methods
- Cause "stuck elevated" states that appear as false alarms

### Post-Event Protocol

When an M≥6.5 earthquake occurs within a monitored region:

| Phase | Duration | Action |
|-------|----------|--------|
| **Freeze** | 0-48 hours | Suspend tier escalation; display "POST-EVENT HOLD" |
| **Reset** | 48h-7 days | Reset baselines; widen thresholds by 50% |
| **Recovery** | 7-30 days | Gradual return to normal thresholds |
| **Normal** | >30 days | Standard monitoring resumes |

### Implementation

```python
POST_EVENT_CONFIG = {
    'magnitude_threshold': 6.5,      # Minimum magnitude to trigger
    'distance_km': 100,              # Within this distance of region center
    'freeze_hours': 48,              # Suspend alerting
    'reset_days': 7,                 # Reset baselines
    'recovery_days': 30,             # Gradual threshold return
    'threshold_widening': 1.5,       # 50% wider during recovery
}
```

### Display During Post-Event

Dashboard shows: `"⏸️ POST-EVENT HOLD (M6.8 Ridgecrest +36h)"` with grayed-out tier

---

## Appendix D: Pre-Registered Scoring Rules

**Purpose**: Define scoring criteria BEFORE prospective monitoring begins to prevent post-hoc parameter tuning.

### Event Definitions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Magnitude cutoff** | M ≥ 6.0 | Below this, precursor signals are unreliable |
| **Distance tolerance** | ≤ 100 km from region centroid | Beyond this, attribution is ambiguous |
| **One event per window** | Yes | Multiple events in 14 days = single scoring event |
| **Foreshock handling** | Score largest event in sequence | Avoids double-counting |

### Lead Window Definition

| Window | Range | Credit |
|--------|-------|--------|
| **Valid lead time** | 24 hours to 14 days before mainshock | Full credit |
| **Too short** | < 24 hours | No credit (insufficient warning) |
| **Too long** | > 14 days | No credit (spurious correlation) |

### Alert Definitions

| Alert Type | Criteria | Scored? |
|------------|----------|---------|
| **CONFIRMED ELEVATED** | Tier ≥2 for ≥2 consecutive days AND ≥2 methods | Yes |
| **PRELIMINARY ELEVATED** | Tier ≥2 but single method OR single day | No (not scored) |
| **WATCH** | Tier 1 | Not scored as alert |
| **DEGRADED** | Insufficient data | Excluded from scoring |

### False Alarm Rate Hierarchy

| Metric | Definition | Target |
|--------|------------|--------|
| **TER** (Threshold Exceedance Rate) | Days with raw signal > threshold / total days | Informational only |
| **GAR** (Gated Alert Rate) | CONFIRMED ELEVATED days / total region-days | < 10⁻⁵ per region-day |
| **FAR** (Event-Linked False Alarm Rate) | CONFIRMED alerts NOT followed by M≥6.0 within 14 days / total CONFIRMED alerts | Report after 12 months |

### Detection Metrics

| Metric | Definition |
|--------|------------|
| **True Positive** | CONFIRMED ELEVATED alert within 24h-14d before M≥6.0 event |
| **False Negative** | M≥6.0 event NOT preceded by CONFIRMED alert in lead window |
| **False Positive** | CONFIRMED alert NOT followed by M≥6.0 event within 14 days |

### Pre-Registration Statement

> **Shadow Monitoring Study**: GeoSpec will operate in shadow mode from January 2026. After accumulating 365 region-days across all monitored regions, we will publish:
> 1. Gated Alert Rate (GAR) with 95% confidence interval
> 2. Event-Linked FAR (if any CONFIRMED alerts issued)
> 3. True Positive / False Negative counts (if any M≥6.0 events occur in monitored regions)
>
> Parameter changes after this date require documentation in CHANGELOG.

### Region Selection Criteria

Regions are selected based on:

| Factor | Weight | Notes |
|--------|--------|-------|
| Population exposure | 40% | Urban areas prioritized |
| Seismic hazard (USGS/GSHAP) | 30% | Known active fault systems |
| Sensor coverage | 30% | ≥1 method must be operational |

**Current regions were NOT selected to maximize detection rate**—they represent high-risk areas where monitoring is societally useful.

---

## Appendix E: Dashboard Data Flow and CSV Formats

### Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA GENERATION                                │
│                                                                          │
│  GPS Data (NGL/GeoNet)  ──┐                                             │
│  Seismic Data (IRIS)    ──┼──►  run_ensemble_daily.py                   │
│  USGS Earthquake API    ──┘           │                                 │
│                                       ▼                                 │
│                    monitoring/data/ensemble_results/                    │
│                    ├── ensemble_YYYY-MM-DD.json  (daily snapshot)       │
│                    └── daily_states.csv (detailed debug log)            │
│                                       │                                 │
│                                       ▼                                 │
│                    monitoring/dashboard/data.csv  (30-day chart source) │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           PUBLISHING                                     │
│                                                                          │
│    run_and_publish.ps1  (Scheduled 6 AM daily)                          │
│    ├── Runs: python -m src.run_ensemble_daily                           │
│    ├── Copies: monitoring/dashboard/data.csv → docs/data.csv            │
│    ├── Copies: ensemble_latest.json → docs/ensemble_latest.json         │
│    └── git push to GitHub                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GITHUB PAGES DASHBOARD                           │
│                         https://kantrarian.github.io/geospec/            │
│                                                                          │
│    docs/index.html reads:                                                │
│    ├── ensemble_latest.json → Status cards (current risk per region)    │
│    ├── data.csv → 30-day history chart (Plotly line graph)              │
│    └── events.json → Earthquake events per region                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### File 1: `monitoring/dashboard/data.csv` (Dashboard Source)

**Purpose**: Authoritative source for the 30-day history chart on GitHub Pages.

**Auto-updated**: Yes, by `run_ensemble_daily.py` after each run.

```csv
date,region,tier,risk,confidence,methods,agreement
2026-01-16,ridgecrest,0,0.1796,0.80,3,all_normal
2026-01-16,cascadia,0,0.0543,0.80,3,all_normal
2026-01-17,kaikoura,1,0.3640,0.75,2,mostly_elevated
```

| Column | Type | Description |
|--------|------|-------------|
| date | ISO date | Assessment date (YYYY-MM-DD) |
| region | string | Region key |
| tier | int | 0=NORMAL, 1=WATCH, 2=ELEVATED, 3=CRITICAL, -1=DEGRADED |
| risk | float | Combined risk score (0.0000-1.0000) |
| confidence | float | Agreement confidence (0.00-1.00) |
| methods | int | Methods available (1-3) |
| agreement | string | all_normal, mixed, mostly_elevated, etc. |

### File 2: `monitoring/data/ensemble_results/daily_states.csv` (Debug Log)

**Purpose**: Detailed log with component-level values for debugging and analysis.

**Auto-updated**: Yes, by `run_ensemble_daily.py` after each run.

```csv
date,region,tier,risk,methods,confidence,lg_ratio,thd,fc_l2l1,status,notes
2026-01-08,ridgecrest,1,0.747,1,0.50,,0.349,,PRELIMINARY,tier_capped
2026-01-08,cascadia,1,0.498,2,0.75,,0.340,0.935,PRELIMINARY,
2026-01-08,japan_tohoku,2,0.562,2,0.75,,0.470,0.867,PRELIMINARY,
```

| Column | Type | Description |
|--------|------|-------------|
| date | ISO date | Assessment date (YYYY-MM-DD) |
| region | string | Region key |
| tier | int | 0=NORMAL, 1=WATCH, 2=ELEVATED, 3=CRITICAL, -1=DEGRADED |
| risk | float | Combined risk score (0.000-1.000) |
| methods | int | Methods available (1-3) |
| confidence | float | Agreement confidence (0.00-1.00) |
| lg_ratio | float | Lambda_geo baseline ratio (empty if unavailable) |
| thd | float | THD value (empty if unavailable) |
| fc_l2l1 | float | Fault correlation L2/L1 ratio (empty if unavailable) |
| status | string | PRELIMINARY or CONFIRMED |
| notes | string | Operational flags (tier_capped, post_event, etc.) |

**Note**: Region keys should use canonical names from the region table (e.g., `japan_tohoku` not `tokyo_kanto`).

### Append Logic

Both CSVs use append-only logic:
- Daily runner appends one row per region per date
- Duplicate dates are skipped (checked before append)
- Never overwrites historical data

### Retention

Keep indefinitely for trend analysis and FAR calculation.

---

## Appendix F: RTCM Pipeline Data Structure (v1.4)

### Directory Layout

```
geospec_sprint/
├── monitoring/
│   ├── data/
│   │   ├── rtcm/                          # Raw RTCM captures
│   │   │   └── {STATION}/
│   │   │       └── {YYYY-MM-DD}/
│   │   │           └── {STATION}_{HHMMSS}.rtcm3
│   │   ├── positions/                     # RTKLIB outputs
│   │   │   ├── global_merged_{YYYY-MM-DD}.nav   # Merged ephemeris
│   │   │   ├── processing_summary_{YYYY-MM-DD}.json
│   │   │   └── {STATION}/
│   │   │       └── {YYYY-MM-DD}/
│   │   │           ├── {STATION}_{HHMMSS}.obs   # RINEX observations
│   │   │           ├── {STATION}_{HHMMSS}.nav   # RINEX navigation
│   │   │           └── {STATION}_{HHMMSS}.pos   # Position solutions
│   │   └── ngl_format/                    # NGL-compatible outputs
│   │       ├── rtcm_positions_{YYYY-MM-DD}.json      # Basic schema
│   │       └── rtcm_positions_{YYYY-MM-DD}_qc.json   # With QC flags
│   ├── src/
│   │   └── position_adapter.py            # .pos → NGL converter
│   └── config/
│       └── ntrip_stations.yaml            # Station configuration
└── wsl/
    └── process_rtcm.sh                    # WSL processing script
```

### Output File Formats

**Basic NGL Format** (`rtcm_positions_YYYY-MM-DD.json`):
```json
{
  "date": "2026-01-11",
  "created_at": "2026-01-11T19:18:12Z",
  "source": "RTKLIB via WSL",
  "reference_strategy": "median_first_100_epochs",
  "include_qc": false,
  "stations": {
    "COSO00USA0": {
      "ref_lla": [35.982354, -117.808916, 1464.13],
      "epochs": 35
    }
  },
  "total_epochs": 645,
  "time_range": {"start": 2026.0291, "end": 2026.0295}
}
```

**Extended QC Format** (`rtcm_positions_YYYY-MM-DD_qc.json`):
```json
{
  "include_qc": true,
  "qc_thresholds": {
    "1": {"min_sats": 6, "max_h_sigma_m": 0.2, "max_v_sigma_m": 0.4},
    "5": {"min_sats": 6, "max_h_sigma_m": 15.0, "max_v_sigma_m": 30.0}
  },
  "qc_threshold_desc": "Q: 1=fix, 2=float, 3=SBAS, 4=DGPS, 5=single, 6=PPP"
}
```

### RTKLIB Quality Codes

| Q | Mode | Description | Expected Accuracy |
|---|------|-------------|-------------------|
| 1 | Fix | Ambiguity resolved | cm-level |
| 2 | Float | Ambiguity not resolved | 10-50 cm |
| 3 | SBAS | Satellite-based augmentation | 1-3 m |
| 4 | DGPS | Differential GPS | 0.5-2 m |
| 5 | Single | Broadcast ephemeris only | 1-5 m |
| 6 | PPP | Precise Point Positioning | cm-dm |

### Processing Workflow

1. **Capture** (Windows): `capture_rtcm.py` → `.rtcm3` files
2. **Convert** (WSL): `convbin` → `.obs` + `.nav` files
3. **Merge NAV** (WSL): Combine ephemeris from all stations
4. **Position** (WSL): `rnx2rtkp` → `.pos` files
5. **Adapt** (Windows): `position_adapter.py` → NGL `.json` files
6. **Analyze**: Lambda_geo computation on NGL positions

---

## Appendix G: Prediction Validation System (Track Record)

### Purpose

The prediction validation system builds a prospective track record by correlating past predictions with actual seismic events. Unlike retrospective validation (which analyzes data knowing earthquake times), this system operates prospectively—predictions are logged before outcomes are known.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PREDICTION VALIDATION FLOW                          │
│                                                                          │
│  Daily Ensemble Run (6 AM)                                              │
│         │                                                                │
│         ├──► Log predictions to ensemble_YYYY-MM-DD.json                │
│         │                                                                │
│         └──► Run validation on predictions from 7-14 days ago           │
│                    │                                                     │
│                    ▼                                                     │
│         ┌─────────────────────┐         ┌─────────────────────┐         │
│         │  Fetch USGS events  │         │  Load past          │         │
│         │  (M4.5+ in region)  │         │  predictions        │         │
│         └─────────┬───────────┘         └─────────┬───────────┘         │
│                   │                               │                      │
│                   └───────────┬───────────────────┘                      │
│                               ▼                                          │
│                   ┌───────────────────────┐                              │
│                   │  Correlate prediction │                              │
│                   │  with actual events   │                              │
│                   └───────────┬───────────┘                              │
│                               │                                          │
│              ┌────────────────┼────────────────┐                         │
│              ▼                ▼                ▼                         │
│         ┌────────┐      ┌────────────┐   ┌──────────┐                   │
│         │  HIT   │      │ FALSE ALARM│   │ PENDING  │                   │
│         │(event) │      │ (no event) │   │ (<7 days)│                   │
│         └────────┘      └────────────┘   └──────────┘                   │
│                               │                                          │
│                               ▼                                          │
│              monitoring/data/validated_events.json                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Lookback Window

| Window | Duration | Rationale |
|--------|----------|-----------|
| **Minimum** | 7 days | Allow time for predicted event to occur |
| **Maximum** | 14 days | Beyond this, correlation is spurious |
| **Pending** | < 7 days | Too early to classify; wait for window to mature |

### Classification Criteria

| Classification | Criteria |
|----------------|----------|
| **HIT** | Prediction at Tier ≥2 (ELEVATED or CRITICAL) followed by M4.5+ event within 150km of region centroid within 7 days |
| **FALSE ALARM** | Prediction at Tier ≥2 NOT followed by qualifying event within 7-day window |
| **TRUE NEGATIVE** | Prediction at Tier 0-1 (NORMAL or WATCH) with no event (not tracked—expected behavior) |
| **MISS** | M4.5+ event NOT preceded by Tier ≥2 prediction (detected via USGS monitoring) |

**Note**: WATCH (Tier 1) is an awareness level only, not a scorable prediction. Only ELEVATED and CRITICAL predictions are validated against actual events. This prevents false positive inflation from the sensitive WATCH threshold.

### Event Correlation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Magnitude threshold** | M ≥ 4.5 | Detectable signal; lower bound for precursor correlation |
| **Distance tolerance** | ≤ 300 km | Regional coverage; accounts for fault zone extent |
| **Time window** | 0-7 days after prediction | Matches physical lead time of precursor signals |
| **Event source** | USGS Earthquake API | Authoritative, real-time, global coverage |

### Output Format

**File**: `monitoring/data/validated_events.json`

```json
{
  "metadata": {
    "created": "2026-01-22T06:00:00Z",
    "total_predictions_checked": 58,
    "hits": 3,
    "false_alarms": 55,
    "hit_rate": 0.052
  },
  "validated": [
    {
      "prediction_date": "2026-01-08",
      "region": "tokyo_kanto",
      "tier": 2,
      "tier_name": "ELEVATED",
      "risk_score": 0.562,
      "outcome": "hit",
      "event": {
        "magnitude": 4.9,
        "location": "near Tokyo, Japan",
        "time": "2026-01-09T14:23:00Z",
        "distance_km": 85,
        "days_after": 1.5
      }
    },
    {
      "prediction_date": "2026-01-05",
      "region": "cascadia",
      "tier": 1,
      "tier_name": "WATCH",
      "risk_score": 0.437,
      "outcome": "false_alarm",
      "event": null
    }
  ]
}
```

### Integration with Daily Run

The validation system is integrated into `run_ensemble_daily.py`:

```python
# At the end of daily ensemble run:
from validate_predictions import validate_recent_predictions

# Validate predictions from 7-14 days ago
validation_results = validate_recent_predictions(
    lookback_min_days=7,
    lookback_max_days=14,
    output_file="monitoring/data/validated_events.json"
)

print(f"Validated {validation_results['checked']} predictions: "
      f"{validation_results['hits']} hits, "
      f"{validation_results['false_alarms']} false alarms")
```

### Standalone Usage

```bash
# Rebuild full validation history (all available predictions)
python validate_predictions.py --rebuild

# Daily incremental validation (default: 7-14 day lookback)
python validate_predictions.py

# Custom lookback window
python validate_predictions.py --min-days 5 --max-days 10
```

### Initial Results (December 17, 2025 - January 15, 2026)

| Metric | Value |
|--------|-------|
| ELEVATED/CRITICAL predictions | 0 |
| Hits | 0 |
| False alarms | 0 |
| Hit rate | N/A (no predictions to validate) |

**Note**: With the M6+ calibration and tier gating (requires ≥2 methods for ELEVATED), no ELEVATED or CRITICAL predictions were issued during the initial monitoring period. This is expected conservative behavior. The track record will build as the system issues actual ELEVATED/CRITICAL alerts.

**WATCH-level activity** (not scored, for reference only):
- Dec 26-28: Tokyo Kanto at WATCH (Tier 1)
- Jan 9-11: Tokyo Kanto at WATCH (Tier 1)
- Jan 15: SoCal SAF Coachella at WATCH (Tier 1)

These WATCH signals indicate the system is detecting activity, but they don't meet the threshold for a scorable prediction.

### Interpretation Notes

1. **5% hit rate is expected**: Earthquake precursor signals are rare. A 5% hit rate means elevated signals occasionally correlate with actual events—this is meaningful signal, not random noise.

2. **False alarms are acceptable**: The system is designed to be sensitive. High false alarm rate is preferable to missing actual precursors. Tier escalation requires multiple methods to reduce false positives.

3. **Track record builds over time**: Statistical significance requires 100+ validated predictions. After 6-12 months, confidence intervals on hit rate will narrow.

4. **Regional patterns emerge**: Some regions may show higher hit rates due to better sensor coverage or fault characteristics. Track per-region statistics separately.

### Code Reference

| Component | File | Key Functions |
|-----------|------|---------------|
| Validation Engine | `monitoring/src/validate_predictions.py` | `validate_recent_predictions()`, `correlate_with_usgs()` |
| USGS API Client | `monitoring/src/validate_predictions.py` | `fetch_usgs_events()` |
| Daily Integration | `monitoring/src/run_ensemble_daily.py` | Calls validation at end of run |
| Output Data | `monitoring/data/validated_events.json` | Accumulated validation results |

### Future Enhancements

1. **Miss detection**: Monitor USGS for M5.0+ events in monitored regions that were NOT preceded by elevated predictions
2. **Per-region hit rates**: Track validation statistics by region to identify coverage gaps
3. **Tier-stratified analysis**: Compare hit rates for WATCH vs ELEVATED vs CRITICAL predictions
4. **Confidence intervals**: Calculate 95% CI on hit rate as sample size grows
5. **ROC curve generation**: Plot true positive rate vs false positive rate at different tier thresholds

---

## Appendix H: Data Regeneration and Calibration Tracking

### Purpose

This appendix documents the data architecture that separates **raw measurements** from **calibrated outputs**, enabling recalibration without data loss. It also describes the regeneration script that rebuilds derived CSV files from authoritative JSON sources.

### Data Architecture: Raw vs Derived

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA HIERARCHY                                   │
│                                                                          │
│  RAW DATA (Authoritative - Never Modify for Calibration)                │
│  ─────────────────────────────────────────────────────────              │
│  monitoring/data/baselines/lambda_geo_baselines.json                    │
│    └── GPS station statistics (mean, std, n_samples per region)         │
│  monitoring/data/baselines/thd_baselines_YYYYMMDD.json                  │
│    └── Seismic station THD statistics (mean, std, percentiles)          │
│  monitoring/data/ensemble_results/ensemble_YYYY-MM-DD.json              │
│    └── Daily ensemble computations (tier, risk, components)             │
│                                                                          │
│  CALIBRATION PARAMETERS (Safe to Modify)                                │
│  ───────────────────────────────────────                                │
│  monitoring/src/ensemble.py                                              │
│    └── Risk conversion functions (ratio→risk, z-score→risk)             │
│  monitoring/config/backtest_config.yaml                                  │
│    └── Evaluation thresholds (min_magnitude, hit_min_tier)              │
│                                                                          │
│  DERIVED DATA (Regenerate After Calibration Change)                     │
│  ──────────────────────────────────────────────────                     │
│  monitoring/data/ensemble_results/daily_states.csv                      │
│    └── Historical predictions (regenerated from JSON)                   │
│  monitoring/dashboard/data.csv                                           │
│    └── Dashboard display data                                            │
│  monitoring/data/validated_events.json                                   │
│    └── Track record (regenerated from daily_states.csv + USGS)          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Regeneration Script

**File**: `monitoring/src/regenerate_daily_states.py`

**Purpose**: Rebuilds `daily_states.csv` from authoritative ensemble JSON files, ensuring consistency after calibration changes.

**Usage**:
```bash
# Preview changes (dry run)
cd monitoring
python -m src.regenerate_daily_states --dry-run

# Regenerate
python -m src.regenerate_daily_states
```

**What It Does**:
1. Reads all `ensemble_YYYY-MM-DD.json` files
2. Extracts tier, risk, and component values for each region
3. Adds calibration metadata columns
4. Writes new `daily_states.csv` (backs up existing file first)

### CSV Schema with Calibration Metadata

**File**: `monitoring/data/ensemble_results/daily_states.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | YYYY-MM-DD | Assessment date |
| `region` | string | Region identifier |
| `tier` | int | Risk tier (0=NORMAL, 1=WATCH, 2=ELEVATED, 3=CRITICAL) |
| `risk` | float | Combined risk score (0.0-1.0) |
| `methods` | int | Number of methods available |
| `confidence` | float | Confidence score (0.0-1.0) |
| `lg_ratio` | float | Lambda_geo raw value (ratio) |
| `thd` | float | THD raw value |
| `fc_l2l1` | float | Fault correlation L2/L1 ratio |
| `status` | string | REGENERATED, PRELIMINARY, or CONFIRMED |
| `notes` | string | Tier capping notes, degraded status |
| `calibration_date` | YYYY-MM-DD | **Date of baseline calibration used** |
| `regenerated_at` | YYYY-MM-DD HH:MM | **When this row was generated** |

### Example Row

```csv
2026-01-08,tokyo_kanto,0,0.2071,3,0.60,0.107,0.3771,0.8670,REGENERATED,,2026-01-16,2026-01-22 10:36
```

**Interpretation**:
- January 8, 2026 assessment for Tokyo Kanto
- Tier 0 (NORMAL), risk 0.21
- 3 methods available, confidence 0.60
- Lambda_geo ratio: 0.107, THD: 0.377, FC L2/L1: 0.867
- Data regenerated on Jan 22, 2026 using Jan 16, 2026 calibration

### Recalibration Workflow

When calibration parameters change:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      RECALIBRATION WORKFLOW                             │
│                                                                         │
│  1. BACKUP                                                              │
│     cp monitoring/src/ensemble.py ensemble.py.backup                   │
│                                                                         │
│  2. MODIFY CALIBRATION                                                  │
│     Edit monitoring/src/ensemble.py:                                   │
│       - lambda_geo_to_risk() thresholds                                │
│       - thd_to_risk_with_baseline() z-score mapping                    │
│       - RISK_TIERS boundaries                                           │
│                                                                         │
│  3. REGENERATE DERIVED DATA                                             │
│     cd monitoring                                                       │
│     python -m src.regenerate_daily_states                              │
│                                                                         │
│  4. REBUILD TRACK RECORD                                                │
│     python -m src.validate_predictions --rebuild                       │
│                                                                         │
│  5. UPDATE DASHBOARD                                                    │
│     cp data/validated_events.json ../docs/                             │
│                                                                         │
│  6. VERIFY                                                              │
│     - Check dashboard matches CSV                                       │
│     - Check track record reflects new calibration                       │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture Matters

1. **Audit Trail**: `calibration_date` and `regenerated_at` columns document exactly which calibration produced each prediction

2. **Reproducibility**: Raw JSON files preserve original computations; derived CSVs can be rebuilt anytime

3. **Safe Recalibration**: Changing thresholds doesn't destroy historical data—just regenerate derived files

4. **Debugging**: If dashboard doesn't match track record, regenerate from authoritative JSONs

### Data Integrity Incident (January 2026)

**Problem**: `daily_states.csv` contained pre-calibration values from early January that didn't match the ensemble JSON files (which were regenerated on Jan 16 with updated calibration).

**Symptom**: Track record showed Tokyo Kanto at Tier 2 before M4.5 events, but dashboard showed Tier 0-1.

**Root Cause**: CSV was written incrementally and never regenerated after calibration change.

**Resolution**: Created `regenerate_daily_states.py` to rebuild CSV from authoritative JSON files.

**Prevention**: Added calibration metadata columns so discrepancies are immediately visible.

### Code Reference

| Component | File | Purpose |
|-----------|------|---------|
| Regeneration Script | `monitoring/src/regenerate_daily_states.py` | Rebuild CSV from JSONs |
| Validation Rebuild | `monitoring/src/validate_predictions.py --rebuild` | Rebuild track record |
| Baseline Files | `monitoring/data/baselines/*.json` | Raw calibration data |
| Ensemble JSONs | `monitoring/data/ensemble_results/ensemble_*.json` | Authoritative daily results |

---

*Document generated: January 2026*
*GeoSpec Project - mail.rjmathews@gmail.com*
