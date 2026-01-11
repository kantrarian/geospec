# GeoSpec Three-Method Ensemble: Technical Methods Document

**Version**: 1.2
**Date**: January 2026
**Author**: R.J. Mathews
**Status**: Partial Operational
- THD: Operational (all 8 regions via IU/BK networks)
- Fault Correlation: Operational (Cascadia only); Disabled elsewhere (insufficient coverage)
- Lambda_geo: Retrospective-only (awaiting RT-GNSS integration)

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
- Fault correlation disabled for most regions (insufficient station coverage)
- Lambda_geo awaiting real-time GPS integration (retrospective-only)
- Validation: 16/16 tests passed on 4 major earthquakes (M7.1-M9.0)
  - Each event tested for: ELEVATED reached, CRITICAL reached, ≥72h lead time, sustained until mainshock
  - **Caveat**: All retrospective; prospective validation pending

---

## Three-Method Ensemble Overview

### Architecture

```
                    GPS DATA                         SEISMIC DATA
                       │                                  │
                       ▼                                  ▼
              ┌────────────────┐              ┌─────────────────────┐
              │   Lambda_geo   │              │  Waveform Fetching  │
              │   (NGL IGS20)  │              │  (IRIS/NCEDC/SCEDC) │
              └───────┬────────┘              └──────────┬──────────┘
                      │                                  │
                      │                       ┌──────────┴──────────┐
                      │                       │                     │
                      │              ┌────────▼────────┐  ┌─────────▼─────────┐
                      │              │ Fault Correlation│  │   Seismic THD    │
                      │              │   (Eigenvalues)  │  │   (Harmonics)    │
                      │              └────────┬─────────┘  └─────────┬────────┘
                      │                       │                      │
                      └───────────────────────┼──────────────────────┘
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

| Parameter | Value |
|-----------|-------|
| Source | Nevada Geodetic Laboratory (NGL) |
| Product | IGS20 Rapid Solutions |
| Update Frequency | Daily |
| Latency | 2-14 days |
| Format | TENV3 position files |
| URL | http://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/ |

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

**NOT YET OPERATIONAL** - Awaiting real-time GPS integration. Historical validation used reconstructed ratios from published observations.

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
| Source | IRIS, NCEDC, SCEDC (FDSN Web Services) |
| Data Type | Continuous waveforms (BHZ channel) |
| Window | 24 hours |
| Step | 6 hours |
| Processing | Bandpass 0.01-1.0 Hz, envelope via Hilbert |

### Current Status

**MOSTLY DISABLED** - Working for Cascadia only (2 segments with adequate coverage). Disabled for all other regions due to insufficient station coverage (requires minimum 2 stations/segment, 2 segments/region). Regional station data gaps (CI, NC, KO networks) prevent operation.

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
| Source | IRIS (IU network), NCEDC (BK network) |
| Data Type | Continuous waveforms (BHZ channel) |
| Sample Rate | 40 Hz (decimated to 1 Hz) |
| Window | 24 hours |
| Processing | Detrend, FFT, spectral analysis |

### Current Status

**OPERATIONAL** - Working for all 8 regions using IU global network stations.

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
| Tokyo Kanto | MAJO | IU | IRIS | Matsushiro, Japan | 100% |
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
| IU.MAJO (Matsushiro) | Japan's primary IU station; near Tokyo |
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
| Tokyo | 4 | 0 | **DISABLED** | Requires NIED Hi-net |

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

**Impact**: Regional networks in Turkey and Japan not accessible via FDSN

**Affected Networks**:
| Network | Region | Issue |
|---------|--------|-------|
| KO | Turkey | Kandilli Observatory - not on IRIS |
| TU | Turkey | National network - restricted |
| NIED Hi-net | Japan | Requires registration (submitted) |
| GE | Germany/Turkey | GEOFON - partial availability |

**Current Workaround**: Using IU stations (ANTO, MAJO)

**Impact on Results**: Single station per country. Cannot monitor multiple fault systems independently.

### Problem 4: GPS Data Latency

**Impact**: Lambda_geo method not operational in real-time

**Current Situation**:
- NGL IGS20 data has 2-14 day latency
- No real-time GPS integration implemented
- Historical validation only

**Root Cause**: NGL processes GPS data in daily batches with significant delay

**Required Solution**: Partner with UNAVCO/GAGE for real-time streams

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

#### 1. Complete NIED Hi-net Registration
**Status**: Application submitted January 2026
**Expected Outcome**: Access to Japan's dense seismic network
**Impact**: Enable fault correlation for Tokyo region; improve THD with local stations

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

### Historical Validation (4 Events)

| Event | Magnitude | CRITICAL Lead | Methods | Tests |
|-------|-----------|---------------|---------|-------|
| Ridgecrest 2019 | M7.1 | 336 hours | 3/3* | 4/4 |
| Turkey 2023 | M7.8 | 168 hours | 1/3 | 4/4 |
| Tohoku 2011 | M9.0 | 192 hours | 1/3 | 4/4 |
| Chile 2010 | M8.8 | 168 hours | 1/3 | 4/4 |

*Ridgecrest validated with reconstructed data

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
- Sample size (n=4) is statistically limited; prospective validation needed

### Current Operational Results (January 10, 2026)

| Region | Risk | Tier | Methods | Confidence | Notes |
|--------|------|------|---------|------------|-------|
| NorCal Hayward | 0.489 | WATCH | 1/3 (THD) | 50% | Single method; FC disabled |
| Cascadia | 0.139 | NORMAL | 2/3 (FC+THD) | 80% | — |
| Ridgecrest | 0.059 | NORMAL | 1/3 (THD) | 50% | CI stations unavailable |
| SoCal Mojave | 0.059 | NORMAL | 1/3 (THD) | 50% | CI stations unavailable |
| SoCal Coachella | 0.059 | NORMAL | 1/3 (THD) | 50% | CI stations unavailable |
| Tokyo | 0.038 | NORMAL | 1/3 (THD) | 50% | Awaiting NIED Hi-net |
| Istanbul | 0.038 | NORMAL | 1/3 (THD) | 50% | KO network restricted |
| Turkey Kahramanmaras | 0.038 | NORMAL | 1/3 (THD) | 50% | KO network restricted |

**Interpretation**: Cascadia is the only region with 2+ methods operational. All other regions are THD-only with 50% confidence. If any region showed ELEVATED (≥0.50) with only 1 method, it would be flagged for confirmation.

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

| Component | File | Key Functions |
|-----------|------|---------------|
| Ensemble Integration | `monitoring/src/ensemble.py` | `compute_risk()`, `GeoSpecEnsemble` |
| THD Analysis | `monitoring/src/seismic_thd.py` | `SeismicTHDAnalyzer`, `fetch_continuous_data_for_thd()` |
| Fault Correlation | `monitoring/src/fault_correlation.py` | `FaultCorrelationMonitor`, `compute_correlation_matrix()` |
| Seismic Data | `monitoring/src/seismic_data.py` | `SeismicDataFetcher`, `get_segment_envelopes()` |
| Daily Runner | `monitoring/src/run_ensemble_daily.py` | `run_all_regions()`, `REGIONS` config |
| Region Config | `monitoring/src/fault_segments.py` | `FAULT_SEGMENTS` |
| **API Reference** | `docs/API_REFERENCE.md` | Connection instructions for IRIS, IGS, NIED, ORFEUS |

## Appendix B: Data Center Endpoints

| Data Center | FDSN URL | Networks |
|-------------|----------|----------|
| IRIS | https://service.iris.edu/fdsnws/ | IU, II, CN, UW, GE |
| NCEDC | https://service.ncedc.org/fdsnws/ | BK, NC |
| SCEDC | https://service.scedc.caltech.edu/fdsnws/ | CI, AZ |

---

*Document generated: January 2026*
*GeoSpec Project - mail.rjmathews@gmail.com*
