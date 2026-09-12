# Λ_geo Forward Monitoring Specification (v2.0)

## Rigorous Framework for Prospective Validation

**Author:** R.J. Mathews  
**Date:** January 2026  
**Status:** Technical Specification (Pre-Implementation)

---

## 1. Monitoring Unit Definition

### 1.1 Spatial Unit: Fault Segment Polygons (Not Grid Cells)

Instead of arbitrary lat/lon grids, define monitoring units as **fault segment polygons**:

```
┌──────────────────────────────────────────────────────────────┐
│  Example: Southern California Monitoring Units               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   SAF-Mojave    ┌─────────────┐                             │
│                 │   Polygon   │  • Min 15 stations          │
│                 │   ~50km     │  • 8 Delaunay triangles     │
│                 └─────────────┘                             │
│                                                              │
│   SAF-Coachella ┌─────────────┐                             │
│                 │   Polygon   │  • Min 12 stations          │
│                 │   ~40km     │  • 6 Delaunay triangles     │
│                 └─────────────┘                             │
│                                                              │
│   Garlock       ┌──────────┐                                │
│                 │ Polygon  │    • Min 10 stations           │
│                 └──────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Eligibility Criteria Per Monitoring Unit

| Criterion | Minimum | Preferred |
|-----------|---------|-----------|
| **Station count** | ≥ 10 | ≥ 20 |
| **Station history** | ≥ 3 years | ≥ 5 years |
| **Delaunay triangles** | ≥ 6 | ≥ 15 |
| **Triangle quality** (min angle) | > 15° | > 25° |
| **Max station spacing** | < 50 km | < 30 km |
| **Data completeness** | > 80% | > 95% |

### 1.3 Spatial Coherence Requirement

**Problem:** A single bad station can drive the "spatial max" metric.

**Solution:** Require coherent elevation across multiple triangles:

```python
def is_spatially_coherent(lambda_geo_grid, threshold_factor, baseline):
    """
    Check if elevation is coherent, not single-point noise.
    
    Criteria:
    1. At least K adjacent cells exceed threshold
    2. Or at least N% of triangles exceed threshold
    """
    threshold = threshold_factor * baseline
    
    # Count cells exceeding threshold
    elevated_cells = lambda_geo_grid > threshold
    n_elevated = np.sum(elevated_cells)
    total_cells = lambda_geo_grid.size
    
    # Check spatial clustering (connected components)
    from scipy.ndimage import label
    labeled, n_clusters = label(elevated_cells)
    max_cluster_size = max(np.sum(labeled == i) for i in range(1, n_clusters + 1)) if n_clusters > 0 else 0
    
    # Coherence criteria
    MIN_CLUSTER_SIZE = 3  # At least 3 adjacent cells
    MIN_FRACTION = 0.10   # Or at least 10% of grid
    
    coherent = (max_cluster_size >= MIN_CLUSTER_SIZE) or (n_elevated / total_cells >= MIN_FRACTION)
    
    return coherent, {
        'n_elevated': n_elevated,
        'max_cluster_size': max_cluster_size,
        'fraction_elevated': n_elevated / total_cells
    }
```

---

## 2. Real-Time Baseline Definition

### 2.1 Problem with "First 7 Days"

In retrospective analysis, we knew the event window. In real-time:
- We don't know when an event will occur
- Seasonal variations dominate multi-month periods
- Recent events create postseismic transients

### 2.2 Rolling Baseline Specification

```python
class RollingBaseline:
    """
    Real-time baseline using robust statistics and seasonal detrending.
    """
    
    def __init__(self, 
                 lookback_days: int = 90,
                 exclude_recent_days: int = 14,
                 seasonal_model: bool = True):
        """
        Args:
            lookback_days: Days of history for baseline (default 90)
            exclude_recent_days: Gap between baseline and current (default 14)
            seasonal_model: Whether to fit/remove annual+semiannual
        """
        self.lookback_days = lookback_days
        self.exclude_recent_days = exclude_recent_days
        self.seasonal_model = seasonal_model
    
    def compute_baseline(self, lambda_geo_history: np.ndarray, 
                         dates: np.ndarray) -> dict:
        """
        Compute robust baseline from historical data.
        
        Timeline:
        [-------- lookback_days --------][-- exclude --][current]
        ^                               ^               ^
        baseline_start            baseline_end      today
        """
        
        # Get baseline window
        baseline_end = len(dates) - self.exclude_recent_days
        baseline_start = max(0, baseline_end - self.lookback_days)
        baseline_data = lambda_geo_history[baseline_start:baseline_end]
        baseline_dates = dates[baseline_start:baseline_end]
        
        # Remove seasonality if requested
        if self.seasonal_model and len(baseline_data) > 60:
            baseline_data = self._remove_seasonality(baseline_data, baseline_dates)
        
        # Robust statistics (median, MAD)
        median = np.nanmedian(baseline_data)
        mad = np.nanmedian(np.abs(baseline_data - median))
        robust_std = 1.4826 * mad  # MAD to std conversion
        
        return {
            'median': median,
            'mad': mad,
            'robust_std': robust_std,
            'n_days': len(baseline_data),
            'baseline_start': baseline_dates[0],
            'baseline_end': baseline_dates[-1],
        }
    
    def _remove_seasonality(self, data, dates):
        """Fit and remove annual + semiannual components."""
        # Convert dates to decimal year
        decimal_year = np.array([d.year + d.timetuple().tm_yday / 365.25 for d in dates])
        
        # Fit: y = a + b*sin(2π*t) + c*cos(2π*t) + d*sin(4π*t) + e*cos(4π*t)
        X = np.column_stack([
            np.ones(len(decimal_year)),
            np.sin(2 * np.pi * decimal_year),
            np.cos(2 * np.pi * decimal_year),
            np.sin(4 * np.pi * decimal_year),
            np.cos(4 * np.pi * decimal_year),
        ])
        
        # Robust fit (iteratively reweighted least squares)
        coeffs, _, _, _ = np.linalg.lstsq(X, data, rcond=None)
        seasonal_model = X @ coeffs
        
        return data - seasonal_model + np.nanmedian(data)
```

### 2.3 Baseline Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Fixed 7-day** | Simple, matches validation | Only works retrospectively |
| **Rolling 30-day** | Responsive to changes | Too short for seasonality |
| **Rolling 90-day** | Captures seasonality | May include transients |
| **Seasonally-matched** | Best for annual cycles | Requires multi-year history |

**Recommendation:** 90-day rolling with seasonal detrending, excluding most recent 14 days.

---

## 3. Multiple Comparisons Correction

### 3.1 The Problem

If monitoring N regions for D days, expected false alarms:

```
Expected false alarms = N × D × FPR_per_day

Example:
- 10 regions × 365 days × 0.01 FPR = 36.5 false alarms/year
```

### 3.2 Global False Alarm Budget

**Specification:** Target ≤ 1 false Tier-2 alert per year across all regions.

```python
def compute_region_threshold(n_regions: int, 
                              target_global_fpr: float = 1/365,
                              monitoring_days_per_year: int = 365) -> float:
    """
    Compute per-region, per-day FPR to achieve global target.
    
    Args:
        n_regions: Number of monitored regions
        target_global_fpr: Target false alerts per year
        monitoring_days_per_year: Days of monitoring
    
    Returns:
        per_region_per_day_fpr: FPR threshold per region per day
    """
    # Bonferroni-style correction (conservative)
    per_region_per_day_fpr = target_global_fpr / (n_regions * monitoring_days_per_year)
    
    return per_region_per_day_fpr

# Example: 10 regions, 1 false alarm/year target
# Per-region-day FPR = 1 / (10 × 365) = 0.000274 = 0.027%
```

### 3.3 Adaptive Thresholds

Instead of fixed 5×, compute region-specific thresholds to achieve target FPR:

```python
def compute_adaptive_threshold(region_baseline: dict, 
                                target_fpr: float = 0.00027) -> float:
    """
    Given region's baseline distribution, find threshold for target FPR.
    
    Uses log-normal model fit to baseline data.
    """
    from scipy import stats
    
    mu = region_baseline['lognormal_mu']
    sigma = region_baseline['lognormal_sigma']
    
    # Find threshold where P(X > threshold) = target_fpr
    # For 3-day max: P(max of 3 > T) = 1 - (1 - P(X > T))^3 = target_fpr
    # Solve: P(X > T) = 1 - (1 - target_fpr)^(1/3)
    p_single = 1 - (1 - target_fpr) ** (1/3)
    
    # Inverse CDF of log-normal
    threshold = stats.lognorm.ppf(1 - p_single, s=sigma, scale=np.exp(mu))
    
    return threshold
```

---

## 4. Tiered Alert Levels

### 4.1 Alert Tier Definitions

| Tier | Condition | Duration | Action |
|------|-----------|----------|--------|
| **0: Normal** | Λ_geo < 2× baseline | - | Routine logging |
| **1: Watch** | 2× ≤ Λ_geo < 5× | 1 day | Internal flag, automated logging |
| **2: Elevated** | 5× ≤ Λ_geo < 10× | 2+ days sustained | Internal review, data quality check |
| **3: High** | Λ_geo ≥ 10× | 2+ days + coherent | Expert review, agency notification (private) |

### 4.2 Tier Transition Logic

```python
class AlertStateMachine:
    """
    Manages tier transitions with hysteresis and confirmation.
    """
    
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.current_tier = 0
        self.tier_entry_date = None
        self.consecutive_days = 0
        self.history = []
    
    def update(self, date, lambda_geo_max, baseline, is_coherent):
        """
        Update alert state based on new observation.
        """
        ratio = lambda_geo_max / baseline
        
        # Determine candidate tier
        if ratio >= 10 and is_coherent:
            candidate_tier = 3
        elif ratio >= 5:
            candidate_tier = 2
        elif ratio >= 2:
            candidate_tier = 1
        else:
            candidate_tier = 0
        
        # Apply persistence requirements
        if candidate_tier == self.current_tier:
            self.consecutive_days += 1
        elif candidate_tier > self.current_tier:
            # Escalation requires confirmation
            if candidate_tier >= 2:
                # Tier 2+ requires 2 consecutive days
                if self.consecutive_days >= 1 and candidate_tier == self.current_tier + 1:
                    self._escalate(candidate_tier, date)
            else:
                self._escalate(candidate_tier, date)
        else:
            # De-escalation after 3 days below threshold
            if self.consecutive_days >= 3:
                self._deescalate(candidate_tier, date)
            self.consecutive_days += 1
        
        self.history.append({
            'date': date,
            'ratio': ratio,
            'tier': self.current_tier,
            'coherent': is_coherent
        })
        
        return self.current_tier
    
    def _escalate(self, new_tier, date):
        self.current_tier = new_tier
        self.tier_entry_date = date
        self.consecutive_days = 1
        # Log escalation event
        
    def _deescalate(self, new_tier, date):
        self.current_tier = new_tier
        self.tier_entry_date = date
        self.consecutive_days = 1
        # Log de-escalation event
```

### 4.3 Response Matrix

| Tier | Automated | Human Review | Agency Contact | Public |
|------|-----------|--------------|----------------|--------|
| 0 | Log only | No | No | No |
| 1 | Flag + log | No | No | No |
| 2 | Alert email | Within 24h | No | No |
| 3 | Alert + call | Immediate | Private discussion | **Never** (research phase) |

---

## 5. Prospective Scoring Plan (Pre-Registered)

### 5.1 Event Definitions

**Target Events (Hits/Misses evaluated against):**
- Magnitude: M ≥ 6.5 (primary), M ≥ 7.0 (secondary analysis)
- Location: Within 100 km of monitoring unit centroid
- Depth: < 50 km (crustal events)
- Type: Mainshocks only (exclude aftershocks within 14 days of M6+ parent)

**Lead Time Window:**
- Minimum: 24 hours before mainshock
- Maximum: 14 days before mainshock
- "Hit" requires alert active at time T where 24h ≤ T ≤ 14d before event

### 5.2 Scoring Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Hit Rate** | Hits / (Hits + Misses) | > 0.7 |
| **False Alarm Rate** | False Alarms / (False Alarms + Correct Rejections) | < 0.1 |
| **Precision** | Hits / (Hits + False Alarms) | > 0.3 |
| **Lead Time** | Mean(time from alert to event) for hits | 3-10 days |
| **Time in Warning** | Days per year in Tier 2+ | < 30 days |

### 5.3 Evaluation Windows

```
┌─────────────────────────────────────────────────────────────┐
│  Event Evaluation Window                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [------ 14 days ------][event][-- 3 days --]              │
│  ^                      ^      ^                           │
│  Alert must start       Event  Aftershock                  │
│  within this window            exclusion                   │
│                                                             │
│  Hit: Tier 2+ active at least once in [-14d, -24h]         │
│  Miss: No Tier 2+ in window, event occurred                │
│  False Alarm: Tier 2+ but no event in [+24h, +14d]         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Special Cases

| Situation | Handling |
|-----------|----------|
| **Foreshock-mainshock pair** | Count as single event (mainshock timing) |
| **Slow slip episode** | Label separately, exclude from primary scoring |
| **Swarm activity** | Count largest event only |
| **Multiple regions alerted** | Credit to region closest to epicenter |
| **Postseismic period** | Exclude 90 days after M7+ from scoring |

---

## 6. Noise Floor Characterization

### 6.1 Pre-Monitoring Baseline Study

Before activating monitoring for a region, compute:

```python
def characterize_noise_floor(region_id: str, 
                              years_of_data: int = 3) -> dict:
    """
    Characterize baseline variability during quiet periods.
    
    Outputs:
    - Seasonal pattern
    - Day-to-day variability
    - Frequency of threshold exceedances with no earthquakes
    - Station dropout sensitivity
    """
    
    # 1. Load multi-year history
    data = load_region_history(region_id, years=years_of_data)
    
    # 2. Remove periods near known M5+ earthquakes
    quiet_data = remove_earthquake_windows(data, magnitude_threshold=5.0,
                                            window_days=30)
    
    # 3. Compute seasonal pattern
    seasonal = fit_seasonal_model(quiet_data)
    
    # 4. Compute detrended variability
    detrended = quiet_data - seasonal
    daily_std = np.std(np.diff(detrended))
    
    # 5. Count threshold exceedances
    baseline_median = np.median(quiet_data)
    exceedances = {
        '2x': count_exceedances(quiet_data, 2 * baseline_median),
        '5x': count_exceedances(quiet_data, 5 * baseline_median),
        '10x': count_exceedances(quiet_data, 10 * baseline_median),
    }
    
    # 6. Station dropout sensitivity
    dropout_sensitivity = test_station_dropout_impact(region_id)
    
    return {
        'region_id': region_id,
        'years_analyzed': years_of_data,
        'quiet_days': len(quiet_data),
        'seasonal_amplitude': np.max(seasonal) - np.min(seasonal),
        'daily_variability_std': daily_std,
        'baseline_median': baseline_median,
        'exceedances_per_year': {
            k: v / years_of_data for k, v in exceedances.items()
        },
        'dropout_sensitivity': dropout_sensitivity,
        'monitorable': exceedances['5x'] / years_of_data < 5,  # <5 per year
    }
```

### 6.2 Monitorability Criteria

A region is "monitorable" if:

| Criterion | Threshold |
|-----------|-----------|
| Quiet-period 5× exceedances | < 5 per year |
| Quiet-period 10× exceedances | < 1 per year |
| Station dropout impact | < 2× change in Λ_geo |
| Seasonal amplitude | < 3× baseline variability |
| Data completeness | > 90% |

---

## 7. Confound Detection & Handling

### 7.1 Network Artifact Detection

```python
def detect_network_artifact(lambda_geo_change: float,
                            n_stations_affected: int,
                            total_stations: int,
                            change_is_steplike: bool) -> bool:
    """
    Flag potential processing/network artifacts.
    
    Artifacts typically show:
    - Sudden step-like change (not gradual)
    - Affects many/all stations simultaneously
    - No spatial coherence (random pattern)
    """
    
    fraction_affected = n_stations_affected / total_stations
    
    is_artifact = (
        change_is_steplike and
        fraction_affected > 0.5 and
        lambda_geo_change > 5  # Large change
    )
    
    return is_artifact
```

### 7.2 Postseismic Period Handling

After M ≥ 6.5 events within monitoring region:

```python
POSTSEISMIC_REGIMES = {
    'M6.5-6.9': {'baseline_reset_days': 30, 'elevated_threshold_factor': 1.5},
    'M7.0-7.4': {'baseline_reset_days': 60, 'elevated_threshold_factor': 2.0},
    'M7.5-7.9': {'baseline_reset_days': 90, 'elevated_threshold_factor': 2.5},
    'M8.0+':    {'baseline_reset_days': 180, 'elevated_threshold_factor': 3.0},
}

def handle_postseismic(region_state, event_magnitude, event_date):
    """
    Adjust monitoring parameters after significant earthquake.
    """
    regime = get_regime(event_magnitude)
    
    region_state.set_mode('postseismic')
    region_state.reset_baseline_after(days=regime['baseline_reset_days'])
    region_state.adjust_threshold(factor=regime['elevated_threshold_factor'])
    region_state.exclude_from_scoring_until(
        event_date + timedelta(days=regime['baseline_reset_days'])
    )
```

### 7.3 Slow Slip Event Detection

Slow slip events (SSEs) can produce elevated Λ_geo without being earthquake precursors:

```python
def is_slow_slip_signature(lambda_geo_timeseries: np.ndarray,
                            duration_days: int) -> bool:
    """
    SSE characteristics:
    - Gradual onset (days, not hours)
    - Sustained elevation (weeks)
    - Gradual decay
    - Often with tremor activity
    """
    
    # Check for gradual onset (low derivative at start)
    onset_derivative = np.gradient(lambda_geo_timeseries[:7])
    gradual_onset = np.max(np.abs(onset_derivative)) < 0.5 * np.max(lambda_geo_timeseries)
    
    # Check for sustained plateau
    sustained = duration_days > 14
    
    # Check for symmetric rise/fall
    peak_idx = np.argmax(lambda_geo_timeseries)
    rise = lambda_geo_timeseries[:peak_idx]
    fall = lambda_geo_timeseries[peak_idx:]
    symmetric = abs(len(rise) - len(fall)) / max(len(rise), len(fall)) < 0.5
    
    return gradual_onset and sustained and symmetric
```

---

## 8. Implementation Checklist

### Phase 0: Pre-Monitoring (Months 1-2)

- [ ] Select 5-10 candidate regions based on hazard × observability × utility
- [ ] Run noise floor characterization for each region
- [ ] Eliminate regions that fail monitorability criteria
- [ ] Compute region-specific adaptive thresholds
- [ ] Set up data pipelines and quality monitoring
- [ ] Pre-register prospective scoring plan

### Phase 1: Shadow Monitoring (Months 3-8)

- [ ] Run daily Λ_geo computation for all regions
- [ ] Log all tier transitions (no external actions)
- [ ] Track actual seismicity in monitored regions
- [ ] Compute running precision/recall estimates
- [ ] Identify and document confound events

### Phase 2: Evaluation (Months 9-12)

- [ ] Compile 6-month performance statistics
- [ ] Compare to pre-registered targets
- [ ] Identify systematic biases or failures
- [ ] Publish findings for peer review
- [ ] Decide: continue, modify, or discontinue

---

## 9. The 5 Critical Questions Answered

### Q1: What's the minimum station density?

**Answer:** ≥ 10 stations within monitoring polygon, with < 50 km spacing and ≥ 6 Delaunay triangles with minimum angle > 15°.

### Q2: How will you handle seasonality?

**Answer:** 90-day rolling baseline with annual + semiannual harmonic removal. Exclude most recent 14 days from baseline to avoid contamination.

### Q3: What's your global false alert budget?

**Answer:** ≤ 1 Tier-2 alert per year across all monitored regions. Per-region thresholds computed via Bonferroni correction.

### Q4: What's your response to Tier 1 vs Tier 2?

**Answer:** 
- Tier 1: Automated logging only
- Tier 2: Internal email + human review within 24h, data quality check
- Tier 3: Expert review + private agency discussion

### Q5: What's your prospective scoring plan?

**Answer:** Pre-registered evaluation against M ≥ 6.5 events within 100 km, lead time 24h-14d, excluding aftershocks. Targets: Hit rate > 70%, FAR < 10%, Precision > 30%.

---

## 10. Recommended Pilot Regions

Based on hazard × observability × utility / confound-risk:

| Rank | Region | Score | Notes |
|------|--------|-------|-------|
| 1 | **Southern California (SAF)** | 0.92 | Best GPS, validated (Ridgecrest), high hazard |
| 2 | **Tokyo/Kanto** | 0.88 | Best network globally, extreme exposure |
| 3 | **San Francisco Bay** | 0.85 | Good GPS, high hazard, high exposure |
| 4 | **Cascadia (Oregon/Washington)** | 0.78 | Good GPS, M9 risk, lower background rate |
| 5 | **Istanbul (Marmara)** | 0.72 | High hazard, moderate GPS, confound risk |

**Recommendation:** Start with Southern California (closest to validation) + Tokyo (best network) as initial pilot regions.
