# GeoSpec Calibration Reference Document

**Version**: 1.0.0
**Generated**: 2026-01-22
**Purpose**: Document all calibration parameters, thresholds, and data file locations for auditing and recalibration.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [File Locations](#file-locations)
3. [Risk Tier Thresholds](#risk-tier-thresholds)
4. [Method 1: Lambda_geo (GPS Strain)](#method-1-lambda_geo-gps-strain)
5. [Method 2: Fault Correlation](#method-2-fault-correlation)
6. [Method 3: Seismic THD](#method-3-seismic-thd)
7. [Per-Region Calibration Data](#per-region-calibration-data)
8. [Raw vs Calibrated Data Separation](#raw-vs-calibrated-data-separation)
9. [Sensitivity Analysis](#sensitivity-analysis)
10. [Recalibration Procedure](#recalibration-procedure)

---

## Executive Summary

The current calibration is tuned for **M6.0+ detection** based on 5 historical events (M6.8-M9.0). The thresholds are:

| Method | WATCH Trigger | ELEVATED Trigger | CRITICAL Trigger |
|--------|---------------|------------------|------------------|
| Lambda_geo | ratio > 1.5x baseline | ratio > 2.5x baseline | ratio > 4.0x baseline |
| Seismic THD | z-score > 1.0 | z-score > 2.0 | z-score > 3.0 |
| Fault Correlation | L2/L1 < 0.25 | L2/L1 < 0.15 | L2/L1 < 0.10 |

**Issue Identified**: These thresholds were calibrated for M6+ events. M4.5-M5.9 events may not produce sufficient precursor signals to trigger even WATCH-level alerts.

---

## File Locations

### Baseline Data (RAW - Do Not Modify Directly)

| File | Description | Last Updated |
|------|-------------|--------------|
| `monitoring/data/baselines/lambda_geo_baselines.json` | Lambda_geo 90-day baselines per region | 2026-01-16 |
| `monitoring/data/baselines/thd_baselines_20260112.json` | THD station baselines (mean, std, percentiles) | 2026-01-12 |

### Calibration Configuration

| File | Description |
|------|-------------|
| `monitoring/src/ensemble.py` | Risk conversion functions (lines 141-270) |
| `monitoring/config/backtest_config.yaml` | Evaluation thresholds (min_magnitude: 6.0) |
| `monitoring/config/regions.yaml` | Region definitions and global settings |
| `monitoring/src/regions.py` | Fault polygons and station eligibility |

### Output Data (DERIVED - Safe to Regenerate)

| File | Description |
|------|-------------|
| `monitoring/data/ensemble_results/ensemble_YYYY-MM-DD.json` | Daily ensemble computations |
| `monitoring/data/ensemble_results/daily_states.csv` | Historical predictions (STALE - needs regeneration) |
| `monitoring/dashboard/data.csv` | Dashboard display data (regenerated) |

---

## Risk Tier Thresholds

**Source**: `monitoring/src/ensemble.py` lines 117-123

```python
RISK_TIERS = {
    0: {'name': 'NORMAL',   'min_risk': 0.00, 'max_risk': 0.25},
    1: {'name': 'WATCH',    'min_risk': 0.25, 'max_risk': 0.50},
    2: {'name': 'ELEVATED', 'min_risk': 0.50, 'max_risk': 0.75},
    3: {'name': 'CRITICAL', 'min_risk': 0.75, 'max_risk': 1.00},
}
```

**Tier Gating Rule**: Requires >=2 methods available for Tier >=2 (ELEVATED/CRITICAL)
- Single method can only reach Tier 1 (WATCH) maximum
- This prevents false positives from single noisy method

---

## Method 1: Lambda_geo (GPS Strain)

### Risk Conversion Function

**Source**: `monitoring/src/ensemble.py` lines 141-176

```
Raw Input: Lambda_geo ratio (current / baseline mean)

ratio < 1.0  → risk = 0.00
ratio 1.0-1.5 → risk = 0.00-0.20 (linear)
ratio 1.5-2.5 → risk = 0.20-0.45 (WATCH zone)
ratio 2.5-4.0 → risk = 0.45-0.70 (ELEVATED zone)
ratio 4.0-8.0 → risk = 0.70-1.00 (CRITICAL zone, saturates at 8x)
```

### Calibration Events (Observed Ratios)

| Event | Magnitude | Observed Ratio | Mapped Tier |
|-------|-----------|----------------|-------------|
| Tohoku 2011 | M9.0 | 5.6x | CRITICAL |
| Morocco 2023 | M6.8 | 3.3x | ELEVATED |
| Ridgecrest 2019 | M7.1 | 2.3x | WATCH |
| Chile 2010 | M8.8 | 2.1x | WATCH |
| Turkey 2023 | M7.8 | 2.0x | WATCH |

### Per-Region Baseline Statistics

**Source**: `monitoring/data/baselines/lambda_geo_baselines.json`

| Region | Mean | Std Dev | Max | Stations | Triangles | Quality |
|--------|------|---------|-----|----------|-----------|---------|
| ridgecrest | 0.173 | 0.165 | 0.788 | 30 | 48 | good |
| socal_saf_mojave | 0.055 | 0.050 | 0.238 | 35 | 56 | good |
| socal_saf_coachella | 0.024 | 0.039 | 0.225 | 36 | 60 | good |
| norcal_hayward | 0.171 | 0.292 | 1.203 | 23 | 37 | good |
| tokyo_kanto | 0.018 | 0.017 | 0.087 | 41 | 72 | good |
| istanbul_marmara | 0.010 | 0.014 | 0.062 | 5 | 4 | good |
| cascadia | 0.895 | 0.879 | 4.333 | 29 | 48 | good |
| campi_flegrei | 0.018 | 0.017 | 0.075 | 11 | 14 | good |
| kaikoura | 0.719 | 0.806 | 3.945 | 28 | 47 | good |
| anchorage | 0.498 | 0.696 | 3.424 | 27 | 45 | good |
| kumamoto | 0.214 | 0.202 | 0.911 | 43 | 73 | good |
| turkey_kahramanmaras | N/A | N/A | N/A | 0 | 0 | no_data |
| hualien | N/A | N/A | N/A | 1 | 0 | no_data |

**Note**: Tokyo Kanto has very low baseline (0.018) meaning small absolute changes map to large ratios. A ratio of 1.5x only requires Lambda_geo = 0.027.

---

## Method 2: Fault Correlation

### Risk Conversion Function

**Source**: `monitoring/src/ensemble.py` lines 179-208

```
Raw Inputs:
- L2/L1 ratio (eigenvalue ratio, lower = more decorrelated = higher risk)
- Participation ratio (PR, lower = more concentrated = higher risk)

L2/L1 contribution (50% weight):
  L2/L1 > 0.30 → risk = 0.00-0.15
  L2/L1 0.15-0.30 → risk = 0.15-0.50
  L2/L1 0.05-0.15 → risk = 0.50-0.80
  L2/L1 < 0.05 → risk = 0.80-1.00

PR contribution (50% weight):
  PR > 2.0 → risk = 0.00
  PR 1.5-2.0 → risk = 0.00-0.30
  PR 1.0-1.5 → risk = 0.30-0.60
  PR < 1.0 → risk = 0.60-1.00

Final risk = 0.5 * l2l1_risk + 0.5 * pr_risk
```

### Interpretation

- **Normal**: L2/L1 > 0.3, PR > 2.0 (distributed, uncorrelated strain)
- **Watch**: L2/L1 0.2-0.3 or PR 1.5-2.0 (emerging correlation)
- **Elevated**: L2/L1 0.1-0.2 or PR 1.0-1.5 (significant correlation)
- **Critical**: L2/L1 < 0.1 or PR < 1.0 (highly correlated, concentrated)

---

## Method 3: Seismic THD

### Risk Conversion Function (Baseline-Relative)

**Source**: `monitoring/src/ensemble.py` lines 234-280

```
Raw Input: THD value (Total Harmonic Distortion of seismic signal)

z-score = (THD - baseline_mean) / baseline_std

Z-score to risk mapping:
  z < 0   → risk = 0.05-0.10 (below baseline)
  z 0-1   → risk = 0.10-0.25 (normal variation)
  z 1-2   → risk = 0.25-0.50 (WATCH)
  z 2-3   → risk = 0.50-0.75 (ELEVATED)
  z > 3   → risk = 0.75-1.00 (CRITICAL)
```

### Station-to-Region Mapping

**Source**: `monitoring/config/backtest_config.yaml`

| Region | Network | Station | Sample Rate | Notes |
|--------|---------|---------|-------------|-------|
| ridgecrest | IU | TUC | 40 Hz | 400km proxy (shared with SoCal) |
| socal_saf_mojave | IU | TUC | 40 Hz | Shared proxy |
| socal_saf_coachella | IU | TUC | 40 Hz | Shared proxy |
| norcal_hayward | BK | BKS | 40 Hz | Berkeley, on fault |
| cascadia | IU | COR | 40 Hz | Corvallis, 100km from zone |
| tokyo_kanto | HINET | N.KI2H | 100 Hz | Primary (fallback: IU.MAJO) |
| istanbul_marmara | IU | ANTO | 40 Hz | Ankara, 300km proxy |
| turkey_kahramanmaras | IU | ANTO | 40 Hz | Shared proxy |
| campi_flegrei | IV | CAFE | 100 Hz | Direct station |

### THD Baseline Statistics

**Source**: `monitoring/data/baselines/thd_baselines_20260112.json`

| Station | Mean THD | Std THD | n_samples | Quality |
|---------|----------|---------|-----------|---------|
| IU.TUC | 0.3407 | 0.0400 | 31 | acceptable |
| IU.MAJO | 0.3088 | 0.0302 | 31 | acceptable |
| IU.COR | ~0.32 | ~0.04 | 31 | acceptable |
| BK.BKS | ~0.31 | ~0.03 | 31 | acceptable |

**Note**: A THD of 0.377 at IU.MAJO (Tokyo) gives z-score = (0.377 - 0.309) / 0.030 = 2.27, which maps to ELEVATED risk ~0.57.

---

## Per-Region Calibration Data

### Tokyo Kanto (Key Region for Recent Discrepancy)

| Parameter | Value | Source File |
|-----------|-------|-------------|
| **Lambda_geo** | | |
| Baseline mean | 0.0176 | lambda_geo_baselines.json |
| Baseline std | 0.0175 | lambda_geo_baselines.json |
| Stations | 41 | lambda_geo_baselines.json |
| GPS Source | NGL GEONET (G-series) | lambda_geo_baselines.json |
| **THD** | | |
| Primary station | HINET N.KI2H (100Hz) | backtest_config.yaml |
| Fallback station | IU.MAJO (40Hz) | backtest_config.yaml |
| Baseline mean | 0.3088 | thd_baselines_20260112.json |
| Baseline std | 0.0302 | thd_baselines_20260112.json |
| **Region Bounds** | | |
| Lat range | 34.5° - 36.5° | regions.py |
| Lon range | 138.5° - 141.0° | regions.py |

### Sensitivity Calculation for Tokyo Kanto

To trigger **WATCH** (risk > 0.25):

| Method | Required Raw Value | Required Change |
|--------|-------------------|-----------------|
| Lambda_geo | ratio > 1.5x → Lambda > 0.027 | +0.009 from baseline |
| THD | z > 1.0 → THD > 0.339 | +0.030 from baseline |
| Fault Corr | L2/L1 < 0.25 | Significant decorrelation |

To trigger **ELEVATED** (risk > 0.50):

| Method | Required Raw Value | Required Change |
|--------|-------------------|-----------------|
| Lambda_geo | ratio > 2.5x → Lambda > 0.044 | +0.026 from baseline |
| THD | z > 2.0 → THD > 0.369 | +0.060 from baseline |
| Fault Corr | L2/L1 < 0.15 | Severe decorrelation |

---

## Raw vs Calibrated Data Separation

### Raw Data (NEVER Modify for Calibration)

These files contain actual measurements and should not be changed:

```
monitoring/data/baselines/lambda_geo_baselines.json  ← GPS station raw statistics
monitoring/data/baselines/thd_baselines_*.json       ← Seismic station raw statistics
monitoring/data/ensemble_results/ensemble_*.json     ← Daily raw computations
```

### Calibration Parameters (SAFE to Modify)

These files contain thresholds and can be adjusted:

```
monitoring/src/ensemble.py                           ← Risk conversion functions
  - lambda_geo_to_risk() lines 141-176
  - fault_correlation_to_risk() lines 179-208
  - thd_to_risk_with_baseline() lines 234-280
  - RISK_TIERS dict lines 117-123

monitoring/config/backtest_config.yaml               ← Evaluation thresholds
  - min_magnitude: 6.0  ← Change to 4.5 for M4.5+ detection
  - hit_min_tier: "WATCH"
```

### Derived Data (Will Be Regenerated After Calibration Change)

```
monitoring/data/ensemble_results/daily_states.csv    ← Historical tiers (REGENERATE)
monitoring/dashboard/data.csv                        ← Dashboard display (REGENERATE)
docs/data.csv                                        ← Public dashboard (REGENERATE)
```

---

## Sensitivity Analysis

### Current Thresholds vs M4.5 Detection

Based on the calibration events (all M6.8+), the current thresholds are tuned for large events:

| Magnitude Range | Expected Lambda_geo Ratio | Expected THD z-score | Detection Likely? |
|-----------------|---------------------------|----------------------|-------------------|
| M7.0+ | 2.0x - 5.6x | z > 2.0 | Yes (WATCH to CRITICAL) |
| M6.0 - M6.9 | 1.5x - 2.5x | z = 1.0 - 2.0 | Possible (WATCH) |
| M5.0 - M5.9 | 1.0x - 1.5x | z = 0.5 - 1.5 | Unlikely (NORMAL) |
| M4.5 - M4.9 | 0.8x - 1.2x | z < 1.0 | No (below threshold) |

**Conclusion**: The current calibration is likely too conservative for M4.5-M5.9 events.

### Options for Recalibration

1. **Lower Lambda_geo thresholds**:
   - Current: WATCH at 1.5x
   - Option: WATCH at 1.2x (may increase false alarms)

2. **Lower THD z-score thresholds**:
   - Current: WATCH at z > 1.0
   - Option: WATCH at z > 0.5 (may increase false alarms)

3. **Reduce Tier gating**:
   - Current: Require 2+ methods for ELEVATED
   - Option: Allow single method to reach ELEVATED (more sensitive, less precise)

4. **Region-specific calibration**:
   - Different thresholds for high-activity regions (Japan, Taiwan)
   - Lower thresholds where more frequent validation is possible

---

## Recalibration Procedure

### Step 1: Backup Current Calibration

```bash
cd C:/GeoSpec/geospec_sprint
cp monitoring/src/ensemble.py monitoring/src/ensemble.py.backup_20260122
cp monitoring/config/backtest_config.yaml monitoring/config/backtest_config.yaml.backup_20260122
```

### Step 2: Modify Thresholds

Edit `monitoring/src/ensemble.py`:
- Adjust `lambda_geo_to_risk()` thresholds
- Adjust `thd_to_risk_with_baseline()` z-score mapping

Edit `monitoring/config/backtest_config.yaml`:
- Change `min_magnitude: 6.0` to desired value

### Step 3: Regenerate Historical Data

```bash
cd C:/GeoSpec/geospec_sprint/monitoring
python -m src.run_ensemble_daily --date 2025-12-17  # Regenerate each date
# OR create a script to regenerate all dates
```

### Step 4: Rebuild Validation Track Record

```bash
cd C:/GeoSpec/geospec_sprint/monitoring
python -m src.validate_predictions --rebuild
```

### Step 5: Verify Against Known Events

Compare new predictions against known USGS M4.5+ events to validate improved sensitivity.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-22 | Initial calibration reference document |

---

*GeoSpec Project - mail.rjmathews@gmail.com*
