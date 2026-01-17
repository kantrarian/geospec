# GeoSpec Ensemble JSON Output Schema

This document defines the JSON output format for the GeoSpec monitoring ensemble.
The primary output file is `ensemble_latest.json`, updated daily.

## Version

Schema version: 2.0.0
Last updated: January 2026

## Top-Level Structure

```json
{
  "generated": "2026-01-12T10:00:00",
  "date": "2026-01-10",
  "version": "2.0.0",
  "regions": {
    "<region_key>": <RegionResult>,
    ...
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `generated` | ISO8601 string | Timestamp when this file was created |
| `date` | YYYY-MM-DD string | Target assessment date |
| `version` | string | Schema version |
| `regions` | object | Map of region keys to RegionResult objects |

---

## RegionResult

Each monitored region has a result object:

```json
{
  "region": "ridgecrest",
  "date": "2026-01-10T00:00:00",
  "combined_risk": 0.35,
  "tier": 1,
  "tier_name": "WATCH",
  "confidence": 0.75,
  "agreement": "mostly_elevated",
  "methods_available": 3,
  "notes": "...",
  "coverage": <CoverageInfo>,
  "effective_weights": {...},
  "components": {
    "lambda_geo": <MethodResult>,
    "fault_correlation": <MethodResult>,
    "seismic_thd": <MethodResult>
  }
}
```

### Risk Tiers

| Tier | Name | Risk Range | Color | Meaning |
|------|------|------------|-------|---------|
| 0 | NORMAL | 0.00 - 0.25 | green | Baseline activity |
| 1 | WATCH | 0.25 - 0.50 | yellow | Above normal, monitoring |
| 2 | ELEVATED | 0.50 - 0.75 | orange | Significant anomaly |
| 3 | CRITICAL | 0.75 - 1.00 | red | Strong anomaly cluster |
| -1 | DEGRADED | N/A | gray | Insufficient data |

### Agreement Types

| Agreement | Description |
|-----------|-------------|
| `all_critical` | All available methods show critical levels |
| `all_elevated` | All available methods show elevated levels |
| `all_normal` | All available methods show normal levels |
| `mostly_elevated` | Most methods elevated, some normal |
| `mixed` | Methods disagree significantly |
| `single_method` | Only one method available |
| `no_data` | No methods produced valid results |

---

## MethodResult (Generic)

Base structure for all method results:

```json
{
  "name": "method_name",
  "available": true,
  "raw_value": 0.123,
  "raw_secondary": 0.456,
  "risk_score": 0.35,
  "is_elevated": false,
  "is_critical": false,
  "notes": "Human-readable description"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Method identifier: `lambda_geo`, `fault_correlation`, `seismic_thd` |
| `available` | bool | Whether this method produced a valid result |
| `raw_value` | float | Primary metric value (method-specific) |
| `raw_secondary` | float? | Secondary metric if applicable |
| `risk_score` | float | Normalized risk score 0.0-1.0 |
| `is_elevated` | bool | True if risk_score >= 0.50 |
| `is_critical` | bool | True if risk_score >= 0.75 |
| `notes` | string | Human-readable diagnostic string |

---

## Method-Specific Fields

### Lambda_geo (GPS Strain)

```json
{
  "name": "lambda_geo",
  "raw_value": 15.3,        // Baseline ratio (1.0 = normal)
  "notes": "ratio=15.3x"
}
```

**raw_value interpretation:**
- `< 3×`: NORMAL - typical background variation
- `3-10×`: WATCH - above normal strain rotation
- `10-100×`: ELEVATED - significant anomaly
- `> 100×`: CRITICAL - extreme strain (historical precursors: 485×-7999×)

### Fault Correlation (Seismic Decorrelation)

```json
{
  "name": "fault_correlation",
  "raw_value": 0.15,          // L2/L1 eigenvalue ratio
  "raw_secondary": 1.8,       // Participation ratio
  "notes": "L2/L1=0.1500, PR=1.80, 3/4 segments"
}
```

**raw_value (L2/L1) interpretation:**
- `> 0.30`: NORMAL - segments correlated
- `0.10-0.30`: ELEVATED - partial decorrelation
- `< 0.10`: CRITICAL - strong decorrelation (approaching failure)

### Seismic THD (Harmonic Distortion)

The THD method includes additional baseline context fields for diagnostics:

```json
{
  "name": "seismic_thd",
  "raw_value": 0.18,           // THD value (dimensionless)
  "raw_secondary": 12.5,       // Signal-to-noise ratio
  "risk_score": 0.72,
  "notes": "sta=IU.TUC, THD=0.1800, z=5.20, baseline_mean=0.0680, baseline_std=0.0215, n=85, rate=40Hz",

  // Structured baseline context (v2.0.0+)
  "baseline": {
    "mean": 0.068,
    "std": 0.0215,
    "n_samples": 85,
    "window": "2025-10-15 to 2026-01-05",
    "quality": "good"
  },
  "z_score": 5.2,
  "sample_rate_hz": 40.0
}
```

#### Baseline Object

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Median THD during calibration period |
| `std` | float | MAD-based standard deviation (MAD × 1.4826) |
| `n_samples` | int | Number of valid days in calibration window |
| `window` | string | Calibration date range |
| `quality` | string | QA grade: `good`, `acceptable`, `poor`, `missing` |

#### Baseline Quality Grades

| Grade | N Samples | Interpretation |
|-------|-----------|----------------|
| `good` | ≥ 60 | Production-ready baseline |
| `acceptable` | 30-59 | Usable, monitor for drift |
| `poor` | < 30 | Preliminary, high uncertainty |
| `missing` | 0 | No baseline available (using absolute thresholds) |

#### Z-Score Interpretation

The z-score measures standard deviations above baseline:

| Z-Score | Risk | Interpretation |
|---------|------|----------------|
| < 0 | 0.00-0.10 | Below baseline (normal) |
| 0-1 | 0.10-0.25 | Normal variation |
| 1-2 | 0.25-0.50 | WATCH - elevated |
| 2-3 | 0.50-0.75 | ELEVATED - significant anomaly |
| > 3 | 0.75-1.00 | CRITICAL - strong anomaly |

---

## Coverage Information

```json
{
  "coverage": {
    "segments_defined": 4,
    "segments_working": 3,
    "segment_names": ["seg1", "seg2", "seg3"],
    "coverage_pct": 75.0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `segments_defined` | int | Total fault segments defined for region |
| `segments_working` | int | Segments with sufficient data |
| `segment_names` | string[] | Names of working segments |
| `coverage_pct` | float | Percentage of defined segments working |

---

## Effective Weights

After renormalization (when some methods unavailable):

```json
{
  "effective_weights": {
    "lambda_geo": 0.571,       // 0.4/(0.4+0.3) when THD unavailable
    "fault_correlation": 0.429,
    "seismic_thd": 0.0
  }
}
```

---

## Example Complete Output

```json
{
  "generated": "2026-01-12T10:30:00",
  "date": "2026-01-10",
  "version": "2.0.0",
  "regions": {
    "ridgecrest": {
      "region": "ridgecrest",
      "date": "2026-01-10T00:00:00",
      "combined_risk": 0.42,
      "tier": 1,
      "tier_name": "WATCH",
      "confidence": 0.75,
      "agreement": "mostly_elevated",
      "methods_available": 3,
      "notes": "",
      "coverage": {
        "segments_defined": 4,
        "segments_working": 3,
        "segment_names": ["ridgecrest_main", "little_lake", "searles_valley"],
        "coverage_pct": 75.0
      },
      "effective_weights": {
        "lambda_geo": 0.40,
        "fault_correlation": 0.30,
        "seismic_thd": 0.30
      },
      "components": {
        "lambda_geo": {
          "name": "lambda_geo",
          "available": true,
          "raw_value": 5.2,
          "risk_score": 0.24,
          "is_elevated": false,
          "is_critical": false,
          "notes": "ratio=5.2x"
        },
        "fault_correlation": {
          "name": "fault_correlation",
          "available": true,
          "raw_value": 0.22,
          "raw_secondary": 2.1,
          "risk_score": 0.45,
          "is_elevated": false,
          "is_critical": false,
          "notes": "L2/L1=0.2200, PR=2.10, 3/4 segments"
        },
        "seismic_thd": {
          "name": "seismic_thd",
          "available": true,
          "raw_value": 0.32,
          "raw_secondary": 15.2,
          "risk_score": 0.68,
          "is_elevated": true,
          "is_critical": false,
          "notes": "sta=IU.TUC, THD=0.3200, z=3.10, baseline_mean=0.2000, baseline_std=0.0387, n=85, rate=40Hz",
          "baseline": {
            "mean": 0.2,
            "std": 0.0387,
            "n_samples": 85,
            "window": "2025-10-15 to 2026-01-05",
            "quality": "good"
          },
          "z_score": 3.1,
          "sample_rate_hz": 40.0
        }
      }
    }
  }
}
```

---

## Pipeline Configuration Reference

The THD pipeline parameters are locked in `monitoring/config/pipeline_lockfile.json`.
Baselines are calibrated using `monitoring/src/calibrate_thd_baselines.py`.

Key parameters:
- Fundamental frequency: M2 tidal (12.42h period)
- Harmonics analyzed: 2f through 6f
- Window: 24 hours
- Sample rate: Decimated to 1 Hz
- Statistics: Median + MAD (robust to outliers)
