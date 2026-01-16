# Legacy Synthetic Backtests - ARCHIVED

**Date Archived**: January 14, 2026
**Reason**: These files contain HARDCODED "literature-derived" Lambda_geo time series values that do not represent actual GPS computations.

## Why These Were Archived

These backtest scripts used **hardcoded Lambda_geo time series** that claimed to represent earthquake precursor signals:

```python
# Example from backtest_tohoku_gps.py (lines 46-67)
LAMBDA_GEO_TIMESERIES = {
    '2011-02-20T12:00:00': 0.8,
    '2011-02-22T12:00:00': 1.2,
    # ... exponentially increasing values ...
    '2011-03-11T00:00:00': 7234.8,  # Pre-mainshock "signal"
}
```

**These values were NOT computed from real GPS data.** They were manually constructed to demonstrate expected behavior patterns based on literature analysis. This created "Artifactual Confidence" - claiming an 80% detection rate based on synthetic signals.

## The Problem

| Metric | Literature-Derived Claim | Real GPS Computation |
|--------|-------------------------|---------------------|
| Tohoku 2011 Lambda_geo | 7,234x baseline | 5.6x baseline |
| Hit Rate | 80% (4/5 events) | Requires recalibration |
| Data Source | Hardcoded values | Computed from .tenv3 files |

## What Replaced These

1. **Real Data Backtest**: `validation/backtest_real_data_only.py` computes Lambda_geo from actual NGL GPS station data (321 .tenv3 files across 10 earthquakes).

2. **Region-Specific Baselines**: `monitoring/data/baselines/lambda_geo_baselines.json` contains 90-day baselines computed from real NGL GPS data.

3. **Corrected Documentation**: `METHOD_DOCUMENT.md` now distinguishes "Theoretical Validation" (literature) from "Operational Validation" (real data).

## Files Archived

- `backtest_tohoku_gps.py` - Tohoku 2011 M9.0 (hardcoded Lambda_geo)
- `backtest_turkey_gps.py` - Turkey 2023 M7.8 (hardcoded Lambda_geo)
- `backtest_chile_gps.py` - Chile 2010 M8.8 (hardcoded Lambda_geo)
- `backtest_morocco_gps.py` - Morocco 2023 M6.8 (hardcoded Lambda_geo)

## Data Integrity Reference

Per CLAUDE.md:
> **NEVER use simulated, synthetic, or "literature-derived" data when real data is required.**

These archived files violated this principle by claiming validation results from synthetic data.

---

*Archived by: Claude / R.J. Mathews*
*January 14, 2026*
