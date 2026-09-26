# Backtest Validation: Next Steps for Paper-Grade Results

**Status:** Current validation is directionally sound but requires additional work for publication.
**Date:** January 2026

## Current State Summary

| Metric | Value | Target | Status | Notes |
|--------|-------|--------|--------|-------|
| Hit Rate | 80% (4/5) | ≥60% | ✓ PASS | Wide CI: [38%, 96%] due to n=5 |
| Precision | 100% | ≥30% | ✓ PASS | Event-centric (no null windows) |
| FAR | 0.00/year | ≤1.0 | ✓ PASS | Inflated by event-centric design |
| Time in Warning | 44.6% | ≤15% | ✗ FAIL | Expected for event windows only |
| Lead Time | 6.3 days | 3-10d | ✓ PASS | Range: 4.1-7.7 days |

**Key Limitation:** Current metrics are computed over event-centric windows only, not continuous monitoring. This inflates time-in-warning and makes FAR artificially low.

---

## Priority 1: True Null-Heavy Run

### Why It Matters
- Current FAR/time-in-warning are dominated by event windows
- Need continuous daily replay to get realistic "year-round" metrics
- Will reveal true false positive rate

### Action Items
1. **Run continuous daily replay** over 2019-2020 for Ridgecrest region
   - Use `monitoring/src/backtest.py` with `--start 2019-01-01 --end 2020-12-31`
   - This will produce daily tier states for ~730 days
   - Score against USGS catalog for M≥6 events in region

2. **Compute true metrics:**
   - Time-in-warning: (days at WATCH+) / (total days)
   - FAR: (ELEVATED+ days with no event within 14d forward) / (total days)
   - Expected: time-in-warning << 15% for null-heavy period

3. **Required data:**
   - THD baselines for 2019-2020 (already have for Ridgecrest)
   - Lambda_geo requires GPS data (may need to fetch from NGL)
   - Fault correlation requires seismic cache (limited availability)

### Expected Outcome
- Time-in-warning: ~5-10% (vs current 44.6%)
- FAR: ~0.5-1.5/year (may fail if too sensitive)
- This will be a true test of operational viability

---

## Priority 2: Add More Events (Especially M6-7)

### Why It Matters
- n=5 gives wide confidence intervals
- All current events are M6.8-9.0 (high end)
- Need M6-7 to establish detection floor

### Candidate Events (with NGL GPS coverage)

| Event | Mag | Date | Region | GPS Stations | Seismic |
|-------|-----|------|--------|--------------|---------|
| Napa 2014 | M6.0 | 2014-08-24 | NorCal | ~15 | TBD |
| Searles Valley 2019 | M6.4 | 2019-07-04 | Ridgecrest | 14 | YES |
| Petrolia 2021 | M6.2 | 2021-12-20 | NorCal | ~10 | TBD |
| Ferndale 2022 | M6.4 | 2022-12-20 | NorCal | ~10 | TBD |

### Action Items
1. Run data inventory for candidate events
2. Download GPS data from NGL
3. Run Lambda_geo validation (GPS-only initially)
4. With 8-10 events, CIs narrow significantly

---

## Priority 3: Expand Seismic Coverage

### Current State
- THD: Only Ridgecrest (16 days cached)
- Fault Correlation: Only Ridgecrest
- Lambda_geo: All 5 events

### Why It Matters
- Ensemble claim requires all 3 methods
- Single-event seismic validation is insufficient

### Action Items
1. **Cache seismic data for Japan (Tohoku region)**
   - Hi-net integration already complete (N.KI2H)
   - Can backfill March 2011 data if available

2. **Cache seismic data for Turkey (East Anatolian)**
   - IU.ANTO station available via IRIS
   - Backfill January-February 2023

3. **Cache seismic data for NorCal**
   - BK.BKS station (baseline already calibrated)
   - Enable validation of additional California events

---

## Priority 4: Train/Test Split Protocol

### Why It Matters
- Addresses "retrospective leakage" concerns
- Pre-registration prevents threshold tuning to test set
- Essential for publication

### Proposed Split
| Set | Events | Purpose |
|-----|--------|---------|
| Training | Ridgecrest 2019, Chile 2010 | Tune thresholds, weights |
| Test | Tohoku 2011, Turkey 2023, Morocco 2023 | Blind evaluation |

### Protocol
1. Lock all thresholds using training set only
2. Run test set without looking at results
3. Report test set metrics as primary finding
4. Training set metrics are secondary

### Documentation
- Record threshold values before test run
- Git commit hash as timestamp
- JSON lockfile for reproducibility

---

## Priority 5: Station-Density Gating

### Why It Matters
- Morocco failure shows sparse networks → false confidence
- System should refuse to make predictions when data is inadequate
- Improves precision, prevents "crying wolf"

### Proposed Rule
```python
MINIMUM_STATION_REQUIREMENTS = {
    'lambda_geo': {
        'min_stations': 4,
        'max_distance_km': 200,
        'min_triangle_quality': 0.2,
    },
    'thd': {
        'min_stations': 1,
        'max_distance_km': 400,
    },
    'fault_correlation': {
        'min_segments': 2,
        'min_waveform_days': 7,
    }
}
```

### Implementation
1. Add geometry check before Lambda_geo computation
2. Return `available=False` with reason if inadequate
3. Ensemble gracefully degrades to available methods
4. Dashboard shows "insufficient coverage" warning

### Expected Impact
- Morocco would show "Lambda_geo: insufficient coverage" instead of weak detection
- Prevents false confidence from sparse networks
- Improves time-in-warning (no alerts in inadequate regions)

---

## Priority 6: Uncertainty Quantification

### Current State
- Bootstrap CI: [40%, 100%]
- Wilson CI: [38%, 96%]
- Very wide due to n=5

### Target State
- n≥10 events: CI width ~±15%
- Per-method uncertainty propagation
- Ensemble confidence reflects data quality

### Action Items
1. Add more events (Priority 2)
2. Propagate method-level uncertainty to ensemble
3. Report confidence alongside tier

---

## Timeline Suggestion

| Priority | Effort | Impact | Dependency |
|----------|--------|--------|------------|
| P5: Station-density gating | 1 day | High | None |
| P2: More M6-7 events | 2-3 days | High | GPS data |
| P1: Null-heavy run | 2-3 days | Critical | THD baseline |
| P4: Train/test split | 1 day | High | P2 |
| P3: Expand seismic | 1 week | Medium | Data caching |
| P6: Uncertainty | 2-3 days | Medium | P2 |

---

## Files to Update

When implementing these priorities:

1. `monitoring/src/ensemble.py` - Add station-density gating
2. `monitoring/src/backtest.py` - Enable continuous daily replay
3. `validation/acceptance_criteria.yaml` - Add train/test split config
4. `docs/CALIBRATION_BACKTEST_REPORT.md` - Update with new results

---

*This document captures the path from current "directionally correct" validation to publication-ready backtesting.*
