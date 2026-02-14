# Investigation: Dashboard Display Issues

**Date**: 2026-01-22
**Reporter**: User observation
**Status**: ROOT CAUSES IDENTIFIED

---

## Issues Reported

1. **Hualien backtest chart**: Shows 3 post-event days but FC and THD lines stop at April 2nd (pre-event)
2. **Today's values**: Card and 30-day history chart didn't load at all

---

## Issue 1: Hualien Post-Event Data Not Displaying

### Root Cause: Y-Axis Range Clips Negative THD Values

**Location**: `monitoring/dashboard/index.html` line 1110

```javascript
yaxis: {
    gridcolor: '#2a2a3a',
    title: 'Risk / FC / THD',
    range: [0, 1.1],  // <-- PROBLEM: Clips negative values
    side: 'left'
},
```

**Data Analysis**:
```
Hualien post-event THD values in backtest_timeseries.json:
- 2024-04-04: thd = -0.27  (z-score BELOW baseline)
- 2024-04-05: thd = -0.32  (z-score BELOW baseline)
- 2024-04-06: thd = -0.30  (z-score BELOW baseline)

FC values:
- 2024-04-04: fc_l2l1 = null
- 2024-04-05: fc_l2l1 = null
- 2024-04-06: fc_l2l1 = null
```

**Why it doesn't show**:
- THD values ARE present in the JSON (verified: -0.27, -0.32, -0.30)
- Y-axis has fixed range [0, 1.1]
- Negative THD z-scores are CLIPPED because they fall below y=0
- FC is genuinely null (data unavailable from IRIS)

### Fix Required

Change line 1110 in index.html:
```javascript
// FROM:
range: [0, 1.1],

// TO (auto-scale to show negative values):
range: [-1, 1.1],
// OR use autorange:
autorange: true,
```

---

## Issue 2: Today's Values Not Loading

### Root Cause: data.csv Missing 2026-01-22

**Location**: `monitoring/dashboard/data.csv`

```
Last entries in data.csv:
2026-01-21,ridgecrest,1,0.3541,...
2026-01-21,hualien,0,0.0851,...
(No 2026-01-22 entries)
```

**Why it failed**:
- Dashboard expects today's date (2026-01-22) in data.csv
- data.csv only has data through 2026-01-21
- The daily monitoring script didn't run for today

### Data Sources Status

| File | Last Date | Status |
|------|-----------|--------|
| data.csv | 2026-01-21 | Missing today |
| ensemble_latest.json | 2026-01-12 | Stale (10 days old) |
| backtest_timeseries.json | N/A (static) | OK |

### Fix Required

Run the daily monitoring pipeline to generate 2026-01-22 data:
```powershell
cd C:\GeoSpec\geospec_sprint\monitoring
.\run_daily.ps1
```

Or manually update with current ensemble values.

---

## Summary of Fixes

| Issue | File | Fix |
|-------|------|-----|
| THD negative values clipped | index.html:1110 | Change `range: [0, 1.1]` to `range: [-1, 1.1]` or `autorange: true` |
| Missing today's data | data.csv | Run daily monitoring script |
| Stale ensemble_latest.json | ensemble_latest.json | Run ensemble update |

---

## Verification Steps

After fixes:
1. Refresh dashboard
2. Check Hualien chart - THD line should now show in post-event period (will be below x-axis for negative values)
3. Check region cards - should show today's values
4. Check 30-day history - should include today

---

## Related Files

- `monitoring/dashboard/index.html` - Chart rendering logic
- `monitoring/dashboard/data.csv` - 30-day history data
- `monitoring/dashboard/ensemble_latest.json` - Today's values
- `monitoring/dashboard/backtest_timeseries.json` - Historical event data
- `monitoring/run_daily.ps1` - Daily update script

---

*Investigation completed: 2026-01-22*
