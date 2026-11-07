# Dashboard Data Discrepancy Investigation & Fix Plan

*Created: January 16, 2026*

## Overview
Investigation and fix for three data discrepancies in the GeoSpec monitoring dashboard.

## Issues Identified

### Issue 1: Kaikoura & Anchorage Flatline After Jan 4
**Symptom**: Risk values show constant pattern (0.3166 and 0.6531) after January 4, 2026.

**Root Cause**: Only Lambda_geo method available. Seismic THD stations returning "Insufficient data":
- Kaikoura: NZ.HSES station - no seismic waveform data
- Anchorage: AK.SSL station - no seismic waveform data

**Evidence** (data.csv lines 74-101):
```
2026-01-04,kaikoura,1,0.3166,0.50,1,single_method
2026-01-05,kaikoura,1,0.3166,0.50,1,single_method  # Identical
...repeats through Jan 16 with same values
```

### Issue 2: Original Regions Missing Data Before Jan 11
**Symptom**: The 9 original regions (ridgecrest, socal_saf_mojave, etc.) only appear from Jan 11 onward.

**Root Cause**: These regions were added to monitoring on Jan 11. No historical backfill performed.

**Evidence** (data.csv structure):
- Lines 2-101: Only kaikoura, anchorage, kumamoto, hualien (Dec 17 - Jan 10)
- Lines 102+: All 13 regions appear (Jan 11+)

### Issue 3: CSV Values Don't Match Chart Display
**Root Cause**: Incomplete data coverage causes chart interpolation artifacts.

## Implementation Plan

### Step 1: Add Fallback Seismic Stations
**File**: `geospec_sprint/monitoring/src/regions.py`

Add backup stations for regions with data gaps:
```python
'kaikoura': {
    'seismic_stations': ['NZ.HSES', 'NZ.WEL', 'NZ.BKZ'],  # Add fallbacks
},
'anchorage': {
    'seismic_stations': ['AK.SSL', 'AK.COLA', 'AK.BMR'],  # Add fallbacks
}
```

### Step 2: Backfill All 13 Regions (Dec 17 - Jan 10)
```powershell
cd C:\GeoSpec\geospec_sprint
$dates = @('2025-12-17','2025-12-18','2025-12-19','2025-12-20','2025-12-21',
           '2025-12-22','2025-12-23','2025-12-24','2025-12-25','2025-12-26',
           '2025-12-27','2025-12-28','2025-12-29','2025-12-30','2025-12-31',
           '2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05',
           '2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-10')

foreach ($date in $dates) {
    .venv\Scripts\python.exe -m src.run_ensemble_daily --date $date --all
}
```

### Step 3: Regenerate Dashboard CSV
```powershell
cd C:\GeoSpec\geospec_sprint\monitoring
..\..\.venv\Scripts\python.exe generate_dashboard_csv.py
```

### Step 4: Deploy Updated Dashboard
```powershell
Copy-Item monitoring\dashboard\data.csv docs\data.csv
git add docs/data.csv
git commit -m "Update dashboard with complete backfill data"
git push
```

## Files to Modify

| File | Change |
|------|--------|
| `monitoring/src/regions.py` | Add fallback seismic stations |
| `monitoring/data/ensemble_*.json` | Regenerate via backfill |
| `monitoring/dashboard/data.csv` | Regenerate after backfill |
| `docs/data.csv` | Copy updated CSV |

## Verification

1. **Check flatline resolved**: Kaikoura/Anchorage show varying values
2. **Check full coverage**: All 13 regions present Dec 17 - Jan 16
3. **Check methods column**: Shows >1 where seismic data available
4. **Visual check**: Chart displays match CSV values

## Expected Outcome

After fix:
- 30-day chart shows all 13 regions with complete data
- No flatline patterns (or documented as "seismic unavailable")
- CSV row count: ~390 rows (13 regions x 30 days)

## Notes

- Backfill will take time (~25 dates x 13 regions)
- Some flatlines may persist if seismic stations truly unavailable
- Consider caching seismic data to speed future runs
