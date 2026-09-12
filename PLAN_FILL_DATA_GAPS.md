# Plan: Fill Historical Backtest Data Gaps

**Created**: 2026-01-22
**Author**: R.J. Mathews / Claude
**Status**: ✅ PARTIALLY COMPLETE (2026-01-22)

## Completion Summary

| Task | Status | Notes |
|------|--------|-------|
| Add Ridgecrest to THD fetch | ✅ DONE | Added to HISTORICAL_EVENTS dict |
| Fetch Ridgecrest THD | ✅ DONE | Peak THD 2.81 on Jul 4 (36h before event) |
| Merge Ridgecrest FC | ✅ DONE | Added to all_historical_fc.json |
| Re-fetch Ridgecrest FC post-event | ⚠️ BLOCKED | SCEDC stations not accessible from IRIS |
| Regenerate all_historical_thd.json | ✅ DONE | All 9 events now included |
| Update backtest_timeseries.json | ✅ DONE | THD post-event data filled |
| Hualien day 3 | ⚠️ INCOMPLETE | Window calculation ends at 04-05 |

### Ridgecrest Post-Event Now:
- LG: 3/3 ✅
- FC: 0/3 (SCEDC access issue - pre-event FC data preserved)
- THD: 2/3 ✅ (07-07: -0.73, 07-08: -0.76)

---

## Executive Summary

Analysis of `backtest_timeseries.json` reveals gaps in post-event data (3 days after each earthquake) for the FC (Fault Correlation) and THD (Total Harmonic Distortion) methods. The screenshot from the dashboard shows these gaps visually - missing orange (THD) and green (FC) lines in the post-event period.

---

## Gap Analysis

### Data Availability by Event

| Event | LG Status | FC Status | THD Status | Post Days |
|-------|-----------|-----------|------------|-----------|
| **Ridgecrest 2019** | ✓ Complete (3 days) | ❌ ALL NULL | ❌ ALL NULL | 3 |
| **Tohoku 2011** | Partial (day 1 null) | Partial (day 4 null) | Partial (days 1,4 null) | 4 |
| **Turkey 2023** | Partial (day 1 null) | Partial (day 4 null) | Partial (days 1,4 null) | 4 |
| **Chile 2010** | ✓ Complete | ❌ Expected null | Partial (day 3 null) | 3 |
| **Morocco 2023** | ✓ Complete | ❌ Expected null | ✓ Complete | 3 |
| **Kaikoura 2016** | Partial (day 1 null) | Partial (day 4 null) | Partial (days 1,4 null) | 4 |
| **Anchorage 2018** | ❌ ALL NULL (no NGL) | Partial (day 4 null) | Partial (day 1 null) | 4 |
| **Kumamoto 2016** | Partial (day 1 null) | Partial (day 4 null) | Partial (days 1,4 null) | 4 |
| **Hualien 2024** | ❌ ALL NULL (no NGL) | ❌ ALL NULL | Missing day 3! | 2 |

### Root Cause Analysis

#### 1. Ridgecrest FC/THD Missing (CRITICAL)
- **FC**: `ridgecrest_2019_fc.json` exists (fetched 2026-01-15) but was NOT merged into `all_historical_fc.json`
- **FC**: Ridgecrest FC data stops at 2019-07-05 (before event 2019-07-06) - no post_event flag
- **THD**: No `ridgecrest_2019_thd.json` file exists - THD fetch was never run for this event

#### 2. Event Day Nulls (Day 1 of post-event)
- Event day waveforms often saturated/clipped from mainshock
- THD/FC algorithms fail on corrupted data
- This is expected behavior, not a bug

#### 3. Day 4 Nulls
- Scripts configured with `post_days=3`, but window stepping means some days get partial/no coverage
- The 4 days shown in JSON are: event day + 3 following days, but day 4 often has incomplete windows

#### 4. Hualien Missing Day 3
- `backtest_timeseries.json` only has 2024-04-04 and 2024-04-05
- Missing 2024-04-06 entirely
- Need to re-run THD fetch and verify FC fetch includes this day

#### 5. Anchorage/Hualien No Lambda_geo
- NGL does not have GPS data for Alaska PBO stations or Taiwan stations
- This is a data availability limitation, not fixable

---

## Intermediate Result Files

### What Exists

**fc_historical/**
```
ridgecrest_2019_fc.json   <- Exists but NOT in all_historical_fc.json!
tohoku_2011_fc.json       <- In all_historical_fc.json
turkey_2023_fc.json       <- In all_historical_fc.json
all_historical_fc.json    <- Contains: tohoku, turkey, kaikoura, anchorage, kumamoto, hualien
```

**thd_historical/**
```
anchorage_2018_thd.json
hualien_2024_thd.json
kaikoura_2016_thd.json
kumamoto_2016_thd.json
all_historical_thd.json   <- Contains all 8 events EXCEPT ridgecrest!
```

**lg_historical/**
```
All 9 events present with individual files
all_historical_lg.json complete
```

### What's Missing

| File | Status | Action Needed |
|------|--------|---------------|
| `ridgecrest_2019_thd.json` | MISSING | Run THD fetch |
| `ridgecrest_2019` in `all_historical_fc.json` | MISSING | Re-run FC fetch with post_days or merge manually |
| `ridgecrest_2019` in `all_historical_thd.json` | MISSING | Add after THD fetch |
| Hualien 2024-04-06 | MISSING | Verify in source data, add to backtest |

---

## Implementation Plan

### Phase 1: Fix Ridgecrest Data (PRIORITY 1)

#### Step 1.1: Fetch Ridgecrest THD
```powershell
cd C:\GeoSpec\geospec_sprint
.venv\Scripts\Activate.ps1
python validation/fetch_historical_thd.py --event ridgecrest_2019
```

**Expected Output**: `validation/results/thd_historical/ridgecrest_2019_thd.json`

**Data Source**: IRIS FDSN
**Station**: CI.WBS or CI.CCC (same as FC)

#### Step 1.2: Re-fetch Ridgecrest FC with Post-Event Data
The existing ridgecrest_2019_fc.json was fetched on 2026-01-15 before post_days was added.

```powershell
python validation/fetch_historical_fc.py --event ridgecrest_2019
```

**Expected**: Updated file with `post_event: true` entries for 2019-07-07, 2019-07-08, 2019-07-09

#### Step 1.3: Merge Ridgecrest into All-Historical Files
Either:
a) Re-run `--all` to regenerate the all_historical files, OR
b) Manually merge ridgecrest into existing files

Option (a) is cleaner but takes longer. Option (b) preserves existing data.

### Phase 2: Fix Hualien Day 3 (PRIORITY 2)

#### Step 2.1: Verify THD Source Data
```powershell
python validation/fetch_historical_thd.py --event hualien_2024
```

Check if 2024-04-06 appears in the output.

#### Step 2.2: Verify FC Source Data
```powershell
python validation/fetch_historical_fc.py --event hualien_2024
```

Check if 2024-04-06 appears in the output.

#### Step 2.3: Update Backtest
If day 3 data exists in source files, run:
```powershell
python validation/update_backtest_all_methods.py
```

### Phase 3: Regenerate All-Historical Files (PRIORITY 3)

Run the complete fetch pipeline to ensure all data is captured:

```powershell
# Fetch all THD (includes Ridgecrest now)
python validation/fetch_historical_thd.py --all

# Fetch all FC (includes Ridgecrest with post-event)
python validation/fetch_historical_fc.py --all

# Lambda_geo already complete, but verify
python validation/fetch_historical_lambda_geo.py --all

# Merge everything
python validation/update_backtest_all_methods.py
```

### Phase 4: Verify Results

#### Step 4.1: Check Post-Event Coverage
```python
import json
with open('docs/backtest_timeseries.json') as f:
    data = json.load(f)

for event_key, event in data['events'].items():
    post = [e for e in event['timeseries'] if e.get('post_event')]
    lg = [e for e in post if e.get('lg_ratio') is not None]
    fc = [e for e in post if e.get('fc_l2l1') is not None]
    thd = [e for e in post if e.get('thd') is not None]
    print(f"{event_key}: {len(post)} days, LG={len(lg)}, FC={len(fc)}, THD={len(thd)}")
```

#### Step 4.2: Visual Verification
1. Push updated `backtest_timeseries.json` to GitHub Pages
2. Check https://kantrarian.github.io/geospec/
3. Verify Ridgecrest shows orange (THD) and green (FC) lines in post-event period

---

## Data Sources Reference

### THD Data Sources
| Event | Primary Station | Backup Station | Network |
|-------|-----------------|----------------|---------|
| Ridgecrest 2019 | CI.WBS | CI.CCC | SCEDC/IRIS |
| Tohoku 2011 | IU.MAJO | II.ERM | IRIS |
| Turkey 2023 | IU.ANTO | GE.ISP | IRIS |
| Chile 2010 | IU.LVC | II.LCO | IRIS |
| Morocco 2023 | IU.PAB | GE.MARJ | IRIS |
| Kaikoura 2016 | IU.SNZO | II.TAU | IRIS |
| Anchorage 2018 | IU.COLA | II.KDAK | IRIS |
| Kumamoto 2016 | IU.MAJO | II.ERM | IRIS |
| Hualien 2024 | IU.TATO | IU.GUMO | IRIS |

### FC Data Sources
| Event | Station Pairs | Network |
|-------|---------------|---------|
| Ridgecrest 2019 | CI.WBS-CI.SLA-CI.CCC | SCEDC |
| Tohoku 2011 | IU.MAJO-PS.TSK | IRIS |
| Turkey 2023 | IU.ANTO-GE.ISP-GE.CSS | IRIS/GEOFON |
| Kaikoura 2016 | IU.SNZO-II.TAU | IRIS |
| Anchorage 2018 | IU.COLA-II.KDAK | IRIS |
| Kumamoto 2016 | IU.MAJO-II.ERM | IRIS |
| Hualien 2024 | IU.TATO-TW.NACB | IRIS |

### Lambda_geo Data Sources
| Event | GPS Network | Status |
|-------|-------------|--------|
| Ridgecrest 2019 | PBO/NGL | ✓ Available |
| Tohoku 2011 | GEONET/NGL | ✓ Available |
| Turkey 2023 | CORS-TR/NGL | ✓ Available |
| Chile 2010 | CSN/NGL | ✓ Available |
| Morocco 2023 | IGS regional | ✓ Available |
| Kaikoura 2016 | GeoNet NZ | ✓ Available |
| Anchorage 2018 | PBO Alaska | ❌ Not in NGL |
| Kumamoto 2016 | GEONET | ✓ Available |
| Hualien 2024 | CWB Taiwan | ❌ Not in NGL |

---

## Data Integrity Safeguards

Per CLAUDE.md rules:

1. **NEVER fabricate data** - If IRIS returns no data for a date, mark as `null`
2. **Document gaps honestly** - Notes field should explain why data is missing
3. **Preserve existing data** - Always backup before overwriting
4. **Use real data sources only** - IRIS, SCEDC, NCEDC, NGL archive

### Backup Commands
```powershell
# Before any changes
cp docs/backtest_timeseries.json docs/backtest_timeseries.json.bak.20260122
```

---

## Expected Outcomes After Implementation

| Event | Post-Event Days | LG | FC | THD |
|-------|-----------------|----|----|-----|
| Ridgecrest 2019 | 3 | 3 | 3 | 3 |
| Tohoku 2011 | 3-4 | 2-3 | 2-3 | 2-3 |
| Turkey 2023 | 3-4 | 2-3 | 2-3 | 2-3 |
| Chile 2010 | 3 | 3 | 0 | 2-3 |
| Morocco 2023 | 3 | 3 | 0 | 3 |
| Kaikoura 2016 | 3-4 | 2-3 | 2-3 | 2-3 |
| Anchorage 2018 | 3-4 | 0 | 2-3 | 2-3 |
| Kumamoto 2016 | 3-4 | 2-3 | 2-3 | 2-3 |
| Hualien 2024 | 3 | 0 | 2-3 | 3 |

*Note: Day 1 (event day) often has null values due to waveform saturation - this is expected.*

---

## Checklist

- [ ] Backup existing backtest_timeseries.json
- [ ] Fetch Ridgecrest THD data
- [ ] Re-fetch Ridgecrest FC data with post_days
- [ ] Verify Hualien day 3 in source data
- [ ] Regenerate all_historical_thd.json
- [ ] Regenerate all_historical_fc.json
- [ ] Run update_backtest_all_methods.py
- [ ] Verify gaps filled in backtest_timeseries.json
- [ ] Push to GitHub Pages
- [ ] Visual verification on dashboard

---

*Last Updated: 2026-01-22*
