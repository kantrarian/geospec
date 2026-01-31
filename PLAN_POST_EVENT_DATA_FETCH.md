# Plan: Fetch Historical Lambda_geo and FC Data for Post-Event Periods

**Created**: 2026-01-22
**Author**: R.J. Mathews / Claude
**Status**: IMPLEMENTED - Scripts created and verified

## Objective

Extend the historical backtest timeseries to include 3 days of post-event data for all three methods:
- Lambda_geo (GPS strain)
- FC (Fault Correlation L2/L1)
- THD (Seismic Total Harmonic Distortion) - **Already completed**

## Current State

### THD Post-Event Data (COMPLETED)
- Script: `validation/fetch_historical_thd.py`
- Status: Extended to fetch 3 days post-event
- Data: Real z-scores from IRIS FDSN
- Result: Post-event entries added to `backtest_timeseries.json` with `post_event: true`

### Lambda_geo Post-Event Data (TO DO)
- Current script: `validation/backtest_real_data_only.py`
- Data source: NGL GPS .tenv3 archive files
- Current coverage: Pre-event only
- Post-event data: Available in NGL archive (GPS stations continue operating after events)

### FC Post-Event Data (TO DO)
- Current script: `validation/fetch_historical_fc.py`
- Data source: IRIS/SCEDC FDSN seismic waveforms
- Current coverage: Pre-event only (14 days lead window)
- Post-event data: Available via FDSN (continuous seismic recording)

---

## Implementation Plan

### Phase 1: Lambda_geo Post-Event Data

#### 1.1 Data Source Verification
- **Source**: Nevada Geodetic Laboratory (NGL) .tenv3 files
- **URL**: http://geodesy.unr.edu/gps_timeseries/tenv3/
- **Format**: Daily position solutions in IGS14/IGb14 reference frame
- **Availability**: Continuous daily solutions, including post-event periods

#### 1.2 Existing Pipeline (from `backtest_real_data_only.py`)
```
GPS .tenv3 Files → Parse Displacement Time Series
  → Delaunay Triangulation (3+ stations)
  → Velocity Gradient (7-day sliding window)
  → Strain Tensor E = (∇v + ∇vᵀ) / 2
  → Commutator [E, Ė] = E·Ė - Ė·E
  → Lambda_geo = ||[E, Ė]||_F (Frobenius norm)
  → Ratio = value / baseline_mean
```

#### 1.3 Modifications Required

**File**: `validation/fetch_historical_lambda_geo.py` (NEW)

Create a dedicated script that:
1. Reads event configuration from `HISTORICAL_EVENTS` dict
2. Downloads .tenv3 files for event region (if not cached)
3. Computes Lambda_geo for `lead_days + 3` days (extend window)
4. Outputs daily Lambda_geo ratios including post-event
5. Saves results to `validation/results/lg_historical/`

**Key Parameters**:
```python
HISTORICAL_EVENTS = {
    'ridgecrest_2019': {
        'date': datetime(2019, 7, 6, 3, 19, 53),
        'lead_days': 14,
        'post_days': 3,  # NEW: 3 days after event
        'gps_stations': ['CCCC', 'P595', 'P594', 'P580', 'P591'],
        'region_bounds': (35.5, 36.5, -118.5, -117.0),  # lat_min, lat_max, lon_min, lon_max
    },
    # ... other events
}
```

#### 1.4 GPS Station Requirements per Event

| Event | Stations | Data Directory | Post-Event Availability |
|-------|----------|----------------|------------------------|
| Ridgecrest 2019 | CCCC, P595, P594, P580, P591+ | `data/raw/ridgecrest_2019/gps/` | ✓ Continuous |
| Tohoku 2011 | GEONET stations | `data/raw/tohoku_2011/gps/` | ✓ GEONET archive |
| Turkey 2023 | CORS-TR network | `data/raw/turkey_2023/gps/` | ✓ IGS stations |
| Chile 2010 | CSN stations | `data/raw/chile_2010/gps/` | ⚠️ Limited pre-2010 |
| Morocco 2023 | IGS regional | `data/raw/morocco_2023/gps/` | ⚠️ Sparse |
| Kaikoura 2016 | GeoNet NZ | `data/raw/kaikoura_2016/gps/` | ✓ GeoNet archive |
| Anchorage 2018 | PBO Alaska | `data/raw/anchorage_2018/gps/` | ✓ PBO archive |
| Kumamoto 2016 | GEONET | `data/raw/kumamoto_2016/gps/` | ✓ GEONET archive |
| Hualien 2024 | CWB Taiwan | `data/raw/hualien_2024/gps/` | ✓ CWB archive |

#### 1.5 Output Format
```json
{
  "event_key": "ridgecrest_2019",
  "data_source": {
    "type": "NGL_GPS",
    "stations": ["CCCC", "P595", "P594"],
    "fetch_date": "2026-01-22T15:30:00"
  },
  "statistics": {
    "baseline_mean": 0.0012,
    "baseline_std": 0.0003,
    "peak_ratio": 6134.0,
    "peak_date": "2019-07-05"
  },
  "timeseries": [
    {"date": "2019-06-22", "lambda_geo": 0.0011, "ratio": 0.92},
    {"date": "2019-06-23", "lambda_geo": 0.0013, "ratio": 1.08},
    // ... through event date ...
    {"date": "2019-07-07", "lambda_geo": 0.0089, "ratio": 7.42, "post_event": true},
    {"date": "2019-07-08", "lambda_geo": 0.0045, "ratio": 3.75, "post_event": true},
    {"date": "2019-07-09", "lambda_geo": 0.0023, "ratio": 1.92, "post_event": true}
  ]
}
```

---

### Phase 2: FC (Fault Correlation) Post-Event Data

#### 2.1 Data Source Verification
- **Source**: IRIS/SCEDC/NCEDC FDSN Web Services
- **Format**: miniSEED continuous waveforms (BHZ channel, 20-40 Hz)
- **Availability**: Continuous recording, including post-event periods

#### 2.2 Existing Pipeline (from `fetch_historical_fc.py`)
```
Event Configuration → Fetch Waveforms (FDSN)
  → Bandpass Filter (0.01-1.0 Hz)
  → Hilbert Transform (envelope)
  → Cross-Correlation Matrix (station pairs)
  → SVD Decomposition → [λ₁, λ₂, ...]
  → L2/L1 = λ₂ / λ₁
  → Tier Classification (NORMAL/ELEVATED/CRITICAL)
```

#### 2.3 Modifications Required

**File**: `validation/fetch_historical_fc.py` (MODIFY)

Update existing script to:
1. Extend `end_time` from `event_date` to `event_date + 3 days`
2. Add `post_days` parameter to event configuration
3. Mark post-event entries with `post_event: true`

**Code Change**:
```python
# Current (line ~180):
end_time = event_date + timedelta(hours=12)

# Modified:
post_days = event.get('post_days', 3)
end_time = event_date + timedelta(days=post_days)
```

#### 2.4 Station Pairs per Event

| Event | Fault Segment | Station Pairs | Network |
|-------|--------------|---------------|---------|
| Ridgecrest 2019 | Main rupture | CI.WBS-CI.SLA-CI.CCC | SCEDC |
| Ridgecrest 2019 | Garlock junction | CI.CLC-CI.JRC2 | SCEDC |
| Tohoku 2011 | Japan Trench | IU.MAJO-II.ERM | IRIS |
| Turkey 2023 | East Anatolian | IU.ANTO-GE.ISP | IRIS/GEOFON |
| Kaikoura 2016 | Alpine Fault | NZ.SNZO-NZ.WEL | GeoNet |
| Kumamoto 2016 | Futagawa Fault | IU.MAJO-II.ERM | IRIS |
| Hualien 2024 | Longitudinal Valley | IU.TATO-TW.NACB | IRIS/CWB |

**Note**: Chile 2010 and Morocco 2023 have insufficient station coverage for FC.

#### 2.5 Output Format
```json
{
  "event_key": "ridgecrest_2019",
  "fault_segments": [
    {
      "name": "ridgecrest_main",
      "stations": ["CI.WBS", "CI.SLA", "CI.CCC"],
      "timeseries": [
        {"date": "2019-06-22", "l2l1": 0.234, "tier": "NORMAL"},
        // ... through event ...
        {"date": "2019-07-07", "l2l1": 0.021, "tier": "CRITICAL", "post_event": true},
        {"date": "2019-07-08", "l2l1": 0.045, "tier": "CRITICAL", "post_event": true},
        {"date": "2019-07-09", "l2l1": 0.089, "tier": "ELEVATED", "post_event": true}
      ]
    }
  ]
}
```

---

### Phase 3: Integration into Backtest Timeseries

#### 3.1 Update Script: `validation/update_backtest_all_methods.py` (NEW)

Create unified script that:
1. Loads THD results from `validation/results/thd_historical/`
2. Loads Lambda_geo results from `validation/results/lg_historical/`
3. Loads FC results from `validation/results/fc_historical/`
4. Merges all three into `docs/backtest_timeseries.json`
5. Adds post-event entries with all available methods

#### 3.2 Final Backtest Entry Format
```json
{
  "date": "2019-07-07",
  "tier": null,
  "tier_name": "POST_EVENT",
  "risk": null,
  "lg_ratio": 7.42,
  "fc_l2l1": 0.021,
  "thd": 3.45,
  "post_event": true
}
```

---

## Execution Steps

### Step 1: Create Lambda_geo Fetch Script
```bash
# Create new script
validation/fetch_historical_lambda_geo.py

# Run for all events
python validation/fetch_historical_lambda_geo.py --all
```

### Step 2: Modify FC Fetch Script
```bash
# Edit existing script to extend post-event window
validation/fetch_historical_fc.py

# Re-run for all events
python validation/fetch_historical_fc.py --all
```

### Step 3: Create Integration Script
```bash
# Create unified update script
validation/update_backtest_all_methods.py

# Run to merge all methods
python validation/update_backtest_all_methods.py
```

### Step 4: Verify Results
```bash
# Check backtest_timeseries.json for post-event entries
python -c "
import json
with open('docs/backtest_timeseries.json') as f:
    data = json.load(f)
for event_key, event in data['events'].items():
    post = [e for e in event['timeseries'] if e.get('post_event')]
    print(f'{event_key}: {len(post)} post-event entries')
"
```

---

## Data Integrity Requirements

Per CLAUDE.md rules:

1. **Lambda_geo**: Must compute from actual NGL GPS .tenv3 files
2. **FC**: Must compute from actual IRIS/SCEDC seismic waveforms
3. **THD**: Already using real IRIS data ✓
4. **If data unavailable**: Mark as `null`, do NOT fabricate values
5. **Document gaps**: Note which events have incomplete post-event data

---

## Expected Outcomes

| Event | LG Post-Event | FC Post-Event | THD Post-Event |
|-------|--------------|---------------|----------------|
| Ridgecrest 2019 | ✓ Full | ✓ Full | ✓ Complete |
| Tohoku 2011 | ✓ Full | ⚠️ Limited | ✓ Complete |
| Turkey 2023 | ⚠️ Limited | ⚠️ Limited | ✓ Complete |
| Chile 2010 | ⚠️ Limited | ❌ None | ✓ Complete |
| Morocco 2023 | ⚠️ Limited | ❌ None | ✓ Complete |
| Kaikoura 2016 | ✓ Full | ✓ Full | ✓ Complete |
| Anchorage 2018 | ✓ Full | ⚠️ Limited | ✓ Complete |
| Kumamoto 2016 | ✓ Full | ✓ Full | ✓ Complete |
| Hualien 2024 | ✓ Full | ✓ Full | ✓ Complete |

---

## Timeline Estimate

| Phase | Task | Complexity |
|-------|------|------------|
| 1.1 | Create Lambda_geo fetch script | Medium |
| 1.2 | Download/cache GPS data | Low (automated) |
| 1.3 | Run Lambda_geo computation | Low (automated) |
| 2.1 | Modify FC fetch script | Low |
| 2.2 | Re-run FC computation | Low (automated) |
| 3.1 | Create integration script | Medium |
| 3.2 | Merge and verify | Low |

---

## Implementation Status

- [x] Plan reviewed and approved
- [ ] Data sources verified accessible (requires GPS data download)
- [x] Scripts created and tested
  - `validation/fetch_historical_lambda_geo.py` - NEW
  - `validation/fetch_historical_fc.py` - MODIFIED (added post_days, new events)
  - `validation/update_backtest_all_methods.py` - NEW
- [ ] Results validated against known patterns (requires data fetch)
- [x] Documentation updated
- [ ] Changes committed to repository

---

## References

1. `validation/backtest_real_data_only.py` - Existing Lambda_geo computation
2. `validation/fetch_historical_fc.py` - Existing FC computation
3. `validation/fetch_historical_thd.py` - THD computation (reference implementation)
4. `docs/BACKTEST_DATA_SOURCES.md` - Data source documentation
5. `docs/METHOD_DOCUMENT.md` - Technical methodology
