# Dashboard Data Architecture

**Version**: 1.0
**Date**: January 23, 2026
**Status**: PROPOSED

---

## Executive Summary

This document defines a unified data architecture for the GeoSpec dashboard to ensure all components display consistent earthquake event information from a single source of truth.

---

## Problem Statement

### Current State (Inconsistent)

| Dashboard Component | Data Source | Update Mechanism | Issues |
|---------------------|-------------|------------------|--------|
| Region Cards (events) | `ensemble_latest.json` → `earthquake_events` | Python daily run | Missing regions (hualien, kaikoura, anchorage, kumamoto) |
| 30-Day Chart (M6.5+ markers) | Live USGS API fetch (JavaScript) | Real-time browser fetch | Different bounds, may fail silently |
| Backtest Charts (events) | `backtest_timeseries.json` | Manual | Static historical data |

### Identified Inconsistencies

1. **Bounding Box Mismatch**: Python (`earthquake_events.py`) and JavaScript (`index.html`) define different `REGION_BOUNDS`

   | Region | Python Bounds | JavaScript Bounds |
   |--------|---------------|-------------------|
   | hualien | `(23.0, 25.5, 120.5, 122.5)` | `[23.0, 25.0, 120.5, 122.5]` |
   | cascadia | `(42.0, 49.0, -130.0, -122.0)` | `[42.0, 49.0, -128.0, -122.0]` |

2. **Missing Regions in ensemble_latest.json**:
   - `hualien`, `kaikoura`, `anchorage`, `kumamoto` not fetched during daily run

3. **Duplicate USGS Fetches**:
   - Python fetches M4+ during daily run → cards
   - JavaScript fetches M6.5+ at page load → chart markers
   - Same API, different results

4. **Silent Failures**: JavaScript USGS fetch may fail due to CORS or network issues without user notification

---

## Unified Architecture (Proposed)

### Principle: Single Source of Truth

**All earthquake event displays pull from `ensemble_latest.json`** - no live JavaScript USGS fetches.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED DATA FLOW                                     │
│                                                                          │
│  Daily Ensemble Run (Python)                                             │
│         │                                                                │
│         ├──► fetch_earthquake_events() for ALL monitored regions         │
│         │    (Uses REGION_BOUNDS from earthquake_events.py)              │
│         │                                                                │
│         ├──► Filter: M4+ for cards, flag M6.5+ for chart markers         │
│         │                                                                │
│         └──► Store in ensemble_latest.json:                              │
│                {                                                         │
│                  "earthquake_events": {                                  │
│                    "hualien": {                                          │
│                      "events": [...],                                    │
│                      "m65_plus_events": [...],  ← NEW: For chart markers │
│                      "largest_event": {...},                             │
│                      "most_recent_event": {...}                          │
│                    },                                                    │
│                    ...                                                   │
│                  }                                                       │
│                }                                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DASHBOARD (JavaScript)                                │
│                                                                          │
│  loadData() reads ensemble_latest.json                                   │
│         │                                                                │
│         ├──► renderCards() uses earthquake_events[region]                │
│         │    • Shows largest_event, most_recent_event                    │
│         │    • Shows event_count (M4+ in 90d)                            │
│         │                                                                │
│         └──► renderChart() uses earthquake_events[region].m65_plus_events│
│              • Draws vertical lines for M6.5+ events                     │
│              • NO live USGS fetch needed                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Benefits

1. **Consistency**: All displays use same data source with identical bounding boxes
2. **Reliability**: No browser-side API calls that may fail silently
3. **Auditability**: All event data logged in JSON files
4. **Performance**: Single fetch during daily run, not per-page-load

---

## Implementation Plan

### Phase 1: Update Python Data Generation

**File**: `monitoring/src/earthquake_events.py`

1. Ensure ALL monitored regions are in `REGION_BOUNDS` (add missing ones)
2. Add `m65_plus_events` field to filter M6.5+ events for chart markers

**File**: `monitoring/src/run_ensemble_daily.py`

1. Ensure `fetch_earthquake_events()` is called for ALL regions in `REGIONS` dict
2. Add `m65_plus_events` to output structure

### Phase 2: Update Dashboard JavaScript

**File**: `monitoring/dashboard/index.html`

1. Remove `fetchUSGSEvents()` function (live USGS fetch)
2. Remove `REGION_BOUNDS` JavaScript constant (use Python's via JSON)
3. Update `renderChart()` to read M6.5+ events from `ensemble_latest.json`
4. Keep `matchEventToRegion()` but use bounds from loaded JSON

### Phase 3: Synchronize Region Configuration

Create single source for region bounds:

**File**: `monitoring/config/region_bounds.json` (NEW)

```json
{
  "ridgecrest": {"minlat": 35.0, "maxlat": 36.5, "minlon": -118.5, "maxlon": -116.5},
  "hualien": {"minlat": 23.0, "maxlat": 25.5, "minlon": 120.5, "maxlon": 122.5},
  ...
}
```

Both Python and JavaScript read from this file.

---

## Region Configuration (Canonical)

The following regions should be monitored with these bounding boxes:

| Region Key | Min Lat | Max Lat | Min Lon | Max Lon | Notes |
|------------|---------|---------|---------|---------|-------|
| ridgecrest | 35.0 | 36.5 | -118.5 | -116.5 | Eastern CA Shear Zone |
| socal_saf_mojave | 34.0 | 36.5 | -118.5 | -115.5 | SAF Mojave segment |
| socal_saf_coachella | 33.0 | 34.5 | -117.0 | -115.0 | SAF Coachella segment |
| norcal_hayward | 37.0 | 38.5 | -123.0 | -121.5 | SF Bay faults |
| cascadia | 42.0 | 49.0 | -130.0 | -122.0 | Subduction zone |
| anchorage | 59.0 | 63.0 | -152.0 | -147.0 | Alaska |
| tokyo_kanto | 34.5 | 37.0 | 138.5 | 141.5 | Japan Trench |
| kumamoto | 31.5 | 34.0 | 129.5 | 132.0 | Kyushu |
| hualien | 23.0 | 25.5 | 120.5 | 122.5 | Taiwan east coast |
| kaikoura | -43.5 | -41.5 | 172.0 | 175.0 | NZ South Island |
| istanbul_marmara | 40.0 | 41.5 | 27.5 | 30.5 | Marmara Sea |
| turkey_kahramanmaras | 36.5 | 38.5 | 36.0 | 38.5 | East Anatolian |
| campi_flegrei | 40.5 | 41.0 | 13.8 | 14.5 | Naples caldera |

---

## Data Schema Updates

### ensemble_latest.json Structure

```json
{
  "date": "2026-01-23",
  "timestamp": "2026-01-23T06:00:00Z",
  "regions": { ... },
  "earthquake_events": {
    "hualien": {
      "region": "hualien",
      "bounds": {
        "minlat": 23.0,
        "maxlat": 25.5,
        "minlon": 120.5,
        "maxlon": 122.5
      },
      "last_updated": "2026-01-23T06:00:00Z",
      "event_count": 26,
      "largest_event": {
        "event_id": "us7000rl2n",
        "time": "2025-12-27T09:25:55Z",
        "latitude": 24.6876,
        "longitude": 122.0478,
        "depth_km": 63,
        "magnitude": 6.6,
        "mag_type": "mww",
        "place": "30 km ESE of Yilan, Taiwan",
        "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000rl2n"
      },
      "most_recent_event": { ... },
      "events": [ ... ],
      "m65_plus_events": [
        {
          "event_id": "us7000rl2n",
          "time": "2025-12-27",
          "magnitude": 6.6,
          "place": "30 km ESE of Yilan, Taiwan"
        }
      ]
    }
  }
}
```

### Dashboard CSV (data.csv)

No changes required - continues to store daily risk scores.

---

## Deployment Notes

**IMPORTANT**: GitHub Pages serves from the `docs/` folder, NOT `monitoring/dashboard/`.

After making changes to dashboard files:
```powershell
# Sync files from development to deployment folder
cp monitoring/dashboard/index.html docs/index.html
cp monitoring/dashboard/ensemble_latest.json docs/ensemble_latest.json
cp monitoring/dashboard/data.csv docs/data.csv

# Commit and push
git add docs/
git commit -m "Sync dashboard to docs/"
git push
```

## Migration Checklist

- [x] Archive current data (DONE: `archive_2026-01-23/`)
- [x] Update `earthquake_events.py` to include all regions
- [x] Add `m65_plus_events` field to Python output
- [x] Update `run_ensemble_daily.py` to fetch events for all REGIONS
- [x] Remove JavaScript `fetchUSGSEvents()` function
- [x] Update JavaScript `renderChart()` to use JSON data
- [x] Test dashboard with new architecture
- [x] Update METHOD_DOCUMENT.md with new architecture
- [x] Sync files to docs/ folder for GitHub Pages

---

## Rollback Plan

If issues arise:
1. Restore from `archive_2026-01-23/` directory
2. Revert `index.html` to archived version
3. Re-run daily ensemble to regenerate JSON

---

*Document created: January 23, 2026*
*Author: GeoSpec Development*
