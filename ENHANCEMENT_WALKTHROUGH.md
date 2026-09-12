# GeoSpec Enhancement Walkthrough

This document outlines the enhancements made to stabilize data acquisition and enable real-time monitoring for the GeoSpec earthquake prediction system.

## 1. Stabilized Data Acquisition (SCEDC/Seismic)

We addressed availability issues with the Southern California Earthquake Data Center (SCEDC) and other regional networks.

### New Components

**`scedc_diagnostic.py`**: A standalone tool to verify connectivity to SCEDC servers.
- Usage: `python geospec_sprint/scedc_diagnostic.py`
- Verifies both HTTP API and FDSN service status.

**`seismic_waveform_fetcher.py`**: A robust fetcher for Methods 2 (Fault Correlation) and 3 (Seismic THD).
- Feature: Automatic fallback logic. If SCEDC fails for a station, it tries IRIS, then NCEDC, etc.
- Feature: Priority mapping for international networks (TU, GE, JP).
- Usage:
```python
from seismic_waveform_fetcher import SeismicWaveformFetcher
fetcher = SeismicWaveformFetcher()
st = fetcher.get_waveforms("CI", "WBS", "*", "BHZ", t1, t2)
```

### Updates

**`data_integrator.py`**: Replaced placeholder logic for `_integrate_local_network_data` with a functional SCEDC Event Catalog fetcher using ObsPy.

## 2. Real-Time GPS (Lambda_geo)

We implemented the infrastructure for real-time strain analysis, moving beyond the 2-day latency of NGL Rapid products.

### New Components

**`ntrip_client.py`**: A pure Python NTRIP 2.0 Client.
- Connects to IGS-IP or other NTRIP casters to receive real-time GNSS streams.
- Eliminates dependency on external `ntripclient.exe` binaries.

### Updates

**`live_data_fetcher.py`**: Integrated a `RealTimeGNSSStreamer` class.
- Consumes valid NMEA/Solution streams from the NTRIP client.
- Yields live `(lat, lon, height)` tuples for immediate processing by Lambda_geo.
- Factory function `create_realtime_streamer_from_env()` loads credentials from `.env`

## 3. International & Backup Stations

We expanded the system's reach and resilience.

### Connectivity Verification
- **Japan (F-net/JMA)**: Verified accessible via IRIS (JP network). Added tokyo_kanto station list (e.g., TSKB, 3009).
- **GEOFON (GE)**: Verified accessible (e.g., GE.WLF).
- **Turkey (TU)**: Direct access to TU network proved restricted.
  - Solution: Confirmed IU.ANTO (Ankara) and GE stations as reliable fallbacks.

### Backup Configuration
Updated `live_data_fetcher.py` with station lists for Tokyo and Istanbul, enabling Lambda_geo monitoring for these critical regions.

## 4. Summary of Files Created/Modified

| File | Purpose |
|------|---------|
| `scedc_diagnostic.py` | SCEDC connectivity check |
| `src/seismic_waveform_fetcher.py` | Robust waveform acquisition with fallback |
| `src/ntrip_client.py` | Real-time NTRIP data streaming |
| `src/international_connectivity_test.py` | Verification of global network access |
| `src/data_integrator.py` | [Mod] Added SCEDC Event fetching |
| `src/live_data_fetcher.py` | [Mod] Added Real-Time streamer & Backup stations |

## 5. Configuration

### NTRIP Credentials (in `.env`)
```
IGS_NTRIP_USER=<username>
IGS_NTRIP_PASSWORD=<password>
IGS_NTRIP_CASTER=igs-ip.net
IGS_NTRIP_PORT=2101
```

## 6. Dashboard History Backfill (Completed)

**Problem**: Dashboard originally only showed Ridgecrest data because `run_daily.ps1` was running without the `--all` flag, and historical JSON files for other regions were missing.

**Action**:
1. Verified `run_ensemble_daily.py` supports generating data for all regions.
2. Manually triggered backfill for Jan 12, 13, and 14 using:
```bash
python -m src.run_ensemble_daily --date 2026-01-12
python -m src.run_ensemble_daily --date 2026-01-13
python -m src.run_ensemble_daily --date 2026-01-14
```
3. Regenerated `data.csv` using `generate_dashboard_csv.py`.

**Result**: `data.csv` now contains 36+ rows covering Jan 11-14 for all 9 configured regions.

## 7. Historical Expansion Verification

We successfully integrated and validated four new historical events:

| Region | Event | Date | Detection Result |
|--------|-------|------|------------------|
| New Zealand | Kaikoura (M7.8) | 2016-11-13 | Critical (0.83) |
| Alaska | Anchorage (M7.1) | 2018-11-30 | Elevated (0.32) |
| Japan | Kumamoto (M7.0) | 2016-04-16 | Watch (0.48) |
| Taiwan | Hualien (M7.4) | 2024-04-03 | Precursor (0.56) |

### Key Findings
- **Taiwan (Hualien)** provided the strongest validation, with a clear 3-day precursor ramp-up.
- **Deep Events (Alaska)** showed delayed and attenuated surface signals, requiring lower baselines.
- **High Sensitivity**: All events were detectable using the Lambda_geo metric, validating the codebase expansion.

### Regional Risks (Jan 14):
- Ridgecrest: NORMAL (Risk 0.22)
- SoCal Mojave: NORMAL (Risk 0.17)
- NorCal Hayward: NORMAL (Risk 0.08)
- Campi Flegrei: WATCH (Risk 0.34, elevated Seismic THD)

## 8. Deployment and Version Control (Completed)

- **Deployment**: Copied `index.html` and `data.csv` to `docs/` directory to enable GitHub Pages deployment.
- **Version Control**: All source code now tracked in git (previously excluded by `.gitignore`).
- **Dashboard URL**: https://kantrarian.github.io/geospec/

## 9. Backtest Data Integration

Historical FC (Fault Correlation) and THD (Total Harmonic Distortion) data integrated into `backtest_timeseries.json`:

| Event | Methods Available |
|-------|-------------------|
| Ridgecrest 2019 | LG, THD, FC (3/3) |
| Tohoku 2011 | LG, THD, FC (3/3) |
| Turkey 2023 | LG, THD, FC (3/3) |
| Chile 2010 | LG, THD (2/3) |
| Morocco 2023 | LG, THD (2/3) |

---

*Last Updated: January 15, 2026*
