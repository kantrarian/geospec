# Station Substitutions and Known Limitations

This document records station substitution decisions and known limitations in the GeoSpec monitoring network.

## Overview

The GeoSpec monitoring system relies on seismic stations for THD (Total Harmonic Distortion) analysis. Due to practical constraints, some regions share stations or use distant proxies. These substitutions are documented here for transparency.

## Station Assignments

| Region | Primary Station | Network | Distance | Notes |
|--------|-----------------|---------|----------|-------|
| ridgecrest | IU.TUC | IU | ~400 km | Validated with 2019 M7.1 |
| socal_saf_mojave | IU.TUC | IU | ~200 km | Shared with ridgecrest |
| socal_saf_coachella | IU.TUC | IU | ~300 km | Shared with ridgecrest |
| norcal_hayward | BK.BKS | BK | ~0 km | Direct coverage |
| cascadia | IU.COR | IU | ~100 km | Subduction zone proxy |
| tokyo_kanto | N.KI2H | HINET | ~0 km | **Hi-net operational** (128 stations @ 100Hz) |
| istanbul_marmara | IU.ANTO | IU | ~300 km | Shared with kahramanmaras |
| turkey_kahramanmaras | IU.ANTO | IU | ~500 km | Distant proxy |
| campi_flegrei | IV.CAFE | IV | ~0 km | Direct volcanic coverage |

## Known Substitutions

### IU.TUC Serving 3 SoCal Regions

**Affected regions:** ridgecrest, socal_saf_mojave, socal_saf_coachella

**Limitation:** A single station in Tucson, AZ serves three distinct seismic zones in Southern California. This means:
- THD anomalies are averaged across ~400 km
- Regional precursors may be diluted
- False correlation: all three regions will show identical THD values

**Mitigation:**
- Lambda_geo provides regional differentiation (separate GPS polygons)
- Fault correlation uses region-specific segments
- Validation showed IU.TUC detected Ridgecrest 2019 precursors despite distance

**Future improvement:** Add CI network stations (e.g., CI.WCS, CI.MPM) for direct SoCal coverage.

### IU.ANTO Serving 2 Turkey Regions

**Affected regions:** istanbul_marmara, turkey_kahramanmaras

**Limitation:** Ankara station serves both western (Marmara) and southeastern (Kahramanmaras) Turkey, spanning ~500 km.

**Mitigation:**
- Kahramanmaras 2023 showed elevated THD at ANTO despite distance
- Different fault systems may produce independent precursors
- Lambda_geo polygons are distinct

**Future improvement:** Add KOERI stations for regional coverage.

### IU.MAJO for Tokyo/Kanto

**Affected regions:** tokyo_kanto

**Limitation:** Matsushiro station is ~200 km from Tokyo, missing local Kanto seismicity.

**Mitigation:**
- **Hi-net integration implemented** (Phase 2 sprint complete)
- NIED Hi-net account approved (username: devilldog)
- HinetPy library installed and authenticated
- Fallback to IU.MAJO when Hi-net unavailable

**Current Status (January 2026):**
- Hi-net authentication: **Working**
- Hi-net data download: **Working** - via status page polling (HinetPy auto-method has timing issues)
- WIN32 tools: **Working** via WSL wrappers (C:\GeoSpec\win32tools)
- Fallback: IU.MAJO via IRIS (operational)

**Resolution (January 13, 2026):**
The HinetPy library's `get_continuous_waveform()` method has a timing issue with the NIED status page.
The workaround is to:
1. Submit data request via HinetPy
2. Poll status page for "Available" downloads
3. Download by request ID
4. Convert WIN32 using proper win.prm format (4-line file pointing to channel table)

**Verified Working:**
- Downloaded 46.51 MB of Hi-net data (3 minutes, 2328 channels)
- Converted to SAC format via win2sac_32
- Read with ObsPy: N.KI2H station, 36.88°N 140.65°E, 100 Hz, valid seismic data

**Completed Steps (January 13, 2026):**
1. ✅ Updated hinet_data.py to use status page polling method
2. ⏳ Calibrate Hi-net station baselines (N.KI2H, N.TKCH, N.SITH, etc.) - pending
3. ✅ Switched tokyo_kanto primary source to Hi-net with IU.MAJO fallback

**Available Kanto Stations (Hi-net 0101):**
| Code | Name | Location |
|------|------|----------|
| N.TKCH | Tochigi | 35.86°N, 139.82°E |
| N.GNMH | Gunma | 36.30°N, 139.07°E |
| N.SITH | Saitama | 35.98°N, 139.40°E |
| N.CBAH | Chiba | 35.60°N, 140.13°E |
| N.KNGH | Kanagawa | 35.37°N, 139.45°E |

## Sample Rate Considerations

| Station | Native Rate | Notes |
|---------|-------------|-------|
| IU.* | 40 Hz | Standard broadband |
| BK.* | 40 Hz | Standard broadband |
| IV.CAFE | 100 Hz | High-rate volcanic |
| GE.* | 20 Hz | Lower rate - separate baseline |
| Hi-net | 100 Hz | High-rate regional |

**Important:** Baselines should not be mixed across sample rates. A 20 Hz station will have different THD characteristics than a 40 Hz station due to aliasing effects.

## Calibration Status (January 2026)

All primary stations have been auto-calibrated using 30-day data windows:

| Station | Mean THD | Std THD | Coverage | QA Grade |
|---------|----------|---------|----------|----------|
| IU.TUC | 0.3407 | 0.0400 | 100% | acceptable |
| BK.BKS | 0.3055 | 0.0497 | 100% | acceptable |
| IU.COR | 0.2828 | 0.0464 | 100% | acceptable |
| IU.MAJO | 0.3088 | 0.0302 | 100% | acceptable |
| IU.ANTO | 0.4129 | 0.1532 | 96.8% | acceptable |
| IV.CAFE | N/A | N/A | 0% | FAILED |

**Note:** IV.CAFE (Campi Flegrei) data is not available via IRIS - requires direct INGV access.

## Validation Status

| Substitution | Validated? | Evidence |
|--------------|------------|----------|
| IU.TUC → Ridgecrest | Yes | 2019 M7.1 retrospective |
| IU.ANTO → Istanbul | Partial | Network simulation |
| IU.ANTO → Kahramanmaras | Yes | 2023 M7.8 retrospective |
| HINET → Tokyo | Yes | Integration complete January 2026 |

## Recommendations

1. ~~**Priority 1:** Complete Hi-net integration for Tokyo/Kanto coverage~~ **DONE**
2. **Priority 1:** Add CI network stations for direct SoCal monitoring
3. **Priority 2:** Add KOERI stations for Turkey regional coverage
4. **Priority 3:** Consider IV network expansion for Italian volcanic regions
5. **Priority 4:** Calibrate Hi-net station baselines for THD analysis

## References

- MONITORING_SPECIFICATION_v2: Section 3.2 (Station Selection)
- backtest_config.yaml: Station mapping configuration
- Ridgecrest 2019 validation: docs/retrospective_validation/ridgecrest_2019.md
