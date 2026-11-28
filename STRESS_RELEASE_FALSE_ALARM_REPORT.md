# Stress-Release Drop Detector — False-Alarm Analysis

- **Source:** real ensemble dir data/ensemble_results
- **Window:** 2025-10-18 → 2026-06-07 (233 days, 233 region-days)
- **Detector params:** min_elevated_days=3, min_delta_z=1.5, sync_filter=True

## Firing rate

- **Total firings:** 34
- **By confidence:** high=27, moderate=7
- **Firing rate:** 53.26 per region-year (target ≈ 1/region/year)

## Hit / false-alarm split (domain-aware association)

Drops are scored with the tectonic-domain association rule (grassmann 2026-06-09): a drop in a correlation group is a HIT if a qualifying event occurred **anywhere on that group's plate-boundary domain** within the domain window. This captures the teleseismic case — the western-Pacific monitors (Hualien, Tokyo Kanto, Kumamoto) → the M7.8 **Mindanao** rupture ~2000 km away, which region-local scoring scored as a false alarm. Solo monitors keep region-local + buffer scoring.

- **western_pacific** {hualien, kumamoto, tokyo_kanto}: M≥6.5 within 7d, bounds lat[5.0, 50.0] lon[120.0, 155.0]
- **cascadia_norcal** {cascadia, norcal_hayward}: M≥6.0 within 7d, bounds lat[35.0, 52.0] lon[-132.0, -118.0]
- **socal** {ridgecrest, socal_saf_coachella, socal_saf_mojave}: M≥6.0 within 7d, bounds lat[30.0, 37.0] lon[-122.0, -114.0]
- **solo monitors:** M≥6.0 within 14d, region bounds +1.0° buffer

- **Hits:** 5
- **False alarms:** 29
- **False-alarm rate:** 85.3%  (precision 14.7%)

### Per-firing detail

| region | drop_date | conf | assoc | class | matched event |
|---|---|---|---|---|---|
| anchorage | 2025-11-17 | high | solo | hit | M6.0 12 km WNW of Susitna, Alaska (2025-11-27) |
| kaikoura | 2025-11-26 | moderate | solo | false_alarm | — |
| kaikoura | 2025-12-09 | moderate | solo | false_alarm | — |
| tokyo_kanto | 2025-12-29 | high | western_pacific | false_alarm | — |
| kumamoto | 2025-12-29 | high | western_pacific | false_alarm | — |
| hualien | 2025-12-29 | high | western_pacific | false_alarm | — |
| tokyo_kanto | 2026-01-12 | high | western_pacific | false_alarm | — |
| kumamoto | 2026-01-12 | high | western_pacific | false_alarm | — |
| ridgecrest | 2026-01-22 | high | socal | false_alarm | — |
| socal_saf_mojave | 2026-01-22 | high | socal | false_alarm | — |
| socal_saf_coachella | 2026-01-22 | high | socal | false_alarm | — |
| cascadia | 2026-01-22 | high | cascadia_norcal | false_alarm | — |
| tokyo_kanto | 2026-01-28 | high | western_pacific | false_alarm | — |
| kumamoto | 2026-01-28 | high | western_pacific | false_alarm | — |
| ridgecrest | 2026-02-07 | high | socal | false_alarm | — |
| socal_saf_mojave | 2026-02-07 | high | socal | false_alarm | — |
| socal_saf_coachella | 2026-02-07 | high | socal | false_alarm | — |
| ridgecrest | 2026-03-06 | high | socal | false_alarm | — |
| anchorage | 2026-03-08 | moderate | solo | false_alarm | — |
| tokyo_kanto | 2026-03-12 | high | western_pacific | false_alarm | — |
| kumamoto | 2026-03-12 | high | western_pacific | false_alarm | — |
| ridgecrest | 2026-03-25 | moderate | socal | false_alarm | — |
| socal_saf_mojave | 2026-03-25 | moderate | socal | false_alarm | — |
| tokyo_kanto | 2026-03-27 | high | western_pacific | false_alarm | — |
| kumamoto | 2026-03-27 | high | western_pacific | false_alarm | — |
| tokyo_kanto | 2026-04-10 | high | western_pacific | false_alarm | — |
| kumamoto | 2026-04-10 | high | western_pacific | false_alarm | — |
| tokyo_kanto | 2026-04-25 | high | western_pacific | false_alarm | — |
| kumamoto | 2026-04-25 | high | western_pacific | false_alarm | — |
| anchorage | 2026-05-01 | high | solo | false_alarm | — |
| tokyo_kanto | 2026-05-10 | moderate | western_pacific | hit | M6.7 44 km ESE of Ōfunato, Japan (2026-05-15) |
| kumamoto | 2026-05-10 | moderate | western_pacific | hit | M6.7 44 km ESE of Ōfunato, Japan (2026-05-15) |
| tokyo_kanto | 2026-06-05 | high | western_pacific | hit | M7.8 26 km SW of Kablalan, Philippines (2026-06-07) |
| kumamoto | 2026-06-05 | high | western_pacific | hit | M7.8 26 km SW of Kablalan, Philippines (2026-06-07) |
