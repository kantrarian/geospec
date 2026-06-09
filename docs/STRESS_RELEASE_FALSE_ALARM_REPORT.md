# Stress-Release Drop Detector — False-Alarm Analysis

- **Source:** CSV tier-proxy (C:\geospec\docs\data.csv)
- **Window:** 2025-10-18 → 2026-06-07 (233 days, 3106 region-days)
- **Detector params:** min_elevated_days=3, min_delta_z=1.5, sync_filter=True

> ⚠️ **Tier-proxy run.** docs/data.csv has no z-scores, so the detector uses its tier-fallback delta_z path (which skips the delta_z gate). This is an **UPPER BOUND** on the z-score-driven production detector. Re-run with `--ensemble-dir` on the production box for the faithful number. Pre-2025 validated events are outside this window.

## Firing rate

- **Total firings:** 74
- **By confidence:** high=1, moderate=73
- **Firing rate:** 8.70 per region-year (target ≈ 1/region/year)

## ⚠️ Methodological caveat: region-local scoring vs. a teleseismic hypothesis

This scorer marks a drop a HIT only if an event fell **inside the dropping region's own bounds**. But the stress-release hypothesis is *teleseismic*: the motivating case is the western-Pacific monitors (Hualien, Tokyo Kanto, Kumamoto) signalling the M7.8 **Mindanao** rupture ~2000 km away. Under region-local scoring that very event — the finding's headline success — scores as a **false alarm** (see the 2026-06-05 tokyo_kanto / kumamoto rows). So the false-alarm rate below is inflated by the scoring geometry on top of the tier-proxy permissiveness. A faithful analysis needs a monitor→tectonic-domain association rule (e.g. a correlation group maps to a plate-boundary magnitude/forward window), which is a modelling decision for the hypothesis owner — deliberately NOT invented here. Treat the number below as a loose upper bound under the strictest (local-only) association.

## Hit / false-alarm split (region-local association)

- **Scoring:** event M≥6.0 within 14d, region bounds +1.0° buffer
- **Hits:** 2
- **False alarms:** 72
- **False-alarm rate:** 97.3%  (precision 2.7%)

### Per-firing detail

| region | drop_date | conf | class | matched event |
|---|---|---|---|---|
| tokyo_kanto | 2025-11-02 | moderate | false_alarm | — |
| kumamoto | 2025-11-02 | moderate | false_alarm | — |
| anchorage | 2025-11-17 | moderate | hit | M6.0 12 km WNW of Susitna, Alaska (2025-11-27) |
| kaikoura | 2025-11-26 | moderate | false_alarm | — |
| kaikoura | 2025-12-09 | moderate | false_alarm | — |
| mexico_guerrero | 2025-12-26 | moderate | hit | M6.5 0 km W of San Marcos, Mexico (2026-01-02) |
| tokyo_kanto | 2025-12-29 | moderate | false_alarm | — |
| kumamoto | 2025-12-29 | moderate | false_alarm | — |
| hualien | 2025-12-29 | moderate | false_alarm | — |
| tokyo_kanto | 2026-01-12 | moderate | false_alarm | — |
| kumamoto | 2026-01-12 | moderate | false_alarm | — |
| ridgecrest | 2026-01-22 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-01-22 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-01-22 | moderate | false_alarm | — |
| cascadia | 2026-01-22 | moderate | false_alarm | — |
| norcal_hayward | 2026-01-26 | moderate | false_alarm | — |
| tokyo_kanto | 2026-01-28 | moderate | false_alarm | — |
| kumamoto | 2026-01-28 | moderate | false_alarm | — |
| anchorage | 2026-02-03 | moderate | false_alarm | — |
| norcal_hayward | 2026-02-05 | moderate | false_alarm | — |
| cascadia | 2026-02-05 | moderate | false_alarm | — |
| ridgecrest | 2026-02-07 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-02-07 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-02-07 | moderate | false_alarm | — |
| kaikoura | 2026-02-07 | moderate | false_alarm | — |
| norcal_hayward | 2026-02-17 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-02-19 | moderate | false_alarm | — |
| ridgecrest | 2026-02-20 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-02-20 | moderate | false_alarm | — |
| cascadia | 2026-02-20 | moderate | false_alarm | — |
| anchorage | 2026-02-23 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-03-04 | moderate | false_alarm | — |
| norcal_hayward | 2026-03-05 | moderate | false_alarm | — |
| ridgecrest | 2026-03-06 | high | false_alarm | — |
| cascadia | 2026-03-07 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-03-08 | moderate | false_alarm | — |
| anchorage | 2026-03-08 | moderate | false_alarm | — |
| tokyo_kanto | 2026-03-12 | moderate | false_alarm | — |
| kumamoto | 2026-03-12 | moderate | false_alarm | — |
| ridgecrest | 2026-03-18 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-03-18 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-03-18 | moderate | false_alarm | — |
| norcal_hayward | 2026-03-21 | moderate | false_alarm | — |
| cascadia | 2026-03-21 | moderate | false_alarm | — |
| anchorage | 2026-03-22 | moderate | false_alarm | — |
| ridgecrest | 2026-03-25 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-03-25 | moderate | false_alarm | — |
| tokyo_kanto | 2026-03-27 | moderate | false_alarm | — |
| kumamoto | 2026-03-27 | moderate | false_alarm | — |
| ridgecrest | 2026-03-30 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-03-30 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-03-30 | moderate | false_alarm | — |
| cascadia | 2026-04-01 | moderate | false_alarm | — |
| anchorage | 2026-04-04 | moderate | false_alarm | — |
| tokyo_kanto | 2026-04-10 | moderate | false_alarm | — |
| kumamoto | 2026-04-10 | moderate | false_alarm | — |
| ridgecrest | 2026-04-15 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-04-15 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-04-15 | moderate | false_alarm | — |
| cascadia | 2026-04-15 | moderate | false_alarm | — |
| ridgecrest | 2026-04-22 | moderate | false_alarm | — |
| socal_saf_mojave | 2026-04-22 | moderate | false_alarm | — |
| socal_saf_coachella | 2026-04-22 | moderate | false_alarm | — |
| tokyo_kanto | 2026-04-25 | moderate | false_alarm | — |
| kumamoto | 2026-04-25 | moderate | false_alarm | — |
| anchorage | 2026-05-01 | moderate | false_alarm | — |
| tokyo_kanto | 2026-05-10 | moderate | false_alarm | — |
| kumamoto | 2026-05-10 | moderate | false_alarm | — |
| hualien | 2026-05-10 | moderate | false_alarm | — |
| hualien | 2026-05-21 | moderate | false_alarm | — |
| tokyo_kanto | 2026-05-22 | moderate | false_alarm | — |
| kumamoto | 2026-05-22 | moderate | false_alarm | — |
| tokyo_kanto | 2026-06-05 | moderate | false_alarm | — |
| kumamoto | 2026-06-05 | moderate | false_alarm | — |
