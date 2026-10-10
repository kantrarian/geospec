# Method amendment R3 — rolling lambda_geo baseline recalibration (registered 2026-07-29)

**Status:** registered and effective **prospectively** from the first recalibrated run after
2026-07-29. Companion to amendment R2 (`AMENDMENT_2026-07-29_scoring_window.md`).

## The change

| | before | after (R3) |
|---|---|---|
| baseline calibration | manual one-shot (`calibrate_lambda_geo_baselines.py` run by hand; static thereafter) | **automatic weekly** recalibration in the daily job |
| baseline window | 90 days ending at the (arbitrary) manual run date | 90 days ending **30 days before today** (rolling, lagged) |

## Why

1. **Staleness.** The baseline was computed once and never refreshed; the live `lambda_geo` ratio was
   being measured against a months-old (winter) reference, so the ratio carried the full seasonal drift
   of the signal. Audit of the daily series showed the region-level ratios inheriting a strong seasonal
   component (in the 15-year Kumamoto reference series, July's climatological median is ~3.3× the
   annual median — a naive fixed threshold is ~8× easier to exceed in July than in January).
2. **The 30-day lag is a safety property, not a detail.** A rolling baseline that included the most
   recent days would absorb any genuine slow precursory buildup into its own reference and suppress the
   very signal the system monitors. The window `[today−120d, today−30d]` keeps the reference current
   while guaranteeing the last 30 days can never dilute themselves.

## Honest limitations, stated at registration

- A lagged 90-day rolling window **reduces but does not remove** seasonal bias (it tracks the seasonal
  cycle with roughly a 2–3-month phase lag). A full **seasonal reference** (month-matched climatology or
  rain-gauge-regressed residual) is planned as a separate amendment once enough multi-year live data
  exists to build it in the live system's own units; it will be registered before use like this one.
- **The risk series will show a step discontinuity** at the first recalibrated run (values before/after
  are measured against different baselines). Cross-baseline comparisons of raw ratio values are not
  meaningful across that boundary; tier/alarm statistics restart their comparability there. The
  prospective scoring record (R2) is unaffected in its rules, but alarm *rates* before vs after R3 are
  not directly comparable.
- Recalibration failures fall back to the existing baseline file (fail-open to stale rather than
  fail-closed to no monitoring), logged in the run output.

*Registered by the project maintainers, 2026-07-29. Append-only; further method changes get new dated
amendment files.*
