# Shared-support claim correction — 2026-09-17

**Research status: INCONCLUSIVE. Not an earthquake warning or forecasting service.**

This dated correction withdraws unsupported interpretations previously displayed
on the dashboard. It is an append-only incident record: original documents,
datasets and Git history remain available. Later resolutions should be added as
dated follow-ups, not substituted for this record.

## Claims withdrawn

The dashboard's assertions that regional correlations identify a single tectonic
unit, establish propagation at approximately 190 km/hr or LAIC/cascade
triggering, reject tidal aliasing, or demonstrate a 76% false-detection reduction
while preserving all validated precursors are **not supported as stated**.
The claimed pre-rupture stress-release mechanism and 1–3 day predictive lead time
are likewise not established. Correlation of region-labeled series does not by
itself establish independent measurements, causality or forecast skill.
This notice supersedes those interpretations wherever they remain in historical
research material; it does not assert that every observed correlation is an artifact.

## Publicly inspectable shared support

The public [`REGIONS` configuration in `run_ensemble_daily.py`](https://github.com/kantrarian/geospec/blob/e207b477b665ab56d4b7d69dbc4018cbd9502621/monitoring/src/run_ensemble_daily.py#L115)
declares the following THD station assignments:

| Station | Region labels using that support |
|---------|---------------------------------|
| IU.MAJO | `kumamoto`; `tokyo_kanto` when its IU.MAJO fallback is used (Tokyo's configured primary is Hi-net) |
| IU.TUC | `ridgecrest`, `socal_saf_mojave`, `socal_saf_coachella` |
| IU.ANTO | `istanbul_marmara`, `turkey_kahramanmaras` |

These are THD-component source relationships, not a claim that every component
or entire regional ensemble is identical. Actual support must be established
for each record, including fallback selection, time window and processing.
Copies or transformations of the same station record cannot count as independent
regional corroboration. A map of region names is not a map of independent sensors.

## Experimental drop display, not a forecast

The existing [`detectStressReleaseDrops` implementation](https://github.com/kantrarian/geospec/blob/e207b477b665ab56d4b7d69dbc4018cbd9502621/docs/index.html#L1107)
groups rows by region label. After its initial tier-drop screen it retains drops
in at least two labels on the same date. It also retains solo drops with at
least five preceding elevated samples or prior tier at least 2. Thus it neither
always requires multiple regions nor verifies independent station support.

**Source-aware detector recomputation has not been performed by this correction.**
The number of independently supported detections is unknown. Existing markers,
heuristic confidence badges and retrospective event matches are experimental
display outputs, not validated precursors, calibrated probabilities or warnings.
Absence of a marker is not an all-clear.

A future evaluation would need record-level source/window/processing provenance,
shared-support grouping before corroboration counts, a declared null and
false-alarm denominator, and fixed rules evaluated on data not used to choose
them. No such evaluation is claimed here. Detector logic, thresholds, calibration,
collected data and monitoring runs are unchanged.

## Calibration is not out-of-sample validation

The public [Calibration and Backtest Report, section 4.2.4](CALIBRATION_BACKTEST_REPORT.md#424-training-set-performance)
explicitly identifies 5/5 as **training/calibration performance**. Its separate
sections 4.3–4.4 report 2/5 on an unseen-event retrospective set. Those historical
figures are not interchangeable and do not establish prospective operational
forecast skill. This correction labels the displayed 5/5 accordingly; it does
not rerun or independently validate either result. The former dashboard's May
2026 recalibration date and equivalence to current thresholds have not been
verified here and are no longer asserted in its calibration panel.

## Publication surfaces and preservation

The research-status notice precedes metrics in the README and both dashboard
entry points. The README template in `run_and_publish.ps1` carries the same
notice so daily generation does not erase it. Method descriptions loaded from
older JSON are qualified at display time; underlying records are preserved.
No operational detector repair, recalibration, new scientific result or history
rewrite is part of this correction.
