# M-F4 annex — monitor risk-delta skill — freeze candidate v1 (2026-08-21)

Instantiates prereg v0.3 §4 (M-F4). ACCRUAL lane (SEALED_PREDICTION_ACCRUAL). All constants
frozen here. **Admission gate: requires daily-monitor continuity authority at freeze (the
~08-25 renewal); absent → typed NOT_ADMITTED_DATA_CONTINUITY.**

## Features (issue-time only)

For region r on issue day d (using only rows dated ≤ d):
- `drisk = risk(d) − risk(d−1)` (missing prior day → typed no-prediction for (r, d))
- `roll_z_risk = (risk(d) − mean(risk over the prior 7 available days)) / (std + 1e−9)`;
  requires ≥ 4 prior days, else typed no-prediction
- `recent_event ∈ {0,1}`: any qualifying event in `(d−7, d)` per the frozen catalog view AT
  issue time (the persistence feature)
The `confidence` field is **EXCLUDED** (artifact check unresolved at authoring; a pre-freeze
archive-only resolution may amend this by disclosed amendment only).

## Model + baseline

Model: L2 logistic regression (C = 1.0, max_iter = 2000) on {drisk, roll_z_risk, recent_event};
scaler (mean/std) and coefficients fit ONCE on the CALIBRATION interval, then FROZEN
(apply-never-refit). **Calibration interval (codex freeze-review fix 1 — label maturity):**
issue days from 2025-10-18 through
`calibration_issue_end = min(freeze_day − H, the latest issue day whose complete (d, d+H] label
is present in the PINNED pre-freeze calibration-catalog snapshot)`.
The calibration-catalog receipt, exact training-row digest, admitted region set, scaler,
coefficients, and fit diagnostics are all pinned at freeze. Any candidate training row after
`calibration_issue_end` refuses typed `CALIBRATION_LABEL_NOT_MATURE` — never dropped silently,
never admitted later. **KAT (W-MF4)**: appending an event inside the 7-day tail after the
snapshot must leave the training digest and coefficients byte-identical. Baseline:
`recent_event` alone (persistence).

## Predictions

One immutable signed row per (region, issue_day): `p_model`, `p_persistence`, feature vector,
issue timestamp — emitted daily during accrual, append-only, embargoed; label horizon
**H = 7 days**; late or revised rows are typed refusals.

## Labels + catalog

Qualifying event: USGS ComCat, magnitude ≥ 4.0, origin time in `(d, d+7]`, epicenter in the
region's registered bbox (FAULT_SEGMENTS polygons' union bbox, pinned by bytes). Catalog
SNAPSHOT taken once at the maturity tail (`evaluation_end + 7d`); version + query receipts
committed; completeness policy: ComCat as-is, completeness caveat disclosed per region;
no post-snapshot revisions enter scoring.

## Estimand + inference (prereg v0.3 §4 verbatim, instantiated)

Admitted regions: the monitor's region set at freeze minus typed exclusions. Estimand: the
equal-weight MACRO mean over admitted regions of `AUC_r(model) − AUC_r(persistence)` with
midrank ties. A region lacking either label class in the window → typed
`REGION_UNSCORABLE_ZERO_CLASS`; **no-drop rule: if > 1/3 of admitted regions are unscorable, the
whole endpoint is typed ENDPOINT_UNSCORABLE** (no verdict). Inference: synchronized circular
calendar-block bootstrap over issue days — block length **14 days**, blocks drawn synchronously
across regions, **B = 999** replicates, seeds derived from the freeze sha (`lane=MF4`);
percentile CI. **Rejection: the one-sided 95% bootstrap lower bound of the macro mean > 0.**
Alpha 0.05, one claim, no omnibus.

## Power fixtures (mandatory)

Zero-class region; missing region; shared cross-region event; planted-skill class (a synthetic
feature-label dependence at registered effect sizes) through the ENTIRE pipeline including the
no-drop rule and block inference.
