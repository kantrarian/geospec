# AMENDMENT 2026-10-05 — method qualification: only qualified measurements score (effective scored day 2026-10-07 UTC)

- **date (UTC):** 2026-10-05 (registration and source publication; this commit)
- **authors:** cayley, with codex (review) and grassmann (measurement support)
- **status:** REGISTERED. The rule is **date-gated**: it applies to scored days **on or after 2026-10-07 UTC**
  (`monitoring/src/calibration_eligibility.py`: `ELIGIBILITY_RULE_ACTIVE = True`, `EFFECTIVE_SCORED_DAY = "2026-10-07"`).
  Scored days before 2026-10-07 keep the earlier (rule-off) behaviour; issued history is not rescored.
- **owner approval:** "Approve the M1/M4 qualification release with effective scored day 2026-10-07 UTC"
  (recorded 2026-10-05T02:51:07Z; sha256 of the quoted text `7627a1559033413cd2c1a4f798742a0b62305504394f330fbfcfb5ceccbd166e`)
- **rule version / contract:** `calibration-eligibility-v3` / `method-comparability-v1`

## Purpose and permitted interpretation

Only qualified measurements may contribute to the combined method score or tier. Qualification is determined by
calibration, provenance, processing compatibility, observation support, freshness and freezes, never by score amplitude.
A valid zero retains its weight. A missing or invalid value is not zero. The combined score is a conditional weighted
mean of the included methods, not an earthquake probability or a validated comparison of regional hazard.

## Qualification and comparison

Each component carries one state and its reason: VALID, VALID_ZERO, UNAVAILABLE, FROZEN, EXPIRED, STALE,
INVALID_DEFAULT, NOT_QUALIFIED or UNKNOWN. Only the first two qualify. Reports retain the raw measurement, method score,
calibration status and source evidence whether or not the component qualifies.

Reports state the included methods, excluded reasons, nominal and effective weights, weighted coverage and observation
support. Comparison identity is the method set, the effective weights, the estimator/normalization code identity and a
compatible calibration class. Policy age limits alone do not establish compatibility. Unknown support is not comparable
support; unknown station sharing is not evidence of independence; shared-station regions remain linked, not independent
samples.

Maxima are reported only within compatible descriptive groups. A single global maximum is withheld when groups differ,
support is incomplete, legacy and new regimes are mixed, or no method qualifies. Exact ties are preserved. A genuine zero
remains a numeric observation and may be a group's maximum.

The existing single-method WATCH cap, all-invalid DEGRADED handling and freeze rules remain. No score threshold is
lowered. No method becomes admissible because a historical visual pattern looks promising.

## Confirmation across days

Consecutive qualifying tiers count only within the same prospective regime and a compatible measurement basis.
Confirmation resets on a rule/contract transition, a method-set change, an effective-weight change, a station or carrier
change, unbound support, or a missing intervening day. Issued prior tiers are retained separately for inspection; old
days are not rescored to manufacture continuity.

## Effective day: what the gate does and does not do

The implementation is a **date gate**, not a gate on a prior successful run. The boundary 2026-10-07 is a UTC scored day
whose midnight is strictly after this registration and source publication. Scheduled runs on the new source with scored
days before the boundary are **expected** rule-off checks; they are not proof that any run succeeded. A missed or failed
run is reported as such, never described as passed, never silently rerun, and the published boundary is never moved
backward. The date gate does not stop or roll back production by itself. The rule applies on the production daily path
per scored day; scripts that construct the ensemble without a scored day follow the module flag.

## Changes that take effect with this source, independent of the rule's boundary

Rule-off scoring before 2026-10-07 does not imply whole-pipeline parity with earlier source. With this source:

- **IU.SNZO** (kaikoura's THD station) is measured by a bound daily operator (location- and channel-bound fetch, no gap
  fill, refusal by name), the same operator its calibration uses; kaikoura's THD raw value or availability may differ
  from the earlier wildcard-merge path.
- **The weekly THD recalibration** preserves a station's prior record when its recalibration fails (with the failed
  attempt recorded), writes an explicit calibration date on each entry, and stamps the operator identity.
- **Station baselines** read each entry's own calibration date (used only by the rule's lag check).
- **Exception text** in component notes is redacted of credential-shaped content.
- **Structured station attempts** (provider, requested and returned station/channel/location, waveform epoch, sampling,
  outcome and reason) are recorded for every THD station tried.

Historical replay uses the commit that produced the day being replayed.

## Method-specific boundaries in this increment

- **THD:** n=0/default and other ineligible calibrations remain visible but cannot tier. No new baseline is installed by
  this amendment; a daily-operator baseline candidate for IU.SNZO exists, is not installed, and needs its own decision.
  A failed recalibration must not re-date a prior baseline.
- **Lambda_geo:** remains unqualified until a bound support/operator identity reaches its provenance; full-support epoch
  selection, finite numerator and denominator, matching carrier/operator and visible observation lag are prerequisites.
- **Fault correlation:** an expired calibration capsule is not extended by editing its expiry; the current expired and
  missing statuses remain until qualified replacements exist.

## Scientific limits

This change improves measurement validity and observability. It demonstrates neither earthquake prediction nor
prospective skill. Retained-cache reconstruction is retrospective and may include later-arriving or revised
observations. Qualified data and honest missingness are prerequisites to later exploratory comparisons, not scientific
outcomes in themselves.
