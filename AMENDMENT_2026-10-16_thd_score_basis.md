# AMENDMENT 2026-10-16 — THD scores are not computed without an empirical baseline

- **date (UTC):** 2026-10-16 (registration and source publication; this commit)
- **authors:**
  - grassmann (record contract and method implementation)
  - cayley (contract proposal; export and viewer)
  - codex (review; stress-reader repair)
- **status:** REGISTERED.
  - The change takes effect **with this source**. There is no date gate in code.
  - **First affected scored day (planned): 2026-10-15 UTC.** This is the scored day of the first scheduled run on this source,
    named before that run.
  - The actual activation is recorded separately: the run, its scored day and its outcome. This amendment is not edited
    to match it.
- **owner approval:** "approve the score-basis release window" (recorded 2026-10-16T14:59:10Z in cayley's session, in reply to the score-basis release-window request; sha256 of the quoted text `e397c6ccea4238c7d907ed0854a289eb34cd701c6457b78d12ca12be470b1970`)
- **versions:**
  - `thd-score-basis-v1`: the method record, plus the stress-release reader's handling of the marker. Reviewed method
    implementation: 3a807ba1c37b8a16a086e9dc83b1579517c62f5f. The later activation receipt names the final public
    merge/amendment commit.
  - Collection maintenance (section below): implementation fbb0bea4d98ca593c06b33c0430ec1b83b24652d, one commit on
    3a807ba1 (codex 4e93a285). The same activation receipt binds the final public source.
  - Display/export implementation: bd30c3212164bfe478145bdef00740ed4b206119, integrated on the lunar successor branch at
    89c20c78e09dd37701a137d9f7e29aeaab33fd54. The final D3 release id, manifest, image digest, exact published method
    commit and host pin are bound by the separate compiled-release readiness and activation receipt before operational
    use. This amendment is not edited to insert those later identities.
  - Unchanged:
    - the qualification rule (`calibration-eligibility-v3`)
    - the comparison contract (`method-comparability-v1`)
    - score thresholds, ensemble weights and tier rules

## Purpose and permitted interpretation

The qualification rule can judge a THD baseline as:
- `NO_BASELINE`: no station baseline;
- `ZERO_SAMPLE_DEFAULT`: n = 0, a default or manual estimate.

In either case the component was already excluded from tiering. Its record still carried a z score, a risk score and
elevation flags. These were computed against that default, or with no baseline at all, and the viewer displayed them as
method scores. They are not evidence about the station.

The last record before this amendment (scored 2026-10-13) carried three such components:

| region | station | rule code | values carried |
|---|---|---|---|
| kaikoura | IU.SNZO | `ZERO_SAMPLE_DEFAULT` | z 2.89, risk 0.60, elevated |
| campi_flegrei | IV.CAFE | `NO_BASELINE` | z 0.0, risk 0.78, elevated and critical |
| norcal_hayward | BK.BRK | `NO_BASELINE` | z 0.0, risk 0.91, elevated and critical |

The recorded tier of each region was DEGRADED.

From this source on, the record says such a score was **not computed**. That makes the record more honest. It
demonstrates neither earthquake prediction nor prospective skill.

## Changes that take effect with this source

- **Method record.** These changes apply only when `calibration.eligible_for_tiering` is false because of `NO_BASELINE`
  or `ZERO_SAMPLE_DEFAULT`:
  - `z_score`, `risk_score`, `is_elevated` and `is_critical` are null (not computed). None of them is 0 or false.
  - A new field, `score_basis`, is `NOT_COMPUTED_NO_EMPIRICAL_BASELINE`. It is written only in this case.
  - In the notes, the z token becomes `z=not computed (no empirical baseline)`.
  - Kept: the raw value, units, source, baseline block, exclusion code and reason, tier, ensemble weights and
    `methods_available`.
- **Unchanged cases:** stale or lagged empirical baselines, every other method, the qualification rule, DEGRADED
  semantics, and every earlier report.
- **Readers.**
  - The stress-release reader no longer recomputes a z from the retained default block.
  - It carries the marker. When either boundary's z is withheld by the marker and no other method supplies a numeric z,
    it emits no z-based diagnostic.
  - Unmarked records, and other methods' numeric z, behave as before.
- **Display and export** (implementation above; final release bound separately):
  - A withheld score is labelled "not computed (no empirical baseline)". It is never shown or exported as 0, and never
    drawn.
  - Elevation flags are exported tri-state: null is never false.
  - The observation CSV has a `score_basis` column.
  - The SQL schema is unchanged. The marker reaches SQL only through the notes.
- **Legacy records.** An older record without the marker only lacks the marker. It does not show that the baseline was
  calibrated.

## Prospective effects on comparison and confirmation

- **No tier input changes.** Eligibility, weights and thresholds are untouched.
  - Confirmation follows the region's tier (`run_ensemble_daily.py`, consecutive days at WATCH or higher).
  - So this amendment should not move a tier or a confirmation count. Any observed difference is recorded as observed.
- **When a station gains an empirical baseline, its score is computed again under the existing rule.**
  - Expected cases: IU.SNZO through the owner's calibration addition of 2026-10-14; IV.CAFE or BK.BRK through a later
    scheduled recalibration.
  - The score is computed from that scored day on, and the marker no longer appears.
  - These are expectations, not promises.

## Collection maintenance (not a method or qualification change)

The weekly THD recalibration no longer fetches BK.BKS, GE.ARPR or GE.CSS, which no region configures. Each one's newest
recalibrated record is carried forward unchanged and expires by the existing age rules. No baseline is deleted or
installed; no score input changes.

## Boundaries

- No baseline is installed by hand.
- No threshold, weight or qualification change.
- No earlier report is rescored. Historical replay uses the commit that produced the replayed day.
- A withdrawal of this source is recorded by a new dated amendment and does not edit this one.
