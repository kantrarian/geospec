# AMENDMENT 2026-10-22 — THD baseline station check, baseline provenance and the declared estimator

- **date (UTC):** 2026-10-22. Registered by publication of this source commit in the owner-approved coordinated release
  window; this is not an activation or successful-run receipt.
- **authors:**
  - cayley (method change, record block and estimator declaration)
  - codex (review)
  - grassmann (independent producer checks)
- **status:** REGISTERED with the public source commit carrying this finalization.
  - The change takes effect **with this source**. There is no date gate in code.
  - **First affected scored day (planned):** **2026-10-21**, at the scheduled 2026-10-23 11:00Z run, provided the
    coordinated release is activated and resumed before that run. If the window misses it, a replacement first
    affected day must be named before activation. No scheduled run is implied by this registration.
  - The actual activation (run, scored day, outcome) is recorded separately. This amendment is not edited to match it.
- **owner approval:** the owner approved all four items of the coordinated-release request at 2026-10-22 12:31Z:
  `approve all 4`. SHA256 of those UTF-8 bytes, without a newline:
  `03e92ce8d0bf26ac58857d6167140acfab78d815a47a0d40459e8ff196e6b4ea`.
  Scope: this amendment, grey retrospective display, the planned first affected scored day above, and one coordinated
  stop/drain/source-publication/pin-binding/release/host-check/resume window. The owner separately stopped the schedule
  at 2026-10-22 15:13:37Z; drain was observed at 15:14:46Z. Subsequent activation and resume require their own receipts.
- **versions:**
  - `thd-baseline-station-check-v1`, `thd-baseline-provenance-v1` and `thd-robust-estimator-v1`. Reviewed
    implementation: cayley ee395f80, on public 290d9163; codex review 11b05e0f. The activation receipt names the final
    public commit.
  - The D3 display and export of these fields, and the later provenance annotation of earlier records, are a separate
    D3 release. That release's id, manifest, image and host pin are bound by its own readiness and activation receipt.
  - Unchanged:
    - the qualification rule (`calibration-eligibility-v3`)
    - the comparison contract (`method-comparability-v1`)
    - the score-basis rule (`thd-score-basis-v1`)
    - the baseline selection rule (`thd-baseline-as-of-v1`)
    - the z formula, score thresholds, ensemble weights and tier rules

## Purpose and permitted interpretation

A THD value is normalized against a station baseline. The producer looked the baseline up by station key and, for a
declared-channel station, required its exact selector (`thd-station-channel-v1`). It did not:

- check, as one typed result, that the baseline it is about to use belongs to the measured station and was calibrated
  under the measurement's convention;
- say in the record which file the baseline came from, when it was calibrated, or which estimator produced its
  moments.

The record's `baseline.mean` / `baseline.std` carry the median and 1.4826 × MAD, not an arithmetic mean and standard
deviation.

This amendment makes the record state those facts. It demonstrates neither earthquake prediction nor prospective skill.

## Changes that take effect with this source

- **Producer-boundary check (`thd-baseline-station-check-v1`).** The check runs on the baseline AS LOOKED UP, before any
  normalization, for `seismic_thd` only.
  - The baseline's own station must equal the measured station. Otherwise `STATION_MISMATCH`.
  - A declared-channel station's baseline must carry its exact selector. Otherwise `SELECTOR_MISMATCH`, the existing
    `thd-station-channel-v1` outcome, unchanged.
  - Where the baseline states a calibration convention, it must equal the measurement's convention. Otherwise
    `CONVENTION_MISMATCH`. The measurement's convention is the shared daily measurement version for an unbound station,
    and the bound daily operator's version for a bound station: run_thd_recal's own rule.
  - Unknown stays unknown. A baseline stamped with no convention is `UNKNOWN`: it is used, and it is neither a
    mismatch nor a match.
  - A failed check treats the baseline as ABSENT. The raw value is retained and the record carries the no-baseline
    block. Eligibility then sees no baseline, so the existing score-basis rule withholds z/risk (`NOT_COMPUTED`).
    Nothing is zeroed, and nothing is scored against a substitute.
  - The measurement identity is recorded, not compared. A code change between a recalibration and a scored day changes
    it without changing the quantity.
- **Record block (`thd-baseline-provenance-v1`).** The provenance block is emitted only when the eligibility rule is
  active and the measurement reaches baseline lookup. With the rule inactive the block is omitted. The separate
  producer-boundary check can still change handling of a mismatched baseline; no general byte-identical-output claim is
  made. Written into the `seismic_thd` component, the block carries the baseline the producer looked up and what happened
  to it:
  - `disposition`: `SCORED_AGAINST`, `REJECTED_BY_CHECK`, `REJECTED_STALE` or `NO_BASELINE`;
  - the typed check;
  - the source:
    - the dated file and the sha256 of the bytes read, or
    - `BUILT_IN_DEFAULT`;
  - the calibration date, labelled window, n and center/scale;
  - the DECLARED estimator, or none;
  - the convention, measurement identity and selector.

  An absent block establishes neither the reason for absence nor which baseline was used. Earlier code, inactive
  eligibility and early unavailable returns are distinct possible paths; retain UNKNOWN unless independently
  documented.
- **Estimator declaration (`thd-robust-estimator-v1`).**
  - `calibrate_thd_baselines` declares the estimator it computes: center = median, scale = `MAD_TO_SIGMA` × MAD, with
    `MAD_TO_SIGMA` = 1.4826, at 6 stored decimals.
  - The recalibration entry carries that declaration.
  - The stored numbers are computed exactly as before.

## Expected prospective effect (an expectation, not a promise)

- **No baseline in use today fails the check.**
  - Every entry of the newest recalibration file names its own station.
  - The conventions stated there are the measurement's own (the daily measurement version; the bound operator's for
    IU.SNZO).
  - Entries with no convention (e.g. the retained BK.BKS record) are `UNKNOWN`, not rejected.
- So no qualification, score or tier is expected to change. Any observed difference is recorded as observed.

## Display-only changes in the same source (no method, score or tier change)

- **The GitHub history chart shows grey retrospective points** (codex 170e6a2c, wording 3ec764e2). They appear only on
  report days with no qualified method score.
- **The display policy is `retained-component-weighted-mean-v1`:** a mean of retained, measured, unfrozen component
  scores under the published display weights. It is labelled "not issued", with:
  - the clamp floor of a clipped LG ratio;
  - calibration look-ahead where a baseline was calibrated after the scored day.
- **Original scores, tiers, alerts, eligibility and the raw history files are unchanged.**
- **The decline display and calendar event relations** (codex eca09d61): grey dashed connectors join a reconstructed
  point to its immediately adjacent available values. Each displayed tier crossing shows its retrospective peak,
  first decrease, crossing and local low (L, or unconfirmed L?). Retained M5+ events are related only within the same
  region; different regions crossing on the same day are no longer collapsed. Scored/report days are not measurement
  or input-availability timestamps; same-day order is unknown. The browser display now counts only consecutive valid
  days, rejects a tier change without a numeric score decrease, and orders heuristic badges correctly. Thus displayed
  markers and earlier cross-region event matches may change. The comparison window remains anchored to the first NORMAL
  report day, never to the earlier peak; later observations may confirm the local low. Production detection/scoring
  rules, stored scores, tiers, alerts and input-availability records are unchanged. Grey reconstructions remain excluded
  from the detector. These are display corrections, not a precursor validation, warning or backdated detection.

## Boundaries

- No baseline is installed by hand.
- No threshold, weight or qualification change.
- No earlier report is rescored. Historical replay uses the commit that produced the replayed day.
- Earlier records receive provenance only through the separate, labelled D3 annotation, never by rewriting.
- A withdrawal of this source is recorded by a new dated amendment and does not edit this one.
