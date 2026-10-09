# AMENDMENT 2026-10-09 — THD station requests, routing, one shared daily measurement and scored-day baseline selection

- **date (UTC):** 2026-10-09 (registration and source publication; this commit)
- **authors:** cayley, with codex (review) and grassmann (retained evidence and measurement support)
- **status:** REGISTERED.
  - The changes take effect **with this source**; there is no date gate in code.
  - **First affected scored day (planned): 2026-10-08 UTC.** This is the scored day of the first
    scheduled run on this source. It is named here before that run.
  - The actual activation (the run, its scored day and its outcome) is recorded separately. This amendment is not
    edited to match it.
- **owner approval:** "i approve.  give me the needed lines and be sure the backslashes are correct and it doesn't take more than one line each" (recorded 2026-10-09T00:49Z, in reply to the release request (handoff section 7: amendment and method push, guarded release, the scheduled calibration-convention change, default reversible rollback); sha256 of the quoted text `4ece11a1986b4c229ee3014e0f871741540bbcf6717a030a75367e14608dbcab`)
- **owner approval, scope extension:** "approved adding both" (recorded 2026-10-09T14:54:29Z, in reply to the request to add the
  scored-day baseline selection and the publication of DEGRADED regions to this release; sha256 of the quoted text
  `2f4d93a8b0d3fd7d7bbf05e69f499dcbf05729bebc3e5d15214d86734ccb22c7`)
- **versions:**
  - `thd-provider-routing-v1`
  - `thd-coverage-facts-v2`
  - `thd-daily-measurement-v1`
  - `thd-operator-class-v1`
  - `thd-baseline-as-of-v1`
  - station attempts `thd-station-attempts-v3`
  - The qualification rule (`calibration-eligibility-v3`) and comparison contract (`method-comparability-v1`) are
    unchanged.

## Purpose and permitted interpretation

This amendment registers:
- how THD station waveforms are requested and from which provider
- which recorded station location is used
- that the daily score and the weekly baseline recalibration measure the same quantity

It improves measurement validity and observability. It demonstrates neither earthquake prediction nor prospective skill,
and it does not explain the earlier difference between installed and recomputed baselines.

## Changes that take effect with this source

- **Provider routing.**
  - Each network has an explicit, documented provider route; a network without one is refused by name rather than sent
    to every provider.
  - For the configured THD stations the first provider is unchanged except network IV (now its own data centre). Some
    networks lose earlier fallback providers.
  - Registered-access networks (Hi-net) are not requested.
  - Each provider attempt keeps a typed outcome, its error class and its HTTP status.
- **Station location (selector).** Eight stations are requested at the exact location code that retained daily records
  show was served, instead of any location:
  - BK.BKS, IU.ANTO, IU.COLA, IU.COR, IU.MAJO, IU.TATO and IU.TUC at `00`
  - MX.TLIG at the blank location

  The other configured stations keep the wildcard and record it as not pinned.
- **Anchorage.**
  - IU.COLA is the primary THD station.
  - AK.BMR is the fallback. It has no baseline of its own, so its value cannot qualify.
  - AK.SSL is no longer requested.
- **Recorded, not admitted.** Each station attempt now records:
  - the selector basis
  - coverage measured from usable samples within the requested window
  - the retained response epoch, where response metadata was retained (unknown outside that metadata's query window)
  - a digest of the samples as returned
  - the measurement identity

  A day on which returned data fail local processing is recorded as a local processing error, not a provider error.
  None of these records admits or refuses a value.
- **One shared daily measurement.**
  - The daily score is unchanged. Its value equals the earlier computation on the same input: window `[D − 25 h, D]`,
    a 12-hour floor, resampling to 1 Hz, and the existing estimator.
  - The weekly THD recalibration now uses that same measurement. Before, it used `[D, D + 25 h]` at the native rate.
  - Each recalibrated baseline records its calibration convention, its measurement identity and a per-day receipt:
    the value, input identity and coverage, or the reason no value was produced.
  - A digest identifies an input; it does not replace the retained bytes needed to replay a waveform.
- **Baseline selection for a scored day.** From the first affected scored day, when the selected THD calibration is
  dated after the scored day, the newest readable dated baseline file on or before that scored day is selected instead.
  - Existing per-entry lag, age and calibration-eligibility checks remain mandatory. No eligible baseline means not
    qualified.
  - Before this, a recalibration run on the day a report is issued was dated after the day it scores, because scoring
    runs two days behind. It was then refused by the registered lag rule in every THD region, on each recalibration
    day and the day after.
  - Prior issued scores are unchanged. The 2026-10-07 report is not rescored.
  - A dated file name records when a recalibration file was written for. It is not proof that its bytes were available
    on that day.
- **Offline scoring.** The event scorer used for backtests counts a record only by its publication instant before the
  event. Event times from the USGS catalogue are read as UTC. The daily runner does not use this scorer.

## Prospective effects on comparison and confirmation

- **The comparison identity of a THD station changes when what decides its value changes.** That covers a selector,
  coverage-admission, response-handling or estimator change, or the calibration convention of the baseline it is
  scored against.
- **It does not change with provenance:** source hashes, retained digests, dates, per-day receipts or routine
  recalibrations within one convention.
- **Under the existing rule, confirmation resets on such a change. Two transitions are expected:**
  1. **With this source:** the eight stations above change selector. Regions they can serve may reset confirmation:
     anchorage, cascadia, hualien, istanbul_marmara, kumamoto, mexico_guerrero, norcal_hayward, ridgecrest,
     socal_saf_coachella, socal_saf_mojave, tokyo_kanto and turkey_kahramanmaras.
  2. **At the first scheduled weekly recalibration on this source:** new baseline moments are computed under the shared
     convention. Their first selection may change baseline values and reset confirmation again, for regions whose THD
     is scored against such a baseline. That is up to all regions except kaikoura, whose station keeps its separately
     bound measurement.
- **One transition, not two,** only if the new baseline is produced and selected before the first scoring run on this
  source. Otherwise the transitions are recorded separately.
- **These are affected sets, not promises.** A station that returns no usable value, or keeps an unchanged basis, is
  recorded as such; no reset is reported that did not occur.
- **Prior reports are preserved and not rescored.** Historical replay uses the commit that produced the day being
  replayed.

## Boundaries

- No baseline is installed by hand. The first new baselines come from the existing scheduled recalibration step and its
  existing age and eligibility rules.
- Response metadata is not response removal. No coverage-admission rule is added.
- Not in this amendment:
  - IV.CAFE's other channel
  - Hi-net access
  - additional data centres
  - a longer baseline window
  - any change to score thresholds, weights or the qualification rule
- A withdrawal of this source is recorded by a new dated amendment and does not edit this one.

## Scientific limits

- Selector pins rest on the location served in retained daily records over a short period. They are evidence of what
  was served, not a proof for every day.
- The installed-versus-recomputed baseline offset remains unexplained. Neither the shared measurement nor a longer
  baseline is a measured fix for it.
- Nothing here is evidence of predictive skill.
