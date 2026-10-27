# AMENDMENT 2026-10-27 — a THD value outside the shared eligibility rule is not scored

- **date (UTC):** 2026-10-27. Registered by publication of this source commit in the owner-approved F1 window. This is
  not an activation or successful-run receipt.
- **authors:**
  - cayley (implementation)
  - codex (review c1612936: WORKS-WITH-FIX; the finite-domain repair, the honest wording and this registration)
  - grassmann (independent producer and host checks)
- **status:** REGISTERED with the public source commit carrying this amendment.
  - The change takes effect **with this source**. There is no date gate in code, as in AMENDMENT 2026-10-16.
  - **First affected scored day (planned): 2026-10-26 UTC**, at the scheduled 2026-10-28 11:00Z run. This holds
    provided the paired D3 successor is activated and the same schedule resumed before that run. If the window misses
    it, a replacement first affected day is named before activation. No scheduled run is implied by this registration.
  - The actual activation (run, scored day, outcome) is recorded separately. This amendment is not edited to match it.
- **owner approval:** the owner authorized the F1 window at 2026-10-27T13:25Z in grassmann's session:
  `Yes, authorize the F1 window now (today)`. SHA256 of those UTF-8 bytes, without a newline:
  `5eeaf3b448a4c1a4fa6920a0b2f73c68bb1520fc121065968702bb7007feea8f`.
  - The authorization covers one bounded sequence:
    - drain;
    - this public source publication with this amendment;
    - the host-pin observation and successor activation;
    - the same-schedule resume before 2026-10-28 10:00Z.
- **versions:**
  - **New: `thd-value-eligibility-v1`.** It is recorded:
    - in every region's `thd_station_configuration.versions.value_eligibility`;
    - on every recalibrated baseline (`value_eligibility`);
    - in the notes of every refused component.
  - **Reviewed implementation:** cayley aac2a692 (on bb742f06), reviewed by codex in c1612936 and accepted in da2a1bdb.
    It is published as the two source commits immediately preceding this one, with byte-identical files. The
    activation receipt names the final public commits. This amendment is not edited to insert them.
  - **Unchanged:**
    - the measurement (`thd-daily-measurement-v1`; `measure()`, `analyze_window` and the bound operator are untouched);
    - the estimator and comparison identities, so every admitted value keeps its key and no station episode resets;
    - `calibration-eligibility-v3` and `method-comparability-v1`;
    - thresholds, weights and tier rules;
    - the schema labels `thd-station-configuration-v1` and `thd-station-attempts-v3`. The first gains one versions
      entry, the second one outcome.

## Purpose

The weekly recalibration has always refused a daily value unless `thd > 0` and `p1 > 0` (`NONPOSITIVE_THD_OR_P1`). The
daily scorer did not apply that rule.

- When the fundamental band power is not positive, THD = sqrt(sum P_n / P1) is undefined. The analyzer then returns
  0.0 as a placeholder, which the daily path could score as a very low THD.
- A dead channel delivering zeros, or a flat line, produces exactly this.
- A scored placeholder also stopped the station search, so the configured fallback was never asked.

The recal's inline rule also admitted `+inf` in either column.

## Changes that take effect with the source

- **One shared rule** (`thd_daily_measurement.unusable_value`), applied in this order:
  1. no value: not judged;
  2. missing `p1`: refused, `NONPOSITIVE_THD_OR_P1`;
  3. non-finite `thd` or `p1`: refused, `NONFINITE_THD_OR_P1` (new);
  4. `p1 <= 0`: refused, `NONPOSITIVE_THD_OR_P1`, "ratio undefined";
  5. `thd <= 0`: refused, `NONPOSITIVE_THD_OR_P1`, "defined THD outside the positive-only scoring domain".

  The positive-only domain is the recal's established rule. Step 3 is an intentional eligibility repair.
- **Daily scorer.** A refused value is unavailable (`available=false`, `raw_value 0.0` as for every unavailable THD
  component).
  - The notes read: `Unusable THD from NET.STA (<reason>, thd-value-eligibility-v1: <case>; thd=…, p1=…); not scored`.
  - The station's attempt outcome is `THD_NOT_SCORABLE`.
  - The next configured station is tried.
- **Recal.** It applies the same rule.
  - Every finite record shape decides exactly as before.
  - A non-finite value or power is refused as `NONFINITE_THD_OR_P1`. Before, `+inf` entered a baseline, and NaN was
    refused as `NONPOSITIVE_THD_OR_P1`.
- **Legacy records.** A record without `value_eligibility` only lacks the marker.

## Prospective effects

- **A region whose primary station is unusable** takes its THD value from the next configured station that supplies
  one.
- **A region with no usable configured station** has no THD component value. Its tier then follows the existing
  rules: on the synthetic flat-line check, the region was NORMAL with the scored placeholder, and is DEGRADED once it is
  refused.
- These are expectations. Any observed difference is recorded as observed.

## Audit before registration (derived; reviewed inventory)

Rows affected = available THD rows whose `raw_value` is non-finite or `<= 0`.

- **0 affected rows** among 3,343 available THD rows: the newest parseable version of each of 268 report dates,
  2026-01-08..2026-10-22, at runner d251072e.
- **0 affected rows** in all 297 parseable versions of `docs/ensemble_latest.json`.
- **0 affected rows** among 673 available THD rows in the 58 reports of the d251072e `docs/` tree.
- **Not exercised:** two versions of `docs/ensemble_latest.json` are empty files (0 bytes, commits d48c4786 and
  83fee4ec). This audit cannot say which report date either belonged to.
- **`p1` is not published.** No existing report will be rewritten. The reviewed retained raw-THD inventory found no
  inadmissible available raw values, but retained data do not establish historical P1 eligibility.

## Boundaries

- No baseline is installed or edited.
- No threshold, weight or qualification change.
- No earlier report is rescored. Historical replay uses the commit that produced the replayed day.
- **Consumers.** The paired D3 production successor recognises the new note prefix. Its export and snapshot builder
  (codex c0d5af78) are compiled and activated in the same owner-authorized window, and its identity is bound by the
  separate activation receipt. A D3 release without that change reads a refused component as `UNCLASSIFIED` (raw value
  basis) and `NOT_RECORDED` (station support): unknown, never a measurement.
- A withdrawal is recorded by a new dated amendment and does not edit this one.
