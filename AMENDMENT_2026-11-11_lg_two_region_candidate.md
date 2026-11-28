# AMENDMENT 2026-11-11 — LG for Tokyo Kanto and Kumamoto from a matched, fixed-origin candidate

- **date (UTC):** 2026-11-11. Registered by publication of this source commit in the owner-authorized window. This is
  not an activation or successful-run receipt.
- **authors:**
  - cayley (producer packaging, fixed-origin repair, daily wiring)
  - codex (integration policy and consumers; review through eca1291a1, closure of the numerical repair)
  - grassmann (numerator support rule v3, independent host checks)
- **status:** REGISTERED with the public source commit carrying this amendment.
  - **Operation `lg-candidate-operation-v1`, regions `tokyo_kanto` and `kumamoto` only.**
  - **Effective scored day E: 2026-11-12 UTC.** The gate is in code: `monitoring/config/lg_candidate_operation.json`
    names E, and every scored day before E keeps the legacy LG path exactly as before, even with this source deployed.
  - **First eligible normal publication (planned):** the scheduled 2026-11-14 run, scoring 2026-11-12. This holds
    provided the paired D3 successor is active, the schedule resumed, and a prospective baseline calibrated on or
    before 2026-11-12 exists. Otherwise the first LG-eligible publication is later; it is recorded separately, and this
    amendment is not edited to match it.
- **owner approval:** the owner authorized the two-region rollout in codex's session, relayed in 0c71239f1 at
  2026-11-09T18:45:56Z: `authorized. continue until cayley, grassmann and yourself have verified it all up and working`.
  SHA256 of those UTF-8 bytes, without a newline: `f5c7985cfd3222316751ae844fb2fbdd333e633d931b3270d3c458c36191180b`.
  - The authorization covers one bounded scope:
    - this source and its registration;
    - a bounded refresh from the existing NGL source for fresh matched baselines;
    - the D3 drain, pin successor, activation and observation;
    - verification across reports, charts and downloads.
  - It does not cover equipment or step metadata lookups, other regions, other providers or raw-input uploads.
- **versions:**
  - **New:**
    - `lg-candidate-operation-v1`: the daily connection, allowlist, E gate, weekly calibration and off switch;
    - `lg-producer-candidate-v1` with `fixed-origin-v1`: the measurement and matched dated baselines;
    - `lg-support-rule-v3` (EDGE_MARGIN 5);
    - `ngl-refresh-candidate-v3`;
    - `lg-candidate-runner-v1`;
    - `lg-integration-candidate-v1`: the eligibility policy.
  - **Unchanged:**
    - every other region's LG path and every other method;
    - the LG ratio clamp [0.1, 50] and the ratio-to-score mapping;
    - weights and tier rules;
    - `calibration-eligibility-v3` and `method-comparability-v1`;
    - issued history.

## Purpose

For these two regions, the daily LG ratio was not eligible for the tier: its baseline carried no matched provenance.
The candidate replaces it with a matched, as-of measurement.

- The numerator and baseline share one operator identity: provider, frame, declared station carrier, processing
  version and code.
- Each scored day is measured as of the end of that day.
- The baseline is a dated, immutable, prospective file selected as of the scored day.

`fixed-origin-v1` removes a numerical defect, verified on retained inputs (codex eca1291a1).

- Absolute NGL northings near 3.9e9 mm lost about 1e-6 relative precision in float64 smoothing and differencing.
- Each station now has one fixed E/N origin, recorded in the baseline and reused at scoring.

These are **experimental operating rules, not scientifically validated warning thresholds**.

## What takes effect from E (allowlisted regions only)

- **Measurement:**
  - support margin 5;
  - 90-day windows on both sides;
  - calibration window ending 30 days before the calibration day;
  - declared-carrier admission 7 days;
  - baseline statistic: the mean of the daily maximum over rule epochs, valid zeros kept;
  - the fixed origin as above.
- **Eligibility** (`CandidatePolicy`):
  - baseline maximum age 45 days;
  - calibration minimum lag 30 days;
  - observation maximum age 45 days;
  - at least 30 baseline epochs;
  - matched operator identities, a ratio that reproduces from the retained numerator and baseline, and no support
    after the scored day.
- **Calibration:**
  - prospective only: `calibrated_on` is the run day, and the file is generated that day.
  - **The first baseline** is a separate operator step on the host: the NGL catalog plus at most 50 candidate station
    products per region, at most 100 in all.
  - **Afterwards:** a weekly re-calibration that refreshes only the declared carriers of the previous baseline and
    fetches no catalog, plus a daily freshness check against the actual last accepted epoch.
- **Refresh:**
  - run-day keyed;
  - transient failures only are retried: 3 attempts per file (0/5/30 s), within 10 extra attempts per run;
  - validated, atomic installs;
  - verified TLS;
  - a candidate-owned cache. The legacy runner cache and the evidence copies are forbidden.
- **Failure:**
  - every failure is a **typed, visible exclusion** of that region's LG. Examples: `LG_CANDIDATE_OFF`,
    `LG_CANDIDATE_STATE_INVALID`, `LG_CANDIDATE_RUN_FAILED`, `BOUND_CATALOG_UNAVAILABLE`,
    `DECLARED_CARRIER_STATION_UNAVAILABLE`, and the eligibility codes `LG_*`.
  - The rest of the daily service continues.
  - A malformed operation config refuses the run before any fetch.
- **Off switch and rollback:**
  - an `OFF` file in the host's candidate state root excludes both regions' LG with `LG_CANDIDATE_OFF` and makes no
    request;
  - the rollback is the previous D3 release and runner commit.

## Not changed and not claimed

- The ratio-to-score mapping is unchanged. There is no change-sensitive channel and no tuning on outcomes.
- No physical attribution of any LG change is made. Station and smoothing sensitivity, and predictive value, remain
  open questions.
- Original issued reports are unchanged. Any retrospective computation is labelled retrospective.

## Prospective effects (expectations)

- On and after E, Tokyo Kanto and Kumamoto either include LG, under a matched baseline with its provenance retained on
  the record, or show a named exclusion.
- New values are not expected to match the 2026-11-04 retrospective replay.
- Any observed difference is recorded as observed.
