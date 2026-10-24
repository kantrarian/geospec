# PRESTART SCHEDULE — successor w2r1 (drafted 2026-08-30T22:08Z)

**SUCCESSOR ARTIFACT** — supersedes `prestart_schedule_2026-08-28.md`
(append-only; the 08-28 artifact stands as history). Trigger: the
composed prestart completed 2026-08-30 (typed `prestart_overall =
REFUSE`), missing the registered 08-28 PRESTART; the 08-28 artifact's
no-silent-absorption clause and the calendar v3 rule ("a missed 08-28
PRESTART supersedes the schedule, never slides these dates") require
a successor, and codex's Gate-2 ruling (2026-08-30T18:44Z,
`REFUSE_CURRENT_REVISION` at public `8cc564e7`, repair 2) requires an
append-only successor schedule + calendar with `evaluation_start` the
first UTC day after a future completed PASS prestart. Nothing is
backdated to 2026-08-29 and no v3 date is silently carried forward.

- **authority**: asylum (owner), in-session 2026-08-30 — WORD GIVEN
  (fixes evaluation_start 2026-09-04, hence PRESTART 2026-09-03; the
  proposed constant is unchanged).
- **quote (verbatim)**: "proceed eval starting on 9/04"
- **quote_sha256**: `56a08a166c7d5c0dc71cd29be113c5b923472632b4ea1d9e2fc6d1a10e745f18`
- **decision context**: codex Gate-2 exception list (4 repairs) with
  owner-authorized successor path; owner adopted the option-1 MAG
  disposition (terminal exclusion, see
  `mag_primary_terminal_exclusion_v1.md`) and authorized drafting the
  consolidated successor packet 2026-08-30 in-session ("draft the
  consolidated successor packet (all four repairs, one routing to
  codex) and hand grassmann the power-cert execution slice").

## Derived window (prereg v0.3 §1 semantics, unchanged rules)

| item | value |
|---|---|
| PRESTART completes | **2026-09-03** (UTC day) |
| `evaluation_start` | **2026-09-04** (first UTC day after) |
| `evaluation_end` | **2027-01-13** (start + 131 d, 132-day span) |
| maturity tail end | **2027-01-20** (end + H_max 7 d) |
| availability cutoff | **2026-09-02** (last COMPLETE UTC day at execution) |
| selection lookback | **[2026-06-05, 2026-09-02]** (90 days inclusive) |
| calibration interval | **2026-01-01 → 2026-09-02** |
| renewal term binds | through **2027-01-20** (the 08-22 owner renewal is
  relational — "through the WINDOW-2 CLOSE = evaluation_end + H_max,
  binding when PRESTART fixes evaluation_start" — so it covers this
  successor without a fresh ask) |

Cutoff interpretation carried forward verbatim from the 08-26/08-28
artifacts: selection/calibration execute ON the PRESTART day, so the
cutoff is the last complete UTC day at execution time
(2026-09-02 < 2026-09-04 ✓).

## Lane set and reuse (Gate-2 repairs 1 + Gate-2 reuse rule)

- **MAG primary set: TERMINALLY EXCLUDED** from this successor's
  accrual lane set per `mag_primary_terminal_exclusion_v1.md`
  (owner option 1). The typed dispositions stand unedited:
  `family_b = FILTER_SUPPORT_INSUFFICIENT`,
  `mag_primary_set = UNTESTABLE_NO_ADMISSIBLE_PRIMARY`. No
  interpolation, fallback, or reinterpretation.
- **M-F4 fit artifacts: REUSED AS CANDIDATES**, not rerun, per the
  Gate-2 reuse rule: model/rules unchanged (byte-identities attested
  in `mf4_reuse_attestation_v1.json`) and fit cutoff 2026-08-20
  strictly pre-evaluation (2026-08-20 < 2026-09-04 ✓). Any failure of
  that proof at landing replaces/rebinds explicitly — never a silent
  refit.

## Derived authority artifacts (this successor round)

- calendar authority **v4** (`w2-calendar-v4-noncal`), producer
  `monitoring/src/w2_calendar_authority_gen_v4_cayley.py`: baseline
  60 = 2026-07-05..2026-09-02; excluded [2026-09-03]; evaluation 132
  = 2026-09-04..2027-01-13; engine 192; B1B 16×12 (structure
  identical to v3 — only the dates move).
- `w2_power_harness_cayley.py` + `w2_geometry_capsule_gen_cayley.py`
  v4-cal amendments (authority constants + schema literal + KAT
  dates; no other behavior change; selftests PASS incl. the
  cross-derivation equality check against the committed v4 artifact).
- selection registries regenerate at cutoff 2026-09-02 on the
  PRESTART day via their registered producers (the
  `selection_records_*_2026-08-27.*` set stands as history).

## Schedule

- **on landing → 09-02**: calendar v4 + amendments + governance
  records landed (asylum-worded pushes); Tier-S smoke (~1.5–2 h
  parallelized) → committed selector artifact; Tier-C certification
  (~5 h wall at 7 procs per `power_cert_sizing_v2.json`); power-cert
  RESULT PACKAGE produced, receipted, independently verified, and
  bound per `power_cert_result_contract_v1.md`; execution manifest
  re-pinned to those exact output bytes.
- **09-03**: selection registries + calibration surfaces produced at
  cutoff 09-02; composed prestart rerun — **PASS required**; Gate-2
  scientific-admission re-review (codex); OWNER SEAL; PRESTART
  completes; ledger enters ACCRUAL.
- **09-04**: `evaluation_start`; sealed accrual begins.

No slippage margin is silently absorbed: if any gate is not honestly
green on 09-03, PRESTART moves and this artifact is superseded by a
successor (append-only). No value access, publication, or claim is
authorized by this schedule; Λ_geo INCONCLUSIVE.
