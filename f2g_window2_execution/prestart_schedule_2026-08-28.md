# PRESTART SCHEDULE — owner decision 2026-08-28 (recorded 2026-08-24T16:44Z)

**SUCCESSOR ARTIFACT** — supersedes `prestart_schedule_2026-08-26.md`
(append-only; the 08-26 artifact stands as history). Trigger: the
convergence loop consumed the 08-24 staging margin; codex's technical
close landed 2026-08-24T15:41Z (FINAL CLOSE PASS @ manifest `9b59062`);
the 08-26 artifact's no-silent-absorption clause requires a successor,
never a slide. Grassmann's staging critical-path read (1627Z) and
cayley's concurrence recommended redate; the owner chose 08-28.

- **authority**: asylum (owner), in-session
- **quote (verbatim)**: "restart 8-28 is fine"
- **quote_sha256**: `c2fdcf7650e74b96446d65f9c616af041386d10a2be564829b357dd0f79c8150`
- **decision context**: hold-08-26 required ~1,782 first-pass captures +
  v3 freeze review + Tier-S + parallelized Tier-C all inside 08-25 with
  zero slack; the owner took 08-28 for real margin over 08-27.

## Derived window (prereg v0.3 §1 semantics, unchanged rules)

| item | value |
|---|---|
| PRESTART completes | **2026-08-28** (UTC day) |
| `evaluation_start` | **2026-08-29** (first UTC day after) |
| `evaluation_end` | **2027-01-07** (start + 131 d, 132-day span) |
| maturity tail end | **2027-01-14** (end + H_max 7 d) |
| availability cutoff | **2026-08-27** (last COMPLETE UTC day at execution) |
| selection lookback | **[2026-05-30, 2026-08-27]** (90 days inclusive) |
| calibration interval | **2026-01-01 → 2026-08-27** |
| renewal term binds | through **2027-01-14** (the 08-22 owner renewal is
  relational — "through the WINDOW-2 CLOSE = evaluation_end + H_max,
  binding when PRESTART fixes evaluation_start" — so it covers this
  successor without a fresh ask) |

Cutoff interpretation carried forward verbatim from the 08-26 artifact:
selection/calibration execute ON the PRESTART day, so the cutoff is the
last complete UTC day at execution time (2026-08-27 < 2026-08-29 ✓).

## Derived authority artifacts (this successor round)

- calendar authority **v3** (`w2-calendar-v3-noncal`): baseline 60 =
  2026-06-29 → 2026-08-27; excluded [2026-08-28]; evaluation 132 =
  2026-08-29 → 2027-01-07; engine 192; B1B 16×12/60 unchanged in shape.
- staged expected contracts **v3**: cutoff 2026-08-27; census **1,794**
  = 4×90 selection + 3×239 MAG + 3×239 MF4 (239 = 2026-01-01→08-27).
- All v2 artifacts remain committed as history; production code and the
  admission census move to v3; codex verification round required before
  any staging fire.

## Back-schedule (UTC days; margin restored)

- **08-24 (late)**: successor artifacts + code shift committed (cayley);
  grassmann authors capture specs (~1–2 h); codex successor round opens.
- **08-25**: codex successor verification + v3 static contract freeze
  (freeze BEFORE first HTTP) + codex freeze review; captures begin after
  the freeze review (1,782→1,794 keyed captures, typed-refusal
  discipline); S/T/E tree + boundary bind + admission.
- **08-26 → 08-27**: Tier-S smoke (~1.5–2 h parallelized) + Tier-C
  certification (~5 h wall at 7 procs; R40 extensions absorbable);
  reruns/retries absorbable inside the margin.
- **08-28**: certification complete + certified S committed; calibration
  ledgers + selection registries produced at cutoff 08-27; manifest
  CLOSED (12/12); execution verifier `--prestart` PASS; admission
  capsule assembled; OWNER SEAL; PRESTART completes; ledger enters
  ACCRUAL.
- **08-29**: `evaluation_start`; sealed accrual begins.

No slippage margin is silently absorbed: if any gate is not honestly
green on 08-28, PRESTART moves and this artifact is superseded by a
successor (append-only). No value access, publication, or claim is
authorized by this schedule; Λ_geo INCONCLUSIVE.
