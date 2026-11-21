# PRESTART SCHEDULE — owner decision 2026-08-26 (recorded 2026-08-23T01:13Z)

- **authority**: asylum (owner), in-session
- **quote (verbatim)**: "prestart 8/26 it is"
- **quote_sha256**: `5b44934a4e6ce9a2809978e7d4282238a15c63efa3ce24ef1bb6ee0364ab46d1`
- **decision context**: 8/23-vs-8/26 pros/cons brief (8/23 infeasible against the
  composed admission gate; 8/26 = earliest date every gate passes on its own terms).

## Derived window (prereg v0.3 §1 semantics)

| item | value |
|---|---|
| PRESTART completes | **2026-08-26** (UTC day) |
| `evaluation_start` | **2026-08-27** (first UTC day after) |
| `evaluation_end` | **2027-01-05** (start + 131 d, 132-day span) |
| maturity tail end | **2027-01-12** (end + H_max 7 d) |
| availability cutoff | **2026-08-25** (interpretation below) |
| selection lookback | **[2026-05-28, 2026-08-25]** (90 days inclusive) |
| calibration interval | **2026-01-01 → 2026-08-25** |
| renewal term binds | through **2027-01-12** (owner authorization `7d18d49`) |

**Cutoff interpretation (flagged for ratification with the calendar design)**:
selection/calibration execute ON 2026-08-26, so the cutoff must be the last COMPLETE
UTC day at execution time = 2026-08-25. This satisfies the frozen "last full UTC day
strictly before `evaluation_start`" (08-25 < 08-27 ✓) with "full" read at execution
time — the alternative (08-26) would have selection consuming a partial day, which
the cutoff-stability rule exists to prevent.

## Back-schedule (all times UTC days)

- **08-23**: grassmann bar REV 11 (five LOCO KATs) → codex amendment close; the
  window-2 CALENDAR-AUTHORITY design settled (cayley's proposal routed with this
  schedule; grassmann + codex rulings); PRODUCER TRUST-BOUNDARY decision (grassmann
  chooses acquisition-code vs staged-envelope; if envelope, cayley authors the
  schema amendment same day, codex round).
- **08-24**: acquisition/staging build + REAL data staged (grassmann: 90-day
  day-records for selection; magnetometer + SYM-H/Kp/OMNI series 2026-01-01→08-25;
  producer receipts per the closed contracts). Calendar-authority artifact + seam
  binding implemented (cayley) + codex round. Anticipated-mask geometry envelope
  drafted from live telemetry.
- **08-25**: envelope committed + manifest-pinned (with `loco_registry_carrier` +
  effect grids); **Tier-S smoke + Tier-C certification FIRED** (16–33 h projected;
  local CPU).
- **08-26**: certification completes + certified S committed; calibration ledgers +
  selection registries produced at cutoff 08-25 data; manifest reaches CLOSED
  (12/12); execution verifier `--prestart` PASS; admission capsule assembled; OWNER
  SEAL (asylum: the authorization binding the exact manifest blob/lanes/lease/
  window); PRESTART completes; ledger enters ACCRUAL.
- **08-27**: `evaluation_start`; sealed accrual begins.

No slippage margin is silently absorbed: if any gate is not honestly green on
08-26, PRESTART moves and this artifact is superseded by a successor (append-only).
No value access, publication, or claim is authorized by this schedule; Λ_geo
INCONCLUSIVE.
