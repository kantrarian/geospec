# R5 activation prerequisites — registered checklist (2026-07-30)

R5 is in **SHADOW MODE** (computes + dual-publishes; never substitutes the operational R3 statistic —
`run_ensemble_daily` holds substitution). Per amendments R5 §activation + R6 §5, R5 activates only from
shadow, at a dated marker registered BEFORE the first active day, after ALL items below pass. This file
is that gate; adversarial review (codex 2026-07-30, recheck R5-R1..R5-R5 + R6 §3) sourced it.

## Closed already (shadow-safe now)
- **R5-R1** future-dated model rejected in historical replay (`0 <= age <= 7` + window causality). KAT K6.
- **R5-R3** predictor envelope guards BOTH sides (lower + upper). KAT K7.
- **shadow labeling** the R5 record carries `r5_computed=True, r5_active=False` while held (no self-claim
  of operational status).

## OWED before activation (do NOT lift shadow until each passes with a red-KAT)
1. **R5-R2 — cache schema versioning.** Persisted models must carry `schema_version` + fit-code/amendment
   id and be validated before use (n ≥ min after trim, finite consistent-length beta/residual/ratio
   arrays, cond ≤ ceiling, max_leverage ≤ ceiling, both predictor ranges present+ordered, window/date
   causality). Any missing/invalid field → fresh fit or `fallback_r3:<reason>`. Invalidate all pre-v2
   caches on upgrade. (A stale v1 cached model currently bypasses the v2 gates.)
2. **R5-R4 — residual ties.** The rank-remap identity holds for unique residuals (epsilon-guarded), but a
   tie block maps to its first quantile. Either register a deterministic secondary tie key + test on
   clamped data, OR narrow the amendment's "identical in distribution" claim to the no-tie condition and
   publish tie mass + the induced distribution defect as telemetry.
3. **R5-R5 / R6 §3 — record capsule (producer side).** The ensemble serializer and the R4 `data.csv` must
   emit, per region-day: `raw_r3_ratio`, `method_epoch` (`r3` / `r5_shadow` / `r5` / `fallback_r3:<reason>`),
   fallback reason, model/schema id, availability, publish receipt. Without this the R5-1 lineage fix is
   not end-to-end (the loader can consume the fields but the producer cannot emit them) and R4 cannot
   produce the R6 §3 epoch-stratified trajectories.
4. **R6 §1 publication receipt (the hit-clock).** A git commit timestamp is client-controlled and is NOT
   proof of public availability. Define a durable server-stamped receipt (alarm artifact hash + commit SHA
   + workflow/deploy id + server `created_at`); name the exact availability field. No receipt for a day →
   that day's alarms are ineligible for hit credit (never synthesize an earlier availability). This is a
   correction owed to R6 §1 and is registered here as such.
5. **Lag-sensitivity telemetry (R3/R5 §4, already owed).** Fixed 30/90/180-day shadow fits published with
   each weekly refit; divergence surfaced as an instability indicator. Not yet implemented.

## Activation procedure (unchanged, restated)
Run R5 in shadow until 1–5 pass with red-KATs → commit the named receipts (15-year legacy replay, April
FA-1/FA-2 invariance, measured step discontinuity, tie-mass, lag-sensitivity) → register a dated
`stats-independent` activation marker in a new amendment → activate on the following run. Until then the
operational statistic is R3's and `descriptive_only` stays True (gated separately by R6 §5 / R4-R4).

*Registered 2026-07-30. Append-only.*
