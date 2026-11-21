# fault2graph WINDOW-2 preregistration — DRAFT v0.3 (cayley, 2026-08-21)

**Status: DRAFT v0.3, folding all five codex R2 repairs (1958Z note). NOT frozen. Owner
directives bound: b2 standing; 3 carriers + cascadia; PoC candidates graduate only here.**

## 1. Window — genuinely prospective

`evaluation_start` = the first UTC day AFTER the PRESTART barrier (§2 stage 1) completes.
`evaluation_end` = `evaluation_start + 131 days` (132-day span). The **maturity tail** extends to
`evaluation_end + H_max` where `H_max` = the frozen maximum label horizon (7 days for M-F4).
Data from 2026-07-11 to the barrier is CALIBRATION/COVERAGE ONLY. The freeze binds: the freeze
commit timestamp, the exact evaluation and excluded-prefix date lists, `H_max`, and typed
refusals for any verdict-bearing row predating the barrier or any scoring before the maturity
tail closes.

## 2. Global barrier — TWO-STAGE state machine (codex R2 fix 1)

**Stage 1 — PRESTART (before `evaluation_start`):** bind EVERY lane: all code, models,
calibration fits, hypothesis registries, adapters, the M-F4 model + scaler (calibration-only),
an ANTICIPATED-MASK POWER ENVELOPE for accrual lanes, global window UUID + remote lease + lane
UUIDs, and OWNER AUTHORIZATION for blinded prospective accrual.

**Stage 2 — ACCRUAL (the evaluation window):** M-F4 performs `SEALED_PREDICTION_ACCRUAL` — one
signed, append-only, EMBARGOED prediction per (region, issue_day), emitted before its label
matures; this is not a lane fire. Graph/MAG producers mechanically acquire only the registered
raw inputs + support receipts ("mechanical canonicalization, validation, hashing only, with
access receipts"). Analysts read neither predictions, labels, measurement values, nor interim
statistics.

**Stage 3 — POST-ACCRUAL SUPPORT BARRIER (after `evaluation_end + H_max`):** the catalog
snapshot and realized masks CLOSE; a NON-ANALYST role runs only the frozen support/scorability
and true-mask power rules; all certifications and typed no-runs are frozen and EVERY final
value-fire owner seal is obtained BEFORE any final statistic is opened.

**Stage 4 — FINAL FIRE / RELEASE:** graph/MAG lanes consume values; M-F4 pairs its committed
predictions with matured labels. ALL results remain embargoed until every admitted lane is
terminal and every verifier passes. Post-first-fire failures are typed terminals for window 3;
exploratory access begins only after full release.

Typed refusals (KAT'd): late/revised prediction; early label access; semantic support
inspection; missing maturity tail; cross-lane release before all terminals; missing lane
authorization; late lane addition; reused/incorrect global lease; post-first-fire source change.

## 3. Carriers + registered selection

Carriers: istanbul_marmara, socal_coachella, turkey_kahramanmaras, **cascadia**.
Selection algorithm (cutoff-stable): frozen pool blobs + receipts; exact identity requirements;
availability cutoff strictly BEFORE `evaluation_start`, lookback `[cutoff−89, cutoff]`; exact
presence formula (present-days / lookback-days); **churn score = mean adjacent-day Jaccard
SIMILARITY of measured station sets, sorted DESCENDING (higher = more stable = preferred)**
(codex R2 fix 4); floors, per-carrier caps/minima, stable sort keys, tie-breaks, typed
INSUFFICIENT_POOL. One immutable registry per carrier at freeze. Evaluation-period outages =
mask absences only; realized masks bound scorability/power geometry, never registries/families/
thresholds/grids; missingness carried identically through power and production.
**Cascadia carrier capsule committed BEFORE freeze** (domain polygon, UW/PNSN/CN query rules +
receipts, duplicate resolution, channel/location precedence, segments, caps, edge construction,
UTC calendar, station-index digest formula).

## 4. Families (annex contracts at freeze)

- **B2A / B3A (carried)**: semantics pinned BY BYTES in the freeze manifest (§7) — engine
  functions at their registered blob, not "unchanged" prose.
- **B2B (churn-tolerant runs)**: annex per v0.2 — adjacent-pair intersection `I_d`, ≥⅔ overlap
  floor (typed INTERSECTION_BELOW_FLOOR), induced-subgraph label-invariant partition comparison
  on `I_d`, registered switch event/segment minimums/run-break rule, raw-panel recomputation in
  every draw, variable-support + adversarial-churn KATs.
- **B1B (robust burst)**: annex per v0.2 — pre-evaluation-only health admission, named
  winsorization applied identically to observed/null/LOCO/injected, frozen calibration
  baseline/MAD/zero-scale refusal/cap/aggregation, new-registry LOCO set.
- **M-F4 (monitor risk-delta skill, ACCRUAL lane)**: predictions per §2 stage 2. **Estimand
  (codex R2 fix 3): the equal-weight MACRO mean over predeclared admitted regions of the paired
  within-region difference `AUC(model) − AUC(persistence)`, midrank tie handling.** A region
  lacking both label classes in the window emits a predeclared typed terminal; the registered
  no-drop rule decides whether the whole endpoint is unscorable. Frozen: region weights, catalog
  close at the maturity-tail boundary (catalog + version + completeness policy), synchronized
  calendar-block resampling (block length ≥ 7 days, drawn synchronously across regions),
  replicate count + seed derivation, CI construction, one-sided rejection inequality. Power
  fixtures MUST include zero-class, missing-region, and shared-event cases. Pilot provenance is
  NON-QUANTITATIVE (§8). Admission gate: renewal continuity absent at freeze → typed
  NOT_ADMITTED_DATA_CONTINUITY, no later addition.
- **MAG-1 (companion, ACCRUAL-compatible)**: its R1-passed design + coverage admission, pinned
  by bytes (§7); internal multiplicity imported as a digest-pinned object.

## 5. Rejection graph — non-circular selector (codex R2 fix 2)

Graph lane: `S = { h ∈ {B2A,B2B,B1B,B3A} : CP_LCB(power_h) ≥ 0.80, where power_h is evaluated
under the FULL FROZEN FOUR-MEMBER Holm graph at h's registered MDE/effect object }`.
`S` is computed ONCE from synthetic data; the complete four-member power result is committed;
production applies **Holm at family alpha 0.05 over the immutable `S`**. Certification is NEVER
iterated after removing a member (the 4→3 threshold-relaxation counterexample is a required
refusing KAT). Members outside `S` report typed CANNOT_DETERMINE_NO_POWER and never enter Holm.
M-F4: the single §4 macro endpoint, alpha 0.05 one-sided. MAG-1: its pinned internal
allocation. Separate named claims; omnibus "Window-2 positive" prohibited.

## 6. Power certification

Full window-1 instrument (Tier-S→C, CP ≥0.80, R=20/40 stopping, registered coordinate orders,
bound-mode byte-verification), executed through the ENTIRE §5 rejection graph and §3
selection/admission gates. Two registered stages: the PRESTART anticipated-mask ENVELOPE
(accrual authorization), and the Stage-3 TRUE-MASK certification (frozen before any statistic
opens). A realized mask outside the envelope for an accrual lane = typed no-run, no post-window
recertification.

## 7. Freeze manifest — byte-pinned dependencies (codex R2 fix 4)

The freeze replaces every inherited reference with (repository, full commit, path, canonical
SHA-256, schema/version, and imported section/object digest): engine B2A/B3A semantics; the
sealed-run instrument (driver/instrument/verifier blobs); MAG-1 design + coverage admission;
Phase-B statistical conventions; the selection-algorithm constants; this preregistration itself.
The verifier reopens those bytes and refuses on absence, mismatch, dirty substitution, or
unlisted dependency.

## 8. Pilot provenance (codex R2 fix 5)

All Corpus-A and Corpus-B pilot point estimates, comparisons, confidence statements, and
directional conclusions are NON-QUANTITATIVE; the tables remain historical hypothesis-provenance
only. No clean-rerun conclusion is claimed. The frozen M-F4 design, not any pilot direction, is
the sole basis for prospective evaluation.

## 9. Sequence & dependencies

v0.3 → codex R2 byte-level close → freeze package (numerical annexes + cascadia capsule + MAG-1
instantiation + byte-pin manifest; a SEPARATE pre-fire verification, not a design reopening) →
grassmann bars (incl. cross-lane, churn, non-identity, 4→3 KATs) → producer/cascadia builds →
PRESTART barrier → accrual → support barrier → seals → fires → verification → release →
verdicts → exploratory access. Renewal (~08-25) = M-F4 admission gate.

## 10. Non-claims

Λ_geo INCONCLUSIVE unchanged; no forecast/precursor/displacement claims; window-1 exploratory
findings are motivation only; mechanisms motivate effect shapes only; publication
owner-controlled.
