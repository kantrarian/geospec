# fault2graph WINDOW-2 preregistration — DRAFT v0.2 (cayley, 2026-08-21)

**Status: DRAFT v0.2, folding all five codex R1 repairs (1935Z note). NOT frozen. Owner
directives bound: b2 standing; 3 carriers + cascadia; PoC candidates graduate only here.**

## 1. Window — genuinely prospective (codex R1 fix 1, recommended option)

**`evaluation_start` = the first UTC day AFTER all of: the final freeze commit, registry/
adapter/bar closure, and GLOBAL EMBARGO ACTIVATION (§2). `evaluation_end` = evaluation_start +
131 days (132-day span).** Data from 2026-07-11 up to the freeze barrier is CALIBRATION/COVERAGE
ONLY and never enters verdict-bearing statistics or power geometry. The freeze binds: the actual
freeze commit timestamp, the exact evaluation date list, the excluded-prefix date list, and a
typed refusal if any verdict-bearing row predates the barrier. No claim of prospectivity is made
for any pre-freeze data.

## 2. Global Window-2 embargo barrier (codex R1 fix 2)

1. EVERY admitted lane's freeze, hypothesis registry, adapters, power result, instrument +
   verifier, and owner fire decision are committed BEFORE the first lane reads or emits any
   result.
2. One GLOBAL window UUID + remote lease, plus per-lane UUIDs; the orchestrator atomically
   reserves all admitted lanes. ALL lane result capsules remain EMBARGOED until every admitted
   lane has completed or emitted a predeclared typed no-run.
3. A failure after the first fire is NEVER repaired with revealed data: that lane emits a typed
   terminal and waits for window 3.
4. Pre-fire input handling is "no semantic/value-bearing analysis; mechanical canonicalization,
   validation, and hashing only, with access receipts" (a non-analyst producer role).
5. Exploratory access begins only after ALL admitted lane outputs are committed and
   independently verified — never after the first lane finishes.
Cross-lane KATs required: missing lane authorization; late lane addition; result release before
all terminals; reused/incorrect global lease; post-first-fire source change — each refuses.

## 3. Carriers + registered selection (codex R1 fix 3)

Carriers: istanbul_marmara, socal_coachella, turkey_kahramanmaras, **cascadia**.

**Selection algorithm (cutoff-stable, executable):** frozen candidate-pool blobs + provider
query receipts; exact network/station/location/channel identity + coordinate/segment
requirements; **availability cutoff strictly BEFORE evaluation_start** with the presence/churn
lookback fixed to `[cutoff−89, cutoff]`; exact presence formula (present-days / calendar-days in
lookback), exact churn formula (mean day-over-day measured-set Jaccard), floors, per-carrier
caps/minima, stable sort keys and tie-breaks, and a typed INSUFFICIENT_POOL result. One
IMMUTABLE selected registry per carrier at freeze. **Evaluation-period outages create mask
absences only — never replacement or reselection.** Realized masks may bound scorability and
power geometry but cannot change registries, families, thresholds, or effect grids; missingness
is carried identically through power and production.

**Cascadia carrier capsule (committed BEFORE freeze):** geographic domain polygon; UW/PNSN/CN
station-query rules + receipts; duplicate-identity resolution; channel/location precedence;
segment assignment; station caps; edge construction; UTC calendar convention; canonical
station-index digest formula. ("Reuse Phase-A verbatim" is replaced by this explicit capsule.)

## 4. Families (annex contracts required at freeze; codex R1 fix 5)

- **B2A** (carried): unchanged semantics; re-certifies on window-2 true geometry.
- **B2B — churn-tolerant runs (annex)**: for each adjacent accepted day pair, `I_d` = the
  intersection of MEASURED station sets; overlap floor `|I_d| ≥ ceil(2/3 · |registry|)` (typed
  INTERSECTION_BELOW_FLOOR); induced subgraph on `I_d`; partition computed on the induced
  graph; LABEL-INVARIANT partition comparison restricted to `I_d`; registered switch event,
  segment minimums, absence/refusal typing, and run-break rule. Every null/power draw recomputes
  intersection → admission → partition → run statistic FROM RAW PANELS. Variable-support planted
  fixtures + adversarial churn KATs mandatory.
- **B1B — robust burst (annex)**: station-health admission uses ONLY pre-evaluation telemetry/
  instrument flags (never evaluation-window values); the z-cap is a NAMED winsorization
  transform applied IDENTICALLY to observed, null, LOCO, and injected panels (never a
  post-value station deletion); frozen calibration baseline, robust scale (MAD w/ zero-scale
  refusal), cap value, aggregation, and the new-registry LOCO set.
- **B3A** (carried): typed secondary, unchanged.
- **M-F4 — monitor risk-delta skill (annex)**: one immutable prediction per (region, issue_day)
  EMITTED BEFORE its 7-day label matures; issue-time features only; scaling + model fit on
  calibration dates only; PRIMARY = one pooled macro statistic (paired AUC difference vs the
  persistence baseline) with a region-aware moving-block procedure spanning ≥ the 7-day label
  overlap; frozen event catalog + version + completeness policy; missing-region and
  renewal/data-gap terminals typed. **The exploratory archive delta (0.590 vs 0.545) is
  NON-QUANTITATIVE hypothesis-generation provenance only — the pilot had a full-data scaler fit,
  a non-temporal pooled split, and overlapping labels (codex-identified leaks, acknowledged).**
  The `confidence`-field decision is resolved pre-freeze on archive data only.
  **Admission gate: if daily-monitor continuity authority (the ~08-25 renewal) is absent at
  freeze, M-F4 is typed NOT_ADMITTED_DATA_CONTINUITY and cannot join later.**
- **MAG-1** (companion): its R1-passed design + coverage admission; its frozen internal
  carrier/endpoint multiplicity is imported verbatim.

## 5. Rejection graph (codex R1 fix 4 — single explicit choice)

**Graph lane = ONE family: Holm at family alpha 0.05 over every POWER-CERTIFIED member of the
predeclared set {B2A, B2B, B1B, B3A}.** Certification is decided by synthetic power only and is
IMMUTABLE after any real access; uncertified members report typed CANNOT_DETERMINE_NO_POWER and
never enter the Holm set. (The 0.05/3 convention is retired with window 1.) Rationale: the
improvement candidates ARE the science of window 2; a B2A-gatekept sequence would structurally
silence them.

M-F4: single pooled macro endpoint, alpha 0.05, one-sided (improvement over persistence).
MAG-1: its own frozen allocation. Lanes make SEPARATE NAMED claims; a combined "Window-2
positive" omnibus claim is PROHIBITED. Power certification for every family executes this exact
rejection graph end-to-end, including no-power terminals.

## 6. Power certification

As v0.1 (full window-1 instrument: Tier-S→C, CP ≥0.80, R=20/40, registered coordinate orders,
bound-mode byte-verification) but explicitly THROUGH the §5 rejection graph and §3 selection/
admission gates, on the true post-mask geometry of the §1 evaluation window.

## 7. Run protocol

The window-1 sealed instrument as standard (auth record binding the codex pass note + source
blobs; remote atomic lease; common-dir lock; canonical checkpoint; schema-closed evidence;
frozen recomputing verifier; codex + grassmann verification) — now ORCHESTRATED under the §2
global barrier: one global lease, per-lane fires, all results embargoed until all terminals,
fresh owner seal per lane before any real-value read.

## 8. Sequence & dependencies

v0.2 → codex freeze rounds → annex contracts (B2B/B1B/M-F4) + cascadia capsule + MAG-1
instantiation → grassmann bars (incl. churn/variable-support KATs + cross-lane embargo KATs) →
producer runs (grassmann; cascadia build) → EMBARGO ACTIVATION fixes evaluation_start → window
runs → power campaigns → certifications → owner seals → orchestrated fires → verification →
verdicts → exploratory access. Renewal (~08-25) is an M-F4 admission gate (§4).

## 9. Non-claims

Λ_geo INCONCLUSIVE unchanged; no forecast/precursor/displacement claims; window-1 exploratory
findings are motivation only; mechanism names motivate effect shapes only; publication
owner-controlled.
