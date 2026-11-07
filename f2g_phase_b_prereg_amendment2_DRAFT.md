# fault2graph Phase B — Preregistration AMENDMENT 2 (calendar-frame temporal transforms) — DRAFT pending codex freeze

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT. Authored to codex's exact executable frame (`c33dc41f` items 3–4) after the day-count seam stop (`73d4f572`): the real registered sequences are 111/108/111 days on misaligned starts with within-span gaps, so Amendment 1's 110-position frame is undefined on the run object. Enters force only on codex adjudication + freeze under the F2G-PB-R2 amendment lane. **Supersession: Amendment 1's §A0b/§A1/§A2 temporal frames are superseded by this text; the prior B2A certification is historical evidence about the wrong 110-day gapless geometry and carries ZERO seal or verdict weight (codex `c33dc41f` item 2).**

## B0. Canonical calendar and mask authority (manifest-hashed)

- **shared_calendar_days** = every civil day `2026-03-01`…`2026-07-10` inclusive — exactly **132 positions** — as bound in `docs/f2g_phase_b_shared_calendar_v1.json` @ geospec `8111805`, sha256 `3aaaec5872933adf3e7f0daa7beb840e11b793a41d7c387ad4800c078fc65111`.
- **Per-carrier registered-day masks** (same authority file): istanbul_marmara 111 days, socal_coachella 108, turkey_kahramanmaras 111 (absent 21/24/21). The masks are DERIVED FROM THE PHASE-A ANCHOR's `output_digests` keys (anchor sha256 `0850cf3d…`, byte-copy committed at `docs/evidence_phase_a_result_anchor.json`) and are independently re-derivable from the anchor bytes alone; the derivation script is committed. No r value was read in deriving them.
- **Capsule rule:** each (carrier, calendar-day) capsule is either the artifact snapshot or typed `NO_REGISTERED_SNAPSHOT`; **no value is ever synthesized**. Intrinsic snapshot/refusal/absence state moves ATOMICALLY under the one common transform; relational adjacency, eligibility, frame mismatch, and runs are recomputed at target positions; every excluded position clears the run/frame reference (the settled A5k semantics, unchanged).

## B1. Split and transforms (codex-specified, verbatim adoption)

- **Baseline** = calendar positions 1–72 (`2026-03-01`…`2026-05-11`); **evaluation** = positions 73–132 (`2026-05-12`…`2026-07-10`). Per-edge testable floor: ≥45 finite baseline observations among the 72 baseline calendar positions.
- **B1A window rule (codex 0044Z repair, adopted verbatim):** a window is each length-7 interval of consecutive TARGET CALENDAR POSITIONS wholly within positions 73–132. `NO_REGISTERED_SNAPSHOT` occupies its calendar position, is non-finite, and is never deleted or compressed. A carrier-edge-window scores iff at least 4 of its 7 positions contain finite z; its mean uses only those finite z values. B1A maximizes only over these exact calendar-position windows. **Mandatory permanent negative (grassmann's bar):** a fixture whose compressed seven-PRESENT-day maximum differs from its seven-CALENDAR-position maximum, requiring the latter.
- **B1A transform:** eleven consecutive **12-calendar-day blocks** (11 × 12 = 132); one common complete `S_11` block action per draw INCLUDING the identity, applied atomically to every bound carrier's full 132-position capsule vector BEFORE the block-aligned **6-block/5-block** split (positions 1–72 / 73–132); complete pipeline rerun per draw. Statistic: Σ over bound carriers of max over the exact calendar-position windows defined above.
- **B2A:** the entire **60-position joint evaluation calendar capsule** (positions 73–132); ONE common `S_60` order per draw including the identity; typed absences EXPOSE and TERMINATE gaps — registered-position compression may never bridge them. Statistic unchanged (Σ maximal identical-partition runs, one-sided LOW).
- **B3A:** the same 60-position evaluation calendar and exact masks; the spatial relabeling null is unchanged in kind; the count/scorability surface (m, K ≥ 2 floor, f threshold) is recomputed on this frame.
- Carrier binding, family-level `CANNOT_DETERMINE_FAMILY_SCORABILITY` withholding, add-one p, 9,999 draws, valid-draw floors, LOCO conjunctive gate, α = 0.05/3: all carry over verbatim.

## B2. Randomness

Substream tokens `B1A`/`B2A`/`B3A` with the same double-pipe grammar; **seed root = the sha256 of THIS amendment's bytes as frozen by codex** (a fresh root; every Amendment-1-rooted stream is consumed evidence).

## B3. Mandatory sequencing before any seal returns (codex item 4, adopted)

Freeze this amendment BEFORE outcomes → grassmann red bars proving the engine on the EXACT 111/108/111 masks, misaligned start, and within-span gaps, binding the calendar-list digest + Phase-A anchor (not lengths alone), with permanent negatives: reordered/missing/extra calendar day, wrong carrier mask, untyped absence, one-carrier truncation, driver/engine calendar-authority mismatch → engine amendment (frozen functions retained as evidence) → **re-estimation of ALL THREE annexes on the exact 132-day frame + frozen masks** (generator emitting mask-true panels), fresh seed root, full + all-35-fold equivalence gates → two-commit close (results, then manifest) → independent codex re-pass → **FRESH asylum seal decision** (the recorded owner quote `435cb785…` is prior context only and does not auto-carry) → at most ONE sealed run.

## B4. Standing

No real-graph access until the fresh seal. All non-claims hold: no forecast skill, no displacement language, no publication; Λ_geo INCONCLUSIVE.

*Amendment 2 DRAFT — cayley. Append-only lane of F2G-PB-R2; no force until codex freeze.*
