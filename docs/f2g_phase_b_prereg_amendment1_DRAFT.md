# fault2graph Phase B — Preregistration AMENDMENT 1 (owner-directed) — DRAFT pending codex adjudication

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT. Enters force only on codex adjudication + freeze under the amendment lane of `F2G-PB-R2-FREEZE-CODEX-20260820T0301Z` ("any change requires an explicit append-only amendment … with a new digest"). Until then the frozen rev-2 statistics stand untouched.

**Owner authorization:** asylum in-session 2026-08-20 04:47Z selected option **(ii) amend-before-seal** from the receipt packet's framing (`38435f29`), owner_quote `"ii"`, owner_quote_sha256 `5d7f49449ab22deac22d767b89549c554134c8e47de4d38e748049875c83503b`.

**Why (evidence, all registered):** the frozen families' power search completed at geospec `98055ca`: Tier-S 0/3,750 recoveries over the full 75-point grid; Tier-C 9/9 FAILED 0/20; terminals `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` ×3. Mechanisms on file: identity/near-identity rotation floor + max-statistic extreme-value fragility (B-1); day-order permutation degeneracy of max-switch, per-carrier lemma p ≡ 1 (B-2); max-over-days null inflation + K-quantization (B-3); and the bounded-correlation noise ceiling (saturation injection |z| ≈ 2.6 < panel noise max ≈ 3.9). Each amended statistic below targets those mechanisms directly.

## A0. Outcome-blindness addendum

No localization statistic has been computed on the Phase-A real graphs by any party at any point. All exposure since rev-2 is synthetic-fixture power surfaces (registered artifacts). Supersession therefore spends no real-data degrees of freedom: the amended families are registered pre-data, same as rev-2 was.

## A1. Family B-1′ — persistent per-edge anomaly (supersedes B-1 for verdict purposes)

- **Statistic:** `T = Σ_carriers max_{(testable edge, window)} meanz(edge, window)` in |z|, where a window is `w = 7` consecutive registered evaluation days, scored iff ≥ 4 finite z cells in it (unscored windows typed, disclosed); z, testability floor, MAD degeneracy exactly as frozen §3 B-1. The per-carrier term is the max over that carrier's scorable (edge, window) pairs of the window mean of |z|.
- **Carrier inclusion (fixed pre-draw):** carriers with ≥1 scorable observed window enter the sum; a carrier without one is typed `CARRIER_NO_SCORABLE_WINDOW` and excluded from observed AND every draw. A draw in which an included carrier has no scorable window is an invalid draw (typed).
- **Null:** joint carrier rotation exactly as frozen (one common offset per carrier per draw, full-vector pre-split rotation, complete pipeline rerun) with ONE registered change: the offset is uniform on `{1, …, n_c − 1}` — **the identity offset 0 is excluded** (it reproduces T_obs by construction and consumed 55%+ of the α budget). Draw semantics, add-one p, N = 9,999, valid-draw floor: unchanged.
- **Why this repairs the mechanisms:** the sum across carriers gives the null ~110³-combination support (near-identity must co-occur in all included carriers, prob ~1e-6, vs the frozen single-max floor ~1/110); the 7-day window mean divides the noise ceiling by ≈ √7 while persistent signal retains its full amplitude (fixture arithmetic: noise window-max ≈ 1.4 vs saturation signal ≈ 2.6 — separation where the frozen max had inversion).
- Persistence sub-statistic: dropped (the window IS the persistence requirement). Sub-`w` transients are registered NON-claims for this family; the power annex quantifies dilution.

## A2. Family B-2′ — temporal community coherence (supersedes B-2)

- **Eligibility, gates, orientation, comparability:** exactly as frozen §3 B-2 (all four typed refusal codes retained).
- **Statistic:** `R_total = Σ_carriers (count of maximal runs of identical partitions over the carrier's comparable eligible-day sequence)`, **one-sided LOW**: temporal regime structure yields FEW long runs; permutation shatters them. (Identical partition = same station set and same signed membership.)
- **Null:** independent within-carrier day-order permutation of the eligible partitions per draw (frozen B-2 null), recomputing adjacency, comparability, and runs. `p = (1 + #{R_null ≤ R_obs}) / (N_valid + 1)`, N = 9,999, floor unchanged. Carrier-inclusion semantics as A1.
- **Why:** the frozen max-switch is inside the null support with probability 1 for two-regime alternatives (§L lemma, p ≡ 1 exactly); the runs count is the classical statistic that day-order permutation genuinely destroys — a planted two-regime carrier moves R from ~2 to permuted-typical ~25, giving p at the add-one floor.

## A3. Family B-3′ — boundary concentration (supersedes B-3)

- **Selection:** exactly as frozen §3 B-3 (deterministic `(−|z|, station_a, station_b)` sort, `K = ceil(0.10·m)`, `INSUFFICIENT_DAILY_EDGES` typing, non-conditioning clause).
- **Statistic:** `C = count of selectable (carrier, day)s with f(d) ≥ (K_d − 1)/K_d` (one-below-saturation at that day's own K), **one-sided HIGH**.
- **Null:** the frozen space null (one within-carrier station→segment label permutation per draw, exact size preservation, selections FIXED), recomputing C per draw. Add-one p, N = 9,999, floor unchanged.
- **Why:** the frozen max-over-days gave the null ~50 chances per draw at one saturated day (base cross-rate 0.727 under the three-segment frame); a COUNT of near-saturated days needs the null to produce many simultaneously — expected ~2 of 132 under permuted labels vs ~k for a persistent planted enrichment.

## A4. Multiplicity, gates, randomness

- Top-level Bonferroni α = 0.05/3 across A1–A3, unchanged. LOCO conjunctive gate, valid-draw floors, add-one rule, diagnostics (raw + Holm + BY): all frozen semantics carry over verbatim.
- Substream family tokens: **`B1A`, `B2A`, `B3A`** (fresh streams; the frozen B1/B2/B3 streams are consumed evidence). Fold/purpose grammar unchanged. `frozen_doc_sha256` in the seed material = the sha256 of THIS amendment's bytes as frozen by codex — the adjudication freeze fixes every amended seed.

## A5. Sequencing to seal (all before any real-data contact)

codex adjudication + freeze of this amendment → grassmann red-KAT amendment 2 (mandatory degeneracy-class reds: planted two-regime must yield B-2′ p at the add-one floor, NOT 1; A1 identity-exclusion + draw-semantics checks; A3 count determinism incl. per-day K) → engine amendment (new family functions; frozen functions retained as evidence surfaces) → **NEW power annexes for B1A/B2A/B3A** under the SAME common protocol machinery (tanh station-factor generator, three-segment geometry, Tier-S/Tier-C, post-LOCO endpoint, Clopper-Pearson stopping, Pareto frontier, flat digests) → **owner seal is offered ONLY IF at least one grid point certifies per verdict-carrying family**; any family that again fails certification types `CANNOT_DETERMINE_NO_POWER` and returns to asylum with options (i)/(iii). No promise of certification is made here — the annexes decide.

## A6. Supersession and standing

Frozen B-1/B-2/B-3 remain registered; their completed power searches stand as evidence; they carry NO verdict weight for the sealed run. Every standing non-claim holds: Phase C owner-gated, no waveform lane, no forecast/displacement language, no publication, Λ_geo INCONCLUSIVE.

*Amendment 1 DRAFT — cayley. Append-only lane of F2G-PB-R2; no force until codex freeze.*
