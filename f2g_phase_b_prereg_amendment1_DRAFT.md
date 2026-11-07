# fault2graph Phase B — Preregistration AMENDMENT 1 **rev-3** (owner-directed) — DRAFT pending codex adjudication

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT rev-3: incorporates codex R1.2 findings 1–5 (0505Z, CLOSED at rev-2 per codex 1313Z) AND the three rev-2 findings (1313Z: equal-block group, atomic A2 day-capsule, pre-bound carrier inventory). Enters force only on codex adjudication + freeze under the amendment lane of `F2G-PB-R2-FREEZE-CODEX-20260820T0301Z`. Until then the frozen rev-2 statistics stand untouched.

**Owner authorization:** asylum in-session 2026-08-20 04:47Z selected option **(ii) amend-before-seal** (`38435f29` framing), owner_quote `"ii"`, owner_quote_sha256 `5d7f49449ab22deac22d767b89549c554134c8e47de4d38e748049875c83503b`.

**Evidence basis:** frozen-family power search complete at `98055ca` (Tier-S 0/3,750; Tier-C 9/9 FAILED 0/20; terminals `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` ×3); mechanisms on file (identity-floor + extreme-value fragility; §L permutation degeneracy; max-over-days inflation + K-quantization; bounded-domain noise ceiling).

## A0. Outcome-blindness addendum

No localization statistic has ever been computed on the Phase-A real graphs by any party. All exposure since the rev-2 freeze is synthetic-fixture power surfaces (registered artifacts). Supersession spends no real-data degrees of freedom.

## A0b. The joint null transform (codex findings 1 + 2, resolved together)

All temporal nulls in this amendment are **COMMON JOINT transforms drawn from a COMPLETE group including the identity**:

- Finding 1 (calibration): the null samples the ENTIRE registered group — no element, identity included, is deleted. Add-one p over 9,999 draws; the frozen valid-draw floor applies.
- Finding 2 (joint structure): one transform per draw acts on the SHARED CALENDAR and carries every included carrier with it — same-day cross-carrier dependence is preserved exactly in every draw. No factorized product null is registered anywhere in this amendment, and no independence justification is claimed.
- **The group (A1): joint EQUAL-BLOCK day permutation (codex 1313Z finding 1).** Partition the registered 110-position sequence into **eleven consecutive 10-day blocks** (positions frozen; all blocks the same size, so the block-permutation action is a genuine complete group — the rev-2 15×7+1×5 mixed partition was NOT closed under composition and is withdrawn). A draw is ONE uniform `S_11` permutation of the 11 block indices, applied atomically and identically to every bound carrier's full day vector (values, station states, missingness move together) BEFORE the positional 60/50 split; the complete pipeline is rerun per draw. H0 is joint cross-carrier day-block exchangeability — within-block temporal dependence (≤ 10 days, comfortably covering the 7-day window scale) is preserved; the group has `11! = 39,916,800` elements including the identity, so no floor (contrast: the frozen per-carrier rotation group had ≤ 110 elements and the identity consumed >55% of α). Exhaustive-orbit KAT required (A5): on a small registered EQUAL-block structure, engine p must equal exact full-group enumeration, including a case where the identity contributes an exceedance.
- **The group (A2): joint calendar permutation** of eligible-day order (§A2), same completeness and commonality requirements.
- **A3's null is spatial, not temporal:** ONE station→segment relabeling of the SHARED (disjoint-union) registry per draw, preserving each carrier's exact segment sizes. Because carrier registries are disjoint, this joint law coincides with simultaneous within-carrier relabelings — stated explicitly rather than silently assumed (finding 2).

## A1. Family B-1′ — persistent per-edge anomaly (supersedes B-1)

- **Statistic:** `T = Σ_{c ∈ INCLUDED} max_{(testable edge, window)} windowmean|z|`, window = `w = 7` consecutive registered evaluation days, scored iff ≥ 4 finite z cells; z, testability floor, MAD degeneracy exactly as frozen §3 B-1.
- **Carrier binding (codex 1313Z finding 3 — pre-outcome, never observed-selected):** the family's carrier set is BOUND PRE-OUTCOME from the seal manifest's registered carrier inventory (the three Phase-A carriers). Every bound carrier must be scorable (≥ 1 scorable window) in the observed panel; if ANY bound carrier is not, the WHOLE family is typed `CANNOT_DETERMINE_FAMILY_SCORABILITY` — family-level withholding, never silent exclusion. In each draw, any bound carrier without a scorable window makes the draw invalid (typed); the 9,900 valid-draw floor governs. KAT mandated (A5): a bound carrier lacking a scorable observed window must produce family-level withholding.
- **Null:** the A0b joint equal-block permutation group, complete, identity included, add-one, 9,999, floor.
- **Support statement (finding 5):** the null support is the group cardinality (`11!` for the registered structure) — not a per-carrier product claim. Window averaging is registered as a POWER HYPOTHESIS (noise-maximum reduction of order √w under weak temporal dependence with the full 7 cells; less when only 4 cells score) to be MEASURED by the annex, not a guaranteed divisor.
- Sub-`w` transients are registered non-claims; the annex quantifies dilution.

## A2. Family B-2′ — temporal community coherence (supersedes B-2)

- **Eligibility, gates, orientation:** exactly as frozen §3 B-2 (all four typed refusal codes retained).
- **Executability contract (finding 3, codex wording adopted):** a carrier enters only with ≥ 2 eligible partitions AND ≥ 1 comparable adjacency, else typed `CARRIER_NO_COMPARABLE_SEQUENCE`. Runs are computed over CALENDAR-ADJACENT eligible days only: any excluded calendar day (gap) or frame mismatch (`NODESET_MISMATCH`) TERMINATES a run and is never bridged — `A,A,[gap],A,A` is two runs (mandatory KAT); mixed node sets never become comparable by deletion.
- **Statistic:** `R_total = Σ_{c ∈ INCLUDED} (count of maximal runs of identical partitions)`, one-sided LOW.
- **Null (codex 1313Z finding 2 — atomic day capsules, wording adopted):** the A2 group acts on the ENTIRE fixed 50-position JOINT evaluation-day capsule. For every carrier, each permuted day carries its partition or typed ineligibility/refusal, its exact station/index frame, and all gap metadata ATOMICALLY; nothing is filtered before permutation. Eligibility, adjacency, mismatch termination, and runs are recomputed only AFTER the common permutation. Complete group, identity included, add-one; `p = (1 + #{R_null ≤ R_obs}) / (N_valid + 1)`. KAT mandated (A5): a paired-carrier fixture with DIFFERENT eligibility masks must show whole-day atomic movement, and `A,A,[gap],A,A` must remain two runs in every transformed panel.
- Carrier binding: pre-outcome from the registered inventory with family-level `CANNOT_DETERMINE_FAMILY_SCORABILITY` withholding, exactly as A1 (a bound carrier here must satisfy the ≥2-eligible/≥1-adjacency contract on the observed panel).

## A3. Family B-3′ — boundary concentration (supersedes B-3)

- **Selection:** exactly as frozen §3 B-3 (deterministic sort, `K = ceil(0.10·m)`, typed refusals, non-conditioning clause).
- **Scorability floor (finding 4):** a selectable day requires `K ≥ 2`; `K = 1` days are typed `DAY_K_UNSCORABLE` (the `(K−1)/K` threshold degenerates to 0 at K = 1), counts disclosed.
- **Statistic:** `C = count of scorable (carrier, day)s with f(d) ≥ (K_d − 1)/K_d`, one-sided HIGH.
- **Null:** the A0b joint relabeling (one shared-registry draw, exact per-carrier segment sizes, selections FIXED), C recomputed per draw, add-one, 9,999, floor.
- **No null-expectation claim is registered** (finding 4: the prior "~2 of 132" figure was false — exact 4/4/4 enumeration for an allowed 7-edge star gives P(cross ≥ 6) ≈ 0.2788, ~37 of 132 days; topology drives the base rate). Exact balanced-label enumeration KATs for star/path/mixed selected-edge topologies are mandated (A5). Power certification remains the sole decision surface.

## A4. Multiplicity, gates, randomness

- α = 0.05/3 across A1–A3; LOCO conjunctive gate, valid-draw floors, add-one, diagnostics (raw + Holm + BY): frozen semantics verbatim.
- Substream tokens `B1A`/`B2A`/`B3A`, fold/purpose grammar unchanged; seed root = the sha256 of THIS amendment's bytes as frozen by codex (the freeze fixes every seed; the RNG seed-source digest is bound in the manifest per A6).

## A5. Sequencing to seal

codex adjudication + freeze → grassmann red-KAT amendment (mandatory classes: exhaustive-orbit calibration KAT incl. identity-exceedance case; paired-carrier synchrony KATs — perfectly synchronous carriers must retain their declared synchrony under every null draw; `A,A,[gap],A,A` two-runs KAT; mixed-nodeset non-bridging; B-3′ exact star/path/mixed enumeration KATs; K=1 unscorable typing; planted two-regime must yield B-2′ p at the add-one floor) → engine amendment (frozen functions retained as evidence surfaces) → **NEW power annexes for B1A/B2A/B3A** on the same common-protocol machinery, with the generator EXTENDED to instantiate the registered joint carrier law: a shared calendar-day factor `G(d) ~ N(0,1)` common to all carriers with registered weight `γ = 0.05` enters every latent `u_ab(d)` (+ a large-γ synchrony fixture for the paired-carrier KATs) → **owner seal offered ONLY IF ≥ 1 grid point certifies per verdict-carrying family**; any family failing again types `CANNOT_DETERMINE_NO_POWER` and returns to asylum with options (i)/(iii). No certification promise — the annexes decide.

## A6. Manifest binding and supersession (finding 5)

On freeze, this amendment binds under the distinct manifest key **`stat_family_amendment_1`** (separate from `red_kat_amendment_1`): path + commit + full LF-blob sha256, the supersession map `B1→B1A, B2→B2A, B3→B3A` (frozen families remain registered evidence with no verdict weight), and the RNG seed-source digest. All standing non-claims hold: Phase C owner-gated, no waveform lane, no forecast/displacement language, no publication, Λ_geo INCONCLUSIVE.

*Amendment 1 rev-2 DRAFT — cayley. Append-only lane of F2G-PB-R2; no force until codex freeze.*
