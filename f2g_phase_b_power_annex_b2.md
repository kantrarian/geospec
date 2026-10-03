# fault2graph Phase B — Power Annex **B-2** (community reorganization) v1 **rev-1.1**

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until hash-bound in the seal manifest. rev-1.1 = codex 0348Z repairs inherited via common rev-1.1; per repair 1, the two-community planted partition below is an EFFECT OBJECT, distinct from the three-segment frozen registry geometry in common §1.
**Inherits:** COMMON PROTOCOL v1 rev-1.1 `docs/f2g_phase_b_power_annex_common.md` @ geospec `d3aa25f`, sha256 `baddf2aa259689356d4d942b6282824ad6ad6d7f50075c54c992a997f216f20d` (normative).
**Frozen family under test:** prereg rev-2 §3 B-2 exactly as frozen (Fiedler-sign partitions with the four identifiability gates, max switch count, day-order permutation null); engine `6034419` unmodified.

## Registered effect (family-specific, identifiability-respecting)

Two-block latent structure on carrier 1 (stations sorted; block A = first 6, block B = last 6): the latent mean of edge (a,b) is `mu0 + 0.9` if a,b share a block and `mu0 − 0.5` otherwise (pre-shock, pre-tanh). This yields positive within-block coherence ≈ tanh(1.25) ≈ 0.85 and near-zero/negative cross-block coherence — a graph that sits on the ACCEPTED side of all four gates on clean days (unique 12-node positive-weight LCC via nonnegative cross residue, healthy two-block eigengap, no zero Fiedler coordinates). Gate refusals induced by MCAR missingness are counted and disclosed, never patched.

- **Effect:** at evaluation position 25, `m` stations from block A and `m` from block B EXCHANGE block membership (their incident latent means switch accordingly), persisting to the end of the window — the canonical persistent community reorganization.

## Registered grid (3 points)

`m ∈ {1, 2, 3}` (3+3 = half of each block switching sides). Grid ordering for ties: `m` ascending. Registered maximum scientifically meaningful effect (outer bound): `m = 3`, persistent from mid-window. Certified results (if any) report the Pareto-minimal certified frontier per common §3 (trivially the smallest certified `m` on this 1-D grid); absent certification the terminal type is `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH`, strengthened for the two-regime class by §L's proof route.

## §L — Mathematical support proof (codex 0330Z item 4 route)

**Lemma (two-regime degeneracy).** Let the eligible evaluation days of a carrier yield partitions taking exactly two values A and B (counts ≥ 1 each), with switch distance `d(A,B) = D > 0` and `d(A,A) = d(B,B) = 0`. Then for EVERY permutation of these days there is at least one adjacent pair with values {A, B} (a two-value sequence with both values present must somewhere place an A next to a B), so the permuted max switch count equals D — identically the observed max. Hence the add-one p-value is exactly 1 for every draw count: the observed statistic sits inside the null support with probability 1. This is precisely the persistent-reorganization alternative this family targets.

**Executable check (bound into the results artifact):** one exact-pipeline run (`n_draws = 9,999`) on the canonical two-regime fixture (m = 3, no missingness) must return `p_value == 1.0` and `max_switches == observed`; plus a 10-replicate confirmation across seeds. My pre-ruling smoke (`f4086fd`) already observed p ≡ 1.0 at 10/10 on the gaussian variant; the tanh generator rerun rides the results artifact.

**Registered consequence:** for the two-regime alternative class the family CANNOT certify at any grid point — supported by proof, independent of replicate counts. Tier-S/Tier-C tables are still produced for the registered grid (mixed/noisy classes can in principle yield p < 1 via gate-refusal asymmetries), but any certification claim would require the tables to contradict the lemma's class, and no certification is expected. Absent certification, frozen §5 types every B-2 nonpositive `CANNOT_DETERMINE_NO_POWER`.

## Recovery definition & 80% rule

Per common §3 (post-LOCO recovery, exact-binomial lower bound ≥ 0.80, registered stopping rule). Tier-S tables are `PRELIMINARY_SMOKE` only.

*Power annex B-2 v1 — cayley. Fixture-only; no real graphs.*
