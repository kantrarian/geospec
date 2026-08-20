# fault2graph Phase B — Power Annex **B-1** (per-edge anomaly) v1

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until hash-bound in the seal manifest.
**Inherits:** COMMON PROTOCOL v1 `docs/f2g_phase_b_power_annex_common.md` @ geospec `bfe20da`, sha256 `91125cc2c51a90834247f405b899f02dfa9ecabdb8244830e4679ade3a29563c` (generator, geometry, randomness, tiers, certification/stopping rules, no-power route, equivalence gate — all normative here).
**Frozen family under test:** prereg rev-2 §3 B-1 exactly as frozen (carrier-rotation null, sole verdict-bearing max |z|, add-one p, valid-draw floor); engine `6034419` unmodified.

## Registered effect (family-specific)

Edge-set latent shift: `delta_lat` is ADDED to `u_ab(d)` (pre-tanh) on the injection support, then `r = tanh(u)` as in the common generator.

- **Injection support:** the `n_e` lexicographically-first canonical edges of carrier 1 (deterministic), on `k` consecutive evaluation days. For `k = 50` the support is the full evaluation window; for `k < 50` the start position is drawn uniformly from the valid evaluation start positions by the replicate stream (seeded, disclosed).
- Missingness is applied BEFORE injection; injected cells that are missing stay missing (no resurrection).

## Registered grid (48 points; Tier-S at every point, Tier-C per common §3)

- `delta_lat ∈ {0.3, 0.6, 1.2, 2.4}` — at `mu0 = atanh(0.30)`: r moves 0.30 → {0.56, 0.74, 0.93, 0.987}. `delta_lat = 2.4` is effectively coherence saturation.
- `k ∈ {3, 10, 25, 50}` consecutive evaluation days.
- `n_e ∈ {3, 10, 33}` edges (33 = half of carrier 1's 66 edges).
- Grid-coordinate ordering for tie-breaks: `(delta_lat, k, n_e)` ascending.

## Registered maximum scientifically meaningful MDE

`(delta_lat = 2.4, k = 50, n_e = 33)` — a persistent, half-network, saturation-level coherence shift. Any effect requiring MORE than this to certify is beyond scientific meaning for this family; per frozen §5 the family then types every nonpositive `CANNOT_DETERMINE_NO_POWER`.

## Recovery definition & 80% rule

Per common §3: Tier-C replicate recovery = full-data family p ≤ 0.05/3 AND post-LOCO gate pass over all 35 folds; certification = one-sided exact-binomial 95% lower bound on post-LOCO recovery ≥ 0.80 under the registered stopping rule. Tier-S tables are `PRELIMINARY_SMOKE` only.

## Prior evidence disclosure (not part of the estimate)

Corner smokes at `f4086fd` (gaussian, pre-ruling generator — superseded by the common tanh generator for all MDE purposes) showed: transient plants capped by ~42% relocation capture with effect-size inversion; identity/near-identity rotation floor ≈ 0.010–0.014 vs α = 0.0167; persistent full-window best case p ≈ 0.028 (n=100 geometry). Registered expectation, to be settled by the tables: certification is unlikely at any grid point. The tables decide, not this paragraph.

*Power annex B-1 v1 — cayley. Fixture-only; no real graphs.*
