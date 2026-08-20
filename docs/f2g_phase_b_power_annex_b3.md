# fault2graph Phase B — Power Annex **B-3** (boundary straddle) v1

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until hash-bound in the seal manifest.
**Inherits:** COMMON PROTOCOL v1 `docs/f2g_phase_b_power_annex_common.md` @ geospec `bfe20da`, sha256 `91125cc2c51a90834247f405b899f02dfa9ecabdb8244830e4679ade3a29563c` (normative).
**Frozen family under test:** prereg rev-2 §3 B-3 exactly as frozen (deterministic top-decile selection by `(−|z|, station_a, station_b)`, `K = ceil(0.10·m)`, fixed-selection space null permuting station→segment labels within carrier, max f across all carrier-days); engine `6034419` unmodified.

## Registered effect (family-specific)

Cross-segment latent enrichment: `delta_lat` is ADDED to `u_ab(d)` (pre-tanh) on the `n_cross` lexicographically-first CROSS-SEGMENT canonical edges of carrier 1 (segment labels per the common geometry: two segments alternating by sorted station index), on `k` consecutive evaluation days (start position seeded by the replicate stream for `k < 50`; full window for `k = 50`). Missingness applied before injection; missing cells stay missing.

The intended alternative: the day's largest |z| cells concentrate on segment-boundary-straddling edges, driving f(d) above its label-permutation expectation.

## Registered grid (24 points; Tier-S at every point, Tier-C per common §3)

- `delta_lat ∈ {0.3, 0.6, 1.2, 2.4}` (same latent scale interpretation as annex B-1).
- `n_cross ∈ {3, 8}` (carrier 1 has K = ceil(0.10·66) = 7 with full measurement — `n_cross = 8` can saturate a full day's selection; with MCAR-reduced m, K shrinks accordingly).
- `k ∈ {10, 25, 50}` consecutive evaluation days.
- Grid-coordinate ordering for tie-breaks: `(delta_lat, n_cross, k)` ascending.

## Registered maximum scientifically meaningful MDE

`(delta_lat = 2.4, n_cross = 8, k = 50)` — persistent saturation-level anomaly on more cross-segment edges than a clean day's entire selection. Any effect requiring more is beyond scientific meaning; per frozen §5 the family then types every nonpositive `CANNOT_DETERMINE_NO_POWER`.

## Recovery definition & 80% rule

Per common §3: Tier-C replicate recovery = full-data family p ≤ 0.05/3 AND post-LOCO gate pass over all 35 folds; certification = one-sided exact-binomial 95% lower bound ≥ 0.80 under the registered stopping rule. Tier-S tables are `PRELIMINARY_SMOKE` only.

## Prior evidence disclosure (not part of the estimate)

Pre-ruling gaussian corner smoke (`f4086fd`): 2/10 at full saturation (T_obs = 1.000, p straddling α) — mechanism: one label permutation per draw but a max over ~50 day-selections gives the null many chances at a near-saturated day, and K = 7 quantizes f coarsely. Registered expectation: certification unlikely; the tanh-generator tables decide.

*Power annex B-3 v1 — cayley. Fixture-only; no real graphs.*
