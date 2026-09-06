# fault2graph Phase B — Power Annex **B-3** (boundary straddle) v1 **rev-1.1**

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until hash-bound in the seal manifest. rev-1.1 = codex 0348Z repair 1 (CRITICAL): all B-3 generation and selection now runs on the THREE-segment frozen registry geometry (4/4/4, 4/4/4, 4/4/3 per common §1 rev-1.1), not the superseded two-segment substitute; plus repairs 2–4 inherited via common.
**Inherits:** COMMON PROTOCOL v1 rev-1.1 `docs/f2g_phase_b_power_annex_common.md` @ geospec `d3aa25f`, sha256 `baddf2aa259689356d4d942b6282824ad6ad6d7f50075c54c992a997f216f20d` (normative).
**Frozen family under test:** prereg rev-2 §3 B-3 exactly as frozen (deterministic top-decile selection by `(−|z|, station_a, station_b)`, `K = ceil(0.10·m)`, fixed-selection space null permuting station→segment labels within carrier, max f across all carrier-days); engine `6034419` unmodified.

## Registered effect (family-specific)

Cross-segment latent enrichment: `delta_lat` is ADDED to `u_ab(d)` (pre-tanh) on the `n_cross` lexicographically-first CROSS-SEGMENT canonical edges of carrier 1 under the THREE-segment frozen geometry (common §1 rev-1.1: seg sizes 4/4/4 by sorted station index; carrier 1 has 48 cross-segment and 18 within-segment edges of 66 total), on `k` consecutive evaluation days (start position seeded by the replicate stream for `k < 50`; full window for `k = 50`). Missingness applied before injection; missing cells stay missing. Base-rate note (disclosed expectation, tables decide): with 48/66 ≈ 0.727 cross edges, the space-null expectation of f is high and saturation f = 1.0 is correspondingly less distinguishable than under a two-way split — the three-segment frame makes B-3 HARDER, not easier.

The intended alternative: the day's largest |z| cells concentrate on segment-boundary-straddling edges, driving f(d) above its label-permutation expectation.

## Registered grid (24 points; Tier-S at every point, Tier-C per common §3)

- `delta_lat ∈ {0.3, 0.6, 1.2, 2.4}` (same latent scale interpretation as annex B-1).
- `n_cross ∈ {3, 8}` (carrier 1 has K = ceil(0.10·66) = 7 with full measurement — `n_cross = 8` can saturate a full day's selection; with MCAR-reduced m, K shrinks accordingly).
- `k ∈ {10, 25, 50}` consecutive evaluation days.
- Grid-coordinate ordering for tie-breaks: `(delta_lat, n_cross, k)` ascending.

## Registered maximum scientifically meaningful effect (outer bound of the surveyed grid)

`(delta_lat = 2.4, n_cross = 8, k = 50)` — persistent saturation-level anomaly on more cross-segment edges than a clean day's entire selection. Certified results are reported as the Pareto-minimal certified frontier (common §3); absent certification the terminal type is `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` and per frozen §5 every nonpositive is typed `CANNOT_DETERMINE_NO_POWER`.

## Recovery definition & 80% rule

Per common §3: Tier-C replicate recovery = full-data family p ≤ 0.05/3 AND post-LOCO gate pass over all 35 folds; candidate selection ranks Tier-S POST-LOCO recovery over the registered top-8 pre-screen; certification = one-sided exact-binomial 95% lower bound ≥ 0.80 under the registered stopping rule. Tier-S tables are `PRELIMINARY_SMOKE` only.

## Prior evidence disclosure (not part of the estimate)

Pre-ruling gaussian corner smoke (`f4086fd`): 2/10 at full saturation (T_obs = 1.000, p straddling α) — mechanism: one label permutation per draw but a max over ~50 day-selections gives the null many chances at a near-saturated day, and K = 7 quantizes f coarsely. Registered expectation: certification unlikely; the tanh-generator tables decide.

*Power annex B-3 v1 — cayley. Fixture-only; no real graphs.*
