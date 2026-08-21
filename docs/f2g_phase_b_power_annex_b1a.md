# fault2graph Phase B — Power Annex **B1A** (persistent per-edge anomaly, amended family) v1

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until hash-bound in the seal manifest.
**Family under test:** frozen AMENDMENT 1 §A1 exactly (freeze `F2G-PB-A1-R3-FREEZE-CODEX-20260820T1325Z`, authority `7c3ca7b`, LF sha `f3d0830b38869d8b6f0b03d113d45ae0f111e8645bd4a2934582b21e48e909e8`): statistic `T = Σ_bound-carriers max_(testable edge, 7-day window) windowmean|z|` (≥4 finite cells per scorable window); null = ONE common `S_11` permutation of the eleven equal 10-day blocks per draw, complete group incl. identity, applied atomically to every bound carrier pre-split, full pipeline rerun; pre-bound carrier inventory with family-level `CANNOT_DETERMINE_FAMILY_SCORABILITY` withholding; add-one p, 9,999 draws, frozen valid-draw floor.
**Inherits:** COMMON PROTOCOL rev-1.5 `docs/f2g_phase_b_power_annex_common.md` @ geospec `9df7761`, sha256 `22ae17cd8b563df0343701e968ac07c7c1e56537eb893d052ba1af6699980aea` (calendar-frame lane sec 6 NORMATIVE for all new estimation) (generator incl. γ=0.05 shared calendar factor, geometry, `B1A` substreams with the frozen-amendment seed root, tiers, certification/stopping, frontier, no-power route, equivalence gate — all normative).

## Registered effect (unchanged effect class from superseded annex B-1, applied to the amended statistic)

Edge-set latent shift: `delta_lat` added to `u_ab(d)` (pre-tanh) on the `n_e` lexicographically-first canonical edges of carrier 1, over `k` consecutive evaluation days (start seeded by the replicate stream for `k < 50`; full window at `k = 50`); missingness applied before injection, missing cells never resurrected.

## Registered grid (48 points; Tier-S at every point, Tier-C per common §3)

`delta_lat ∈ {0.3, 0.6, 1.2, 2.4}` × `k ∈ {3, 10, 25, 50}` × `n_e ∈ {3, 10, 33}`; tie-break order `(delta_lat, k, n_e)` ascending. Registered maximum scientifically meaningful effect (outer bound): `(2.4, 50, 33)`. Note: `k = 3 < w = 7` cells are registered DILUTION probes for the frozen sub-window non-claim — expected uncertifiable by design; their rows document the dilution rather than the family's ceiling.

## Recovery, certification, terminals

Per common §3 verbatim: Tier-C replicate recovery = full-data family p ≤ 0.05/3 AND post-LOCO pass over all 35 folds (fold substreams `loco:<STATION_ID>` under the `B1A` root); Tier-S post-LOCO selector over the registered top-8 pre-screen; certification = exact-binomial 95% LB ≥ 0.80 under the registered stopping rule; Pareto-minimal certified frontier reported; absent certification → `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` and frozen §5 typing. Tier-S tables are `PRELIMINARY_SMOKE` only.

## Registered expectation (disclosed, not verdict-bearing)

The amended statistic addresses all four superseded-family mechanisms (identity floor via 11! support; noise ceiling via window averaging; no z-shift resampling; joint structure preserved). Whether that yields ≥80% post-LOCO power at any grid point is exactly what these tables decide — no promise either way.

*Power annex B1A v1 — cayley. Fixture-only; no real graphs.*
