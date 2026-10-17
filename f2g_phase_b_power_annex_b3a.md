# fault2graph Phase B — Power Annex **B3A** (boundary concentration, amended family) v1

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until hash-bound in the seal manifest.
**Family under test:** frozen AMENDMENT 1 §A3 exactly (freeze `F2G-PB-A1-R3`, authority `7c3ca7b` / `f3d0830b…`): selection per frozen §3 B-3 (deterministic sort, `K = ceil(0.10·m)`, typed refusals, non-conditioning) ON THE 60-POSITION EVALUATION CALENDAR (frame superseded by frozen Amendment 2; absent days typed `NO_REGISTERED_SNAPSHOT`); scorability floor `K ≥ 2` (`DAY_K_UNSCORABLE` typed); statistic `C = count of scorable (carrier, day)s with f(d) ≥ (K_d − 1)/K_d`, one-sided HIGH; null = ONE station→segment relabeling of the shared disjoint-union registry per draw (exact per-carrier segment sizes, selections FIXED), C recomputed; NO null-expectation claim registered (topology drives the base rate — codex enumeration on file); add-one p, 9,999 draws, floor.
**Inherits:** COMMON PROTOCOL rev-1.5 @ `9df7761`, sha256 `22ae17cd8b563df0343701e968ac07c7c1e56537eb893d052ba1af6699980aea` (calendar-frame lane sec 6 normative) (normative; three-segment frozen geometry 4/4/4 / 4/4/4 / 4/4/3).

## Registered effect (unchanged effect class from superseded annex B-3)

Cross-segment latent enrichment: `delta_lat` added pre-tanh on the `n_cross` lexicographically-first cross-segment canonical edges of carrier 1 (48 cross / 18 within of 66 under the frozen three-segment frame), over `k` consecutive evaluation days (seeded start for `k < 50`); missingness before injection.

## Registered grid (24 points; Tier-S at every point, Tier-C per common §3)

`delta_lat ∈ {0.3, 0.6, 1.2, 2.4}` × `n_cross ∈ {3, 8}` × `k ∈ {10, 25, 50}`; tie-break `(delta_lat, n_cross, k)` ascending. Registered maximum meaningful effect (outer bound): `(2.4, 8, 50)`. Mandatory red-KAT support (frozen §A5): exact balanced-label enumeration fixtures for star/path/mixed selected-edge topologies + `K = 1` unscorable typing ride grassmann's bar; these tables measure the count statistic's power over the same generator.

## Recovery, certification, terminals

Per common §3 verbatim (post-LOCO endpoint over all 35 folds, top-8 pre-screen selector, exact-binomial 95% LB ≥ 0.80 with the registered stopping rule, Pareto-minimal certified frontier, `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` terminal, Tier-S = `PRELIMINARY_SMOKE` only). `B3A` substreams under the frozen-amendment seed root.

*Power annex B3A v1 — cayley. Fixture-only; no real graphs.*
