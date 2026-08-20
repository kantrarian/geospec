# fault2graph Phase B — Power-Annex COMMON PROTOCOL v1 **rev-1.1** (shared by annexes B-1/B-2/B-3)

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT until the three family annexes are hash-bound in the seal manifest. This file is referenced BY DIGEST from each family annex; it is not itself a manifest slot. **rev-1.1** incorporates codex 0348Z repairs 1–4 (three-segment frozen geometry; post-LOCO Tier-C selector; `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` terminal type; Pareto-frontier reporting). v1 digest `91125cc2…` superseded.

**Frozen-contract bindings:** prereg rev-2 `a44819d` / sha256 `d1929c3127f6d76b87e95319e0f56c4d8ddf4b0cd6226aa9c2e1fe564e44e04e`; freeze `F2G-PB-R2-FREEZE-CODEX-20260820T0301Z`; engine `monitoring/src/d2_f2g_phase_b_stats.py` @ `6034419` (byte-stable); codex rulings `633be612` (0319Z items 3–5) and `2574cdd8` (0330Z items 1–5), both incorporated normatively below. FIXTURE-ONLY: nothing here touches the Phase-A artifact, any real graph, or any waveform.

## 1. Registered synthetic generator (domain-respecting, codex 0319Z item 5)

**Latent station-factor Fisher-z model.** For carrier c with station set S_c, edge (a,b), day d:

- `u_ab(d) = mu0 + s_a(d) + s_b(d) + eps_ab(d)`, with `s_x(d) ~ N(0, sigma_s^2)` iid per (station, day), `eps_ab(d) ~ N(0, sigma_e^2)` iid per (edge, day).
- **`r_ab(d) = tanh(u_ab(d))`** — guaranteed finite in (−1, 1) with NO post-hoc clipping.
- Registered parameters: `mu0 = atanh(0.30)` (baseline coherence 0.30), `sigma_s = 0.15`, `sigma_e = 0.20`.
- Dependence construction (disclosed): cross-edge dependence enters through the shared station shocks `s_x(d)` — edges sharing a station co-move day-by-day, the same dependence class the admitted bar's G8 locks. Whole carrier-day vectors are generated jointly; no independent edge noise is presented as dependence-realistic.
- **Panel geometry (real-carrier scale):** 3 carriers with 12/12/11 stations (35 total, matching the frozen selected registry count), 110 registered days each (positions 1–60 baseline / 61–110 evaluation). **Segments (codex 0348Z item 1): THREE synthetic segment labels per carrier with the EXACT frozen registry size vectors** — carrier 1: 4/4/4, carrier 2: 4/4/4, carrier 3: 4/4/3 — assigned by sorted station index (first 4 → seg_1, next 4 → seg_2, remainder → seg_3). B-2's planted two-community partition is an EFFECT OBJECT and is distinct from this segment registry. **Missingness:** 8% MCAR at the (edge, day) cell level (matching the Phase-A order of magnitude, 272/3849 ≈ 7%), applied before any effect injection.
- Effects are injected in LATENT units (`delta_lat` added to `u` before tanh) on family-specific supports defined in each family annex. Each results table reports the empirically induced robust-z distribution of the injected cells so latent units are interpretable on the frozen z scale.

## 2. Registered randomness (codex 0330Z item 5)

- **Replicate stream:** `master_r = PCG64(derive_substream_seed(frozen_doc_sha256, <family>, "full", "power"))`. Replicate index r (0-based) uses `rep_seed = the r-th sequential uint64 draw` from `master_r`; the replicate's panel (noise, missingness mask, injection placement where the family grid says "seeded") is generated entirely from `Generator(PCG64(rep_seed))`.
- **Null draws inside the engine** use the frozen null substreams (`purpose=null`, fold-specific) exactly as in production; consequently the null offset/permutation sequences are COMMON across replicates for a fixed frozen doc. Disclosed and permitted per codex 0330Z item 5.

## 3. Estimation tiers (codex 0330Z items 1–4)

- **Tier-S `PRELIMINARY_SMOKE`:** R = 50 replicates per grid point, engine called with `n_draws = 999`. Purpose: full-grid landscape + Tier-C candidate selection ONLY. Tier-S tables are labeled `PRELIMINARY_SMOKE` in every artifact and can NEVER populate a certified MDE or a passed power contract.
- **Tier-C `CERTIFICATION` (exact production pipeline):** `n_draws = 9,999`, frozen valid-draw floor, add-one p, and the final **post-LOCO family-positive rule**: a replicate counts as a recovery only if the full-data family passes α = 0.05/3 AND `loco_gate` passes over ALL 35 station-drop folds (each fold its own frozen `loco:<STATION_ID>` null substream). Pre-LOCO and post-LOCO recovery are reported separately; **all certified-frontier claims follow post-LOCO recovery.**
- **Tier-C candidate selection (registered, deterministic; codex 0348Z item 3 — the selector targets the post-LOCO endpoint):** stage 1: rank all grid points by Tier-S pre-LOCO recovery and keep the top 8 (coordinate tie-break); stage 2: for those 8 points, compute Tier-S **post-LOCO** recovery (R = 50, all 35 folds, `n_draws = 999`); the 3 Tier-C candidates are the highest Tier-S post-LOCO recoveries (coordinate tie-break). Pre-LOCO recovery is reported as a diagnostic only. The stage-1 pre-screen is a registered compute bound and is disclosed as such — points outside the top 8 are smoke-screened only.
- **Certification rule:** a point certifies iff the one-sided exact-binomial 95% LOWER confidence bound on post-LOCO recovery is ≥ 0.80.
- **Replicate cap + deterministic stopping rule:** evaluate at R = 20; if the 95% lower bound ≥ 0.80 → CERTIFIED; else if the one-sided 95% exact-binomial UPPER bound < 0.80 → FAILED; else extend to R_max = 40 and re-apply the lower-bound rule; if the interval still straddles 0.80 at R_max → typed `CANNOT_DETERMINE_POWER_ESTIMATE` (never rounded into certification).
- **Detectable-effect frontier (codex 0348Z item 4):** the grids are multidimensional, so there is no unique scalar MDE. The results report EVERY certified point and the coordinatewise Pareto-minimal certified frontier; the deterministic representative for G5/G16 is the lexicographically first Pareto-minimal certified point. No single point is called "the" scientific MDE.
- **No-power terminal type (codex 0348Z item 2):** if no Tier-C candidate certifies, the family registers **`MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH`** — an explicit statement that the three selected candidates failed the exact pipeline and all OTHER grid cells were smoke-screened, NOT exact-pipeline-bounded. A stronger `MDE_NOT_ATTAINED` claim requires either Tier-C on every grid point or a mathematical support proof encoded and executably checked (B-2 annex §L). Either terminal state satisfies frozen §5: absent a certified point, every nonpositive is typed `CANNOT_DETERMINE_NO_POWER`.

## 4. Computational equivalence (memoized driver)

The engine at `6034419` is byte-stable (codex: no doctoring). Tier-C uses a separate estimation driver that exploits the finite rotation support (≤ 110 distinct offsets per carrier for B-1): per-carrier statistics are precomputed once per distinct offset and the 9,999 frozen-substream draws are exact lookups. **Validity gate:** before any Tier-C table is admissible, the driver must reproduce `E.b1_family`'s add-one p BYTE-EQUAL on a registered small configuration (the equivalence receipt rides the results artifact). The same gate applies to any family the driver memoizes.

## 5. Results artifacts

Each family annex is accompanied by `<annex>_results.json`: tier-labeled tables (every registered grid point for Tier-S; every candidate for Tier-C with pre/post-LOCO recovery, exact-binomial bounds, stopping-rule trace), the induced robust-z report, the equivalence receipt, engine/doc/annex digests, and env lock. Results artifacts are digest-routed with the codex receipt; they are evidence, not registered content — the registered content is the annex bytes.

*Common protocol v1 — cayley. Referenced by digest from annexes B-1/B-2/B-3.*
