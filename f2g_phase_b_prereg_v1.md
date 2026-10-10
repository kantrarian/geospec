# fault2graph Phase B — Localization Statistics Preregistration v1 **rev-2** (DRAFT — pending codex freeze + owner seal)

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT rev-2, incorporating codex R1.2 (note 2026-08-20T02:37Z, reviewed authority `2dd451c` / `16d9e0b7…`) in full: three rulings adopted, five ranked repairs incorporated normatively. Becomes binding on codex contract freeze + asylum seal. Amendments append-only thereafter; re-scores only on codex-adjudicated computation defects.

**Owner authorization:** asylum in-session 2026-08-20 02:18Z, "plan and procede with the next phase" — owner_quote_sha256 `80e581f1814ddb045b11dbf4089f4ef71f2488eac20ba1565acd27032a6c0e7d`. Phase B ONLY; Phase C stays owner-gated.

**Governing rule (load-bearing):** every statistic below is REGISTERED before any evaluation on the real graphs. No localization statistic has been, or will be, computed on the Phase-A artifact until this document freezes. Nothing has been computed between v1 and rev-2.

## 0. Exposure disclosure (outcome-blindness accounting)

- cayley: saw individual r-cell values during Phase-A verification (smoke cells, e.g. one istanbul pair; comparator deltas as aggregates) and edge/delta COUNTS. Never computed any anomaly z, community partition, or straddle fraction.
- grassmann: analogous oracle-comparison exposure (per-cell equality checks). Never computed localization statistics.
- codex: verification-level exposure identical in kind.
No party has ranked edges, days, or stations by any anomaly criterion. This remains true at rev-2.

## 1. Data (frozen inputs)

- Phase-A builder artifact, anchor `phase_a_result.json` sha256 `0850cf3d24602ab0ba420412f5b292c9d33464852fbb9de8d7363a019c7886ad` (producer authority `df1e37ec…`): 330 (carrier, day) snapshots; coherence edges r_ij(d) with n_overlap; typed station states; frozen registries (35 selected stations, segment membership); registered-day sequences per carrier.
- No waveform access in Phase B. No acquisition. Read-only on all roots and the packet. **Per codex ruling (2): no waveform-side phase-randomized surrogate null exists in Phase B; any such lane is a separately owner-authorized, preregistered validation increment.**

## 2. Walk-forward split (frozen; no refit on evaluation data) — codex ruling (1)

Per carrier, over its registered-day sequence in the artifact:
- **Ordering (frozen):** `registered_day` ascending, ISO-8601 date order. This deterministic ordering is the sole day indexing for all of Phase B.
- **Baseline window** = positions 1–60 of the ordering. **Evaluation window** = positions 61–end.
- Capsule calibration windows are preprocessing/admission frames and are NOT reused as analysis frames.
- The run manifest MUST emit the exact per-carrier baseline and evaluation date lists.
- An edge (unordered station pair, canonical `station_a < station_b` lexicographic) is **TESTABLE** iff it has ≥45 finite r observations in the baseline window; otherwise typed `INSUFFICIENT_BASELINE`, excluded from all families, counts disclosed, never silent.
- Baseline statistics are computed once from the baseline window and FROZEN; evaluation days never update them. (Inside null draws, the identical rule is re-executed on the rotated data — §3 B-1.)

## 3. Claim families (each with its own verdict; top-level Bonferroni α = 0.05/3)

### B-1 Per-edge anomaly

- **Statistic:** robust z, `z_ij(d) = (r_ij(d) − median_B(ij)) / (1.4826 · MAD_B(ij))`, median/MAD over the edge's baseline observations. MAD = 0 ⇒ edge typed `DEGENERATE_BASELINE`, excluded, disclosed.
- **Family:** all (testable edge, evaluation day) cells per carrier, carriers pooled for the family verdict.
- **Sole verdict-bearing statistic (codex repair 3):** the family max |z|. Nothing else can trigger a B-1 positive.
- **Primary null — CARRIER-ROTATION (codex repair 1, replaces the v1 time-shift null entirely):** each null draw applies ONE common circular offset per carrier — uniform over that carrier's registered-day count, drawn from the registered substream (§6) — rotating the ENTIRE registered carrier-day snapshot vector BEFORE the 60/remainder split: all of that carrier's edge r values, station states, index/missingness masks, and full day payload move together under the same offset. Offsets may differ across carriers within a draw. **Edges are NEVER shifted independently, and already-computed evaluation z values are NEVER what is resampled** (shifting computed z is a permutation and leaves max |z| invariant — the v1 defect). Each draw reruns the COMPLETE pipeline on the rotated data: positional 60/remainder split, ≥45 finite-support testability rule, median/MAD baseline fit, degeneracy exclusions, evaluation z, family max, persistence count.
- **p-value:** `p = (1 + #{T_null ≥ T_obs}) / (N_valid + 1)` over N = 9,999 seeded draws; a draw is valid iff the family statistic is computable in that draw (invalid draws disclosed with typed reasons). If N_valid < 9,900 the family is typed `CANNOT_DETERMINE_NULL_SUPPORT` — no verdict.
- **Persistence (secondary, NON-verdict-bearing):** count of edges with |z| > 3 on k = 3 consecutive registered evaluation days, calibrated as a max-count against the same carrier-rotation draws, reported as corrected secondary evidence only; it can never promote a verdict.
- **Per-cell diagnostics:** §7 only; never verdict-bearing.

### B-2 Community reorganization

- Per (carrier, evaluation day): weighted graph with w_ij = max(r_ij, 0) over that day's measured stations.
- **Day-eligibility gates (codex repair 2; each failure typed, the day does not contribute to comparisons, all counts disclosed):**
  1. unique largest positive-weight connected component with ≥3 nodes, else `LCC_TIE`;
  2. exact station-ID/index-frame identity across the two compared days, else `NODESET_MISMATCH`;
  3. relative eigengap `(λ3 − λ2)/max(λ3, 1e-12) ≥ 1e-6`, else `FIEDLER_DEGENERATE` (λ ascending eigenvalues of the unnormalized Laplacian of the selected component);
  4. `|v2_i| > 1e-10` for EVERY classified coordinate after unit normalization, else `FIEDLER_ZERO_COORDINATE`.
- **Orientation:** v2 oriented so the coordinate of the lexicographically first station in the component is positive; partition = sign structure (side A = the side containing that station).
- **Statistic:** membership-switch count S(d) between consecutive comparable days (A3 delta-comparability rules retained; index-frame-mismatch days are NOT compared).
- **Null — DAY-ORDER:** each draw permutes whole eligible day partitions within each carrier (per-day structure preserved; temporal ordering broken), recomputes adjacent comparability and switch counts under the permuted order, and pools the maximum across carriers. N = 9,999 seeded draws; add-one p as in B-1; N_valid < 9,900 ⇒ `CANNOT_DETERMINE_NULL_SUPPORT`.

### B-3 Boundary straddle

- Per (carrier, evaluation day), with m = the count of finite B-1 z cells that day (codex repair 4):
  - sort cells by `(−|z|, station_a, station_b)` with canonical unordered station IDs; set `K = ceil(0.10·m)`; take EXACTLY the first K. If m = 0, the day is typed `INSUFFICIENT_DAILY_EDGES`, excluded, disclosed.
  - `f(d) = cross_segment_count / K` (segment membership from the frozen registry). **Registered direction: larger f.**
- **Null — SPACE:** each draw independently permutes station→segment labels within each carrier preserving exact segment sizes; the selected edge set stays FIXED; recompute all f(d); the draw statistic is ONE maximum across all carrier-days. N = 9,999 seeded draws; add-one p; N_valid < 9,900 ⇒ `CANNOT_DETERMINE_NULL_SUPPORT`.
- **Non-conditioning clause (codex ruling 3, load-bearing for the multiplicity treatment):** B-3 consumes B-1 z as a frozen deterministic transform. It runs regardless of the B-1 verdict and may NOT condition on B-1 significance, BY discoveries, or any observed-data-chosen cutoff. The decile is fixed here, pre-outcome. Any conditioning reopens the selection/multiplicity contract.

## 4. Robustness gates (required for any positive verdict)

- **LOCO (codex repair 5 — conjunctive gate ONLY):** a family positive requires (a) the full-data family passing α = 0.05/3 AND (b) EVERY scorable station-drop fold independently passing the same threshold under the identical pipeline with its own registered substream (§6). A missing or unscorable fold is typed `LOCO_FOLD_UNSCORABLE` and WITHHOLDS the positive. LOCO can never create or promote a positive; therefore no extra multiplicity charge.
- **Leakage guard:** baseline and evaluation windows are disjoint and ordered; no statistic uses future data; the 30-day embargo semantics of the source capsules carry over unexamined (the artifact's days are already embargo-compliant by construction).

## 5. Power contracts (codex repair 3 — precondition for any null statement)

- Before the owner seal, one **power-contract artifact per family** (B-1, B-2, B-3) is authored and hash-bound into the contract, each specifying: the synthetic generator, the effect scale/grid, edge/station injection prevalence, injected duration, replicate count, the 80%-power rule, and the maximum scientifically meaningful MDE for that family (B-2 effects defined on community-switch counts; B-3 effects on straddle fractions — not the generic per-edge δ alone).
- Power estimation may use pure synthetic fixtures or the sealed BASELINE windows only — never evaluation values.
- If a family lacks a sealed power contract, or fails its 80% target at (or below) its registered maximum meaningful MDE, any nonpositive result for that family is typed `CANNOT_DETERMINE_NO_POWER` — never reported as "no signal."
- Red-KAT planted-signal recovery is necessary but is NOT an MDE estimate.

## 6. Resampling & RNG contract (codex repair 5 — one executable contract)

- **Draws:** 9,999 per primary family and per LOCO fold. **p-values:** add-one rule everywhere: `p = (1 + #{T_null ≥ T_obs}) / (N_valid + 1)`.
- **Valid draws:** a draw is valid iff the target statistic is computable in it; invalid draws are disclosed with typed reasons; N_valid < 9,900 ⇒ `CANNOT_DETERMINE_NULL_SUPPORT` for that family/fold.
- **Substreams (deterministic, never language `hash()`):** `seed_material = UTF-8("<frozen_doc_sha256_hex_lowercase>||<family>||<fold>||<purpose>")`; seed = the first 8 bytes of `SHA256(seed_material)` interpreted as a big-endian uint64, feeding NumPy PCG64. `family ∈ {B1, B2, B3}`; `fold ∈ {full} ∪ {loco:<STATION_ID>}`; `purpose ∈ {null, power}`. `frozen_doc_sha256` = the sha256 of this document's bytes as frozen by codex's contract — the freeze itself fixes every seed.

## 7. Diagnostics (never verdict-bearing)

Per-cell / per-day listings are diagnostic only: raw p from the family null draws plus **Holm** and **Benjamini–Yekutieli** adjusted values (BY, not BH — dependence-robust per the validation-kit caveat). A cell reports only with adequate null support (its statistic computable in ≥9,900 valid draws). No diagnostic list alters any family verdict.

## 8. Implementation & verification (V-D governance)

- Statistics engine + validation-kit adapter: cayley. Cross-authored red-KATs for the engine: grassmann — including, at minimum (codex freeze disposition): a red test proving that shifting already-computed evaluation z CANNOT calibrate max |z|; spectral-degeneracy and index-frame fixtures for B-2; tie/small-day fixtures for B-3; plus planted-signal recovery and null-uniformity fixtures. No self-audit clearance. Contract + verify-once: codex. Seal: asylum.
- The engine consumes ONLY the Phase-A canonical tables (byte-authority); every run emits a hash-sealed evidence manifest via the validation-kit governance layer; the real-data run happens ONCE after bars are green, its inputs and outputs digest-bound, including the exact per-carrier date lists (§2).
- No waveform-surrogate lane in Phase B (codex ruling 2).

## 9. Standing non-claims

Coherence-structure localization only; no forecast skill; no displacement/tectonic-movement language (geodetic fusion is Phase C, not authorized); Λ_geo remains INCONCLUSIVE; outputs are private evidence artifacts — publication/public claims are a separate owner escalation.

## 10. Revision history

- **v1 DRAFT** — geospec `2dd451c`, sha256 `16d9e0b7bab3f6ed9c5547596a92e2391bd239d80c516b2ecbf50b4ddc82df23`. Registered a time-shift null over already-computed evaluation z — **defective** (shift = permutation ⇒ max |z| invariant ⇒ p ≡ 1; independent edge offsets also destroyed cross-edge dependence). Never executed; caught at review, before any real-data computation.
- **rev-2 (this document)** — incorporates codex R1.2 2026-08-20T02:37Z in full: rulings (1) first-60 baseline with frozen ISO ordering + manifest date lists, (2) no waveform null in Phase B, (3) no extra B-3 multiplicity absent conditioning; repairs (1) carrier-rotation null, (2) B-2 identifiability gates, (3) sole-verdict statistic + family power contracts, (4) B-3 deterministic selection + space-null family, (5) unified resampling/LOCO/diagnostic contract.

*fault2graph Phase B preregistration v1 rev-2 — cayley. DRAFT until codex freeze + asylum seal.*
