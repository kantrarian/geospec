# fault2graph Phase B — Localization Statistics Preregistration v1 (DRAFT — pending codex freeze + owner seal)

**Author:** cayley — **Date:** 2026-08-20 — **Status:** DRAFT. Becomes binding on codex contract freeze + asylum seal. Amendments append-only thereafter; re-scores only on codex-adjudicated computation defects.

**Owner authorization:** asylum in-session 2026-08-20 02:18Z, "plan and procede with the next phase" — owner_quote_sha256 `80e581f1814ddb045b11dbf4089f4ef71f2488eac20ba1565acd27032a6c0e7d`. Phase B ONLY; Phase C stays owner-gated.

**Governing rule (load-bearing):** every statistic below is REGISTERED before any evaluation on the real graphs. No localization statistic has been, or will be, computed on the Phase-A artifact until this document freezes.

## 0. Exposure disclosure (outcome-blindness accounting)

- cayley: saw individual r-cell values during Phase-A verification (smoke cells, e.g. one istanbul pair; comparator deltas as aggregates) and edge/delta COUNTS. Never computed any anomaly z, community partition, or straddle fraction.
- grassmann: analogous oracle-comparison exposure (per-cell equality checks). Never computed localization statistics.
- codex: verification-level exposure identical in kind.
No party has ranked edges, days, or stations by any anomaly criterion.

## 1. Data (frozen inputs)

- Phase-A builder artifact, anchor `phase_a_result.json` sha256 `0850cf3d24602ab0ba420412f5b292c9d33464852fbb9de8d7363a019c7886ad` (producer authority `df1e37ec…`): 330 (carrier, day) snapshots; coherence edges r_ij(d) with n_overlap; typed station states; frozen registries (35 selected stations, segment membership); registered-day sequences per carrier.
- No waveform access in Phase B. No acquisition. Read-only on all roots and the packet.

## 2. Walk-forward split (frozen; no refit on evaluation data)

Per carrier, over its registered-day sequence in the artifact:
- **Baseline window** = the FIRST 60 registered days.
- **Evaluation window** = all subsequent registered days.
- An edge (unordered station pair) is **TESTABLE** iff it has ≥45 finite r observations in the baseline window; otherwise it is typed `INSUFFICIENT_BASELINE` and excluded from all families (counts disclosed, never silent).
- Baseline statistics are computed once from the baseline window and FROZEN; evaluation days never update them.

## 3. Claim families (each with its own verdict; top-level Bonferroni α = 0.05/3)

### B-1 Per-edge anomaly
- Statistic: robust z, `z_ij(d) = (r_ij(d) − median_B(ij)) / (1.4826 · MAD_B(ij))`, with median/MAD over the edge's baseline observations. MAD = 0 ⇒ edge typed `DEGENERATE_BASELINE`, excluded, disclosed.
- Family: all (testable edge, evaluation day) cells per carrier, carriers pooled for the family verdict.
- Primary test: **max-statistic permutation** (dependence-exact): family max |z| against N=1000 draws of the TIME-SHIFT null — each edge's evaluation-day series independently circularly shifted by a uniform random offset over the carrier's registered evaluation sequence. Family p = fraction of null draws whose max |z| ≥ observed.
- Secondary (reported, not verdict-bearing): per-cell empirical p from the same null with **Benjamini–Yekutieli** FDR at q = 0.05 (BY, not BH — dependence-robust per the validation-kit caveat).
- Persistence sub-statistic: count of edges with |z| > 3 on k = 3 consecutive registered evaluation days; tested against the same null (max-count).

### B-2 Community reorganization
- Per (carrier, evaluation day): weighted graph with w_ij = max(r_ij, 0) over that day's measured stations; largest connected component only (smaller components typed `DISCONNECTED`, disclosed); partition = sign structure of the Fiedler vector (second-smallest eigenvector of the unnormalized Laplacian; deterministic sign convention: the partition side containing the lexicographically first station is labeled A).
- Statistic: membership-switch count S(d) between consecutive comparable days (A3 delta comparability rules; index-frame mismatch days are NOT compared).
- Test: observed max S(d) per carrier vs N=1000 draws of the DAY-ORDER null (permute the evaluation-day order; per-day structure preserved, temporal ordering broken). Family verdict via max over carriers with the null applied identically.

### B-3 Boundary straddle
- Per (carrier, evaluation day): among the top-decile |z_ij(d)| testable edges of that day (z from B-1), the fraction f(d) that are CROSS-SEGMENT (segment membership from the frozen registry).
- Test: SPACE null — permute the station→segment assignment within the carrier, preserving segment sizes, N=1000; per-day p; family via max statistic across evaluation days and carriers.

## 4. Robustness gates (required for any positive verdict)

- **LOCO:** recompute the triggering family statistic dropping each selected station in turn; the verdict survives only if significant (same thresholds) in EVERY fold. No claim may depend on a single station.
- **Leakage guard:** baseline and evaluation windows are disjoint and ordered; no statistic uses future data; the 30-day embargo semantics of the source capsules carry over unexamined (the artifact's days are already embargo-compliant by construction).

## 5. Power precondition for any null statement

Before any "no signal" wording: planted-signal synthetic panels (edge-set shift of magnitude δ over k days injected into shift-null-calibrated surrogates) establish the minimum detectable effect (MDE) at 80% power under exactly the pipelines above. A null result is reported ONLY as "no signal at ≥ MDE δ*" — never as an absolute absence (standing no-overclaim rule).

## 6. Implementation & verification (V-D governance)

- Statistics engine + validation-kit adapter: cayley. Producer red-KATs for the engine: grassmann (planted-signal recovery + null-uniformity fixtures — cross-authored, no self-audit clearance). Contract + verify-once: codex. Seal: asylum.
- The engine consumes ONLY the Phase-A canonical tables (byte-authority); every run emits a hash-sealed evidence manifest via the validation-kit governance layer; the real-data run happens ONCE after bars are green, its inputs and outputs digest-bound.
- RNG: PCG64 seeded from the sha256 of this document's frozen bytes (first 8 bytes, big-endian) — the seed is therefore fixed by the freeze itself.

## 7. Standing non-claims

Coherence-structure localization only; no forecast skill; no displacement/tectonic-movement language (geodetic fusion is Phase C, not authorized); Λ_geo remains INCONCLUSIVE; outputs are private evidence artifacts — publication/public claims are a separate owner escalation.

*fault2graph Phase B preregistration v1 — cayley. DRAFT until codex freeze + asylum seal.*
