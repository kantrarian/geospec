# B2B annex — churn-tolerant partition runs — freeze candidate v1 (2026-08-21)

Instantiates prereg v0.3 §4 (B2B). Executable contract; all constants frozen here.

## Statistic

For each carrier, over the evaluation calendar positions in order:
1. A day is a CANDIDATE if its capsule passes the intrinsic gates (identical to the registered
   B2-class gate battery, pinned by bytes).
2. For each ADJACENT candidate pair (d_prev, d): `I_d = MEASURED(d_prev) ∩ MEASURED(d)`.
   **Overlap floor: `|I_d| ≥ ceil(2/3 · |registry_carrier|)`**; below → typed
   `INTERSECTION_BELOW_FLOOR`, run terminates (never bridged).
3. On the subgraph induced by `I_d` for BOTH days: recompute the registered Fiedler partition
   (engine B2 machinery on the induced graph; identical eigengap/coordinate gates, typed).
4. **Label-invariant comparison**: the two partitions, each restricted to `I_d`, are EQUAL iff
   they induce the same bipartition of `I_d` as unordered set pairs. Equal → the run continues;
   unequal → a SWITCH event (run count increments).
5. Segment minimum: each side of a partition must contain ≥ 2 stations of `I_d`
   (else typed PARTITION_DEGENERATE_SIDE, run terminates).
6. Absences (NO_REGISTERED_SNAPSHOT) and any typed refusal terminate the current run
   atomically; gaps are never bridged.
7. **Family statistic** = total run count summed over carriers (one-sided LOW, matching the
   registered B2A direction: fewer runs = more persistence than the null).

## Null

The registered calendar-position permutation of capsules (B2A scheme, pinned by bytes), with
**intersection → admission → induced partition → comparison → runs recomputed FROM RAW PANELS on
every draw**. 9,999 draws, add-one p, valid-draw floor per the pinned Phase-B conventions;
substream seeds from the window-2 freeze sha (`family=B2B`).

## Effect grid (power)

The registered B2A swap classes on the true window-2 geometry — `m ∈ {1,2,3}` block swaps at the
registered onset rule — PLUS two churn-robustness classes: (m=2 swap) × (synthetic churn at
{10%, 25%} random per-day station dropout applied to the injected panel). Certification requires
CP-LB ≥ 0.80 under the FULL four-member Holm graph (v0.3 §5).

## KATs (grassmann bars)

Variable-support planted fixtures (runs must survive registered churn levels); adversarial churn
(alternating dropout defeating naive comparisons must refuse or terminate typed, never
mis-count); INTERSECTION_BELOW_FLOOR boundary (floor−1 refuses, floor passes);
label-permutation invariance (relabeled identical partition ≠ switch); segment-minimum boundary.
