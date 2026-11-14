# Window-2 selection constants — freeze candidate v1 (cayley, 2026-08-21)

Instantiates prereg v0.3 §3. All constants FROZEN at freeze; the algorithm executes once at the
availability cutoff with these exact values.

| constant | value |
|---|---|
| availability cutoff | the last full UTC day strictly before `evaluation_start` |
| lookback | `[cutoff − 89 days, cutoff]` (90 days inclusive) |
| presence(s) | (days in lookback with ≥1 admissible sample for station s) / 90 |
| presence floor | **0.85** |
| churn(s) | mean over adjacent lookback day-pairs of Jaccard SIMILARITY between the measured station sets of the candidate registry containing s (carrier-level; identical for all s in a carrier draft set — used as the carrier-set objective during greedy construction) |
| churn floor (carrier set) | **0.80** |
| per-carrier caps | istanbul_marmara **16**, socal_coachella **20**, turkey_kahramanmaras **14**, cascadia **16** |
| per-carrier minimum | **8** (below → typed INSUFFICIENT_POOL for the carrier) |
| selection order | greedy by (presence DESC, station_id ASC) into the cap, then drop-worst by presence until the carrier-set churn ≥ floor or the minimum is reached (if floor unreachable at minimum → carrier admitted with churn disclosed BELOW_FLOOR, typed) |
| tie-break | lexicographic `station_id` ASC (deterministic) |
| coordinate requirement | provider-metadata-confirmed lat/lon (typed absence stations excluded) |
| segment requirement | assignable per the carrier capsule's segment rule |

Evaluation-period outages = mask absences only (v0.3 §3). Missingness carried identically
through power and production.
