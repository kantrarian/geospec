# WINDOW-2 TIER-C SELECTOR AMENDMENT v1 (registered 2026-08-23)

**Provenance**: codex ruling 2026-08-23T19:09Z item 1 (BLOCKER): the window-2
Tier-C candidate selection is the ALREADY-REGISTERED two-stage selector of
`phaseb_annex_common` rev-1.6 §3 (design pin @ `feb20bb`, blob `44f8ddd9…`),
narrowly mapped onto the window-2 composition (full four-member Holm; B1B-only
LOCO). grassmann endorsed option (b) bar-author-side (frozen rule pre-dating its
data = honest adaptivity; tie-breaks written; the pick derivable by anyone from
rule + smoke output alone). Registered BEFORE any Tier-S smoke runs.

## The binding rule (codex 1909Z, verbatim mapping)

1. Tier-S runs every registered detection point at `R=50`, `n_draws=999`,
   through the full four-member Holm vector. A pre-LOCO recovery is strictly
   `family in holm_rejects(vector)`.
2. Rank by `(-integer_success_count, registered_grid_index)`; denominators must
   be exactly 50 and outcomes strict booleans. Keep `min(8, grid_size)`.
3. For B1B only, apply the registered exact partial-recompute LOCO rule to
   those eight at the same Tier-S quality and rank by
   `(-post_loco_success_count, registered_grid_index)`. Select three. For
   B2A/B2B/B3A (window-2 registers no LOCO fold there), select the top three
   pre-LOCO points — an explicit no-op stage 2. B2A's three-point grid selects
   all three.
4. Append the two fixed B1B specificity obligations, `{gain: 3}` then
   `{gain: 10}` (ascending; taken from the registered grid, never
   smoke-selected).
5. Exact campaign order: B2A top-3, B2B top-3, B1B detection top-3, B3A top-3,
   then the two gain points. Within each top-3, selector rank order.

`registered_grid_index` = the point's index in the frozen effect-grid list of
the bound geometry capsule (gain points keep their indices; detection coverage
excludes them). Tier-S remains `PRELIMINARY_SMOKE`; selection certifies nothing;
points outside the top-8 are smoke-screened only (the annex §3 disclosure
carries over verbatim, including `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH`).

## The selector artifact (closed; produced by the implementation)

`f2g-w2-tier-selector-v1` binds: the effect-grids digest, the smoke artifact
digest, the geometry capsule digest, the exact quality `{R:50, n_draws:999}`,
per-point counts (pre-LOCO everywhere; post-LOCO for the B1B top-8), the top-8
sets, the selected sets, and the ordered 14-point list + its digest. Coverage,
denominator, strict-boolean, stage-2-scope, and gain-count violations refuse
typed (`SELECTOR_*`). The campaign runner consumes ONLY this artifact and
recomputes its digest (codex 1909Z item 2).

## Implementation binding

`monitoring/src/w2_tier_selector_cayley.py` — pure, deterministic; hand-fixture
KATs cover the tie-breaks (equal counts → lower grid index), the B1B stage-2
reordering, B2A select-all-three, coverage/quality/stage-2 doctors, and
determinism of the ordered-list digest. Selftest green py3.14 + py3.11.

No power value, certification, or claim is opened by this registration;
Λ_geo INCONCLUSIVE.
