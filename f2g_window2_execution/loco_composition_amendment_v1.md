# LOCO-COMPOSITION AMENDMENT v1 (2026-08-22) — registered certification semantics

**Provenance**: codex ruling 2026-08-22T19:33Z (option (a), exact contract) ratified
by grassmann as bar-author 2026-08-22T23:28Z (independent review incl the hand-checked
Holm-substitution fixture). Amends the CERTIFICATION composition of prereg v0.3 §6
with annex_b1b's LOCO gate. The frozen design files are NOT edited (amend-only
discipline); this document is the registered text, pinned in the execution manifest
under the `power_harness` slot, and the harness implements it byte-adjacent.

## Registered text (codex 1933Z, verbatim)

> For B1B detection-class power, replicate recovery requires B1B rejection under the
> full four-member Holm vector and under every vector formed by replacing only the
> B1B coordinate with its same-replicate NEW-registry LOCO fold p-value. The other
> three coordinates are reused unchanged. Every fold projects the same raw replicate
> and uses `loco:<STATION_ID>`; no panel is regenerated. Typed no-p is non-recovery;
> fold-set or provenance mismatch refuses the certification artifact. The gain-step
> specificity endpoint is evaluated on the pre-LOCO full Holm vector and is never
> rescued by LOCO.

## Implementation bindings (this harness)

- The bound geometry capsule carries `loco_registry_carrier` (the NEW-registry
  carrier; cascadia in production); the fold set = exactly that carrier's registry.
- Fold projection = remove the named station and its incident edges from the B1B
  view of the SAME replicate; every other raw value byte-identical; fold token
  `loco:<STATION_ID>` extends the registered substream grammar
  (`derive_substream_seed(auth, "B1B", "loco:<id>", "null")`).
- Full-Holm non-rejection early-exits without folds (the conjunction is already
  false). A legitimate typed/no-p fold ⇒ recovery false (a statistical outcome);
  a missing/extra/duplicate/wrong-station/cross-replicate fold ⇒
  `POWER_LOCO_FOLD_SET_INVALID` — the certification artifact refuses (an audit
  failure). The two states are never collapsed.
- `run_artifact_class` (the ≤0.05 gain-step specificity ceiling) evaluates the
  PRE-LOCO full Holm vector only; no LOCO code path exists in it.

## Locking KATs (codex 1933Z; grassmann bar REV 11 cross-authors)

1. Holm substitution, not `p ≤ .05`: `(.001,.010,.024,.8)` rejects B1B; B1B→`.030`
   does not, despite `.030 ≤ .05`.
2. Exact partial recomputation: B2A/B2B/B3A run once per replicate; B1B runs
   `1 + |R_NEW|` after a full positive; equality vs a byte-held reference.
3. Projection/provenance doctors: exact-station removal, all else byte-identical,
   fold-set defects refuse.
4. Typed-fold (recovery false) vs missing-fold (artifact refusal) state split.
5. Specificity anti-rescue: a gain-step full-Holm positive that would turn negative
   under omission of the gain station still counts against the ceiling.

Sizing consequence (accepted by all three): B1B-only folds, ≈×2.5, 16–33 h Tier-C.
No power value is opened by this amendment; Λ_geo INCONCLUSIVE.
