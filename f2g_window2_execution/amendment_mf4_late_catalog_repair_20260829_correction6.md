# CORRECTION 6 (append-only) — MF4 late-catalog repair: count seam (ComCat API 2.7.0) + attempt-1 refusal capsule + attempt-2 discipline

- **UTC registered**: 2026-08-30T02:27:10Z (live clock read)
- **corrects**: the module-pin lineage (original → c1 → … → c5); all
  earlier files untouched. This document is the identity-pin source
  the fire-authorization verifier reads from the committed tree.
- **implements**: codex 2026-08-30T02:19:20Z WORKS-WITH-FIX ruling
  (three items) after the attempt-1 typed refusal
  `MF4_CATALOG_COUNT_MISMATCH` (fired 2026-08-30T02:12:16Z under the
  complete v4 authority; 1/13 queries performed; no snapshot).

## Superseding identities

- acquisition module (runtime == git-blob, eol=lf):
  `64140fc01132d0e190dd96d029d3539c2fdd71b3a979de9277cf61a8ac766220`
- query contract (UNCHANGED — the 13 registered queries are not
  touched by this correction):
  `cf19d414b1ba98a38953332b8940e12460c203babb5eadd139a19ce4bb530095`
- catalog adapter (unchanged from c5):
  `27ce4c023f233c628d7819660e024ac3d4b51b77b46a25a629c9ceaed5692d48`
- archive builder/verifier (unchanged from c5):
  `2a22fea1d22d4c32746e3ae0ffae7636690b0a94dc67436b0d55f2db5407fd0b`
- archive capsule (unchanged from c5):
  `2feea82fb365646ac826858d39f3bac53fc4f43bc53407cce42592ba42ae04a9`
- fire-authorization schema bumped: `geospec-mf4-fire-authorization-v5`
- scope literal bumped (attempt-1 authority cannot replay), exact:
  "MF4 correction-6 acquisition attempt 2: exactly one clean re-fire
  of all 13 registered ComCat queries after the typed count-seam
  refusal; attempt-1 authority is spent; nothing else"

## Item 1 — two registered metadata variants (no generic fallback)

`validate_events(region, bbox, raw, expected_url)` now takes the
exact requested URL and admits exactly TWO metadata frames:

1. **count-present**: the original exact int/non-bool
   `metadata.count == len(features)` check, byte-unchanged;
2. **observed count-absent API-2.7.0 frame**: `count` absent AND
   `api == "2.7.0"` AND int/non-bool `status == 200` AND
   `url == expected_url` AND int/non-bool `limit == 20000` AND
   int/non-bool `offset == 1` AND int/non-bool `generated`; then the
   existing truncation guard `len(features) < limit`.

Any missing or mutated field refuses typed
`MF4_CATALOG_METADATA_FRAME`. No separate count HTTP request is
added (it would expand the registered 13-query acquisition). Locks
C6a–C6n: the observed six-feature 2.7.0 frame passes; wrong/missing
api, status, url, limit, offset, generated each refuse; bool limit
refuses; count-absent truncation refuses; count-present mismatch
still refuses; `--plan` remains zero HTTP.

## Item 2 — attempt-1 immutable refusal capsule

The attempt-1 staging evidence was MOVED byte-unchanged (never
rewritten, never deleted) to
`docs/f2g_window2_execution/mf4_catalog_attempt1_refusal/` and bound
by `ATTEMPT1_REFUSAL_CAPSULE.json`
(schema `geospec-mf4-catalog-attempt1-refusal-capsule-v1`): raw
response `9ce8ff6a…` (5,246 B, HTTP 200, 6 features, exact URL
echo), attempt record `08f4e4fa…` (2,176 B), refusal manifest
`857f161f…` (4,641 B), the exact anchorage query URL, the full
attempt-1 authority chain (head `58782d8e` / tree `fd096a6b` / pass
`d9a394c0` / go `03ea9cce` / auth v4 / module `0afdd095…`), and
explicitly: `queries_fired=1`, `queries_not_fired=12`,
`snapshot_published=false`, `owner_go_status=SPENT`. A `-text` pin
covers the directory. Lock A8 verifies every sealed byte identity
against the capsule, that the sealed metadata is genuinely
count-absent 2.7.0, and that no snapshot directory exists.

## Item 3 — attempt-2 discipline

Attempt 2 is ONE fresh uniform transaction: all 13 regions queried
exactly once under the corrected parser; the sealed anchorage bytes
are NOT reused; output target and staging must be absent (the
attempt-1 evidence now lives under a distinct versioned path).
Execution will run from a FRESH clean worktree at the exact public
tip (no parking of unrelated untracked files). The authority chain
must be rebuilt in full on the v5 schema: this correction lands
publicly on asylum's word → codex bounded re-review of the count
seam + a fresh strict pass → a fresh strictly-later owner
`OWNER_FIRE_GO` binding the new tip/tree/pass and the exact
attempt-2 scope literal above. If attempt 2 refuses, stop again; no
automatic retry.

## Locks

**77/77 PASS** on both store routes (local + s4t share): A1–A8,
B1–B5, C1–C6n, D1–D9k, E1–E19. `--plan` 13/13 zero HTTP. The Gate-2
adapter quarantine (`QUARANTINED_PENDING_GATE2_TRUST_ANCHOR`) is
unchanged and does not ride on this repair. **Λ_geo remains
INCONCLUSIVE.**
