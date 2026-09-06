# CORRECTION 1 (append-only) — MF4 late-catalog repair amendment: acquisition-module digest superseded

- **UTC registered**: 2026-08-29T18:44:24Z (live clock read)
- **corrects**: `amendment_mf4_late_catalog_repair_20260829.md`,
  which pinned `monitoring/src/w2_mf4_catalog_acquire_grassmann.py`
  at sha256 `0cd406e773ab9008e95f4b3cd8044e11c2c64fb7fc07de1a8a4e1d26a0dfc09e`.
  That file is append-only and is NOT edited; this correction
  supersedes its module pin only.

## What changed and why

The codex-required KAT "Tokyo-to-Tohoku mapping refuses" exposed a
seam gap found by my own lock run (B2, 2026-08-29T18:4xZ): a region
configured outside the registered 13 whose alias resolves to an
EXISTING carrier (e.g. `tokyo_kanto -> japan_tohoku`) crashed with a
raw `KeyError` at the pin-table lookup instead of refusing typed.
Production behavior is unchanged (the shipped constants never reach
that path); the repair adds one guard so an unpinned region refuses
`MF4_BBOX_UNREGISTERED` — never resolving through a neighbouring
carrier, per the ruling's "the prior cross-geography mapping must
not return".

- **superseding module sha256**:
  `8f34b1172fd096f0e708a343752d9b47e5a8fec3bc427155aa85472e85b6be10`
- diff scope: one guard clause + message in `build_bboxes()`; no
  query-contract field, bbox value, URL, param, refusal type
  (existing), or receipt field changed.
- Locks after repair: **18/18 PASS**
  (`monitoring/src/w2_mf4_archive_kats_grassmann.py`, incl. B2 now
  typed); `--plan` re-run: 13/13 bbox pins exact, zero HTTP.

Zero HTTP has been performed at any point. The fire still waits on
codex's pre-HTTP review (now of THIS composition) and the in-session
owner go. **Λ_geo remains INCONCLUSIVE.**
