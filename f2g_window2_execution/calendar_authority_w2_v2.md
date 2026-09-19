# WINDOW-2 CALENDAR AUTHORITY v2 (registered 2026-08-23) — `w2-calendar-v2-noncal`

**Provenance**: codex ruling 2026-08-23T14:00Z (calendar option (a) AUTHORIZED with
the non-compression adapter contract), issued against cayley's 0117Z design routing
under the owner PRESTART decision ("prestart 8/26 it is", quote sha `5b44934a`,
schedule artifact `b49ab2c`). codex explicitly authorized replacing the 1358Z item-2
`_cal` certification seam with the PINNED non-cal `b2a_family`/`b3a_family` calls;
the pinned Phase-B engine is NOT touched. Machine-readable authority:
`calendar_authority_w2_v2.json` (producer:
`monitoring/src/w2_calendar_authority_gen_cayley.py` — an independent derivation the
harness selftest cross-checks byte-for-byte).

## Registered objects (codex 1400Z, verbatim)

```
baseline_days   = 2026-06-27 .. 2026-08-25  (60 exact UTC dates)
excluded_days   = [2026-08-26]               (PRESTART; never an engine position)
evaluation_days = 2026-08-27 .. 2027-01-05  (132 exact UTC dates)
engine_days     = baseline_days || evaluation_days  (192 positions)
```

## The non-compression adapter contract (codex 1400Z, binding)

- For B2A/B3A/B1B, every carrier's engine-facing `registered_days` MUST equal
  `engine_days` byte-for-byte. Per-carrier anticipated/realized availability is a
  SEPARATE bound mask; an unavailable date remains a fixed calendar position with no
  value and is never deleted or compacted. A compacted list refuses
  `CALENDAR_MASK_COMPRESSION` (the hazard: removing one baseline availability day
  from a compacted list silently makes 2026-08-27 baseline position 60 and slides
  evaluation to 08-28 — KAT 2 demonstrates it against the pinned
  `walk_forward_split`).
- B2B receives exactly `evaluation_days`; its per-carrier mask continues to yield
  typed `NO_REGISTERED_SNAPSHOT` positions.
- The adapter refuses any mask or value on 2026-08-26 (`CALENDAR_EXCLUDED_DAY`);
  any shifted/extra/missing authority date refuses `CALENDAR_AUTHORITY_MISMATCH` —
  both BEFORE generation or an engine call.
- B1B parameters are AUTHORITY FIELDS, never fallbacks: `n_blocks=16`,
  `block_len=12`, `baseline_positions=60`; exactly five baseline blocks and eleven
  evaluation blocks are asserted.
- B2A/B3A wrapper records add the frame metadata the non-cal results lack: exact
  pinned entrypoint + engine blob sha256, `w2-calendar-v2-noncal`,
  baseline/evaluation/mask digests, and the input-panel digest. The certification
  artifact REFUSES absent or divergent frame metadata (`POWER_CALENDAR_FRAME_INVALID`,
  every expected value recomputed, never trusted from a caller label).

## Cutoff binding (codex ratification)

Cutoff = **2026-08-25**: the greatest UTC date whose complete-day bytes exist before
PRESTART execution begins, strictly earlier than `evaluation_start`; the PRESTART
day is excluded. If PRESTART does not complete on 08-26, the schedule is SUPERSEDED
(append-only successor artifact), never silently slid.

## Implementation binding

`monitoring/src/w2_power_harness_cayley.py` (geometry capsule schema
`f2g-w2-bound-geometry-v2`): `w2_calendar_frame()` derivation,
`_validate_calendar_frame` / `_validate_carrier_mask` adapter refusals,
`make_bound_panels` fixed-grid generation, `replicate_pvalues_bound` non-cal seams +
frame records, `_validate_frame_records` recomputing refusal,
`_b1b_loco_recovery` authority-field geometry. Certification records bind
`calendar_frame_id`, day-array digests, and the engine blob sha.

## Locking KATs (codex 1400Z; implemented in the harness selftest)

1. Exact date arrays/counts/endpoints and explicit 08-26 exclusion — asserted
   against BOTH the in-module derivation and the committed authority JSON.
2. Remove one baseline availability day: engine split remains 60/132 on the fixed
   authority grid; a compressed `registered_days` list refuses
   `CALENDAR_MASK_COMPRESSION`.
3. Any 08-26 mask/value or any shifted/extra/missing authority date refuses before
   generation or an engine call.
4. `_cal` entry points monkeypatched to fail: the bound path completes untouched
   (only non-cal entrypoints run); doctored entrypoint/blob/frame digests refuse.
5. Cross-family alignment: one replicate id/seed produces one 192-position raw
   frame; B2B is the exact 132-position evaluation projection; B2A/B3A/B1B share
   the full fixed frame and preserve mask holes.

Selftest green on Python 3.11 / 3.12 / 3.14 (dual-interpreter discipline). No power
value, certification claim, or window-2 scientific claim is opened by this
registration; Λ_geo remains INCONCLUSIVE.
