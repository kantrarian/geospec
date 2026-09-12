# AMENDMENT — MF4 late calibration-catalog repair (2026-08-29) — APPEND-ONLY

- **UTC registered**: 2026-08-29T18:28:50Z (live clock read)
- **lane status**: **AMENDED_AFTER_FREEZE** — this lane is no longer
  represented as the original untouched preregistration.
- **authority**: asylum (owner), in-session, 2026-08-29 ~18:25Z,
  verbatim "go with option 1, authorize the amendment and acquisition"
  (grassmann session) / "proceed with option 1" (cayley session),
  choosing option 1 of the codex 2026-08-29T17:58:37Z ruling.
- **codex ruling implemented**: the MF4 archive/catalog ruling of
  2026-08-29T17:58:37Z (WORKS-WITH-FIX archive; embedded catalog
  REFUSED; late-repair contract pinned).

## Disclosure (why this amendment exists)

The frozen annex (`docs/f2g_window2_freeze/annex_mf4.md`) requires
calibration labels from a **pinned PRE-FREEZE calibration-catalog
snapshot** with query receipts. **That snapshot was never taken; the
prerequisite was missed before freeze.** No admissible substitute
exists on the monitor host: the per-day embedded dashboard events
are a lossy top-5-per-region projection on unregistered bboxes
(codex CRITICAL 1), and the validation-kit snapshot is M≥5.5 on a
different query set. This amendment repairs the miss LATE and in the
open: the snapshot acquired under it is a **late repair snapshot**,
never a pre-freeze artifact.

Integrity bounds preserved:
- **No 2026-08-29+ MF4 prediction/outcome bytes were opened** before
  this registration; the acquisition window ends 2026-08-28T00:00:00Z,
  strictly before the accrual span (2026-08-29 onward).
- Any evaluation issue day missed before the calibration ledger binds
  remains a typed late/no-prediction day — **never backfilled**.
- Later catalog revisions never enter this calibration snapshot.

## The pinned acquisition contract (fires ONCE, after codex pre-HTTP review + in-session owner go)

- **code**: `monitoring/src/w2_mf4_catalog_acquire_grassmann.py`,
  sha256
  `0cd406e773ab9008e95f4b3cd8044e11c2c64fb7fc07de1a8a4e1d26a0dfc09e`
  (this exact byte identity; the module refuses if its recomputed
  bbox table diverges from the pinned one).
- **provider**: USGS ComCat FDSN event query
  (`https://earthquake.usgs.gov/fdsnws/event/1/query`).
- **queries**: exactly 13, one per registered polygon-union bbox
  (FAULT_SEGMENTS vertices at source sha256
  `13834b75f396ca3c…` full value bound in the module run receipts);
  alias `socal_saf_coachella -> socal_coachella`; typed exclusion
  `tokyo_kanto: MF4_BBOX_UNREGISTERED`. The 13 numeric bboxes are
  pinned verbatim in the module (`PINNED_BBOXES`).
- **temporal**: superset `[2025-10-11T00:00:00Z, 2026-08-28T00:00:00Z]`;
  local admitted filter `2025-10-11T00Z <= t < 2026-08-28T00Z`
  (the early start feeds the first calibration day's `(d-7,d)`
  persistence feature; the end covers all `(d,d+7]` labels through
  issue day 2026-08-20).
- **params**: `minmagnitude=4.0`, `format=geojson`,
  `orderby=time-asc`, `limit=20000` (count == limit refuses
  `MF4_CATALOG_QUERY_LIMIT`).
- **fields kept**: id, exact UTC origin time, lat, lon, magnitude.
- **typed refusals**: query limit; missing/duplicate IDs;
  cross-region inconsistent IDs; malformed/null
  coordinates/magnitude/time; events outside the registered
  temporal/spatial filter.
- **receipts**: per-region exact URL + params, request/response UTC,
  HTTP status, content type, raw bytes + sha256 + count, plus
  acquisition-code identity, fault_segments identity, bbox identity,
  and the completeness policy (ComCat as-is; per-region completeness
  caveat disclosed).
- **outputs**: `docs/f2g_window2_execution/mf4_catalog_snapshot/`
  (13 raw geojson files + `catalog_snapshot_v1.json`, schema
  `geospec-mf4-calibration-catalog-snapshot-v1` +
  `acquisition_receipt_v1.json`, schema
  `geospec-mf4-catalog-acquisition-receipt-v1`).

## Claim ceiling

This amendment repairs a data-provenance prerequisite only. No
scientific admission, publication, promotion, prospective-value or
method claim follows. `producer_boundary=OPEN`,
`calibration_ledgers=OPEN`, `prestart_overall=REFUSE`, Gate 2 OPEN.
**Λ_geo remains INCONCLUSIVE.**
