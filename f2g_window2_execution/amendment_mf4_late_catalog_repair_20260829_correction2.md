# CORRECTION 2 (append-only) — MF4 late-catalog repair amendment: acquisition module v2 per the codex 1901Z pre-HTTP ruling

- **UTC registered**: 2026-08-29T19:16:05Z (live clock read)
- **corrects**: the module pin of
  `amendment_mf4_late_catalog_repair_20260829.md` as already
  superseded by correction 1 (`8f34b117…`). Both earlier files are
  append-only and untouched; this correction supersedes the module
  pin only, now to the v2 identities below.
- **implements**: codex 2026-08-29T19:01:36Z WORKS-WITH-FIX ruling,
  all four fixes, in one composition.

## Superseding identities (both frames bound separately, fix 1)

- module runtime sha256:
  `0732e6c3f7ff75f7985e09183f9137dbecda141030450301420a44db55b3626b`
- module git-blob (LF) sha256:
  `0732e6c3f7ff75f7985e09183f9137dbecda141030450301420a44db55b3626b`
  (equal because `.gitattributes` now pins
  `monitoring/src/w2_mf4_*.py text eol=lf`, closing the
  checkout-frame divergence codex demonstrated)
- query-contract sha256 (canonical constants object):
  `cf19d414b1ba98a38953332b8940e12460c203babb5eadd139a19ce4bb530095`

## The four 1901Z fixes

1. **Fire authority**: `--fire` refuses without
   `--fire-authorization <json>` (schema
   `geospec-mf4-fire-authorization-v1`) binding the exact public
   amendment/correction commit, BOTH module identities, the
   query-contract digest, the codex-pass inbox commit, the owner
   fire-go quote/time/scope, and `output_target_must_be_absent`.
   The module recomputes its own bytes in both frames and refuses
   `MF4_FIRE_AUTH_*` on any mismatch BEFORE any HTTP; the authority
   file + digest bind into every receipt.
2. **Transactional staging**: exclusive staging directory
   (pre-existing target/staging or link/reparse escape refuses);
   every response's raw bytes + attempt metadata sealed with
   exclusive creation BEFORE parsing; any failure writes a terminal
   typed `REFUSAL_MANIFEST.json` binding every attempt and preserves
   the staging evidence; success publishes the directory atomically;
   a second fire refuses staging/target reuse (continuation = a new
   owner decision reusing sealed bytes, never re-querying).
3. **Closed parser**: non-finite JSON constants refused;
   FeatureCollection/Feature/Point types closed; integer non-bool
   `metadata.count == len(features)`; nonempty string IDs; finite
   non-bool lat/lon/mag; integer non-bool epoch ms; `mag >= 4.0`;
   registered bbox/time membership; nondecreasing `(time_ms, id)`;
   content-type allowlist; effective-URL == requested-URL (redirects
   refuse).
4. **Transactional identity binding**: every attempt receipted
   (requested + effective URL, status, headers, exception/refusal
   code, raw length/digest, parser result); ONE canonical global
   event table sorted `(time_ms, id)`, identical cross-region
   duplicates deduplicated (inconsistent shared IDs refuse),
   region membership as a side channel, table digest bound for the
   downstream adapter.

## Archive capsule additions (same ruling)

`mf4_archive_capsule_v1.json` now binds
`file_census = {present_files: 307, usable_files: 306,
malformed_files: 1}` and the explicit 142-entry
`missing_region_cells` list. Capsule rebuilt; independent verifier
green (307 objects, 4,298 rows).

## Locks

Full suite `w2_mf4_archive_kats_grassmann.py`: **22/22 PASS**,
including the new D-series (D1 fire refuses without authorization
pre-HTTP; D2 partial-failure staging leaves a complete refusal
manifest + immutable sealed region-1 bytes + no success snapshot,
and a second fire refuses reuse before any request; D3 the closed-
parser mutation battery — NaN/Infinity constants, boolean magnitude,
mag 3.0 below threshold, metadata.count mismatch, reverse order,
non-Point geometry, non-FeatureCollection; D4 canonical-table dedup
+ region-order digest invariance).

Zero HTTP has been performed at any point. The fire still waits on
codex's single pre-HTTP close of this composition and the in-session
owner go. **Λ_geo remains INCONCLUSIVE.**
