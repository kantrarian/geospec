# CORRECTION 5 (append-only) — MF4 late-catalog repair: the codex 2359Z three-blocker composition

- **UTC registered**: 2026-08-30T00:11:08Z (live clock read)
- **corrects**: the module-pin lineage (original → c1 → c2 → c3 → c4);
  all earlier files untouched. This document is the identity-pin
  source the fire-authorization verifier reads from the committed
  tree.
- **implements**: codex 2026-08-29T23:59:08Z consolidated pre-HTTP
  review (`PRE_HTTP_HOLD` / WORKS-WITH-FIX), all three blockers, one
  composition.

## Superseding identities

- acquisition module (runtime == git-blob, eol=lf):
  `0afdd0950c3e8532f24c7a4fc78e765183605d30f8d1dcc4ccae1c2117c3c846`
- query contract (unchanged):
  `cf19d414b1ba98a38953332b8940e12460c203babb5eadd139a19ce4bb530095`
- catalog adapter (eol=lf):
  `27ce4c023f233c628d7819660e024ac3d4b51b77b46a25a629c9ceaed5692d48`
- archive builder/verifier:
  `2a22fea1d22d4c32746e3ae0ffae7636690b0a94dc67436b0d55f2db5407fd0b`
- archive capsule (rebuilt, physical store path removed from
  identity):
  `2feea82fb365646ac826858d39f3bac53fc4f43bc53407cce42592ba42ae04a9`
- fire-authorization schema bumped:
  `geospec-mf4-fire-authorization-v4`

## Blocker 1 — the owner-go source is now structured authority

Text-substring matching over the committed go source is gone. The
authorization wrapper's `owner_fire_go` carries ONLY the untrusted
source pointer (`source_framework_commit`, `source_file`). The
committed go source itself must be strict JSON with the exact enum
`verdict: "OWNER_FIRE_GO"`, and every semantic field — `quote`,
strict `utc` (`%Y-%m-%dT%H:%M:%SZ`), exact `scope` literal,
`pass_framework_commit`, `public_head_commit`, `public_head_tree` —
is taken FROM that parsed record and compared exactly against the
recomputed values; the wrapper can no longer manufacture them. The
go-source commit must be reachable from origin/main, must DESCEND
from the pass commit, and must NOT be the pass commit itself
(`MF4_FIRE_AUTH_GO_SAME_COMMIT`). Locks: D1e (a HOLD go source
carrying EVERY required token/field refuses pre-opener, textual →
`MF4_FIRE_AUTH_GO_UNPARSEABLE` and structured →
`MF4_FIRE_AUTH_GO_VERDICT`), D1f (a nominally valid go record at the
pass commit refuses; the same record at a later commit verifies —
positive control included), D1d re-pointed to the unparseable-text
refusal.

## Blocker 2 — the real consumer path is receipt-bound

`w2_mf4_catalog_adapter_grassmann.py` no longer consumes bare dicts.
`load_verified_snapshot(snapshot_bytes, receipt_bytes)` is the ONLY
loader: strict-parse both byte streams; exact snapshot and
acquisition-receipt schemas; `sha256(snapshot_bytes)` must equal
`receipt.snapshot_sha256`; the canonical-table digest recomputed
from the table must equal the digest bound in BOTH snapshot and
receipt (an absent digest refuses `MF4_CATALOG_TABLE_DIGEST`); the
temporal role must be exactly `CALIBRATION_LATE_REPAIR` and the
policy must equal the registered `TEMPORAL_ROLE_POLICY` literal
byte-for-byte; the pinned query-contract digest and the
authorization identity must be bound identically in snapshot and
receipt. `calibrate_with_snapshot` takes the two byte streams,
verifies, then runs the frozen `w2_mf4.calibrate` unchanged; the
amended-training binding now also records `snapshot_sha256` and the
authorization sha256. `live_prediction_events` refuses EVERY input
typed: the late snapshot as `MF4_CATALOG_ROLE_VIOLATION` and ALL
`ISSUE_TIME_VIEW` inputs as `MF4_CATALOG_LIVE_UNVERIFIED`, because
no registered issue-time-view receipt verifier exists yet — a truthy
receipt string is not a receipt; the live path gains a positive
branch only when such a verifier is registered and reviewed. Locks:
D8 (reworked: receipt-bound bytes through the REAL frozen calibrate;
digest movement; all three live refusals including codex's exact
forged-view probe) and the D9 battery (D9a–D9k: fake receipt bytes,
wrong receipt schema, tampered snapshot bytes, absent bound digest,
forged snapshot digest, receipt digest mismatch, forged policy,
unbound query contract, authorization divergence, bare-dict
consumption, role forgery through the loader).

## Blocker 3 — capsule identity is portable across physical stores

`local_physical_root` is REMOVED from the capsule (it was
environment-specific state inside exact semantic equality — the
committed capsule could not replay through the routed s4t alias).
Capsule identity now binds only the logical store root
(`s4t://geospec/mf4/risk_archive_v1`) plus the content-addressed
inventory. The physical path at build time is recorded in the build
RECEIPT under `store_observation`, explicitly non-authoritative.
Capsule + receipt REBUILT with the new producer identity; rows file
byte-unchanged (`7e359280…`). Lock A7: one committed capsule
verifies through two physical aliases of the same object store with
byte-identical capsule and identical verifier output, and the
capsule bytes are asserted free of any physical-path field. The
advertised ordinary command now passes 62/62 from BOTH the local
root and `\\192.168.50.1\s4t\geospec_mf4_risk_store_v1`.

## Locks

**62/62 PASS** on both store routes: A1–A7, B1–B5, C1–C5, D1–D9k
(incl. D1b/c/d/e/f, D7 durability, receipt-bound D8), E1–E19.
`--plan` 13/13 zero HTTP. The fire waits on codex's bounded
verification of this tree (ordinary share KAT + the two structured
authority negatives + the receipt-bound calibration/live negatives),
public landing (asylum's word), the structured codex PASS record,
and the fresh in-session owner go. **Λ_geo remains INCONCLUSIVE.**
