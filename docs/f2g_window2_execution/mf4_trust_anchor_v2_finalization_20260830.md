# MF4 trust-anchor repair + archive-capsule v2 finalization (grassmann)

- **UTC registered**: 2026-08-30T02:50:30Z (live clock read)
- **implements**: codex 2026-08-30T00:24Z Gate-2 quarantine repair
  (option 2 chosen: reopen + fully re-verify the committed
  authorization chain) and the v2 catalog binding that follows the
  successful attempt-2 acquisition (fired 2026-08-30T02:39:54Z).
- **the frozen engine `monitoring/src/w2_mf4.py` remains
  byte-untouched**; the acquisition module is UNCHANGED from
  correction 6 (`64140fc0…` — its one fire is spent).

## Identities

- catalog adapter (eol=lf):
  `b0573ae038d153103574785d8be11b889b30f612ff8d4b3a14fee3bf1578cd86`
- archive builder/verifier:
  `5541e89dbeafec91b93ccce31626d93279526d49088f951a900ccfb302dbf1b8`
- archive capsule (rebuilt, catalog binding BOUND_V2):
  `1e5590ec8eb14451d075e1839ccb74d6637f7f87bee24f1649202914dada87e5`
- bound snapshot `490c4077…` / receipt `054002dd…` / canonical table
  `1e4839cf…` (200 events)

## Part 1 — the trust anchor (quarantine repair, codex option 2)

`load_verified_snapshot` now additionally enforces:

- **closed keysets** for receipt AND snapshot (exact registered
  fields; any missing/injected key refuses typed
  `MF4_CATALOG_RECEIPT_KEYSET` / `MF4_CATALOG_SNAPSHOT_KEYSET`);
- **named snapshot path** (`snapshot_file ==
  "catalog_snapshot_v1.json"`) and **strict fired UTC**
  (`%Y-%m-%dT%H:%M:%S.%fZ`);
- **registered row schema** on every canonical event (exact keyset
  id/lat/lon/mag/time_ms/time_utc with typed field checks,
  `MF4_CATALOG_ROW_SCHEMA`);
- **`verify_acquisition_trust_anchor`**: the embedded
  `authorization_content` (exact keyset, registered schema, pinned
  contract digest) is only a CLAIM — the function reopens the
  COMMITTED chain: the codex pass record (reachable from framework
  origin/main, strict JSON, `PRE_HTTP_PASS`, module/contract/tree
  pins equal to the claim), the owner go record (strictly later,
  descends from the pass, strict JSON `OWNER_FIRE_GO`, binds
  pass/head/tree/scope), and the geospec side (the authorized head
  is public history — ancestor of origin/master — with the exact
  claimed tree, and the module blob AT that commit equals the
  pinned identity, which must also equal the receipt's
  `acquisition_code_identity`). Snapshot and receipt must carry the
  identical content; the recomputed anchor (pass/go commits,
  head/tree, module blob) is returned and bound into
  `amended_training_binding` by `calibrate_with_snapshot`. A
  mutually self-issued pair — internally digest-consistent with a
  fabricated authorization — refuses `MF4_CATALOG_TRUST_ANCHOR`
  because its commits are not committed history.

Locks D10a–D10j: the self-issued pair refusal codex required
(D10a, run against the REAL verifiers, no patching), keyset
injection both sides, wrong named file, non-strict UTC, row keyset
injection + bool magnitude, forged authorization module pin,
snapshot/receipt authorization divergence, and the POSITIVE D10j:
the real committed attempt-2 snapshot/receipt pair verifies through
the real committed chain (pass `7601d385`, go `561cfdf4`, head
`f636c234`) and yields the 200-event table. D8/D9 fixtures now
carry full closed keysets and run against a fixture chain whose
pass/go records are valid (the same _FakeGit seam the D1 authority
locks use); the old self-issued D8 positive is gone.

**Quarantine exit is codex's ruling, not mine**: the adapter stays
`QUARANTINED_PENDING_GATE2_TRUST_ANCHOR` until this packet passes
review.

## Part 2 — capsule v2 finalization (catalog binding)

`_catalog_binding()` in the builder binds the committed snapshot +
receipt identities into the capsule deterministically (recomputed
from committed bytes at build AND reconstruction — a tampered
snapshot file diverges capsule equality, lock E20):
`status: BOUND_V2`, snapshot/receipt paths + sha256, canonical
table sha, temporal role, the registered amended-training-digest
formula, and `training_digest_status:
PENDING_ONE_AUTHORIZED_CALIBRATION_RUN` — the
`training_row_digest` closes only at the gated ONE real calibration
run (disclosed, unchanged from the two-stage design). Sandboxes
without the snapshot stay OPEN (fixtures unchanged). A8 was
re-pointed: attempt-1 evidence remains immutable and byte-verified,
and any published snapshot must NOT reuse attempt-1's spent go or
sealed response (checked against the real receipt).

## Locks

**88/88 PASS** on both store routes (local + s4t share): A1–A8,
B1–B5, C1–C6n, D1–D10j, E1–E20. `--plan` 13/13 zero HTTP. No new
HTTP; the one-fire authority is spent. `producer_boundary=OPEN`;
`calibration_ledgers=OPEN`; `prestart_overall=REFUSE`. **Λ_geo
remains INCONCLUSIVE.**
