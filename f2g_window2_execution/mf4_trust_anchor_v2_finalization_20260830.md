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

## Repair rev 2 (codex 2026-08-30T02:57Z WORKS-WITH-FIX, both items)

- **UTC registered**: 2026-08-30T03:08:10Z (live clock read)
- adapter superseded (eol=lf):
  `b9a1a12767a0f2b4cf34069fa0d76600319c8082650e509ea2dbd0ecc2d127b5`

**Blocker 1 — result-byte authentication.** The trust root now
includes the RESULT COMMIT: this later reviewed commit binds
`catalog_commit 4893e632…` (parent == authorized head `f636c234…`),
both registered paths, both Git blob OIDs
(snapshot `4c87f45f…`, receipt `55993672…`), and both byte SHA-256
values (`490c4077…` / `054002dd…`) as adapter constants — no
self-reference (the acquisition commit predates this one).
`authenticate_result_bytes` runs FIRST in the sanctioned loader:
caller bytes must hash to the pinned identities; the catalog commit
must be on the trusted public ref and the registered descendant of
the authorized head; both files are reopened via
`git show <catalog_commit>:<path>` and must equal the caller bytes
byte-for-byte with the pinned blob OIDs. A genuine-chain replay
into forged result bytes refuses
`MF4_CATALOG_RESULT_UNAUTHENTICATED` (lock D10k — codex's exact
reproduction: real pair, table→[], all in-band digests recomputed).
`_validate_pair` is the documented post-authentication stage the
mutation locks exercise (fixture pairs can never authenticate); it
is not a sanctioned consumer entry — consumers get
`load_verified_snapshot` = authenticate + validate.

**Blocker 2 — semantic reconciliation.**
`verify_snapshot_semantics` runs deterministically before the
loader returns: attempts cover exactly the 13 registered regions
and each agrees with the recomputed contract
(region/requested_url/bbox/carrier/status 200/parse OK/event
count); table ids unique + (time_ms, id)-ordered; magnitude >=
registered threshold; time inside the registered window; `time_utc`
the exact millisecond rendering of `time_ms`; membership keys ==
table ids, region lists unique/sorted/registered, every row inside
every listed region bbox; per-region counts equal both the attempt
counts and the membership recompute; total regional counts equal
membership cardinality (dedup disclosed, 203 vs 200 unique). Locks
D10l/m/n (codex's three accepted mutations now refuse typed) +
D10o (positive recompute 203/200/203 on the real pair). D8 reworked
onto the REAL pair through the FULL sanctioned loader into the REAL
frozen calibrate (synthetic risk over the real 13 regions); the
amended-training binding now records the recomputed trust anchor.

**93/93 PASS** on both store routes at rev 2.

## Repair rev 3 (codex 2026-08-30T03:17Z WORKS-WITH-FIX, D10p–t)

- adapter superseded (eol=lf):
  `4a25b8d441a22e351222a489c8509dec2310d3b1e387a914c85c12e846b9784d`
- builder superseded:
  `84a4245b10324de843164facc541558009a8729ee6a710552a2bc8202e708bc4`
- capsule rebuilt (result-commit identity bound):
  `4fe32158b3028a807f7903862a0d1d788ae49de7ca59f63b94ce8d47ca1e5cb3`

**Item 1 — bytes-only operational boundary.**
`load_verified_snapshot` now returns the attestation identity ONLY
(no snapshot object leaves the authenticated stack);
`events_from_snapshot(snapshot_bytes, receipt_bytes, use)` is
bytes-only — it re-runs full result authentication + validation +
raw composition before conversion and returns an immutable tuple; a
dict input refuses typed. `calibrate_with_snapshot` performs the
whole flow inside one stack. `_validate_pair` (the lock seam)
returns pass/refusal diagnostics only — nothing any event or
calibration operation accepts. Locks: D10p (codex probe 1: the
post-load 199-row mutated object cannot reach conversion), D10q
(codex probe 2: the self-consistent 199-row pair refuses at every
event-emitting/calibration entry and the seam returns no
event-bearing object).

**Item 2 — result-commit identity in the attestation.**
`authenticate_result_bytes` returns the RECOMPUTED closed
`result_authentication` record (catalog commit + parent + trusted
ref + both paths + both blob OIDs + both byte shas); it binds into
the loader identity, `amended_training_binding`, and the capsule's
`catalog_binding` (constants re-imported at reconstruction — never
copied from claimed fields). Locks: D8 asserts the exact recomputed
`4893e632` identity in the real calibration ledger; D10r mutates
the capsule's bound result commit and refuses reconstruction.

**Item 3 — exact attempt semantics + raw composition.**
`verify_snapshot_semantics` now enforces the CLOSED attempt keyset
(15 registered fields, typed), exact `params == recomputed`, exact
`effective_url == requested_url`, and `raw_file` naming.
`verify_raw_composition` (permanent, runs on the sanctioned load
path) reopens all 13 committed raw blobs from the catalog commit,
checks raw_file/raw_sha256/raw_bytes per attempt, reparses through
the REGISTERED `validate_events`, canonicalizes, and requires exact
equality with the snapshot table/membership/counts/digest. Locks:
D10s-a/b/c (forged params, forged effective URL, keyset injection
refuse), D10t-a (positive 13-raw → 200-unique/203-regional replay),
D10t-b (raw/receipt divergence refuses).

**101/101 PASS** on both store routes: A1–A8, B1–B5, C1–C6n,
D1–D10t, E1–E20. `--plan` 13/13 zero HTTP. No new HTTP; the
one-fire authority is spent. `producer_boundary=OPEN`;
`calibration_ledgers=OPEN`; `prestart_overall=REFUSE`. **Λ_geo
remains INCONCLUSIVE.**
