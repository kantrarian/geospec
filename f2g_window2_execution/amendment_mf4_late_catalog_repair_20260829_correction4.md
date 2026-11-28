# CORRECTION 4 (append-only) — MF4 late-catalog repair: the codex 2014Z five-item composition

- **UTC registered**: 2026-08-29T20:30:55Z (live clock read)
- **corrects**: the module-pin lineage (original → c1 → c2 → c3); all
  earlier files untouched. This document is the identity-pin source
  the fire-authorization verifier reads from the committed tree.
- **implements**: codex 2026-08-29T20:14:37Z bounded review, all five
  findings, one composition.

## Superseding identities

- acquisition module (runtime == git-blob, eol=lf):
  `cd5a40e7b8014899c532b8a2d552449e09d852f3ff64a61b4ec8fa841e04a694`
- query contract (unchanged):
  `cf19d414b1ba98a38953332b8940e12460c203babb5eadd139a19ce4bb530095`
- catalog adapter (new, eol=lf):
  `2970de4d64b2cb059778cd4635d2d8e8434abd5c0d5f8797b2c9006b064ff173`
- archive builder/verifier:
  `c4c9a84fc76748b4a359a5c063c5964bc820a91922d2cf06aaba56782a3b22e3`
- archive capsule (rebuilt from the exact LF frame):
  `59a928ab8ca52a5c1ab4f900e96ef2e0052e1da94dce6c1e75037a70f9598a1c`

## Item 1 — one explicit EOL frame + selectable store

`mf4_maturity_record_v4.json` and `fault_segments.py` now carry
`text eol=lf` and their working bytes are LF (maturity 2,302 B
`74507258…`; fault_segments 34,410 B `90acfd35…` — exactly the Git
blob identities codex computed). The capsule is REBUILT from that
frame, so its source pins equal the bundled blobs and the ordinary
command reproduces in a fresh LF worktree. The builder/verifier/KATs
accept `--store-dir <path>` or `GEOSPEC_MF4_STORE` so the routed
UNC S4T store is selectable without monkeypatching.

## Item 2 — authority v3: structured verdict + immutable-ref chain

Schema bumped to `geospec-mf4-fire-authorization-v3`. The pass
record must be strict JSON with the exact enum
`verdict: "PRE_HTTP_PASS"` (substring matching is gone) binding
base_commit / bundle_sha256 / result_tree / module sha / query sha;
`result_tree` must equal the CURRENT `HEAD^{tree}` and the reviewed
base must be an ancestor of HEAD; pass and go-source commits must be
reachable from **origin/main** (never mutable local main); the
go-source commit must DESCEND from the pass commit and its committed
bytes must bind the pass commit, current public HEAD, current tree,
the exact scope literal, the exact UTC, and the quote. Locks D1c
(HOLD verdict containing the word PASS refuses pre-opener) and D1d
(valid pass + old unbound go refuses pre-opener).

## Item 3 — complete capsule/receipt verification + risk range

`verify_capsule()` now reconstructs the ENTIRE deterministic capsule
through the single `_construct_capsule()` used by build and requires
exact equality — claim/provenance/locator/status fields, top-level
keyset, inventory object names (`<sha256>.body`) and deterministic
host paths included — plus full receipt validation (schema, strict
built-UTC, capsule path, recomputed counts/digests, empty refusal
list). `combined_risk` is enforced on [0.0, 1.0] at build AND
replay. Locks E9–E19 (field forgeries, key injection, receipt
schema/hidden-refusal, out-of-range risks both sides).

## Item 4 — 28-file durability

`_seal()` fsyncs and reopen-verifies EVERY sealed file (raw
responses and attempt records included). Lock D7: the positive
13-region transaction publishes exactly 28 files with ≥28 fsync
calls; a raw-seal fsync injection refuses typed with no final
target. D7 also exposed and removed a real Windows defect: the prior
finalization fsync used O_RDONLY descriptors (EBADF on Windows) and
its failure had been masked inside D6's expected refusal — disclosed.

## Item 5 — the guard on the real consumer path

New `w2_mf4_catalog_adapter_grassmann.py` (the frozen `w2_mf4.py` is
byte-untouched): `calibrate_with_snapshot()` guards BOTH temporal
roles, runs the frozen `calibrate()` unchanged, and emits
`amended_training_digest = sha256(engine_digest || policy ||
canonical-table sha)` with the full binding recorded —
policy or snapshot change moves the amended digest while the engine
digest stays fixed. `live_prediction_events()` requires a separately
receipted `ISSUE_TIME_VIEW` and refuses `CALIBRATION_LATE_REPAIR`
typed. Lock D8 exercises the REAL frozen calibrate through the
adapter (digest movement, live refusal, receipt-less refusal).

## Locks

**48/48 PASS**: A1–A6, B1–B5, C1–C5, D1–D8 (incl. D1b/c/d, D7
durability), E1–E19. `--plan` 13/13 zero HTTP. The fire waits on
codex's single verification of this tree, public landing (asylum's
word), the structured codex PASS record, and the fresh in-session
owner go. **Λ_geo remains INCONCLUSIVE.**
