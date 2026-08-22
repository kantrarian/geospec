# Window-2 EXECUTION-MANIFEST v1 — schema contract (2026-08-22)

Instantiates the PRESTART deliverable declared in design-manifest-v2.1's
`manifest_class` (codex revision-1 finding 3 / revision-2 positive check): "a closed
execution-manifest-v1 (generator/verifier, selection + adapter + accrual
implementations, typed MAG capsules, calibration ledgers, bars, producer code) plus an
executable verifier ... REQUIRED PRESTART deliverables."

This document is the byte-authoritative schema for
`docs/f2g_window2_execution/execution_manifest.json`. It binds the EXECUTABLE surface
of window-2 the way design-manifest-v2.1 binds the design surface. It does NOT modify
the CLOSED design freeze (target `12161f6` + manifest `5fba544`); it links to it.

## Scope rule (execution closure)

Every byte of code executed during selection, accrual, barrier staging, or evaluation
MUST be committed in THIS repository (`kantrarian/geospec`) and pinned in a BOUND slot
before PRESTART. The allowlist can only close over bytes the manifest can pin;
out-of-repo code is unpinnable and therefore unrunnable. (Runtime allowlist
enforcement lives in the accrual/barrier instruments, as in the sealed run.)

## Top-level fields (closed set)

`schema` = `f2g-window2-execution-manifest-v1`; `generated_utc` (live clock read);
`repository_url`; `execution_target_commit` (all pins resolve last-touch FROM this
commit; must be an ancestor of the manifest's own commit); `target_ref` =
`origin/master`; `design_manifest_commit` + `design_manifest_blob_sha256` +
`design_target_commit` (the linkage block, below); `manifest_state`; `slots`.

## Design linkage (verifier-enforced)

- `design_manifest_commit` resolves; the blob
  `<design_manifest_commit>:docs/f2g_window2_freeze/byte_pin_manifest.json` hashes to
  `design_manifest_blob_sha256`; that parsed manifest's `design_target_commit` equals
  the declared `design_target_commit`; the commit is an ancestor of
  `execution_target_commit`.
- The execution verifier RUNS the full design-pin walk (via the pinned
  `design_pin_verifier`) against `design_manifest_commit` and refuses on anything but
  PASS (`DESIGN_WALK_FAILED`). Before executing it, the verifier compares the
  CRLF→LF-normalized disk bytes of the design verifier to its BOUND pin's blob sha
  (`EXECUTED_BYTES_MISMATCH` on divergence) and attests the executed sha in the
  verdict. Normalization rule: git blobs (LF) are byte-authoritative; `*.py`
  attestation comparisons normalize CRLF→LF only.

## Slots (closed set of 10)

Each slot is `{status, owner, note, pins}` (+ `families`, bars only). `status` ∈
{`BOUND`, `OPEN`}. BOUND ⇒ non-empty `pins`; OPEN ⇒ empty `pins` + an owner + note.
Each pin is the closed set `{path, commit, blob_sha256}` with: pin commit an ancestor
of `execution_target_commit`; pin commit == last-touch of path AT the target; blob
bytes at `commit:path` hash to `blob_sha256`.

| slot | owner | binds |
|---|---|---|
| `execution_generator` | cayley | this manifest's generator |
| `execution_verifier` | cayley | the executable verifier below |
| `design_pin_verifier` | cayley | the design-pin walk executable (landed `b755ce1`) |
| `selection_impl` | cayley | cutoff-stable selection per prereg v0.3 + selection_constants |
| `adapter_impl` | cayley | window-2 family adapter (B2A/B2B/B1B/B3A) over the frozen graph |
| `accrual_impl` | cayley | sealed prediction accrual + two-stage barrier instruments |
| `mag_capsules` | cayley | typed at-freeze station capsules IZN/FRN/TUC (VIC/NEW already design-pinned) |
| `calibration_ledgers` | cayley | MAG-1 subtraction coefficients + diagnostics, committed pre-evaluation |
| `bars` | grassmann | executable bar file(s); when BOUND must declare `families` exactly = {W-SEL, W-CAS, W-B2B, W-B1B, W-MF4, W-MAG, W-BARRIER, W-PIN} |
| `producer_code` | grassmann | accrual producers (seismic + MAG raw byte acquisition) |

## States and the PRESTART gate

`manifest_state` = `CLOSED` iff every slot is BOUND, else `OPEN` (any inconsistency is
`MANIFEST_STATE_WRONG`). Re-binding a slot = regenerate against a new target commit
and commit the new manifest (git history is the binding ledger); pre-PRESTART only.
The verifier's `--prestart` mode refuses ANY open slot (`SLOT_OPEN`). **PRESTART
requires: `manifest_state=CLOSED` + verifier PASS in `--prestart` mode + the design
walk PASS — and separately codex's round over the bound bytes. A verifier PASS here
authorizes NOTHING by itself**: no fire, no prospective-value access; `Lambda_geo`
remains `INCONCLUSIVE`.

## Amendment v1.1 (2026-08-22, codex 1358Z convergence pass)

Schema id becomes `f2g-window2-execution-manifest-v1.1`. The closed slot set grows to
**12**: `power_harness` (cayley — §6 power machinery; the certification entry point
constructs its own config and requires a verified bound-geometry capsule) and
`calibration_runner` (cayley — temporal-boundary + M3-index + provenance-receipt
repaired) join as explicit runtime-allowlisted slots. `mag_capsules` pins target
`docs/f2g_window2_execution/mag_capsules[/receipts]` (the codex 0451Z relocation);
`mag_capsules` and `bars` re-bind ATOMICALLY at the four-group REV-9 green per the
codex 1335Z/1358Z disposition. All other rules unchanged.

## Typed refusal vocabulary

`MANIFEST_COMMIT_UNRESOLVABLE`, `MANIFEST_NOT_IN_COMMIT`, `MANIFEST_UNPARSEABLE`,
`MANIFEST_SCHEMA_MISMATCH`, `TOP_FIELD_MISSING`, `TARGET_UNRESOLVABLE`,
`TARGET_NOT_ANCESTOR_OF_MANIFEST_COMMIT`, `REPO_IDENTITY_MISMATCH`,
`DESIGN_COMMIT_UNRESOLVABLE`, `DESIGN_NOT_ANCESTOR`, `DESIGN_BLOB_SHA_MISMATCH`,
`DESIGN_TARGET_INCONSISTENT`, `DESIGN_WALK_FAILED`, `EXECUTED_BYTES_MISMATCH`,
`SLOT_SET_NOT_CLOSED`, `SLOT_SCHEMA_NOT_CLOSED`, `SLOT_BOUND_WITHOUT_PINS`,
`SLOT_OPEN_WITH_PINS`, `SLOT_OPEN` (prestart mode), `BARS_FAMILY_SET_MISMATCH`,
`PIN_SCHEMA_NOT_CLOSED`, `PATH_NOT_AT_TARGET`, `NON_ANCESTOR_PIN`,
`LAST_TOUCH_MISMATCH`, `BLOB_MISSING`, `BLOB_SHA_MISMATCH`, `MANIFEST_STATE_WRONG`.
