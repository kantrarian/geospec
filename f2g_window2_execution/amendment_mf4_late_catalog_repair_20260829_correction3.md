# CORRECTION 3 (append-only) — MF4 late-catalog repair: the codex 1942Z five-item composition

- **UTC registered**: 2026-08-29T19:59:37Z (live clock read)
- **corrects**: the module pin lineage of
  `amendment_mf4_late_catalog_repair_20260829.md` (original →
  correction 1 → correction 2); all earlier files untouched. This
  correction supersedes the module pin and adds the
  feature-provenance supersession of item 4.
- **implements**: codex 2026-08-29T19:42:50Z consolidated pre-HTTP
  review, all five items, one composition.

## Superseding identities

- module (runtime == git-blob, `text eol=lf`):
  `1ba235423bc3d0a71980696edf4853df2d05e7d3ffc775ef0048d780fc753072`
- query contract:
  `cf19d414b1ba98a38953332b8940e12460c203babb5eadd139a19ce4bb530095`

## Item 1 — externally recomputable fire authority (schema v2)

`geospec-mf4-fire-authorization-v2` binds: the clean public worktree
`HEAD` + `HEAD^{tree}` (HEAD must equal `origin/master`); the module
blob recomputed via `git show HEAD:<path>` (never a runtime copy)
with runtime bytes required to LF-normalize to that exact blob; THIS
correction document in the same committed tree pinning the same
module/query identities; a codex PASS record at a commit reachable
from `agent-framework/main` whose bytes bind PASS + base commit +
patch sha + resulting tree + module sha + query sha; and an owner
fire-go record identifying that PASS commit, carrying a strictly
parsed UTC (`%Y-%m-%dT%H:%M:%SZ`), the EXACT scope literal
("MF4 late-calibration-catalog acquisition: exactly one fire of the
13 registered ComCat queries under
amendment_mf4_late_catalog_repair_20260829 and its corrections;
nothing else"), and a committed source file whose bytes contain the
quote. The go-source digest binds into every receipt. Lock D1b:
correct self-hashes + forged pass/go records refuse
`MF4_FIRE_AUTH_PASS_UNBOUND` before the opener is ever called.

## Item 2 — finalization inside the transaction guard

Snapshot and receipt serialize to canonical bytes, seal with
exclusive creation, reopen/re-hash/re-parse, fsync (file-level;
directory fsync unavailable on Windows, disclosed), then atomic
directory publish — all inside one terminal guard emitting
`MF4_FIRE_FINALIZATION_EXCEPTION` + a refusal manifest and
preserving staging on ANY exception. Lock D6: injected failures at
snapshot seal, receipt seal, and directory publish each refuse
typed, never create the final target.

## Item 3 — full-recompute capsule verifier

`verify_capsule()` now starts from the receipt → capsule byte
digest, then independently recomputes: schema; region partition +
alias; maturity source bytes and bounds (against both the source
file and builder constants); fault-segments bytes + all bboxes; the
full day grid; inventory partition + object count; file census;
every raw object; every row REPLAYED IN BUILD ORDER (content, order,
uniqueness, count in one comparison); rows-file digest/bytes/lines +
row schema; support census; missing-cell list; producer identity.
Locks E1–E8 mutate each bound surface (with the receipt re-bound so
the deeper check under test is the one that fires) — all refuse
typed. The 307-object raw store is being copied to
`\\192.168.50.1\s4t\geospec_mf4_risk_store_v1\` so the real A6
replay is independently runnable.

## Item 4 — temporal-role supersession (feature provenance)

**Verbatim policy**: historical calibration `recent_event` is
RECOMPUTED from the single late-repair snapshot; this supersedes the
annex_mf4 issue-time feature clause ("`recent_event ∈ {0,1}`: any
qualifying event in `(d−7, d)` per the frozen catalog view AT issue
time") as an **AMENDED_AFTER_FREEZE feature-provenance change with
no original-preregistration standing**. The snapshot embeds
`temporal_role: CALIBRATION_LATE_REPAIR` + this policy; the policy
and snapshot digest MUST bind into the training digest at v2
finalization. Live/post-2026-08-29 prediction features use
separately receipted issue-time catalog views;
`calibration_snapshot_role_guard()` refuses the late snapshot for
any live use. Lock D5: calibration_labels / calibration_features
admitted; live use and unbound roles refuse.

## Item 5 — byte-safe transport

This composition routes as a **git bundle** rooted at public
`da938bd7`. Application recipe (the only advertised one):

```text
git -c core.autocrlf=false clone/worktree at da938bd7
git -c core.autocrlf=false fetch <bundle> <tip>
verify: module blob sha, query sha, capsule sha, rows sha
```

Bare `git am`/`git apply` on Windows is NOT sufficient and is no
longer advertised.

## Locks

Full suite: **33/33 PASS** (A1–A6, B1–B5, C1–C5, D1–D6 incl. D1b
forged-chain + D5 role guard + D6 triple injection, E1–E8 capsule
mutations). Zero HTTP has been performed at any point. The fire
waits on codex's single verification of this exact tree, the public
landing (asylum's word), the codex PASS record, and the fresh
in-session owner go bound into the v2 authorization file.
**Λ_geo remains INCONCLUSIVE.**
