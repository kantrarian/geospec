# PREREGISTRATION STATUS REGISTRATION v1 (successor w2r1)

**Append-only owner-authorized status artifact** under codex's
Gate-2 ruling (2026-08-30T18:44Z, finding 4, MODERATE): the pinned
preregistration file self-identifies as "DRAFT v0.3 / NOT frozen",
leaving a hostile-reader ambiguity between the freeze manifest's
intent and the contract's own status declaration. This record
resolves that ambiguity WITHOUT editing the pinned historical bytes.

- **drafted**: 2026-08-30T22:08Z (cayley, successor packet w2r1)
- **owner authorization**: asylum-worded landing push (the owner
  fires geospec public pushes personally; that push binds this
  record). Drafting authorized in-session 2026-08-30 as part of the
  consolidated successor packet.

## The operative preregistration, by exact identity

| field | value |
|---|---|
| pin id | `prereg_v03` (in `docs/f2g_window2_freeze/byte_pin_manifest.json`) |
| path | `docs/f2g_window2_prereg_DRAFT.md` |
| blob sha256 | `0ed87943b11349cf79dcae05a7ed99248f7e63d746242c86194b7fca47903a07` |
| pin commit | `5ccd533a8cc41a4279826f35bea33a7ef7546a27` |
| imported section | whole-file (imported_section_sha256 == blob sha256) |

## Status resolution

1. The operative Window-2 preregistration **is the pinned bytes**
   identified above. Operative status: **FROZEN_BY_REGISTRATION** —
   frozen by the byte-pin manifest registration and sealed by the
   owner's freeze decision, regardless of the legacy header label.
2. The in-file header words "DRAFT v0.3" and "NOT frozen" are a
   **legacy label predating the freeze registration**. They are
   historical bytes, deliberately left unedited (pinned files are
   never rewritten); this record supersedes their status claim.
3. The file's `_DRAFT` filename is likewise historical; the path is
   part of the pinned identity and does not change.

## What the successor imports

The successor revision **imports the pinned prereg unchanged**,
except where these registered append-only mechanisms supersede
specific content — each with its own owner authorization and bound
artifact, none touching the pinned bytes:

- **§1 concrete calendar dates** → superseded by the successor
  schedule (`prestart_schedule_2026-09-03.md`) and calendar
  authority v4, per the schedule-supersession mechanism the prereg
  itself carries (§1 semantics — evaluation_start = first UTC day
  after PRESTART completes — are unchanged; only the dates move).
- **MAG primary set membership** → superseded by
  `mag_primary_terminal_exclusion_v1.md` (owner option 1 under
  Gate-2 finding 1).
- **M-F4 catalog snapshot** → already amended AMENDED_AFTER_FREEZE
  via the bound `amendment_mf4_late_catalog_repair_20260829.md`
  chain (owner option 1 of 2026-08-29; not re-opened here).

No other section is superseded. Anything not listed above binds as
pinned.

## Claim ceiling

Status registration only; no power value, no scientific claim, no
evaluation open; Λ_geo INCONCLUSIVE.
