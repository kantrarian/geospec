# POWER-CERT RESULT CONTRACT v1 (successor w2r1)

Contract for the Window-2 **anticipated-mask power certification
RESULT PACKAGE** required by codex's Gate-2 ruling (2026-08-30T18:44Z,
finding 3, CRITICAL): the CLOSED manifest's `power_harness` slot
binds machinery only; no certification output, power envelope,
selector result, or result receipt exists at `8cc564e7`. This
contract fixes, BEFORE execution, exactly what the run must produce
and bind, so the result lands under reviewed terms. Harness KATs and
slot closure cannot substitute for these produced result bytes.

## Authority preconditions (all BEFORE the run fires)

1. codex one-pass PASS on successor packet w2r1;
2. owner words the prestart date; successor schedule + calendar v4
   + v4-cal amendments + governance records LANDED on public master
   (asylum-fired pushes);
3. the run executes **under the successor calendar v4** (the
   geometry capsule binds `calendar_authority_w2_v4.json` by sha256)
   and under owner authority. Synthetic-only: zero HTTP, zero
   evaluation bytes, opens no window-2 value.

## Execution path (frozen machinery, exact identities)

Pinned in the `power_harness` slot, amended for v4-cal only:
`w2_power_harness_cayley.py`, `w2_geometry_capsule_gen_cayley.py`
(v4-cal successor blobs, re-pinned at landing),
`w2_cert_runner_cayley.py`, `w2_tier_selector_cayley.py`,
`f2g_phase_b_power_estimation_cayley.py`, `…_cal_cayley.py`,
`effect_grids_w2_v1.json` (frame-agnostic — reused unchanged),
`loco_composition_amendment_v1.md`,
`tier_selector_amendment_w2_v1.md`. Sizing per
`power_cert_sizing_v2.json` (timing-only artifact; Tier-C ≈ 5 h wall
at 7 procs). Order: Tier-S smoke → COMMITTED selector artifact
(fire input is `(selector_commit, selector_path)`; 14-point shape,
gains 3 then 10) → Tier-C campaign fire.

## Required outputs — `docs/f2g_window2_execution/power_cert/`

1. `invocation_record.json`, `campaign_summary.json` — the cert
   runner's atomic create-once artifacts (REV 3 semantics:
   invocation_sha256 authenticated by every worker; typed aborts).
2. `power_cert_result_package_v1.json` — the assembly:
   - **identities**: input/code/effect-grid/geometry byte identities
     — every `power_harness`-slot blob sha256, calendar v4 artifact
     sha256, bound geometry capsule digest, selector
     `(commit, path, sha256)`, invocation_sha256;
   - **four-family result**: one member per family
     `B1B / B2A / B2B / B3A` — power surface over the registered
     effect grid, or **typed non-certification** where applicable
     (never omitted, never a silent gap);
   - **certified selector S** — the certified selection with its
     rule identity, or typed non-certification;
   - **per-lane envelopes**: each admitted accrual lane's
     anticipated-mask power envelope and threshold; MAG-dependent
     lanes appear as **typed non-certification consistent with
     `mag_primary_terminal_exclusion_v1.md`** — never power numbers.
3. `power_cert_result_receipt_v1.json` — invocation argv, host,
   interpreter, UTC clock reads (live reads at generation, start and
   end, stated ≤ commit time), exit code, and the sha256 of every
   output byte artifact above.
4. `power_cert_verifier_receipt_v1.json` — INDEPENDENT verifier
   receipt (verifier ≠ executor): rebuilds the package's derivable
   surfaces from committed bytes or returns a typed
   consistency-only status; never a bare `true`.

## Binding

- The execution manifest regenerates with a **new slot**
  `power_certification_result` pinning the exact output bytes above
  (registered-path pattern; duplicate keys refuse; membership read
  from git objects at the target). The `power_harness` slot re-pins
  the two v4-cal amended source blobs. Then the ordinary composed
  prestart verifier reruns — **PASS required** before any accrual.
- Two named producers ride the execution round as candidate bytes
  under THIS contract (flagged in the w2r1 cover; codex may veto in
  the one pass): `w2_power_cert_results_assembly_cayley.py` (builds
  2–3 from 1 + committed inputs; fixtures mirror the real tree) and
  the `f2g_execution_manifest_gen_cayley.py` slot patch.

## Claim ceiling

The result package certifies DESIGN POWER against synthetic
injections under anticipated masks. It is not detection, not
admission, not evidence about the Earth; no evaluation value opens.
Λ_geo INCONCLUSIVE.
