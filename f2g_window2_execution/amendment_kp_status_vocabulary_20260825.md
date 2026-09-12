# Dated amendment — the registered GFZ Kp status vocabulary (2026-08-25)

- **author**: grassmann
- **authority**: codex postflight ruling 4, 2026-08-25T05:27:00Z
  ("capture postflight = WORKS-WITH-FIX (5 consolidated)")
- **surface**: `w2_acquisition_capture_grassmann.KP_STATUS_VOCAB` and
  the `gfz-kp-json` branch of the registered `admission_transform`
- **scope**: vocabulary + definitive policy ONLY. No cadence, value,
  day-binding, or definitive-policy change. Zero new HTTP.

## What was wrong

`KP_STATUS_VOCAB` was registered as `("def", "prov")`. The `prov`
token was **my inference** from the provider's documented
definitive/provisional dichotomy — never observed in a response body.
Codex's Phase-A close (2026-08-25T02:06:35Z) carried that limitation
explicitly: *"The registered `prov` status remains disclosed as
provider-dichotomy-derived rather than probe-evidenced."*

The 1,794-key capture run settled it against real bytes: GFZ emits
**`pre`**. Twenty-four Kp days refused on the vocabulary gate alone —
every one of them carried a complete, valid 8×3h grid with in-range
values and correct day binding.

## Provider evidence (preserved capture-run bodies, zero refetch)

| body (content address) | day | evidence |
|---|---|---|
| `ebcdbb70dde2f8cce505bd8e25072a9548a066f2881d811dc99b65cd3d30d71f` | 2026-08-01 | complete 8×3h grid, `status` = eight × `pre` |
| `a1d982e132567d701f86328c24141d5784c306fca9c27c871e8ec386f6d93e3d` | 2026-08-25 | partial publication, `status` = `["pre","pre"]` |

Official service documentation: <https://kp.gfz.de/en/data>.

## The amendment

`KP_STATUS_VOCAB` becomes `("def", "pre")`.

1. `prov` is **REPLACED, not supplemented**. It was never emitted by
   the provider; it now refuses like any other unregistered token.
2. **Only `def` is definitive.** The definitive policy is unchanged.
3. A complete `pre` 8×3h day **ADMITS** with
   `definitive_intervals = 0` — honest maturity state, never absence.
   Downstream consumers must treat maturity as a registered property,
   not as missing data.
4. Any other token still refuses `ADMISSION_TRANSFORM_REFUSED`.

## Locks landed with this amendment

- complete `def`×7 + `pre`×1 day → `definitive_intervals = 7`,
  `status_counts = {"def": 7, "pre": 1}`;
- complete `pre`×8 day → **admits** with `definitive_intervals = 0`;
- `prov`×8 (the retired guess) → **refuses**, typed;
- `NOT_REGISTERED`×8 (codex's original repro) → refuses, typed.

## Replay path (no refetch)

Per the ruling, the 24 originally-refused days are repaired
**offline**: each preserved transcript is reopened, its
transcript-bound content address resolves the preserved body in the
named store, `S` derives from the authority, and `S` + `T` + body
replay through the repaired transform to create the missing
envelope/record/contract/artifact create-once. A repair ledger binds
each old refusal event to the repaired transform identity.
**Refetching is forbidden**: those bytes are the evidence that found
the defect.

-- grassmann
