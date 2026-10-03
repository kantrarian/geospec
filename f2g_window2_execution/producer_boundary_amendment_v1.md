# PRODUCER TRUST-BOUNDARY AMENDMENT v1 (2026-08-23) — staged-envelope mode

**Provenance**: codex ruling 2026-08-23T14:00Z ruling 2 selects **option (ii)**
(staged-envelope boundary), **subject to grassmann's boundary-owner ratification**;
cayley authors this schema per the same ruling ("Cayley: bind the calendar
capsule/adapter contract and producer-boundary schema for a normal Codex repair
round. Grassmann: please ratify the boundary mode and wire the doctors."). Amends
the execution-manifest schema v1.1 → **v1.2**. Until grassmann ratifies, the slot
stays OPEN and nothing stages.

## Slot amendment (v1.2)

The `producer_code` slot is RENAMED **`producer_boundary`** (codex's preferred
form). The slot object carries a REQUIRED `boundary_mode` field whose only
registered value is `"staged_envelope"`; a missing or divergent mode refuses
(`PRODUCER_BOUNDARY_MODE_UNREGISTERED`). The claim ceiling is structural:

> Acquisition correctness BEFORE the staged bytes is receipt-attested, not
> source-code-attested. The runtime allowlist and all derivation provenance start
> at the content-addressed envelope boundary (producer REV 2/3 verification
> surface). Nothing in a BOUND `producer_boundary` slot asserts that acquisition
> code was attested.

## BOUND conditions (codex 1400Z, binding)

A BOUND staged-envelope slot MUST pin, and the execution verifier refuses
(`PRODUCER_BOUNDARY_PINS_INCOMPLETE`) a BOUND slot missing any class — a note
string or empty pin set can never turn the slot BOUND:

1. **This amendment** (`docs/f2g_window2_execution/producer_boundary_amendment_v1.md`)
   and the envelope verifier code.
2. **The producer transform/aggregation code from the envelope onward**
   (`monitoring/src/` pins; grassmann's producer REV 2/3 surface).
3. **Closed per-lane/per-day envelope records** under
   `docs/f2g_window2_execution/staged_envelopes/`, each binding: raw body SHA-256 +
   byte size; source/endpoint/request parameters; receipt; capture time (UTC);
   exact UTC day; carrier/lane; cutoff; operation parameters; expected key/day
   set; and output digest.
4. **The claim ceiling** (carried by this amendment's registered text above).

## Envelope-record closed schema (registered here; grassmann wires the doctors)

```
f2g-w2-staged-envelope-v1 (closed field set; schema extension refuses):
  schema, lane, carrier, utc_day, raw_body_sha256, raw_body_bytes,
  source, endpoint, request_params, receipt, capture_time_utc,
  cutoff, operation_params, expected_keys, output_sha256
```

`--prestart` recomputes every pin and record from Git objects and refuses:
missing/extra day vs the expected day set; cross-day or cross-carrier replay;
wrong source/request/body digest; non-UTC capture time; non-finite value; any
schema extension; any envelope not admitted by the registered mode. Slot-level
shape doctors live in the execution verifier (cayley); per-record content doctors
live in the producer verification surface (grassmann, on ratification).

## Implementation binding (this amendment)

- `monitoring/src/f2g_execution_manifest_gen_cayley.py` +
  `_verifier_`: schema v1.2, slot rename, `boundary_mode` enforcement, BOUND
  pin-class refusals, KAT cases.
- `monitoring/src/w2_accrual_instrument_cayley.py`: runtime allowlist accepts the
  v1.2 schema id.
- `docs/f2g_window2_execution/execution_manifest_schema_v1.md`: v1.2 appendix.

No staging, acquisition, or data admission is authorized by this amendment alone;
grassmann's ratification + codex's repair round close it. Λ_geo INCONCLUSIVE.

---

## v1.1 APPENDIX (2026-08-23; codex 1843Z item 4) — the external-store successor

The base amendment's `--prestart` clause says pins/records recompute "from Git
objects". codex's 1843Z ruling registers its EXTERNAL-STORE SUCCESSOR before any
staging: raw bodies live on the s4t content-addressed store, NOT in Git history;
the repo commits and manifest-pins a closed `staged_body_inventory`
(`f2g-w2-staged-body-inventory-v1`: logical store id/root + per lane/carrier/day
the DERIVED path `<sha256>.body`, digest, byte size — derived paths make escape
impossible by construction). The prestart admission path must reopen EVERY
object from the named store — path-escaping, missing, extra, unavailable-store,
or content-mismatched objects refuse typed, NEVER PASS — recompute body +
produced artifact, and only then run the S/T/E three-carrier join (static
contract / capture transcript / envelope record; codex 1843Z rulings 1–3,
implemented producer REV 6 + capture REV 2 @ `dfb0772`). An inventory hash is
never mistaken for completed-build availability: availability = the reopen
succeeding at admission time. Implementation:
`w2_acquisition_capture_grassmann.verify_staged_body_inventory` (grassmann) +
the admission wiring in `w2_accrual_instrument_cayley.assemble_prestart_admission`
(cayley). Λ_geo INCONCLUSIVE; nothing here authorizes staging by itself.
