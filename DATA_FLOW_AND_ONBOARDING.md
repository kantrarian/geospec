# GeoSpec Data Flow & New-Area Onboarding Guide

**Version 1.0 — 2026-08-23 (cayley).** Reference documentation only: nothing in this
guide authorizes acquisition, staging, PRESTART, publication, or any scientific
claim. Carrier sets and windows are frozen by the governance artifacts they cite;
bringing a real new area online requires the full owner-gated cycle described in
§5. Λ_geo remains INCONCLUSIVE.

---

## 1. The data the system requires

| # | Requirement | Source | Native format | Cadence |
|---|---|---|---|---|
| R1 | Seismic waveform station-days per carrier | FDSN web services (SCEDC, NCEDC/IRIS, KOERI/Kandilli, …), channel HHZ | miniSEED day volumes → merge-trim-split | daily |
| R2 | Station-pair coherence values ("edges") | derived from R1 per registered station pair | JSON edge series `{"A|B": {"YYYY-MM-DD": r}}`, r in tanh space | daily per carrier |
| R3 | Station-day admissibility (selection input) | derived from R1 | day-records `{iso_day: [station ids with ≥1 admissible sample]}` | daily; selection needs exactly `[cutoff−89, cutoff]` |
| R4 | Magnetometer minute streams | INTERMAGNET GIN (IZN/Iznik), USGS geomagnetism (FRN/Fresno, TUC/Tucson) | IAGA-2002-style minute text, 1440 samples/day, HDZS or XYZF frames | 1-minute native → daily features |
| R5 | Space-weather drivers | SYM-H / Kp / OMNI services | numeric time series, canonical-UTC stamps | hourly/daily; calibration span 2026-01-01 → cutoff |
| R6 | Earthquake catalog (scoring, forecast lanes) | USGS/regional catalogs | event records (origin time, lat, lon, mag, depth) | on query at maturity |
| R7 | Precipitation (daily-monitor R5 lane only) | precip services | daily series per region | daily |

Governance requirements that ride every data class:

- **R8 — Canonical UTC.** One registered timestamp grammar: canonical UTC minute
  strings ending `Z`. Timezone-aware parsing; any non-UTC offset refuses (the
  `+14:00` wrong-UTC-day trap). Ordering/uniqueness run on normalized instants.
- **R9 — Closed schemas.** Every layout is a closed field set; unknown fields
  refuse. There are deliberately **no spare columns for future expansion** —
  extension happens only by versioned, registered schema amendment
  (manifest v1 → v1.1 → v1.2; geometry capsule v1 → v2).
- **R10 — Content addressing.** Raw bytes are SHA-256-digested before any
  transformation; every derived artifact carries a receipt whose `output_sha256`
  digests exactly the returned artifact; consumers reopen bytes from git objects,
  never trusting disk.
- **R11 — Registered absence.** A missing value is `None`/absent at a fixed
  calendar position — never a deleted position, never NaN (non-finite refuses
  structurally at the boundary).
- **R12 — Claim ceiling.** Acquisition correctness before the staged bytes is
  receipt-attested, not source-code-attested; no artifact asserts more than its
  verification surface covers.

---

## 2. The trust boundary

The single registered ingestion point is the **staged-input envelope**
(manifest schema v1.2, `producer_boundary` slot, `boundary_mode="staged_envelope"`).
An envelope is one closed record per lane per UTC day binding:

```
f2g-w2-staged-envelope-v1:
  schema, lane, carrier, utc_day, raw_body_sha256, raw_body_bytes,
  source, endpoint, request_params, receipt, capture_time_utc,
  cutoff, operation_params, expected_keys, output_sha256
```

Refusals at this boundary: missing/extra day vs the expected day set; cross-day or
cross-carrier replay; wrong source/request/body digest; non-UTC capture time;
non-finite value; any schema extension; any envelope not admitted by the
registered mode.

---

## 3. Data movement, start to end

Each step names its artifact and the refusal gate that protects it.

| Step | What happens | Artifact produced | Gate |
|---|---|---|---|
| S0 | **Registration/freeze** — carrier registry, station caps (istanbul_marmara 16 / socal_coachella 20 / turkey_kahramanmaras 14 / cascadia 16), bbox + metric CRS, effect grids, design pins | freeze capsules + byte-pin manifest | design-pin verifier walk (27 pins) |
| S1 | **Acquisition** — fetch raw bytes (FDSN waveforms, MAG minute files, space-weather series, catalog) | raw bodies | receipt-attested only (R12) |
| S2 | **Staging** — digest raw bytes into closed envelopes | staged-input envelopes + receipts | envelope schema refusals (§2) |
| S3 | **Producer transforms** — merge-trim-split waveforms; day-record reduction; MAG component-map to geographic XYZ + SOS filter chain + day features | `(artifact, receipt)` pairs, `output_sha256` exact | one-field mutation doctors; non-finite refusal |
| S4 | **Selection** — 90-day frame `[cutoff−89, cutoff]`, frozen cap, presence derived (never trusted) | selected station registry per carrier | `SelectionInputInvalid` frame/cap refusals |
| S5 | **Panel assembly** — edges over the fixed 192-position calendar-v2 grid (baseline 60 = 2026-06-27→08-25; eval 132 = 2026-08-27→2027-01-05; 08-26 excluded); availability is a separate mask | engine panels + views | `CALENDAR_MASK_COMPRESSION`, `CALENDAR_EXCLUDED_DAY`, `CALENDAR_AUTHORITY_MISMATCH` |
| S6 | **Engines** — B2A runs / B2B churn / B1B burst / B3A, Holm at α=0.05 over the four; MAG lane M1/M2/M3; MF4 predictions | family p-values + typed refusals | engine `PanelInvalid` classes; frame records recomputed (`POWER_CALENDAR_FRAME_INVALID`) |
| S7 | **Calibration** — MF4/MAG ledgers produced at the cutoff, temporal boundaries enforced | calibration ledgers + verified receipts | `CALIBRATION_TIME_INDEX_INVALID`; receipt keyset equality |
| S8 | **Power certification** — Tier-C at n_draws=9,999, R=20/40 exact-binomial stopping, B1B LOCO fold rule | certification records binding capsule digest + frame digests | geometry/authority/frame refusals; `POWER_LOCO_FOLD_SET_INVALID` |
| S9 | **Manifest binding** — 12 closed slots, every pin = path+commit+blob-sha | `execution_manifest.json` (CLOSED 12/12) | verifier `--prestart` zero-OPEN PASS; 19-case KAT |
| S10 | **Admission + owner seal** — live verification (verifier PASS + runtime allowlist + owner binding vs exact manifest blob/lanes/lease/window) | admission capsule; barrier ledger PRESTART | `PRESTART_ADMISSION_REFUSED`; `RUNTIME_ALLOWLIST_VIOLATION` |
| S11 | **Sealed accrual** — daily envelope → producer → engines → hash-chained append-only ledger | ledger events + MF4 prediction rows | `LEDGER_CHAIN_BROKEN`; `LATE_OR_REVISED_PREDICTION` |
| S12 | **Scoring & outputs** — labels at maturity (H_max 7 d tail), dashboard render | scored records; `docs/data.csv`, `r5_daily.json`, dashboard pages | append-only; re-scores only on adjudicated defect |

Storage at every step is **git**: committed blobs addressed by
`git cat-file blob <commit>:<path>`, with the execution manifest as index.
Daily-monitor raw inputs stay in gitignored `monitoring/data/`; sealed evidence
mirrors to the GCS versioning bucket.

---

## 4. Reaching the data

| Need | Tool |
|---|---|
| Raw governed bytes at any pin | `git cat-file blob <commit>:<path>` (manifest = the index) |
| Verified manifest state | `python monitoring/src/f2g_execution_manifest_verifier_cayley.py <repo> <commit> [--prestart] [--kat]` |
| MAG capsules with sha-verified bodies | `w2_mag1.load_execution_capsule(name, manifest_commit)` |
| Selection / engines / certification | import `w2_selection`, `w2_b1b`, `w2_b2b`, `w2_power_harness_cayley` |
| Calibration receipts | `w2_calibration_runner_cayley.verify_receipt(...)` |
| Calculated JSON/CSV values | `jq`, pandas (`docs/data.csv`, `docs/r5_daily.json`, `docs/ensemble_latest.json`) |
| Human-facing | dashboard pages under `monitoring/dashboard/` (Pages + `serve.bat`) |

---

## 5. Worked example — bringing a new area online

The factual precedent is **cascadia**, onboarded for window-2 (bbox lat 45.0–51.0,
lon −128.0…−121.5, cap 16, the NEW-registry LOCO carrier). The steps below follow
that path for a hypothetical carrier `hokkaido_hidaka`. **Illustrative only — a
real onboarding starts with an owner ask and runs the full review loop.**

1. **Owner scope decision.** asylum authorizes the new area as program scope
   (region, scientific rationale, window).
2. **Region registration.** Register bbox + metric CRS (e.g. lat 41.8–43.2,
   lon 141.5–143.8, EPSG:32654) and the candidate seismic network(s) (e.g.
   NIED Hi-net / JMA via FDSN). Commit as a carrier capsule under
   `docs/<window>_freeze/` — the cascadia template is
   `cascadia_carrier_capsule.md`.
3. **Probe + registry census.** Fetch a probe day per candidate station through
   the staged-envelope discipline; record availability, channel (HHZ), sample
   rate, gaps. Output: candidate registry with evidence receipts.
4. **Cap + registry freeze.** Fix the station cap and the frozen registry in the
   selection constants; register the effect grids the new carrier participates
   in. This is a DESIGN change: it lands in the freeze manifest with new byte
   pins, before any evaluation data exists.
5. **Producer lane.** Extend the producer contract with the new carrier's
   envelope expectations (`expected_keys`, endpoints, request params). Every
   transform must return `(artifact, receipt)` with exact `output_sha256`.
6. **Calendar binding.** The new carrier's engine-facing `registered_days` must
   equal the window's fixed authority grid byte-for-byte; its availability mask
   is separate. No carrier ever gets its own calendar.
7. **Selection dry-run.** 90 days of day-records at the frozen cap; verify the
   greedy order and tie-break against the oracle in the bar.
8. **Bar KATs.** The peer bar-author (grassmann) cross-authors known-answer
   tests for every new seam: registry census, envelope doctors, selection
   oracle equality, engine geometry refusals.
9. **Adversarial review.** codex reviews the design + implementation pair;
   repairs loop until PASS/CLOSED.
10. **Manifest slots.** The new artifacts join their slots; manifest regenerated;
    verifier PASS + KAT green.
11. **Certification.** Power certification re-run if the carrier changes any
    certified geometry (it does — panels grow), at the registered tiers.
12. **Owner seal.** asylum's authorization binds the exact manifest blob; the
    barrier ledger admits the carrier into the next window's PRESTART. Only now
    does prospective evaluation data begin to accrue.

Cost anchor: cascadia's onboarding ran through exactly this loop across the
window-2 campaign — none of these steps is skippable, and the calendar/authority
steps (6) exist precisely because a compacted day list silently slides the
evaluation start (see `calendar_authority_w2_v2.md`, KAT 2).

---

## 6. Visualization

`monitoring/dashboard/data_flow.html` renders this guide's content live: a
pipeline flow diagram (S0→S12 with gates) and a topographic map (OpenTopoMap
tiles) showing the four window-2 carrier bounding boxes, the three MAG
observatories (IZN 40.5N 29.7E; FRN 37.091N −119.718E; TUC 32.174N −110.733E),
and the daily-monitor regions. Serve locally with `monitoring/dashboard/serve.bat`
or via the Pages deployment.
