# geo2graph add-ons roadmap v1 (cayley, owner-directed 2026-08-21)

Owner direction: b2 (fresh-window sealed replications) standing until project end; prediction is
the program goal (aftershocks included); add-ons below extend capability and public value.
Status: SCOPING — nothing here is authorized for public exposure without the per-component
asylum go + codex claims pass. Live-verified facts are marked ✓.

---

## MAG-1 — magnetometer channel (research add-on, governed lane)

**Hypothesis space**: crustal-stress electromagnetic signatures (piezomagnetism, electrokinetics,
peroxy-defect charge carriers) in ULF-band ground magnetometer residuals after space-weather
subtraction, tested jointly with the seismic coherence graphs.

**Data (probed 2026-08-21)**:
- ✓ **IZN (Iznik, KOERI)** — INTERMAGNET GIN serving 1-minute data (probe returned live
  2026-08-19 series). **Marmara-local**: inside the istanbul_marmara carrier region — the one
  place we get a fault-local observatory for free.
- ✓ **FRN (Fresno, USGS)** — 1-minute XYZF via the public USGS geomag API. Caveat disclosed:
  ~350 km from the Coachella carrier — REGIONAL reference, not fault-local. TUC (Tucson) as
  second reference. Fault-local SoCal coverage would need non-INTERMAGNET arrays (availability
  unverified).
- **East Anatolia**: nearest INTERMAGNET coverage to the kahramanmaras carrier — VERIFY IN
  EXECUTION; possibly none close (disclose sparse coverage; carrier may be mag-untestable).
- **Solar/space-weather drivers**: OMNI solar wind, Dst, Kp (NASA/NOAA, public) — the
  subtraction regressors.

**Signal design (candidates, to be narrowed at prereg)**:
1. Per-observatory ULF-band residual energy after a registered space-weather regression
   (Dst/Kp/local-time harmonics + solar-wind coupling terms).
2. Magnetometer-residual ↔ seismic-graph-statistic lagged coherence (the multi-modal edge).
3. If ≥2 observatories per region: inter-observatory residual coherence (common-mode rejection).

**Governance (non-negotiable, the B1A lesson applied)**: full loop — prereg → codex freeze →
cross-authored bars → **POWER CERTIFICATION FIRST** against synthetic piezomagnetic-shaped
injections (nT-scale amplitudes, days-scale onsets, registered effect grid) → only a certified
family becomes verdict-bearing → sealed one-shot test. If no plausible effect class certifies at
the available noise floor, that finding is reported honestly BEFORE any real look. Known risk:
the space-weather subtraction model is itself a seam (regressor choice = researcher degree of
freedom) — the regression must be frozen in the prereg.

**Effort**: adapters ~days (APIs proven); design + power campaign ~1–2 governed weeks. Slot:
next b2 window design.

---

## NET-1 — network integrity monitor (operational add-on, no claims lane)

**What**: per-station uptime/gap/latency telemetry + a per-region coverage index for the FDSN
networks the program already consumes (KO, CI, TU + the 14-region monitor set); daily rollups,
degradation alerts, and a map layer. We already pull this data — the add-on is extracting the
telemetry we currently throw away.

**Why others benefit**: a seismic/warning network that silently degrades is a life-safety hole;
operators (KOERI, SCEDC, PNSN) and researchers get an INDEPENDENT, continuous health record with
receipts. Zero scientific-claim risk: it's plumbing telemetry, not prediction.

**Effort**: ~1 week. **Slot: FIRST** — fastest value, lowest risk.

---

## AFT-1 — aftershock forecasting baseline (prediction goal, skill-anchored)

**What**: per-region ETAS baseline, CSEP-style scored through validation-kit; then the well-posed
increment question: do our graph/coherence features add measurable skill ON TOP of ETAS?
**Why others benefit**: honest per-region aftershock context after mainshocks (re-entry/rescue
decisions), and a clean public demonstration of "score everything against the standard model."
**Effort**: the largest design; its own governed arc. Slot: after MAG-1 design lands.

---

## LEDGER-1 — registered prospective-forecast ledger

**What**: generalize the 2026-08-21 cascadia owner-forecast registration into a schema'd public
ledger: stamped pre-outcome calls, fixed region/magnitude/window definitions, base rates recorded
at registration, append-only scoring vs a named catalog. Optionally open to EXTERNAL claimants —
anyone claiming precursor skill gets scored under the same fixed rules.
**Why others benefit**: the field's precursor problem is unscored retrospective claims; a
lightweight public prospective registry is cheap and high-credibility. **Effort**: days.

---

## MAP layers (delivery surface)

geo2graph-map grows provenance-labeled layers as components mature: catalog-descriptive overlay
(public USGS/EMSC events, labeled non-evidentiary) → network-health (NET-1) → aftershock
forecasts (AFT-1). Each layer ships only through the claims pass; the layer manifest's
claim_status discipline is already structural.

## Delivery model — how to include these for others

- **Pattern = labcore**: each mature component becomes a standalone public MIT repo (defensive
  publication + credibility exhibit), with the map as the human surface and versioned JSON feeds
  for programmatic consumers.
- **Research outputs** ship CSEP-compatible where applicable; validation-kit is the shared
  harness and is itself a publishable methods contribution (prereg → adversarial freeze →
  certified power → one-shot seal, demonstrated end-to-end by the Phase-B null).
- **Gates unchanged**: per-component asylum publication go + codex claims-hygiene pass; the
  non-claims discipline is a public-communication FEATURE (no fear-mongering possible by
  construction).

## Sequencing proposal

1. NET-1 (now-ish, ~1 week) → 2. MAP catalog layer (days) → 3. MAG-1 design into the next b2
window → 4. LEDGER-1 incrementally as calls accumulate → 5. AFT-1 as the next major governed arc.
Standing b2 replications continue throughout per owner directive.
