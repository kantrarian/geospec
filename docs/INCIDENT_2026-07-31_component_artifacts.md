# INCIDENT 2026-07-31 — two component-pipeline artifacts inflated the 2026-07-29 WATCH count (registered, same-day)

**Status:** REGISTERED incident record, published same-day. Append-only. This is a monitor-integrity
correction, not a scoring-rule change; the frozen R-series amendments (R2–R6) are untouched.

## What happened

The 2026-07-29 daily run (published 2026-07-31 07:11 ET, commit `31f0a56`) reported **8 regions at
WATCH** (vs 2 the prior day). A same-day integrity check found **6 of the 8 are driven by two pipeline
artifacts, not tectonics**:

### Artifact D1 — stale frozen baseline (anchorage / IU.COLA seismic_thd)
- The IU.COLA THD component was scored against a baseline **frozen since 2026-01-09** (window
  2025-12-09→2026-01-09, n=32, never refreshed — 6.5 months stale).
- Current IU.COLA THD is a day-to-day sawtooth (1.70 → 1.04 → 0.55 → 0.09 → 1.22 over 07-25→07-29);
  against the stale tight std this produced z-scores of ±20–38 on alternating days (z = 26.17 on 07-29).
- A sawtooth that oscillates down-then-up over 5 days is instrument/site noise, not a precursor (and not
  a single seismic event, which spikes once and decays).

### Artifact D2 — degenerate segment-activity data (fault_correlation, 5 regions)
- On 2026-07-29 the fault_correlation raw ratio λ2/λ1 **collapsed simultaneously** in turkey_kahramanmaras
  (0.777→0.123), istanbul_marmara (0.688→0.040), tokyo_kanto (0.625→0.223), socal_saf_coachella
  (0.125→0.032) and ridgecrest (0.099→0.055).
- Segments were PRESENT (2/3–3/3), but participation ratios collapsed to ~1.06–1.43 (near rank-1): the
  per-segment activity envelopes went **near-identical**, degenerating the correlation matrix and
  producing a false "decorrelation" signal. The component's scoring rule (low λ2/λ1 = elevated) behaved
  as designed; the INPUT was degenerate.
- Five independent fault systems do not synchronize in one day; present-but-identical segment envelopes
  indicate a shared upstream data-feed/computation regression on 07-29 (under root-cause investigation).

## Actions (all dated, none retroactive)

1. **Component freeze (effective next run, until root-cause + fix):** the IU.COLA seismic_thd
   contribution (anchorage) and the fault_correlation contributions for the five affected regions are
   frozen (excluded from tier computation), with the exclusion visibly annotated per-region on the
   dashboard. Freeze lifts only with a dated note referencing the fix.
2. **Baseline-refresh amendment (registered herewith):** the R3 rolling-recalibration principle
   (registered 2026-07-29 for the λ_geo baseline) is EXTENDED to component baselines: seismic_thd
   baselines recalibrate on a trailing window on the same weekly cadence, lagged consistently with R3.
   The stale-frozen-baseline failure class is thereby closed for all components, not just λ_geo.
3. **Feed root-cause:** the 07-29 per-segment activity envelopes (`fault_correlation` pipeline) are under
   investigation; findings will be appended here.
4. **Published history stands.** The 2026-07-29 record as published is NOT rewritten (prospective-record
   discipline). This incident note is the correction mechanism; the affected WATCH rows are annotated,
   not deleted. Any future skill analysis choosing to exclude incident-flagged days must do so under a
   registered statistical-plan amendment (R6 §5), never ad hoc.
5. **R4 bookkeeping:** artifact-driven alarm-days inflate τ (the alarm fraction), which is conservative
   for any future skill claim (more alarm days = a worse Molchan score, never a better one). No
   retroactive R4 edits.

## What this incident is NOT

- NOT a change to any registered scoring rule, threshold, or amendment (R2–R6 untouched).
- NOT evidence for or against the method (which remains INCONCLUSIVE; monitor output ≠ validated forecast).
- The remaining 2 WATCH regions (hualien, mexico_guerrero) are NOT part of this incident and stand as
  ordinary monitor output.

*Registered 2026-07-31. Root-cause findings and the freeze-lift note will be appended, dated.*
