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

---

## Root-cause update — D2 (grassmann, 2026-07-31, first forensic pass)

Direct inspection of the **cached 07-29 segment envelopes** (not just the summary λ2/λ1 statistics that
seeded the initial writeup) refines — and partly corrects — the D2 mechanism above. Owning the correction:

**Established:**
- **No pipeline code changed** between the 07-28 and 07-29 runs. The only diff (`0c99e0e..31f0a56`) is
  `run_and_publish.ps1` (a start-of-run self-heal pull) + README — nothing in `seismic_data.py`,
  `fault_correlation.py`, or `fault_segments.py`. So D2 is a **data/computation condition on 07-29, not a
  code regression.**
- The 07-29 envelopes are **NOT degenerate or identical.** For turkey_kahramanmaras the cached envelopes
  (cache keyed by window-start = `20260728`) are **full length** (86401 samples/station, 24 h @ 1 Hz),
  with normal station counts (2/4/2) and **distinct** per-segment activity (central 607, north 1054, south
  384). The earlier "present-but-identical envelopes → degenerate" wording is therefore **too strong** and is
  corrected here.

**What actually moved:** the **cross-segment correlation rose** on 07-29 — raw envelope cross-correlation
≈ −0.08 (07-28) → ≈ +0.48 (07-29) for turkey; the component then reported λ2/λ1 0.777 → 0.123, PR 2.94 → 1.37
(near rank-1). A moderate raw-correlation rise is **amplified** by the activity-index → correlation-matrix
computation into a near-rank-1 result, and — the load-bearing point — it rose **simultaneously across all five
regions**. Five unrelated fault systems' segments becoming correlated on the *same day* is the signature of a
**shared broadband signal**, not local stress decorrelation.

**Leading hypothesis (not yet confirmed): teleseismic contamination.** A large distant earthquake on 07-29
injects a shared arrival into every region's stations globally → correlated activity everywhere → λ2/λ1
collapses across all regions at once → false "decorrelation precursor." This would *also* be consistent with
the D1 IU.COLA THD spike on 07-29 (a teleseism raises broadband THD too). The fault_correlation component has
no teleseismic-signal rejection, so a global event would masquerade as simultaneous local decorrelation — a
known artifact class distinct from a feed bug.

**Still open / next diagnostics:**
1. **Global large-event check for 2026-07-29** — the decisive test of the teleseism hypothesis (a M≈7+ event
   the same day would confirm it). To be run against the pipeline's own catalog / an external source.
2. **`compute_segment_activity_index` amplification** — trace why raw corr ≈ 0.48 maps to λ2/λ1 ≈ 0.12
   (near rank-1); needs an `obspy`-enabled env (unavailable in the forensic venv here).

**Freeze remains justified either way** — teleseismic contamination and a computation-amplified correlation
both mean the 07-29 fault_correlation "decorrelation" signals are **not local stress precursors**. If the
teleseism check confirms, the durable fix is a teleseismic-rejection/regression step in the component (not
just a baseline refresh). — grassmann

## Root-cause update — D2 (grassmann, 2026-07-31, pass 2 — SUPERSEDES the teleseism lead)

The global-event check (cross-region envelope coherence) came back and it **rules out a real seismic source**,
teleseism or eruption. Revised conclusion, with the decisive evidence:

- **Windows are UTC-aligned.** The ensemble derives one common correlation window from the run `date` for
  ALL regions (`src/ensemble.py:546`), so identical sample indices are identical absolute UTC times.
- **Cross-region coherence jumped** from mean inter-region envelope correlation ≈ **+0.02 (07-28) → +0.75
  (07-29)** across turkey / ridgecrest(California) / istanbul.
- The driver is a **broad high-amplitude packet** (FWHM ~60–180 s, 25–65× the daily median) appearing at
  **nearly the same absolute UTC time** in all three: turkey @ 4147 s, ridgecrest @ 4151 s (**4 s** later),
  istanbul @ 4547 s.
- **This cannot be a propagating seismic source.** 4 s between Turkey and California is physically impossible
  (surface-wave differential travel time is ~15–45 min); and Turkey ≈ California (4 s) while Turkey ≠ Istanbul
  (400 s, adjacent regions) is geometrically impossible for ANY single source. The broad shape rules out a
  1-sample injection, but the near-simultaneity across continents rules out a real earthquake/eruption.

**Revised leading cause: a shared NON-TECTONIC transient at a common UTC time on 07-29** (new — 07-28 was
clean at 0.02). Candidate origins, both non-seismic and both consistent with "no code changed + full-length
distinct envelopes":
1. an **upstream data-feed / processing artifact** — a corrupted or duplicated data block, or a network-wide
   telemetry gap + gap-fill/deconvolution transient, served at a common UTC window; or
2. a **global instrument-coupled disturbance** (e.g., a geomagnetic/EM event inducing a simultaneous response
   in seismometer electronics worldwide — genuinely simultaneous across continents, unlike a seismic wave).

**Revised fix implication:** the durable fix is a **spurious-transient / data-quality rejection** step in the
component (outlier-reject anomalous high-amplitude transients and add a data-QC gate before the correlation),
which subsumes teleseismic rejection — NOT a teleseism-specific regression, and NOT (for D2) a baseline
refresh. The freeze stands.

**Still open:** the exact upstream origin. Next: pull the raw traces at ~sample 4147 of the 07-29 window
(≈1.15 h into the common window) to characterize the transient (seismic-shaped vs telemetry glitch vs EM)
and check station/telemetry logs at that UTC. — grassmann

## Root-cause update — D2 (grassmann, 2026-07-31, pass 3 — reconciles; CORRECTS pass 2's "non-tectonic")

Pulled the raw traces. Pass 3 supersedes pass 2 and **restores a real seismic source as the leading cause** —
pass 2's "physically impossible / non-tectonic" verdict rested on a flawed argument, owned below.

**The confirmed mechanism — a component-band design flaw:**
- `process_waveforms` bandpasses to **0.01–1.0 Hz** (`src/seismic_data.py:79-80`, 100 s–1 s period) and the
  activity/correlation is built from the Hilbert-envelope **amplitude** of that band.
- The transient at ~sample 4147 (abs UTC ≈ 2026-07-28T08:09Z, on KO/Turkey among others) is, **in the
  component's 0.01–1.0 Hz band, the single largest minute of the day** (36.9× median, RANK 1 of 1440). In the
  local micro-seismic band (0.5–10 Hz) the same minute is unremarkable (1.8× median, rank 124/1440) and the
  day's real local burst is elsewhere (bin 574, 09:34). So the driver is a **long-period** signal the
  component's band is tuned to and the local band excludes.

**Why pass 2 was wrong (owned):** I read the envelope-amplitude peak as a *phase arrival* and called the 4 s
Turkey–California alignment impossible. But the component correlates **amplitude envelopes, not phase.** A
large teleseism's long-period surface-wave energy **co-elevates amplitude** across continents over a broad
(minutes-wide) window — near-simultaneous at envelope/activity resolution — even though the waveforms are
**not** phase-coherent (measured processed-waveform correlation turkey↔California ≈ −0.14). Amplitude
co-elevation without phase coherence is exactly the teleseismic signature; my phase argument did not apply.

**Reconciled leading cause:** a **large long-period signal — most consistent with a teleseismic earthquake**
(surface waves dominant at 0.01–0.1 Hz) — co-elevated the 0.01–1.0 Hz envelope activity across all five
regions' segments at once → correlation rose → λ2/λ1 collapsed → false "decorrelation precursor." (A large
long-period *coherent-noise* event is the only alternative; a global M≈7+ catalog check for 2026-07-28
~08:09 UTC would confirm which — the fictional-date caveat still applies to external catalogs.)

**Durable fix (revised, and this is the real lesson):** the component is **mis-banded** for its stated purpose.
Local fault-segment stress decorrelation lives in the **local band (≈1–10 Hz)**, which is spatially *incoherent*
between distant regions (good — genuine local signal); the current **0.01–1.0 Hz** band is dominated by
spatially *coherent* long-period energy (teleseisms, microseism, atmospheric), so any large long-period arrival
fires false decorrelation everywhere. Fix: (1) move the analysis band to the local ~1–10 Hz range; (2) add a
long-period/teleseism-transient rejection (flag+exclude windows with anomalous 0.01–1 Hz energy); (3) a
data-QC gate. This is a design correction, not a baseline refresh. **Freeze stands.**

**Meta:** three passes, two self-corrections (pass 1 "identical envelopes" → pass 2 "non-tectonic" → pass 3
"long-period teleseism + mis-banding"). The invariant that held throughout: **the 07-29 fault_correlation
signals are not local stress precursors,** so the freeze and the §5 conservatism were right from the start. — grassmann

### Cross-region confirmation (grassmann, waveform-level, 0.01–1 Hz)
The single long-period arrival at **2026-07-28T08:09 UTC** is the day's dominant minute across regions on
**two continents** — confirming a shared long-period (teleseismic-class) source, not five local events:
| region | continent | stations | transient ×median (rank of 1440) |
|---|---|---|---|
| turkey_kahramanmaras | Anatolia | KO.NURH / KO.MLTY / KO.GAZ | 75× (0) / 37× (1) / 92× (2) |
| istanbul_marmara | Anatolia | KO.NMR8 | 33× (5) |
| ridgecrest | California | CI.LRL / CI.CCC / CI.TOW2 | 20× (2) / 14× (4) / 16× (2) |

tokyo_kanto and socal_saf_coachella have no cached **waveforms** for this run (envelopes only), so they can't
be re-derived at the waveform level here; their 07-29 λ2/λ1 collapse is consistent with the same mechanism,
which is region-independent (the 0.01–1 Hz band captures the same coherent long-period arrival everywhere).
Note: istanbul's cached data is heavily **fragmented** (many gappy sub-segments), a second data-QC concern to
fold into the fix. — grassmann
