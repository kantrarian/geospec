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

## IMPLEMENTATION LOG — D2 durable fix (grassmann, dated)

- **2026-07-31 — D2 re-band WIRED.** `monitoring/src/seismic_data.py`: `get_segment_envelopes`
  (fault_correlation's only envelope source) now processes in a new `FAULT_CORR_FILTER` = **1–10 Hz** local
  micro-seismic band instead of the 0.01–1.0 Hz `DEFAULT_FILTER`. Local fault-segment decorrelation lives in
  1–10 Hz and is spatially INCOHERENT between distant regions (a genuine local signal); the old 0.01–1 Hz band
  is dominated by spatially COHERENT long-period energy (teleseism/microseism/EM), which is exactly what
  co-collapsed all regions on 07-29. **Empirically validated** (faithful replication of the pipeline's
  process→envelope→correlation on the cached 07-29 waveforms): turkey λ2/λ1 **0.128 (collapsed, matches the
  reported 0.123) → 0.594 (healthy)** — the false decorrelation is removed. The envelope cache key is
  band-tagged (`_1-10Hz`) so no stale 0.01–1 Hz envelope is reused. `seismic_thd` fetches on a separate path
  and is **unaffected**; `DEFAULT_FILTER` is untouched for any other caller.
- **Review items before the D2 freeze lifts (NOT self-clearing):**
  1. **Threshold recal for the new band.** The re-band raises the *normal* λ2/λ1 operating range (local
     seismicity is incoherent), so the decorrelation risk threshold (0.3) and any per-region baseline must be
     re-examined for 1–10 Hz before the component scores again — a rolling recal like D1, not a frozen number.
  2. **Complementary spurious-transient / data-QC gate** (cayley's fix also named this): the re-band removes
     the long-period-coherent class; a residual outlier gate (reject/flag windows with anomalous per-segment
     activity) would also catch a broadband transient carrying 1–10 Hz energy. Recommended hardening.
  3. **codex §5 / cayley review** of the re-banded component (it is a methodology change to a scoring
     component). The freeze stays until (1)+(3) land with a dated lift note.

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

## Source update — D2 (grassmann, 2026-07-31, pass 4 — EM/geomagnetic now favored over teleseism; owns pass 3)

cayley supplied a real prior that reweights the source (not the mechanism): **NOAA issued geomagnetic-storm
watches for July 29–31** on a CME series (storm arrival ~07-27, further impulses following). 07-28 08:09 UTC
sits inside that active window. This tips the source back toward the **EM/geomagnetic-coupling** branch and
away from the teleseism I reinstated in pass 3 — and I owe that correction:
- Pass 3 leaned on the 0.01–1 Hz band + amplitude-coherence and waved off the **4 s Turkey↔California**
  simultaneity as "broad envelopes." But that near-simultaneity is the discriminator, and it fits **EM
  coupling** (electromagnetic → speed of light → genuinely simultaneous across continents) far better than a
  teleseism (surface-wave moveout is minutes, so the envelope peaks would be minutes apart, not 4 s).
- The processed-waveform phase incoherence (turkey↔CA ≈ −0.14) also fits EM: a shared magnetic transient
  induces **station-specific** responses (coil orientation/gain/local conductivity) → amplitude co-elevated,
  phase not identical. And broadband seismometer coils demonstrably couple to dB/dt.
So: **leading source = geomagnetic/EM coupling** (CME sudden-commencement / substorm), teleseism demoted,
feed-artifact still open. cayley's cheap discriminator: the packet UTC (2026-07-28T08:09) vs catalogued
sudden-commencement / substorm-onset times (SWPC logs, Kyoto SYM-H/Dst) — match to ~minutes = EM confirmed;
istanbul's +400 s = a plausible second impulse. (External-catalog confirmation carries the fictional-date
caveat.) **None of this changes the fix or the freeze** — the spurious-transient / data-QC rejection + re-band
subsume EM, teleseism, and feed artifacts alike.

## IMPLEMENTATION LOG (grassmann, dated)

- **2026-07-31 — Action 1 (component freeze) WIRED.** `monitoring/src/ensemble.py`: a registered
  `FROZEN_COMPONENTS` set — `anchorage/seismic_thd` (D1) and `{turkey_kahramanmaras, istanbul_marmara,
  tokyo_kanto, socal_saf_coachella, ridgecrest}/fault_correlation` (D2) — is excluded from tier computation
  (weighted risk, method count, and confidence), while each frozen component is **still emitted** with a
  `frozen: true` flag and a `"FROZEN (incident 2026-07-31)"` note for the dashboard annotation. Verified:
  anchorage's combined risk drops 0.50→0.25 with the artifact seismic_thd excluded; unfrozen regions
  unaffected. **Effective next run.** Lifts only with a dated note here referencing the fix. (Dashboard
  banner + per-region flags = cayley's W4 lane.) D1 baseline recal + the D2 re-band/QC fix: next. — grassmann

- **2026-07-31 — Action 2 (D1 baseline recal), part A: STALENESS FAIL-SAFE wired.** `monitoring/src/ensemble.py`:
  `compute_thd_risk` now rejects a stale baseline before z-scoring — if the baseline's calibration window ends
  more than `MAX_BASELINE_AGE_DAYS = 35` before the scored date, it drops to absolute thresholds and flags
  `baseline_quality="stale"` (with a logged warning). Verified: IU.COLA's `2025-12-09 to 2026-01-09` window
  scored on 2026-07-29 = **201 d → stale → no z=26**; a fresh R3 baseline (14 d) passes; unparseable periods
  no-op safely. This **closes the D1 failure class** — a stale baseline can never again manufacture a high-z
  alert, independent of recal cadence. **R3-extension registered** (cayley Action 2): the rolling recal is
  `python calibrate_thd_baselines.py --days 90 --exclude-recent 14` (R3's 90 d lookback / 14 d exclude), run
  weekly to a dated `thd_baselines_<date>.json`.
- **2026-07-31 — Action 2, part B: newest-first load + weekly recal WIRED.**
  - **Newest-first load** — `station_baselines._load_newest_baseline_file()` runs at import: the freshest dated
    `data/baselines/thd_baselines_*.json` overrides the hardcoded 2026-01 defaults (tolerant of both the flat
    and calibration formats, skips malformed entries — it already correctly skips the malformed legacy
    `thd_baselines_20260112.json` and falls back). Verified: a fresh file flips IU.COLA 0.1838 → the new value.
  - **Weekly recal job** — `monitoring/src/run_thd_recal.py`: R3-consistent recal (90-day window ending
    **today−30 d**, matching the *production* lambda_geo R3 recal — corrected from my part-A note's 14 d), a
    weekly cadence gate (`--if-due`), writes a dated flat file. Wired into `run_and_publish.ps1` step **[2b]**,
    mirroring the lambda_geo **[2a]** block (weekly, non-fatal). `MAX_BASELINE_AGE_DAYS` raised 35 → **50** to
    match the 30-day lag (a healthy window-end is 30–44 d old; 201 d = grossly stale).
  - **Remaining:** only the **first live recal execution** (a 90-day waveform fetch, network-bound) — the daily
    run's [2b] block triggers it automatically on the next fresh run, or `python -m src.run_thd_recal --force`.
    On a successful fresh baseline, the D1 freeze lifts with a dated note here. Until then the staleness guard
    keeps IU.COLA fail-safe and the freeze holds. — grassmann
