# MAG-1 magnetometer channel — design DRAFT v0.2 (cayley, 2026-08-21)

**Status: DRAFT v0.2, folding all five codex R1 repairs (1449/1849Z note). NOT frozen. No real
magnetometer residual or seismic alignment is opened before the sealed design exists.**

## 1. Question

Do ground-magnetometer signals — after a frozen space-weather subtraction — carry registered
structure relative to the seismic coherence graphs, for effect classes we FIRST prove we have
power to detect? Mechanism names (piezomagnetism, electrokinetics, peroxy-defect transport)
motivate **injection shapes only** and are not mechanism evidence.

## 2. Typed input capsules (codex R1 fix 3)

One immutable capsule per station/product, committed before any value-bearing use, carrying:
provider + product/version class (reported/adjusted/provisional/definitive); retrieval URL, UTC,
receipt sha256; IAGA code + coordinates; recorded elements and sensor orientation (XYZ/HDZ/…)
with the registered canonical conversion to geographic X_north/Y_east/Z_down in nT (D angular →
converted, never mixed); vector/scalar distinction; sample-mean + timestamp-centering convention;
UTC policy; missing/fill sentinel decoding; baseline-jump/instrument-health flags; resampling +
filter transform hashes; first/last admissible sample + shared mask. For OMNI/SYM-H/Dst/Kp
additionally frozen: units, cadence, release class, GSM/GSE frame, bow-shock target-arrival
timestamp semantics, fill/quality policy, applied lag, resampling rule. **Any cross-carrier or
cross-component comparison REFUSES unless the common frame and support are explicit** (typed
FRAME_NOT_CLOSED). Candidate sources (probed live 2026-08-21): IZN (INTERMAGNET GIN,
Marmara-local ✓), FRN (USGS geomag ✓), TUC (verify), East Anatolia (verify; possibly typed
mag-untestable). Availability decisions happen in a COVERAGE-ONLY admission pass (§7).

## 3. Frozen fit/apply ledger (codex R1 fix 2 — one interpretation only)

1. The subtraction model is FIT ONCE on an immutable CALIBRATION interval using calibration
   bytes only (registered regressors: SYM-H/Dst, Kp, solar-wind coupling term, local-time Sq
   harmonics 24/12/8 h, seasonal harmonics).
2. The design-matrix recipe, coefficient vector, fit diagnostics, and admissibility verdict are
   committed + hashed BEFORE any evaluation bytes are opened.
3. Production, every power replicate, and every null draw APPLY THE FROZEN COEFFICIENTS to
   (possibly transformed) RAW evaluation mag/weather inputs — residual/filter/feature/statistic
   are recomputed; **coefficients are never refit**. Per-draw refitting is NOT admissible.
4. Power injections go into a DISJOINT calibration-noise evaluation capsule, injected BEFORE
   subtraction, so the frozen production seam is exercised without touching or training on the
   future one-shot window.

## 4. Registered statistics (typed; codex R1 fix 4)

- **M1 — event-window band statistic** (per admitted observatory/component/band): registered
  robust statistic of the frozen-subtraction residual in defined event/pseudo-event windows.
- **M2 — cross-modal daily pairing**: a DAILY magnetic feature (registered robust band energy
  over that day's admissible minute samples; frozen min-N and gap rule) paired with the daily
  graph statistic ON THE EXACT SHARED-DAY MASK; frozen lag set, direction, minimum overlap,
  block-null geometry. The daily graph series is NEVER upsampled to minutes.
- **M3 — local innovation**: the local observatory's residual REGRESSED AGAINST registered
  regional reference observatories + space-weather terms (frozen reference set); the statistic
  lives on the innovation. Inter-observatory coherence is demoted to a COMMON-MODE/QC
  DIAGNOSTIC only — high coherence indicates shared regional forcing, not local signal.
- **Effect/statistic families are split**: (A) quasi-DC step/ramp (below the ULF band) and
  (B) ULF band-energy — each with frozen component, amplitude definition (pre-filter nT),
  band edges WITH a Nyquist guard (1-min sampling: guard well below 1/120 Hz; band-B upper edge
  set at freeze ≤ 0.004 Hz), anti-alias/filter order/padding, gap thresholds, edge exclusion,
  and the injected SPATIAL TOPOLOGY + POLARITY across observatories as part of the effect
  object (a step/ramp of days-scale duration is a family-A object, not a band-B one).

## 5. Endpoint-specific null table (codex R1 fix 1 — the shift-identity repair)

| endpoint | transform group | why not a common shift |
|---|---|---|
| M1 | pseudo-onset/control-window SAMPLING from the raw mag+paired-weather record, blocked by season, local solar time, duration, and data-quality mask; frozen subtraction+filter applied to every candidate window; NO circular wrap | a co-shifted full-window energy is shift-invariant (identity) |
| M2 | whole-day ROTATION of the paired raw mag/weather capsule against the graph-day series; zero/near-zero offsets excluded; exact shared-day mask preserved; daily magnetic feature RECOMPUTED post-transform | only the mag↔graph alignment is destroyed; weather pairing preserved |
| M3 | ONE-SIDED support-preserving block rotation of the local observatory's paired raw capsule against the references (or a registered spectral surrogate); never co-shift both sides | co-shifting preserves inter-observatory relations (identity) |

**Executable non-identity KATs per endpoint**: ≥2 allowed transforms must CHANGE the statistic
on a planted fixture; a paired-weather-preservation KAT must remain EQUAL. Shift-raw-then-
recompute everywhere; computed residuals/statistics are never transformed directly.

## 6. Power certification FIRST (through the WHOLE decision rule; codex R1 fix 5)

Injections (per §4 effect objects) into the disjoint calibration-noise corpus; recovery is
scored through the ENTIRE final decision rule — typed-capsule admission, gap/admissibility
gates, frozen-coefficient subtraction, statistic, MULTIPLICITY allocation, and any
station/reference robustness gate. Tier-S → Tier-C with Clopper-Pearson ≥0.80 per the Phase-B
conventions. Terminal wording on failure: `MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH` (never a
global "undetectable" claim). A certified-sensitivity outcome is a **reportable sensitivity
result; any publication decision remains owner-controlled.**

## 7. Hypothesis registry + multiplicity (codex R1 fix 5)

Sequence: (i) COVERAGE-ONLY admission pass (station/product availability; no measurement values)
→ (ii) FREEZE the exact hypothesis registry: admitted observatories/carriers/components/bands/
lags/event-window defs/statistics/directions + availability/typed-untestable rules → (iii) one
PRIMARY endpoint per admitted carrier (or an explicit registered max-T/Holm allocation over the
full registry), alpha, one-/two-sided direction, no-drop rule — all before value-bearing access.
Post-viewing availability changes re-enter at (ii) as a disclosed amendment, never silently.

## 8. Governance path

This v0.2 → codex single close of R1 → freeze → grassmann cross-authored bars (incl. the
non-identity KATs) → typed capsule adapters → calibration fit + ledger commit → power campaign →
certification decision → (if certified) sealed one-shot on the fresh window aligned with the
next b2 campaign (3 carriers + cascadia), fresh owner seal. Never retro-fit to the consumed
Phase-B window. NET-1 telemetry may inform a FUTURE preregistered registry-admission rule only
(per codex boundary note; no historical availability claims).
