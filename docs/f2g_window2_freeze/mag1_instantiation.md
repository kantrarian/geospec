# MAG-1 instantiation — freeze candidate v1 (2026-08-21)

Numerical instantiation of the R1-PASSED MAG-1 design v0.2 (pinned by bytes in the manifest)
under the window-2 v0.3 barrier. ACCRUAL-compatible lane: producers acquire raw magnetometer +
space-weather bytes mechanically during accrual; all analysis post-barrier.

## Admitted products/carriers (per the coverage admission, pinned)

- **istanbul_marmara**: IZN (INTERMAGNET GIN; adjusted-or-reported class, 1-min). Endpoints:
  M1, M2. (No registered reference observatory → no M3.)
- **socal_coachella**: FRN (USGS geomag, adjusted, 1-min XYZF) with **TUC** as the registered
  M3 reference. Endpoints: M1, M2, M3.
- **turkey_kahramanmaras**: typed MAG_UNTESTABLE (no coverage; may amend by disclosed amendment
  only if the AFAD/TUBITAK search lands pre-freeze).
- **cascadia**: NOT admitted for MAG-1 window 2 (no probed observatory; candidate NEW/VIC probe
  is a pre-freeze option, else typed).

## Components, frames, capsules

Canonical geographic X_north/Y_east (nT) after the registered orientation conversion per input
capsule (XYZF for FRN/TUC; IZN's recorded elements converted per its capsule). Statistic
component: the horizontal vector magnitude residual `sqrt(rX² + rY²)` applied after per-
component subtraction. Product classes, sentinels, timestamp conventions, and support masks per
the typed-capsule schema (design §2), instantiated per station in the capsule files at freeze.

## Frozen subtraction (fit/apply ledger)

Regressors: SYM-H (1-min, zero-order-hold from provisional if final absent — class recorded),
Kp (3-h, zero-order-hold), Newell coupling `v^(4/3)·B_T^(2/3)·sin^(8/3)(θ/2)` from OMNI 1-min
(fill-policy per capsule), local-solar-time harmonics (24/12/8 h, sin+cos), seasonal harmonics
(365.25 d and 182.63 d, sin+cos). Linear least squares per observatory per component.
**Calibration interval: 2026-01-01 → the availability cutoff** (strictly pre-evaluation);
coefficients + design recipe + diagnostics committed before evaluation bytes are opened;
apply-never-refit everywhere (including the M3 reference regression, per the codex binding
interpretation).

## Bands, filters, windows

- Family A (quasi-DC step/ramp): daily-mean residual series; detection on day-scale steps/ramps.
- Family B (ULF band-energy): band **0.001–0.004 Hz** (Nyquist guard: upper edge < half of
  1/120 Hz), 4th-order Butterworth applied forward-backward (zero-phase), edge exclusion =
  2 × filter impulse-response span; day admissible iff ≥ 90% of its 1,440 minute samples are
  non-fill.
- M1 event/control windows: duration {1, 3, 7} days; pseudo-onset sampling blocked by season
  (quarter), local solar time (onset hour class), duration, and data-quality mask; n = 999
  control windows per event window; no circular wrap.
- M2 daily magnetic feature: registered robust band-B energy per admissible day (median of
  squared band-passed residual over admissible minutes); lag set {0, ±1, ±2, ±3 days};
  min shared-day overlap 60; block-null: whole-day rotations excluding |offset| ≤ 3.
- M3 (socal): innovation = FRN residual regressed on TUC residual + the space-weather terms
  (frozen at calibration); statistic on the innovation, M1-style windows.

## Effect grid (power)

Amplitudes {0.5, 1, 2, 5} nT; durations {1, 3, 7} d; onsets {step, ramp}; spectral shapes
{family-A drift, family-B band-limited burst}; spatial topology: {IZN-local},
{FRN-local-with-TUC-unaffected}, {FRN+TUC common-mode, aligned polarity} (the last must be
REJECTED by M3 — a specificity class). Injections into disjoint calibration-noise capsules
pre-subtraction. Certification per family CP-LB ≥ 0.80 through the whole decision rule.

## Internal multiplicity (imported into v0.3 §5 as a pinned object)

Two admitted carriers × one PRIMARY endpoint each: istanbul = M2 (band-B), socal = M3 (band-B).
Holm within the MAG-1 lane at alpha 0.05. M1 and family-A endpoints are registered secondaries,
descriptive/typed unless the primaries reject (fixed-sequence gate within the lane). One named
lane claim; no omnibus.
