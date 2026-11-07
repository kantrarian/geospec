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
- **cascadia (DISCLOSED PRE-FREEZE AMENDMENT, 2026-08-22 probes — CAPSULE-BOUND in this same
  commit per the codex atomicity rule)**: **ADMITTED** — VIC (Victoria, NRCan via INTERMAGNET
  GIN; INSIDE the cascadia bbox) local, **NEW** (Newport, USGS geomag, XYZF) as the registered
  M3 reference. Evidence committed AND pinned: probe envelopes
  `receipts/mag_vic_probe.envelope.json` (body sha `24a9bbe3…`, 1,440 samples) and
  `receipts/mag_new_probe.envelope.json` (body sha `2bcdbb12…`, 1,441 samples) + typed input
  capsules `mag_capsule_vic.json` / `mag_capsule_new.json` (identity, coordinates, bbox
  relation, provider, product class, cadence, orientation, sentinels, parser, recomputed
  coverage). Endpoints: M1, M2, M3. **KAT (W-MAG)**: a probe whose body or coordinates are
  absent/mismatched vs its capsule must refuse admission.

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
  1/120 Hz). **Exact filter instantiation (codex freeze-review fix 5)**: 4th-order Butterworth
  bandpass, `scipy.signal.butter(4, [0.001, 0.004], btype='bandpass', fs=1/60, output='sos')`,
  SciPy **1.18.0**; the numeric SOS coefficient array is COMMITTED as the byte-authoritative
  artifact **`docs/f2g_window2_freeze/mag1_band_b_sos.json`** (pinned in design-manifest-v2),
  whose `sos_serialized` field carries the exact coefficients under the declared serialization
  rule — **exact default-spaced Python `json.dumps(sos.tolist())`, NOT compact canonical JSON**
  — with sha256 `77bceec4003b75d11ac671d86fb79342a265a12364fc6101e80decdd6e9a7f29`
  (cross-host reproduced by the reviewer). Applied forward-backward via `sosfiltfilt`,
  `padtype='odd'`, **`padlen=27`** (the fixed integer default for this SOS, probe-verified).
  **Causal span = 266** (last `|h[n]| > 1e-12`); **edge exclusion = 532 samples after each
  contiguous-segment boundary**. **No silent interpolation**: fill/NaN samples SPLIT the series;
  **`usable_N = N − 2·532` per contiguous segment; `FILTER_SUPPORT_INSUFFICIENT` iff
  `usable_N ≤ 0` (N ≤ 1,064)**; the 90% day floor applies only to positive surviving support.
  **KATs (W-MAG)**: impulse boundary (266/267), single NaN, 100-sample gap, segment lengths
  N = 532 / 1,064 / 1,065, byte-equal band energy dual-host.
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

THREE admitted carriers × one PRIMARY endpoint each: istanbul = M2 (band-B), socal = M3
(band-B), cascadia = M3 (band-B). Holm within the MAG-1 lane at alpha 0.05 over the three
primaries. M1 and family-A endpoints are registered secondaries, descriptive/typed unless the
primaries reject (fixed-sequence gate within the lane). One named lane claim; no omnibus.
