# MAG-1 magnetometer channel — design DRAFT v0.1 (cayley, 2026-08-21)

**Status: DRAFT for codex review rounds (owner-directed follow-on to NET-1). NOT frozen, NOT
registered. Everything here is amendable until the codex freeze; no real magnetometer residual is
analyzed against seismic data before the sealed design exists.**

## 1. Question

Do ULF-band ground-magnetometer residuals (after a FROZEN space-weather subtraction) carry
registered structure relative to the seismic coherence graphs and/or catalog events, for effect
classes we can first PROVE we have power to detect? Mechanism families motivating the effect
grid: piezomagnetism (nT-scale, stress-coupled), electrokinetic streaming currents, peroxy-defect
charge transport (Freund/NASA Ames lab line).

## 2. Data sources (probed live 2026-08-21)

| source | role | status |
|---|---|---|
| IZN (Iznik, KOERI) via INTERMAGNET GIN, 1-min | Marmara FAULT-LOCAL observatory | ✓ live probe |
| FRN (Fresno, USGS geomag API), 1-min XYZF | SoCal REGIONAL reference (~350 km from Coachella — disclosed) | ✓ live probe |
| TUC (Tucson, USGS) | SoCal second reference / common-mode | assumed (verify) |
| East Anatolia coverage | kahramanmaras carrier | TO VERIFY; possibly none → carrier disclosed mag-untestable |
| OMNI solar wind, Dst/SYM-H, Kp (NASA/NOAA) | space-weather regressors | public, to adapter |

Sampling: 1-min → accessible band ≈ 0.0001–0.008 Hz (ULF, below Nyquist 1/120 Hz). 1-second
products exist for some observatories (upgrade path).

## 3. Space-weather subtraction (the critical frozen seam)

Per observatory, per horizontal component: a linear model on registered regressors —
{Dst/SYM-H, Kp, solar-wind coupling term (e.g. Newell), local-time diurnal harmonics (Sq: 24/12/8h),
seasonal harmonics} — fit on a REGISTERED baseline window disjoint from evaluation; residual =
observed − model. The regressor list, fit window, and fit method are FROZEN at prereg (researcher
degree of freedom eliminated). Model inadequacy shows up honestly as noise floor in power
certification, never as post-hoc regressor shopping.

## 4. Candidate registered statistics (to be narrowed to ≤2 at freeze)

- **M1**: eval-window ULF residual band-energy vs baseline distribution (per observatory).
- **M2**: lagged coherence between the residual series and the carrier's daily graph statistic
  (the multi-modal edge; joint with the fault2graph lane).
- **M3** (where ≥2 observatories): inter-observatory residual coherence change (common-mode
  rejected local signal).

## 5. Null design (B-1 lesson applied)

Nulls SHIFT RAW INPUTS THEN RECOMPUTE the entire pipeline (regression → residual → statistic):
circular calendar shifts of the magnetometer series relative to the seismic/event calendar, with
the space-weather regressors kept PAIRED to the magnetometer clock (so the null preserves
space-weather structure and destroys only the tectonic alignment). Never shift computed
residuals/statistics (identity-under-permutation trap). Draw counts, add-one p, valid-draw floors
per the Phase-B conventions; frozen substream seeds.

## 6. Power certification FIRST (B1A lesson applied)

Synthetic injected effect classes on REAL magnetometer noise (bound-mode analog: real observatory
series + injected signals): amplitude {0.5, 1, 2, 5} nT; duration {1, 3, 7} days; onset {step,
ramp}; spectral shape {quasi-DC drift, ULF band-limited burst}. Tier-S → Tier-C with
Clopper-Pearson ≥0.80 certification exactly as Phase B. **If NO plausible class certifies at the
real noise floor, that is the reported result and no verdict-bearing real test occurs** — a
certified sensitivity statement about ground-magnetometer precursor detectability is itself a
publishable contribution.

## 7. Governance path

NET-1 ships first (operational lane). MAG-1: this draft → codex R1 rounds → freeze → grassmann
cross-authored bars → adapters + engine lane → power campaign → certification decision →
(if certified) sealed one-shot on the aligned window, fresh owner seal. Multiplicity: MAG-1 runs
on a FRESH data window aligned with the next b2 campaign (never retro-fit to the consumed
Phase-B window).

## 8. Open items

EAF observatory search; TUC verification; 1-sec product availability; regression regressor
finalization; whether M2 joins the b2 seismic prereg or stands alone; observatory instrument
health telemetry folded into NET-1.
