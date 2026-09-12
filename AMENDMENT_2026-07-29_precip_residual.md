# Amendment R5 — precipitation-regressed residual as the geodetic anomaly measure (registered 2026-07-29)

**Status:** registered and owner-signed 2026-07-29 with all recommended parameters adopted: log-OLS with
single 2% trim; uniform per-region application (all 14 regions); dual raw+residual publication; weekly
refit on [today−395 d, today−30 d] (R3-aligned self-absorption lag); fail-open fallback to the R3 ratio
path. Effective prospectively from the first R5-active run; activation is gated on the §4 mechanistic
KATs. Companion to amendments R2 (scoring window), R3 (rolling baseline), R4 (prospective arm).

**Implementation note (binding):** the residual percentile enters the unchanged downstream risk mapping
via a rank-remap onto the training-window ratio distribution (monotone by construction), so thresholds
behave identically in distribution while day-ORDERING reflects rain-adjusted anomaly. Dual publication is
via `docs/r5_daily.json` (raw ratio + residual percentile + coefficients per region, published daily);
the raw ratio also remains in `ensemble_latest.json` as before. Regions with fewer than 90 valid history
days in the fit window remain on the R3 path (per-region automatic activation, logged). The measured step
discontinuity at activation will be appended to this file by the first R5-active run's commit.

---

## 1. The change

The `lambda_geo` component's anomaly statistic becomes the **residual of a per-region log-domain
precipitation regression** instead of the raw baseline ratio:

    log(lg_raw_t) = β0 + β1·API7_t + β2·R30_t + ε_t          (per region)
    anomaly_t     = percentile of ε_t within the training-window residual distribution

- **API7** = 7-day exponential antecedent precipitation index (λ = 0.9); **R30** = trailing 30-day
  precipitation sum. Daily precipitation from the Open-Meteo historical archive at each region's center
  coordinates (public, keyless, no-gap verified 2011–2026 for kumamoto).
- The ensemble's downstream risk mapping and tier logic are **unchanged** — they consume the residual
  percentile where they consumed the ratio-derived statistic. (Implementation binds this at the
  `risk_score` interface; monotonicity preserved, KAT-verified.)
- **Dual publication:** the daily public record carries BOTH the raw ratio and the residual — nothing is
  hidden, and rain-triggering hypotheses stay testable against the raw series forever.

## 2. Fitting discipline (self-absorption guards, aligned with R3)

- Coefficients refit **weekly**, on a trailing window **[today−395 d, today−30 d]** — the same 30-day
  exclusion lag as R3, so a genuine precursory buildup cannot be absorbed into its own regression.
- Log-domain OLS with a single 2%-|residual| trim pass (heavy-tail guard; simple, deterministic,
  registered — no iterative robust machinery).
- **Uniform application to all 14 regions**, per-region coefficients. Dry-regime regions (ridgecrest,
  coachella) will fit β ≈ 0 and reduce to the R3 behavior automatically; the taxonomy's "never transfer
  coefficients across regions" rule is structural (each region fits its own).
- **Fail-open:** if the precipitation fetch fails, the component falls back to the R3 ratio path for
  that run, logged. A weather-API outage must never blind the monitor.

## 3. Effect on the record (disclosed at registration)

- **Step discontinuity** in the lambda_geo component at first R5 run — measured during implementation and
  stated in the amendment (same disclosure pattern as R3). Alarm *rates* before/after R5 are not directly
  comparable; the R4 scoring RULES are unchanged, and the R4 record gets an `r5_active_from` marker.
- Expected qualitative effect (from the W2 offline replay): summer hydro-loaded elevations discounted
  across wet-climate regions; the April-class false alarms (not rain-driven) largely unaffected; the
  July-2026 episode would have scored materially lower (its evidentiary weight already zero per R2 —
  this changes nothing retroactively).

## 4. Pre-activation validation gate (KATs, offline — NOT a skill claim)

1. Residual deseasonalization beats calendar: monthly-median spread of the residual < spread of the
   calendar-deseasonalized series on the 15-y legacy replay (already demonstrated in W2: 61% explained).
2. Non-rain episodes unaffected: April FA-1/FA-2 scores change < 10% under R5 replay.
3. Fallback path exercised (fetch-failure simulation → R3 behavior + log line).
4. Monotone interface: risk mapping order-preserved on the replay.
5. Discontinuity measured + written into the amendment text.

The gate is mechanistic correctness only. Whether R5 *improves the Molchan trajectory* is exactly what
the R4 prospective record will measure — no forward-skill claim is made at activation.

## 5. What R5 does NOT do

No validation claim; no change to R4 scoring rules or the R2 window; no tier-system change; no effect on
any historical classification; NONCONFIRMATORY status and all Class-2 gates unchanged.
