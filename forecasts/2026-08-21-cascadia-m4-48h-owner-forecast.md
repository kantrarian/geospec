# PROSPECTIVE REGISTERED OWNER FORECAST — cascadia M>4, 48h

**Lane: EXPLORATORY / human-call registration. NOT a program claim, NOT a method output, NOT a
Λ_geo or fault2graph prediction. Registered BEFORE outcome so it can be scored honestly.**

- **forecaster**: asylum (owner), by eye from the public daily-monitor history graph
- **quote (verbatim)**: "Glancing at the geospec history graph I believe I'm able to predict that
  cascadia will have a siezmic event greater than magnitude 4 within the next 48 hours."
- **quote_sha256**: `1bc922d578c3a801d9733287a59b562a2a4bcb7f3bf629340363b4472b086229`
- **registered_utc**: 2026-08-21T17:22:07Z (live clock read)
- **window**: 2026-08-21T17:22:07Z → **2026-08-23T17:22:07Z** (48h)
- **event definition**: magnitude > 4.0 (any magnitude type), USGS ComCat authority
  (`earthquake.usgs.gov/fdsnws/event/1/query`), origin time within the window
- **scoring frames** (both scored; frame A primary):
  - **A (program region, primary)**: inside the program's registered cascadia polygons
    (`fault_segments.py` FAULT_SEGMENTS["cascadia"]: vancouver_island, puget_sound,
    olympic_peninsula, columbia_river; bbox lat 45.0–51.0, lon −128.0…−121.5).
    Base rate (USGS, 2021-08-21→2026-08-21): **6 events M≥4 / 5y → P(≥1 in 48h) ≈ 0.7%.**
  - **B (wide offshore, secondary)**: bbox lat 44.0–52.0, lon −131.0…−121.0 (includes
    Explorer/Nootka/Blanco offshore seismicity). Base rate: **235 / 5y → P(≥1 in 48h) ≈ 23%**
    (Poisson, λ≈0.257/48h).
- **instrument state at registration** (for the record): daily monitor ensemble scored day
  2026-08-19 (latest, 2-day lag): cascadia **Tier 0 NORMAL**, combined_risk 0.0198, λ_geo ratio
  0.1×, nothing elevated. The calibrated pipeline does NOT flag cascadia. The fault2graph/B2A
  method has NO cascadia carrier (istanbul/socal/turkey only) and cannot speak to this region.
- **interpretation rules, fixed now**: a frame-A hit is individually surprising (~1-in-150) but a
  single hit is still an anecdote; a frame-B hit is unremarkable (~1-in-4). Neither outcome
  validates or invalidates any method. Value accrues only if owner/eye calls become a REGISTERED
  STREAM scored the same way (Molchan-style), which would be its own preregistered design.
- **scoring**: at window close, query USGS for both frames; record HIT/MISS + events verbatim in
  an appended RESULT section. Append-only; this section is frozen.

---

## RESULT (appended 2026-08-23T17:25Z — FINAL)

Scorer: `monitoring/src/w2_forecast_scorer_cayley.py`, fired 2026-08-23T17:25:00Z
(window closed 17:22:07Z; ~3-min catalog-settling buffer, disclosed). Window
[2026-08-21T17:22:07Z, 2026-08-23T17:22:07Z], strictly M>4.0. Verbatim:

- **Frame A** (registered cascadia bbox 45.0–51.0 N, −128.0…−121.5 E):
  0 events M>4.0 → **MISS**
- **Frame B** (generous 44.0–52.0 N, −131.0…−121.0 E):
  0 events M>4.0 → **MISS**
- Query URLs + empty event lists verbatim in the scorer output (both
  `events: []`, `events_m_gt_4: 0`, `status: FINAL`).

Per the frozen interpretation rules: the null outcome was the ~77%-likely case in
frame B and ~99%-likely in frame A; it validates and invalidates nothing. The
interim checks (2026-08-22, 2026-08-23 pre-close) also showed zero events. No
re-query, no reinterpretation; this section closes the forecast. Λ_geo remains
INCONCLUSIVE.
