# Scoring-rule amendment R2 — hit window 7 → 14 days (registered 2026-07-29)

**Status:** registered and effective **prospectively** from 2026-07-29 (UTC). Applies to
`monitoring/src/validate_predictions.py` (`VALIDATION_CONFIG`).

## The rule change

| | R1 (through 2026-07-28) | R2 (from 2026-07-29) |
|---|---|---|
| hit window (`lead_window_days`) | 7 | **14** |
| validation lookback | 7–14 days | **14–21 days** |
| min magnitude / min tier / region buffer | M5.5 / tier 2 / 100 km | unchanged |

**Why.** R1's 7-day hit window was inconsistent with the system's own stated 7–14-day lead mechanism:
a prediction that led an event by 8–14 days — exactly the mechanism the system claims — was scored a
*false alarm*. The lookback band moves to 14–21 days so that no prediction is classified before its
full hit window has closed.

## What does NOT change

1. **All historical classifications stand.** Every entry already written to `validated_events.json`
   under R1 — including the aggregate record of **0 hits / 22 false alarms** — remains as scored.
   Nothing is reclassified retroactively.
2. **The 2026-07 Kumamoto episode is NOT counted as a hit.** The tier-2 episode of 2026-07-14→18
   (peak risk 0.659, the series maximum) was scored under R1 as 5 false alarms as the windows closed,
   days before the M_JMA 7.1 Kumamoto earthquake of 2026-07-28 (lead 10–14 days). Under R2 those days
   would have classified as hits — but the rule was amended *after* the event, so counting it would be
   hindsight. It is recorded here as the **motivating case**, dual-reported, with zero weight in the
   prospective track record.
3. **The scorecard restarts its evidentiary meaning at R2.** Hits and false alarms accumulated from
   2026-07-29 forward, under a rule fixed *before* the outcomes, are the admissible record.

## Honest context recorded with the amendment

- Kumamoto's alarm base rate is the highest of the 14 monitored regions (tier ≥ 1 on ~39% of days in
  the live record); comparable 14-day elevations occur in ~14% of the series' history. A single
  10–14-day lead on a single event is *consistent with* skill and *individually indistinguishable
  from* base-rate coincidence. Only the prospective record can separate the two.
- The regional proxy shows strong **seasonality** (Kumamoto's ratio is climatologically highest in
  July in the 15-year record — rainy-season hydrological loading is a known confounder). A
  deseasonalization / covariate upgrade is under study and, like this amendment, will be **registered
  before use**, never applied retroactively to manufacture agreement.

*Amended by the project maintainers, 2026-07-29. This file is append-only; further rule changes get
new dated amendment files.*
