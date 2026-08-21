# B1B annex — robust burst statistic — freeze candidate v1 (2026-08-21)

Instantiates prereg v0.3 §4 (B1B). Executable contract; all constants frozen here.

## Health admission (pre-evaluation ONLY)

A station enters the B1B testable set iff its selection-lookback presence ≥ 0.85 (the
selection constant) computed from pre-evaluation telemetry only. No evaluation-window value or
availability observation may alter admission (typed HEALTH_ADMISSION_VIOLATION on any attempt).

## Statistic

As the registered B1A calendar pipeline (blocks, baseline fit, window means — pinned by bytes)
with TWO named transforms applied IDENTICALLY to observed, null, LOCO, and injected panels:
1. **Per-station robust renormalization**: for station s, each of its edges' z-series is divided
   by `max(1, MAD_s / median_carrier_MAD)` where `MAD_s` = the MAD of all |z| values on edges
   incident to s over the BASELINE positions (frozen formula; carrier median over admitted
   stations; zero/degenerate MAD → typed ZERO_SCALE_REFUSAL for that station's edges).
2. **Winsorization**: every |z| value is capped at **c = 8.0** (in robust-z units) BEFORE window
   means. The cap is a statistic transform — never a station or edge deletion.
Aggregation: family T = max over carriers of the max edge window-mean |z| (B1A form). One-sided
HIGH. LOCO = the conjunctive gate over the NEW registry's stations (pinned loco_gate).

## Null

Registered B1A calendar-block permutation (pinned), with renormalization + winsorization
recomputed inside every draw (shift-raw-then-recompute).

## Effect grid (power) + specificity gate

Detection classes: B1A's registered (delta_lat, k, n_e) grid on true window-2 geometry.
**Specificity class (the KOZT lesson): a single-station gain-step artifact (one station's raw
values scaled ×{3, 10} from a registered onset) must NOT produce a familywise positive — the
certified contract requires BOTH CP-LB ≥ 0.80 on detection classes AND ≤ 0.05 positive rate on
the artifact class.** Certification under the full four-member Holm graph.

## KATs

ZERO_SCALE_REFUSAL boundary; cap applied identically across observed/null/LOCO/injected (a
fixture where skipping any one leg changes T must refuse); artifact-class specificity fixture;
health-admission violation refusal; renormalization determinism.
