# Amendment R6 — scoring-contract clarification (registered 2026-07-30)

**Status:** registered and owner-signed 2026-07-30 ("defaults fine" on the four owner choices).
Prospective-only; R2–R5 remain append-only and unmodified. This amendment pins contracts those texts
left open (found by adversarial review, codex 2026-07-30, before any evidence-bearing outcome existed
— zero admissible classifications and zero R5-active days at registration).

## 1. Target capsule (owner choice #1)

- **Catalog:** USGS FDSN event service is THE target catalog. Preferred magnitude = the catalog's
  `mag` (preferred magnitude) as served. Each scoring run records its query parameters, fetch
  timestamp, and a hash of the raw snapshot; classification uses the catalog as-of that snapshot.
  Later catalog revisions do not silently rewrite outcomes; a revision-driven change requires a
  dated correction entry.
- **Deduplication/membership:** events are deduplicated by catalog ID with **region memberships
  merged across all queries** (an event may belong to every region whose 100 km buffer contains it);
  it is scored per membership. Membership never depends on query order.
- **Timestamps:** `event_origin_utc` is the catalog origin time (full UTC). The alarm for local date
  D is `alarm_available_at_utc` = the publish commit time of D's daily run when known, else
  **23:59:59 UTC of D** (conservative). **Hit eligibility: 0 < event_origin_utc −
  alarm_available_at_utc ≤ 14 × 24 h.** Same-date post-event alarms can never credit.
- **Left-censoring:** the alarm history is loaded from ≥ (reset + gap) days before the accumulation
  start. Any episode already active at the start boundary is `left_censored` and ineligible for hit
  credit until 14 observed tier-0 days are followed by a fresh post-start onset.

## 2. Exclusion semantics (owner choice #2): chronological guard state

- **The guard state is chronological and causal.** Events are processed in time order; every
  admissible event, when it occurs on a non-excluded region, opens that region's exclusion window
  (G-K time, 365 d cap) at that moment. **Exclusion lineage is never erased by later batch
  relabelling** — a later, larger event inside an earlier window takes the supersession path
  (fresh-episode requirement) regardless of how batch declustering would label the pair.
- **Exclusion is region-wide**: during a region's exclusion window, the ENTIRE region contributes no
  alarm time, no false alarms, no hits, and no misses ("the region does not exist"), supersession
  excepted. The G-K spatial circle is used for target declustering, not to carve sub-regional
  scoreability.
- **The batch G-K declustered view is a separately reported diagnostic** (cluster counts), never the
  scoring authority.
- **Episodes are sets of scoreable alarm dates**: episode grouping never bridges an exclusion
  boundary, and hit matching runs against scoreable alarm dates only — an excluded alarm day can
  never supply credit through episode-span arithmetic.

## 3. Method epochs + degraded days (owner choice #3): intention-to-monitor

Every region-day in the record carries a frozen `method_epoch` (e.g. `r3`, `r5`, `r5_shadow`,
`fallback_r3:<reason>`) and availability state at publish time. **The primary Molchan record pools
all days under the method actually operating that day** (intention-to-monitor); no post-outcome
removal or selection of degraded days. Epoch-stratified trajectories are reported alongside as
descriptive views.

## 4. Self-absorption wording corrected (finding A-4)

R3/R5's "a genuine precursory buildup cannot be absorbed into its own baseline" is **narrowed**: the
guarantee is only that **the final 30 days cannot enter the fit/baseline**. A buildup persisting
longer than 30 days is progressively attenuated (measured: an identical present signal falls from
residual percentile 1.00 under a clean fit to ≈0.85/0.60/0.38 when a 90/180/250-day buildup enters
the training window). Registered mitigation: **fixed lag-sensitivity telemetry** — shadow fits at
30/90/180-day lags published with each weekly refit; divergence between them is an instability
indicator surfaced in the daily record. The operating lag stays 30 d; no lag is ever selected
retrospectively after seeing outcomes.

## 5. Claims discipline (owner choice #4)

- **R4 output remains descriptive-only until a dedicated statistical-plan amendment** is registered
  naming the test, α, interval construction, null, dependence/overdispersion handling, multiplicity,
  and the exact claim inequality. **Ten pooled mainshocks is the earliest eligibility to RUN that
  plan, not a skill test.**
- **R5 activates only from shadow mode**: the shadow phase produces the named receipts (15-year
  legacy replay, April FA-1/FA-2 invariance, measured step discontinuity) committed to this
  repository, then activation occurs at a dated marker registered BEFORE the first active day.
  Until that marker exists, R5 is telemetry (`r5_daily.json`), and the operational statistic is R3's.

*Registered by the project maintainers, 2026-07-30. Append-only; further changes get new dated
amendment files.*
