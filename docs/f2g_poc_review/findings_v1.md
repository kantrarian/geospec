# PoC review findings v1 -- EXPLORATORY (zero claim weight; next-window prereg inputs only)

Generated 2026-08-21 from the consumed Phase-B window (owner-authorized exploration, A4 disposition).

## F1. Station churn, not physics, fragments B2A on the Turkish carriers
Of 60 eval positions: istanbul lost 16 to NODESET_MISMATCH (+15 absent), turkey 13 (+15); socal
lost just 1. Runs terminate on ANY measured-set change, so istanbul's max run length is 1 --
runs structurally cannot form there. IMPROVEMENT CANDIDATE: churn-tolerant partition comparison
(compare on the intersection nodeset with a registered minimum-overlap floor) -- could roughly
double-to-triple usable sequence on istanbul/turkey. Must be designed + power-certified in the
next-window prereg, never retro-applied.

## F2. B1A's near-miss looks like ONE suspect station
Turkey's carrier max window-mean |z| is 31.5 on KO.KOZT|KO.MLTY, and ALL top-5 turkey edges
involve KO.KOZT (next carrier maxima: socal 4.9, istanbul 2.8). A z of 31 on a tanh-bounded
coherence pipeline smells like an instrument/gain artifact at KOZT, not tectonics. IMPROVEMENT
CANDIDATES: per-station robust variance renormalization; station-level z caps; a registered
station-health screen (NET-1 telemetry as an input!). Also explains why the conjunctive LOCO
gate matters.

## F3. Coverage asymmetry is large and actionable
socal: 66/66 edges every single day, zero churn (and B2A's longest stability run, len 7, lives
there). turkey: median 45, MIN 15 edges/day; KO.KHMN present 15% (and NET-1 shows it data-dark
today). Pool headroom: socal 60 pool vs 12 selected; istanbul 32/12; turkey 18/11. DATA-ADDITION
CANDIDATES: enlarge selected registries (esp. socal), churn-robust station selection for
istanbul, drop-or-replace KO.KHMN, cascadia carrier (owner-directed) built with churn history as
a selection criterion.

## F4 (track d). ML-assisted feature mining -- owner-directed (rogue-wave template)
Plan: assemble a per-carrier-day feature matrix (graph/coherence features + monitor components),
weak labels (M>=2.5/3/4 catalog events within k days, cross-region pooled for label mass),
INTERPRETABLE models (gradient-boosted trees / symbolic regression -- Haefner-style distillation,
not deep nets at this data scale), strict temporal cross-validation INSIDE the exploratory data;
surfaced features graduate ONLY via next-window prereg. Label scarcity is the honest constraint:
anomaly/self-supervised framings and small-event labels are the workarounds.

## F4 RESULTS v1 (2026-08-21) -- EXPLORATORY, zero claim weight

**Corpus A (Phase-A graphs, M>=3.5 ComCat labels, 7d horizon):** label-starved and null.
istanbul: 0 catalog events (ComCat completeness artifact for TR -- NOT evidence of quiet);
socal: 6 events, all AUCs < 0.5 (noise); turkey: 10 events and the PERSISTENCE BASELINE (0.668)
BEATS every model (logistic 0.628, GBT 0.415). The anti-apophenia guardrail worked exactly as
intended: naive "graph features predict events" would have been a clustering artifact. v1 verdict:
NO evidence of graph-feature lift over persistence in this corpus at these label counts.

**Corpus B (14-region monitor archive, 3,536 region-days, 613 positive labels, M>=4.0):** one
weak but real-looking delta -- logistic 0.590 vs persistence 0.545 (GBT 0.557). Permutation
ranking: recent_event (persistence itself), then CONFIDENCE, dRISK (day-change), roll_z_risk;
**risk LEVEL itself has NEGATIVE test importance**. Exploratory hypothesis surfaced for
graduation: *risk-CHANGE/rolling-z features may carry marginal information beyond persistence
while the headline risk level does not* (the 'confidence' importance needs an artifact check --
it encodes method availability/agreement and may proxy active periods).

**Window-2 prereg inputs from F4 v1:** (1) label supply is the binding constraint -- use
AFAD/KOERI regional catalogs for TR completeness (istanbul's 0 is a catalog artifact), lower
thresholds w/ local catalogs, longer windows; (2) candidate registered feature class:
risk-delta/rolling-z (NOT level); (3) confidence-artifact check before anything graduates;
(4) persistence baseline is MANDATORY in any registered skill comparison.

## CORRECTION to F4 RESULTS v1 (2026-08-21, append-only; codex window-2 R1)

codex identified three leaks in the F4 v1 pilot implementation: (1) the logistic scaler
(mean/std) was fit on ALL rows before the temporal cut; (2) corpus-B rows were appended
region-by-region and split at 70% of the CONCATENATED array -- not a global temporal split, so
train/test overlap in time across regions; (3) 7-day labels overlap, invalidating iid CI
reasoning. **The corpus-B delta (logistic 0.590 vs persistence 0.545) is therefore
NON-QUANTITATIVE and stands only as hypothesis-generation provenance for the M-F4 annex.** The
corpus-A null conclusions are unaffected in direction (models UNDERPERFORMED baselines there)
but carry the same implementation caveats. The M-F4 registered design (window-2 prereg v0.2)
supersedes this pilot: issue-time predictions, calibration-only fitting, global temporal
discipline, region-aware moving-block inference.
