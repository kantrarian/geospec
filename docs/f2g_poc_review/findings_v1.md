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
