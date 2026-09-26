# Window-2 power-certification SIZING (v1, 2026-08-22)

**TIMING ONLY** — fixture-authority geometry at the frozen cap sizes (16/20/14/16
stations, 132-day calendar); no power/recovery number exists in this artifact or its
JSON sibling. Owner directive 1924Z (quote sha `7caab14d`). Raw timings:
`power_cert_sizing_v1.json`; script: `monitoring/src/w2_power_sizing_bench_cayley.py`.

## Measured (geomen, py3.14, single process)

| item | seconds |
|---|---|
| bound generator (4 carriers, one replicate panel) | 0.02 |
| B2A @999 draws | 0.1 |
| B3A @999 | 0.3 |
| B2B @999 (memoized pair cache) | 1.1 |
| B1B @999 | **7.9** (the dominant family) |
| **all-four replicate @999 (Tier-S)** | **9.4** |
| **all-four replicate @9,999 (Tier-C, linear fit)** | **84.4** |

## Projections (assumptions in the JSON)

- **Tier-S full-grid smoke** (82 registered points × R=50, all-four rule): **~10.7 h**.
- **Tier-C** (3 candidates × 4 families + 2 artifact points = 14 points):
  **~6.6 h at R=20**, **~13.1 h at R=40** — without per-replicate LOCO.
- **LOCO scenarios** (16 cascadia NEW-registry folds): the JSON's ×17 row
  (~112–223 h) assumes ALL FOUR engines rerun per fold — the conservative bound. If
  LOCO folds rerun ONLY the family under test (the Phase-B loco_gate shape applied to
  B1B), the increment is ≈ +16×7.9 s ≈ +126 s/replicate → **~×2.5, i.e. ~16–33 h**.
  **The LOCO-per-replicate composition for window-2 certification is the ONE open
  sizing variable** and needs the codex/grassmann design settlement before the cert
  fires.

## Scheduling read (for the owner's PRESTART date)

Without per-replicate LOCO: Tier-S + Tier-C ≈ **one day of geomen CPU**. With
B1B-only LOCO folds: ≈ **1.5–2 days**. Only the all-four-per-fold reading (which
nothing in the frozen texts requires) reaches multi-day scale. Compute is local CPU —
no cloud spend. B1B optimization headroom exists (its 7.9 s is renorm+windows over
9,999 permutations; vectorizing the window scan would cut the dominant term) but is
NOT needed for a one-day campaign and would touch a codex-closed engine — not
proposed.

No power claims; Λ_geo INCONCLUSIVE.

---

## APPENDED 2026-08-23 — v2 re-timing at the BOUND window-2 geometry

The calendar-authority ruling (codex 1400Z, option (a): fixed 192-position grid,
baseline 60) changes the certified geometry from the 132-day Phase-B shape this
document timed. Re-timed at v2 (raw: `power_cert_sizing_v2.json`, same box, same
method; v1 retained unedited):

| quantity | v1 (132d) | v2 (192d) |
|---|---|---|
| all-four replicate @999 | 9.4 s | 16.3 s |
| all-four replicate @9,999 (fit) | 84.4 s | 126.7 s |
| B1B @9,999 (dominant) | ~79 s | 102.2 s |
| Tier-S full grid (82 pts × R50 @999) | 10.7 h | 18.6 h |
| Tier-C base R20 / R40 (no LOCO) | 6.6 / 13.1 h | 9.9 / 19.7 h |

**Honest LOCO-composed Tier-C at v2** (amendment rule: B2A/B2B/B3A once per
replicate; B1B runs 1+16 folds ONLY after a full-Holm positive on the 3
B1B-detection candidate points): a certified-quality B1B point goes positive on
most replicates, so folds run nearly every replicate there —
3 pts × R20 × (126.7 + 16×102.2)s ≈ 29.4 h + 11 pts × R20 × 126.7 s ≈ 5.5 h ≈
**~35 h** at R20-clean; R40 extensions push toward ~70 h; a low-positive run
floors near 10 h. This EXCEEDS the 16–33 h range accepted with the amendment
(which was computed on v1 timing).

**Mitigation (no semantic change): per-point process parallelism.** The 14 Tier-C
points are independent certifications; the registered seed grammar is
per-(authority, family, replicate) and point-independent; records are per-point
artifacts. Running points as N parallel processes divides wall-clock by ~min(N,14)
with zero effect on any registered number. At 6–8 processes the worst case fits
comfortably inside the 08-25 fire → 08-26 completion window. Decision routed to
codex/grassmann with the calendar close (it is an EXECUTION arrangement, not a
statistical change, but the runner invocation shape should be recorded before the
fire). Timing only; no power numbers exist in this appendix; Λ_geo INCONCLUSIVE.
