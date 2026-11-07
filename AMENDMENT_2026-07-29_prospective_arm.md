# Amendment R4 — prospective-arm activation: declustering + Molchan accumulation (registered 2026-07-29)

**Status:** registered and effective from 2026-07-29 (UTC). Signed off by the project owner with all
recommended parameters adopted: exclusion cap **365 d**; registered alarm level **tier ≥ 2**;
**symmetric exclusion**; **episode-level bookkeeping**; minimum **10 pooled mainshocks** before any
skill language. Companion to amendments R2 (scoring window) and R3 (rolling baseline).

---

## 1. The problem being solved

After a large mainshock: (i) M5.5+ **aftershocks are near-certain** for weeks (Båth's law: largest
aftershock typically ~M_main−1.2 ≈ M5.9 for the M7.1) — a standing alarm would "hit" trivially;
(ii) **postseismic deformation** (afterslip/viscoelastic) keeps the geodetic channel elevated for
weeks–months — the system will alarm continuously through it. Neither tests precursory skill. Symmetric
danger: if post-event alarms were scored as false alarms, the record would be unfairly punished for
physics it cannot avoid. The rules below exclude such periods from scoring **in both directions**.

## 2. Event set (targets)

- **Admissible events:** M ≥ 5.5 (R2 threshold), within the region's 100 km buffer (current config),
  any depth (subduction-interface events are relevant to these regions; disclosed).
- **Declustering to mainshocks-only** via Gardner–Knopoff (G-K) space–time windows:
  events falling inside a larger event's window are aftershocks/foreshocks — **removed from the target
  set** (not hits, not misses). The largest event of a cluster is its mainshock.
- **G-K windows, with a disclosed cap:** distance L(M) = 10^(0.1238·M + 0.983) km; time per the G-K
  table, **capped at 365 days** (pure G-K gives ~2.5 years for M7.1; the cap avoids multi-year region
  blanking and is declared here as a deviation, chosen before any outcome is known).

## 3. Region exclusion state (the aftershock guard)

- Any admissible mainshock in region R opens an **exclusion window** on R: the G-K duration (capped
  365d) over the G-K distance around its epicenter.
- **During exclusion, region R is unscoreable — symmetrically.** Days in exclusion are removed from
  BOTH the alarm-time numerator and the time base; events inside the window are not scoreable as hits;
  alarms during it are not scoreable as false alarms. The region simply does not exist for the record
  until the window closes.
- **Supersession:** an event LARGER than the window's mainshock re-opens a fresh window keyed to the
  new magnitude. It is scoreable only if a **fresh alarm episode** preceded it — defined as an episode
  whose onset followed ≥14 consecutive tier-0 days in that region ("alarm reset" rule: one standing
  elevation cannot claim a cascade of events).
- **Immediate application, disclosed at registration:** the 2026-07-28 M7.1 opens Kumamoto's exclusion
  at adoption — **Kumamoto is unscoreable until 2027-07-28** (or a superseding larger event). The
  currently elevated post-event state can therefore never be mined as evidence in either direction.

## 4. Alarm bookkeeping (fixes a live honesty defect)

- **Alarm episode** := consecutive tier ≥ 2 days in one region, grouped with gap tolerance ≤ 3 days.
- **One episode ↔ at most one hit; one mainshock ↔ at most one crediting episode.** No per-day
  padding in either direction. (Note: the current validator scores per-day — the existing "22 false
  alarms" are actually ~6 episodes. R4 restates episode-level counts alongside, without touching the
  R1-era records.)
- **Hit:** an admissible mainshock within R2's 14-day window of any day of the episode. **False-alarm
  episode:** an episode whose every day's window closes with no admissible mainshock. **Miss:** an
  admissible mainshock with no episode day in the preceding 14 days.

## 5. Primary metric: the Molchan trajectory

- **Accumulation starts 2026-07-29** (R2's effective date). Nothing earlier enters; the Kumamoto
  episode remains a dual-reported motivating case with zero weight (per R2).
- **τ (alarm fraction)** = tier≥2 days / scoreable days, pooled over the 14 regions, exclusion days
  removed from both numerator and denominator. **ν (miss fraction)** = missed mainshocks / admissible
  mainshocks. Report the (τ, ν) point and its trajectory over time; per-region and pooled; a tier≥1
  sensitivity curve is reported alongside but tier≥2 is THE registered alarm level.
- **Skill assessment:** binomial confidence bounds on the Molchan diagram; **no skill claim of any
  kind until ≥ 10 pooled admissible mainshocks** have accumulated (pre-registered minimum; at current
  base rates this is plausibly 1–3 years). Interim reports are descriptive only and must say so.

## 6. Parameter table (all pinned at registration)

| parameter | value | source |
|---|---|---|
| magnitude threshold | M ≥ 5.5 | R2 (unchanged) |
| hit window | 14 d | R2 (unchanged) |
| alarm level | tier ≥ 2 | current config, now registered |
| episode gap tolerance | 3 d | matches case-study analysis |
| alarm reset gap | 14 tier-0 days | new (supersession guard) |
| G-K distance | 10^(0.1238M+0.983) km | Gardner–Knopoff 1974 |
| exclusion duration | G-K time, cap 365 d | cap = disclosed deviation |
| region buffer | 100 km | current config |
| minimum sample for any skill claim | 10 pooled mainshocks | pre-registered |

## 7. Implementation + ordering (disclosed)

The daily validator currently implements R2 only. R4's declustering runs first in the **analysis
layer** (a scored report generated from the registered rules); the validator code catches up with unit
KATs (G-K window tests, symmetric-exclusion tests, episode-grouping tests, supersession/reset tests)
— target owner grassmann or cayley, gated like all code changes. Until the code lands, the registered
spec governs interpretation; the raw per-day validator output remains published untouched.

## 8. What this spec does NOT do

It does not validate Λ_geo (INCONCLUSIVE stands); it does not make the July episode evidence (zero
weight, per R2); it does not lift the NONCONFIRMATORY cap or any Class-2 gate; it does not authorize
public skill claims — it defines the only bookkeeping under which a future skill claim could ever be
honestly made.
