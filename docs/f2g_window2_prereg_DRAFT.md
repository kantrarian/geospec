# fault2graph WINDOW-2 preregistration — DRAFT v0.1 (cayley, 2026-08-21)

**Status: DRAFT for codex freeze rounds. NOT frozen. Owner directives bound: b2 standing until
project end; window-2 = 3 carriers + CASCADIA; PoC-review candidates graduate ONLY here; ML
candidates require persistence baselines. No window-2 measurement value is read before the
registered gates say so.**

## 1. Window (PROSPECTIVE)

Civil calendar **2026-07-11 → 2026-11-19** (132 days, mirroring window 1's span). The window is
registered BEFORE it completes: this is a prospective design — the strongest form. The campaign
fires only after the window closes and the producer delivers; per-carrier masks derive from
produced snapshot filenames (dual-witnessed, the Amendment-2 mechanism now registered UP FRONT
rather than as a repair).

## 2. Carriers

istanbul_marmara, socal_coachella, turkey_kahramanmaras (as window 1) + **cascadia (NEW,
owner-directed)**. Cascadia build: PNSN/UW + CN network stations; the Phase-A builder pipeline
(REV-2 contracts: station-index digests, typed states, canonical edges) reused verbatim.
Producer coordination with grassmann required (s4t lane).

## 3. Registered station selection (F3 graduation — an ALGORITHM, not hand-picks)

Per carrier, the selected registry is produced by a REGISTERED selection algorithm over the
candidate pool: (i) provider-metadata-confirmed coordinates; (ii) trailing-90-day data-presence
≥ a registered floor (NET-1-class telemetry, PROSPECTIVE rule per the codex boundary — no
retroactive availability claims); (iii) churn score (day-over-day measured-set stability) above
a registered floor; (iv) size caps per carrier (registered at freeze; intent: socal enlarged
toward ~20 from its 60-station pool; istanbul reselected churn-aware). KO.KHMN-class stations
(no coordinates or dark) are excluded by rule (i)/(ii), typed.

## 4. Statistical families

- **B2A (carried, verdict-bearing on certification)**: the window-1 certified partition-run
  statistic, unchanged semantics; power must RE-CERTIFY on window-2's true geometry (masks) via
  the registered synthetic pipeline before it is verdict-bearing.
- **B2B (NEW, F1 graduation — churn-tolerant runs)**: partition comparison on the INTERSECTION
  nodeset with a registered minimum-overlap floor (candidate: ≥⅔ of the selected registry);
  refusals typed INTERSECTION_BELOW_FLOOR (replacing blanket NODESET_MISMATCH termination).
  Motivation (exploratory, zero weight): window-1 istanbul lost 16/60 eval positions to nodeset
  churn — runs structurally could not form. Verdict-bearing ONLY if power-certified.
- **B1B (NEW, F2 graduation — robust burst)**: window-mean |z| with (a) per-station robust
  variance renormalization, (b) a registered station-health/z-cap screen (candidate cap
  registered at freeze; motivation: window-1 KO.KOZT |z|=31.5 artifact suspicion), (c) the LOCO
  gate unchanged. Verdict-bearing ONLY if power-certified (window-1 B1A never certified).
- **B3A (carried as typed secondary)**: unchanged; expected typed non-answers absent
  certification; no new design effort.
- **M-F4 (NEW lane — monitor risk-delta skill test)**: on window-2 monitor data (14 regions,
  disjoint from the graph lane): registered question — do risk-CHANGE features ({drisk,
  roll_z_risk}, logistic, registered spec) beat the PERSISTENCE baseline for M≥4-in-7d,
  AUC difference with registered CI? Persistence baseline MANDATORY; the `confidence` field is
  excluded until its artifact check (pre-freeze, exploratory, on the archive) clears it.
  Motivation (exploratory): archive logistic 0.590 vs persistence 0.545.
- **MAG-1 (companion module)**: joins on the SAME window under its own R1-PASSED design
  (v0.2 + coverage admission: IZN, FRN+TUC; kahramanmaras typed mag-untestable default);
  numerical instantiation in its own freeze.

## 5. Multiplicity (registered before any certification outcome)

Graph lane: family alpha 0.05/3 as window 1. PRIMARY endpoint = B2A. B2B and B1B enter as
REGISTERED SECONDARIES; whichever subset certifies is tested under Holm within the graph-lane
family (allocation finalized at freeze; no post-certification additions; uncertified families
report typed CANNOT_DETERMINE_NO_POWER exactly as window 1). M-F4 and MAG-1 are separate
registered lanes with their own alpha (0.05 each, single primary endpoint per lane). No
availability-driven family changes post-freeze except as disclosed amendments.

## 6. Power certification

Per family, on the TRUE window-2 geometry (post-mask derivation): synthetic effect grids through
the ENTIRE decision rule (selection algorithm, admission gates, statistic, multiplicity,
robustness screens), Tier-S → Tier-C, Clopper-Pearson ≥0.80, R=20/40 stopping, registered
coordinate orders, bound-mode byte-verification — the full window-1 instrument, reused. Terminal
wording on failure: MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH.

## 7. Run protocol (registered = the window-1 sealed instrument, now standard)

Input manifest (byte-bound, anchor digests, zero value reads) → committed fire-authorization
record binding the codex instrument-pass note + all source blobs → remote atomic fire lease +
common-dir lock + canonical checkpoint + history one-shot → schema-closed evidence → hash-sealed
result → FROZEN result verifier (v2.3 pattern: full recomputation, capsule closure, executed-
bytes attestation, lease receipt) → codex + grassmann verification → owner verdict decision.
At most ONE sealed run for the graph lane; one per companion lane. Fresh owner seal required
before any real-value read; exploratory access to window-2 data is PROHIBITED until its sealed
run is consumed (the window-1 ordering, made explicit).

## 8. Sequence & dependencies

draft → codex freeze rounds → grassmann cross-authored bars (incl. B2B/B1B engine lanes + MAG-1
non-identity KATs) → cascadia carrier build + window-2 producer run (grassmann, s4t) → calendar
authorities (dual-witnessed) → power campaigns → certification → fresh owner seal → sealed
run(s) → verification → verdicts. The daily-monitor renewal decision (~08-25) is upstream of
M-F4's data continuity — flagged to asylum.

## 9. Non-claims

Λ_geo INCONCLUSIVE unchanged; no forecast, precursor, or displacement claims; window-1
exploratory findings carry zero evidential weight here beyond having motivated designs;
mechanism names motivate effect shapes only. Publication decisions remain owner-controlled.
