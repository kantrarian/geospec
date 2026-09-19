# AMENDMENT 2026-08-09 — Ridgecrest topology v2 (t2): outcome-blind little_lake redraw

- **date (UTC):** 2026-08-09
- **author:** cayley
- **status:** REGISTERED, NOT COMMISSIONED — production `fault_segments.py` remains t1;
  ridgecrest remains `BLOCKED_TOPOLOGY` in production and in the step-4b BASE campaign.
  This amendment registers the redraw the step-4b contract requires
  (`codex-d2-step4b-2026-08-09-v1`: "ridgecrest is not admitted by an unregistered redraw")
  so a LATER ridgecrest extension can run outcome-blind. Commissioning t2 into production
  is a separate owner decision; nothing here lifts a freeze, fetches data, tunes, or claims.
- **registry artifact:** `monitoring/src/ridgecrest_t2_registry.json`
  (canonical UTF-8 JSON, sorted keys, compact separators, one terminal LF)
- **registry SHA-256:** `449273b866f682d1363806daef5509cac40f1480d003a7fd0731b71a365f2657`
- **bar supplement:** `monitoring/src/test_d2_step4b_ridgecrest_t2_redkats_cayley.py`

## Defect being cured

`validate_topology` (monitoring/src/fault_correlation.py) rejects any region whose correlated
segments share a NET.STA station — the same sensor in two segments trivially inflates their
inter-segment correlation. Ridgecrest t1 shares two: `CI.LRL` (airport_lake + little_lake) and
`CI.WBS` (ridgecrest_mainshock + little_lake). little_lake has exactly one unshared station
(`CI.JRC2`), so no station-drop fix exists inside t1; the region has been `BLOCKED_TOPOLOGY`
in every step-4 disposition.

## Redraw (minimal diff)

Only `little_lake` changes; its two SHARED stations are replaced by two UNSHARED live CI
stations. Polygons, strikes, dips, rakes, and the other two segments are byte-identical to t1.

| segment | t1 stations | t2 stations |
|---|---|---|
| ridgecrest_mainshock | CI.WBS, CI.SLA, CI.CLC | unchanged |
| airport_lake | CI.LRL, CI.CCC, CI.TOW2 | unchanged |
| little_lake | CI.LRL*, CI.WBS*, CI.JRC2 | **CI.WBM, CI.DTP**, CI.JRC2 |

(*shared in t1.) After the swap every NET.STA appears in exactly one segment; the region
passes the `validate_topology` disjointness rule.

## Outcome-blind selection procedure (metadata only, deterministic)

1. **Candidate pool:** live SCEDC FDSN station query, retrieved 2026-08-09T01:56–01:58Z:
   `https://service.scedc.caltech.edu/fdsnws/station/1/query?net=CI&minlatitude=35.2&maxlatitude=35.75&minlongitude=-118.05&maxlongitude=-117.5&channel=BHZ,HHZ,EHZ&level=station&format=text&endafter=2026-08-01T00:00:00`
   returned exactly: CI.DTP, CI.LRL, CI.RRC, CI.SRT, CI.WBM.
2. **Exclude shared:** CI.LRL is in airport_lake — excluded. Remaining pool: DTP, RRC, SRT, WBM.
3. **Rank by distance to the (unchanged) little_lake polygon** (lat 35.3–35.6,
   lon −117.9…−117.6; box-edge distance, 1° lat = 111.19 km, lon scaled by cos φ):
   - **CI.WBM** 35.60839/−117.89049 → **0.93 km** (north edge)
   - **CI.DTP** 35.26742/−117.84581 → **3.62 km** (south edge)
   - CI.RRC 35.372816/−117.989768 → 8.14 km (west)
   - CI.SRT 35.69235/−117.75051 → 10.27 km (north; sits in the airport_lake footprint area)
4. **Take the nearest two:** WBM + DTP. They bracket the segment along its 340° strike
   (WBM north end, DTP south end). No waveform sample from CI.WBM or CI.DTP has ever been
   fetched, processed, or scored anywhere in this program as of this amendment — both are
   absent from t1, from Lane A (ridgecrest incident acquisition was unavailable), and from the
   sealed diagnostic (socal/turkey/istanbul only). Selection used station metadata only.
5. **Channel candidates:** frozen order BHZ→HHZ per the production preference
   (`seismic_data.py` channel order). Channel-level SCEDC query (same retrieval window)
   confirms all nine registry stations carry open-ended (`3000-01-01`) BHZ@40 Hz and
   HHZ@100 Hz epochs; closed EHZ epochs are excluded. Both rates satisfy the 1–10 Hz band.

## What this does and does not enable

- **Enables (later, separately routed):** a step-4b ridgecrest EXTENSION — its own
  outcome-blind plan (carrier `ridgecrest`, `topology_version=t2`, provider `s3://scedc-pds`,
  the two frozen 90-day arms) built against this registry, gated by the bar supplement. The
  BASE campaign's three-carrier policy is unchanged; `build_campaign_plan` still refuses
  ridgecrest.
- **Does not enable:** candidate admission. Ridgecrest has no accepted segmented
  incident/control replay evidence, so under the contract's four-condition rule its
  deterministic ceiling from calibration alone is `BLOCKED_REPLAY_UNAVAILABLE`. A ridgecrest
  segmented replay would be a separately-authorized future diagnostic.
- **Does not touch:** production `fault_segments.py` (stays t1), the five freezes, the
  production registry, deployment, publication, or any claim.

-- cayley
