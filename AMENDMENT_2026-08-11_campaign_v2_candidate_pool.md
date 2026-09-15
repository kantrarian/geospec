# AMENDMENT 2026-08-11 — Campaign-v2 candidate pool (Phase 0.5): coverage-driven registry candidates, frozen pre-probe

- **date (UTC):** 2026-08-11
- **author:** grassmann
- **status:** REGISTERED, PRE-PROBE FREEZE — this pins the campaign-v2 candidate pool
  (contract `codex-d2-campaign-v2-2026-08-10-v1`, codex 1817 §1) BEFORE any
  waveform-availability probe fires. Pool membership is final: no post-probe station
  add, drop, reassignment, channel invention, or provider expansion. Nothing here
  fetches waveforms, lifts a freeze, tunes on outcomes, or claims.
- **pool artifact:** `monitoring/src/d2_campaign_v2_candidate_pool.json`
  (canonical UTF-8 JSON, sorted keys, 2-space indent, one terminal LF)
- **pool SHA-256:** `15d0e32c51c027dc144c5c6d57ec5f100a59374f6248abfc5c56ee38628ddc67`
- **schedules bound (codex 1817):** incident `[2026-03-01, 2026-06-29)` (ref 2026-07-29),
  activation `[2026-03-13, 2026-07-11)` (A = 2026-08-10); both exactly 120 scored days.

## Defect being cured (v1 premise)

v1 registered minimal 2-station segments frozen blind to historical availability; under
fix-A frame-complete scoring one weak station killed a segment-day, and no carrier reached
the 60-day floor (honest 0-candidate null, sealed-v1 close). v2 builds each segment's
candidate pool from the provider's own station metadata so the later selection (Phase 0.5
probes → deterministic 4/3/infeasible rule) can register 3–4 coverage-proven stations per
segment. Selection inputs are availability-only; no waveform bytes, lambda ratios, QC or
replay results, or v1 scores entered this pool.

## Candidate enumeration (metadata only, deterministic, receipts retained)

One channel-level FDSN station query per carrier network, HHZ, over the scheduled-window
union `[2026-03-01, 2026-07-11)`, bbox = carrier's segment-polygon envelope +0.15°
(bbox values are in each receipt URL). Raw responses + hashed receipts:
`E:/GeoSpec/d2_campaign_v2_phase05/station_metadata_receipts/` (routed with the Phase-0.5
evidence bundle; digests also inside the pool artifact under `station_metadata_receipts`).

| carrier | net @ service | status | bytes | body sha256 (16) | attempted UTC |
|---|---|---|---|---|---|
| istanbul_marmara | KO @ eida.koeri.boun.edu.tr | 200 | 4820 | `9bccbf5f013073b1` | 2026-08-11T01:18:52Z |
| turkey_kahramanmaras | KO @ eida.koeri.boun.edu.tr | 200 | 2823 | `6f62ebb68a9652f5` | 2026-08-11T01:18:53Z |
| turkey_kahramanmaras | TU @ eida.koeri.boun.edu.tr | 404 | 400 | `668492e6242a88d0` | 2026-08-11T01:18:54Z |
| socal_coachella | CI @ service.scedc.caltech.edu | 200 | 9106 | `260fb7c440038a20` | 2026-08-11T01:18:54Z |

The TU 404 is conclusive-no-inventory for TU on KOERI's station service; the sole TU
candidate (`TU.ANDN`, v1-retained) stays in the pool and its adequacy is decided by the
availability probe (a conclusive 404 there = ABSENT per §2 of the contract).

## Construction rules (codex 1817 §1, applied verbatim)

1. **v1 stations retain their v1 segment** (source `v1` / `v1+fdsn`); coordinates filled
   from metadata where returned.
2. **New stations** are assigned to the nearest unchanged segment polygon: inside-polygon
   wins (ties by segment name); else nearest polygon centroid (haversine), ties by segment
   name. Assignment method is recorded per station.
3. **Segment polygons are byte-unchanged** from production `fault_segments.py` (t1/t2
   discipline; they are reproduced inside the pool artifact for self-containment).
4. **Every carrier keeps ≥2 segments** (all three have 3) and **no station identity occurs
   in more than one segment pool within a carrier** (codex 2017 wording repair; enforced
   by construction — a station has exactly one `segment` binding per carrier).
5. **Ordered exact NSLC:** v1 stations keep their frozen v1 `ordered_nslc_candidates`;
   FDSN stations get `NET.STA.<loc>.HHZ` for each returned loc code, loc-sorted.

## Frozen pool (110 candidates)

| carrier | segment | candidates | of which v1-retained |
|---|---|---|---|
| istanbul_marmara | marmara_west | 8 | 4 |
| istanbul_marmara | marmara_central | 15 | 4 |
| istanbul_marmara | izmit | 9 | 2 |
| turkey_kahramanmaras | east_anatolian_north | 8 | 4 |
| turkey_kahramanmaras | east_anatolian_central | 7 | 2 |
| turkey_kahramanmaras | east_anatolian_south | 3 | 2 |
| socal_coachella | coachella_north | 20 | 0 |
| socal_coachella | coachella_south | 20 | 2 |
| socal_coachella | brawley_seismic_zone | 20 | 2 |

`coachella_north: 0 v1-retained` is correct — v1 registered only two socal segments
(coachella_south + brawley, 2 stations each); coachella_north is the production topology's
third segment, populated here for the first time. `east_anatolian_south` (3 candidates) is
the thinnest pool and the likely turkey limiter under the ≥3-adequate rule.

## What happens next (not part of this freeze)

Read-only availability probes over the exact scheduled days (KOERI
`availability/1/extent` union-overlap ≥43,200 s; SCEDC day-volume HEADs sum-overlap
≥43,200 s; 404 = ABSENT; timeout/429/5xx/malformed = INDETERMINATE and holds the bundle),
then the deterministic selection rank `(-min(inc,act), -(inc+act), station_id)` with the
4/3/infeasible rule and the ≥60-both-arms carrier gate. Carriers gate to
`ELIGIBLE_METADATA_POTENTIAL` or `COVERAGE_INFEASIBLE`. Phase-1 waveform I/O remains on
asylum's fresh SB-8 hold.
