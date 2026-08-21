# Cascadia carrier capsule — freeze candidate v1 (cayley, 2026-08-21)

Instantiates prereg v0.3 §3 for the NEW carrier. Committed BEFORE freeze per codex R1 fix 3.

- **carrier_key**: `cascadia`
- **geographic domain**: bbox lat 45.0–51.0, lon −128.0…−121.5 (the program's registered
  cascadia region: union bbox of the four legacy sub-polygons)
- **station query (RECEIPTED)**: FDSN station service, networks **UW, CC, CN**, channel **HHZ**,
  level=channel, window 2026-07-11→2026-11-30, endpoint `service.iris.edu` (redirects to
  `service.earthscope.org` — the FINAL effective URL is the receipt's authority). Receipt:
  `docs/f2g_window2_freeze/receipts/cascadia_UW_CC_CN_HHZ.txt`, sha256
  `d4256792bf85edf855a4dbaf7841982824a020cd5e075c103d832248c513a847`, HTTP 200, 203 channel
  rows, **198 unique net.sta** (UW 118, CC 43, CN 37).
- **duplicate-identity resolution**: identity = `NET.STA`; if the same STA code appears under
  multiple networks, all are distinct identities (network is part of identity); multiple channel
  epochs per station collapse to one station row; location codes: prefer `""` then `00` then
  lowest lexicographic.
- **channel precedence**: HHZ only (matches the three existing carriers' receipt convention).
- **segment assignment**: point-in-polygon over the four registered sub-polygons
  (vancouver_island, puget_sound, olympic_peninsula, columbia_river); stations in the bbox but
  outside every sub-polygon are assigned to the sub-polygon with the nearest centroid
  (great-circle); segment names = the four registry names.
- **station cap / minimum**: 16 / 8 (selection_constants.md).
- **edge construction**: identical to Phase-A REV-2 (`build_coherence_edges` contract: finite
  upper-triangle cells, selected-station endpoints, canonical sorted pairs, station-index digest
  per snapshot) — pinned BY BYTES in the manifest, not by prose.
- **UTC calendar convention**: civil UTC days, identical to the shared-calendar mechanism;
  cascadia joins the window-2 shared calendar with its own mask.
- **station-index digest**: `sha256(canonical_json(sorted(MEASURED ids)) + LF)` (the registered
  producer formula, pinned by bytes).
- **producer**: grassmann s4t lane; the producer role is mechanical-only during accrual
  (v0.3 §2 stage 2).
