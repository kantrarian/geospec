# Cascadia sub-polygon OVERLAP TIE amendment v1

- **status**: APPEND-ONLY, PRE-USE. Registered before any realized selection consumes it.
- **authority**: codex ruling 2026-08-31T15:54Z item 2, adopting cayley's option (b).
- **author**: cayley
- **supersedes**: nothing. `cascadia_carrier_capsule.md` stands unchanged; this amendment
  *adds* the containment tie rule that document never registered.
- **claim ceiling**: a geometry derivation rule only. It certifies nothing, admits nothing,
  opens no window-2 value, and fires no capture. `prestart_overall=REFUSE`; Λ_geo INCONCLUSIVE.

## Why this exists

The four registered cascadia sub-polygons **overlap**. Measured from committed bytes at public
`963d7b9`, over the 198 unique `NET.STA` HHZ stations of the frozen receipt
`docs/f2g_window2_freeze/receipts/cascadia_UW_CC_CN_HHZ.txt` (authenticated against its
envelope's `body_sha256` before any coordinate was read):

```
vancouver_island     lat 48.5..51.0   lon -128.0..-123.5
puget_sound          lat 47.0..48.5   lon -124.0..-122.0
olympic_peninsula    lat 46.5..48.0   lon -125.0..-122.5
columbia_river       lat 45.0..46.5   lon -124.5..-121.5

puget_sound n olympic_peninsula = lat 47.0..48.0, lon -124.0..-122.5   (non-empty)

single containing polygon   115
BOTH puget_sound and olympic_peninsula    28
no containing polygon        55
```

`cascadia_carrier_capsule.md` registers point-in-polygon plus a nearest-centroid fallback for
stations outside **every** sub-polygon. It says nothing about a station inside **two**. Without
this amendment the derivation refuses and no cascadia station→segment map can be produced.

## The registered rule

Applied in this order, to every cascadia station:

1. **Boundary counts as contained** (closed polygon semantics). A point on an edge shared by two
   polygons is contained by **both**.
2. **Exactly one** containing polygon → assign that segment. Reason `single`.
3. **More than one** containing polygon → assign the containing polygon whose registered centroid
   is at minimum registered great-circle distance from the station. Reason `overlap_nearest`.
4. **Exact distance tie** → assign the lexicographically smallest registered segment id.
5. **No containing polygon** → nearest registered centroid across **all** registered polygons,
   with the same exact-tie rule. Reason `outside_fallback`.

Ambiguous stations are **never dropped** (that would make selection depend on the downstream
geometry partition) and the registered polygons are **never redrawn**.

## Bound definitions

| element | binding |
|---|---|
| polygon coordinates | verbatim from the DESIGN-PINNED `monitoring/src/fault_segments.py` (`CASCADIA_SEGMENTS`, design pin `region_polygons`); reproduced in the generated bundle |
| boundary predicate | collinear with an edge (`abs(cross) <= eps`) **and** within that edge's bounding box expanded by `eps` |
| epsilon | `1e-9` degrees |
| centroid | arithmetic mean of the sub-polygon's vertices |
| distance | haversine great-circle, earth radius **6371.0088 km**, units kilometres |
| segment-id ordering | byte-wise lexicographic on the registered segment name |
| registered segment ids | `columbia_river`, `olympic_peninsula`, `puget_sound`, `vancouver_island` |
| implementation | `monitoring/src/w2_power_geometry_inputs_gen_cayley.py`: `contains_closed`, `_on_edge`, `_centroid`, `_haversine_km`, `assign_one_station` (module identity pinned in the manifest) |

The boundary predicate is defined **here** rather than inherited: the design-pinned
`FaultSegment.contains_point` is an *open* ray cast whose result exactly on an edge is arbitrary.
That module is not modified by this amendment.

## Retained per station

Every station's decision record carries its candidate containing segments, the great-circle
distance to **every** registered centroid, the selected segment, the reason
(`single` | `overlap_nearest` | `outside_fallback`), and whether a lexicographic tie-break fired —
so any reviewer can re-derive the choice rather than trust it.

## Registered controls

| doctor | what it proves |
|---|---|
| D1 | all three branches exercised by the real receipt (115 / 28 / 55); the overlap population is asserted to still be the measured 28, so a drift in the registered polygons fails loudly |
| D2 | every one of the 28 overlap decisions selects the **nearest containing** polygon, never a non-containing one |
| D3 | a point exactly ON the shared edge is contained by **both** polygons and routes through the overlap rule |
| D4 | an exact distance tie resolves to the lexicographically smallest id |
| D5 | the anticipated 16-station registry maps under the amendment |
| D6 | the amendment does **not** alter the anticipated 16 (zero overlap routings) — asserted, not assumed, because that is the claim the packet makes |
| D7 | an empty registered segment (`olympic_peninsula` takes none of the anticipated 16) is disclosed, and the remaining active-segment count is checked against the registered engine minimum rather than filled by synthesizing a station |

## Effect on the anticipated set

None. All 28 ambiguous stations are `UW.*`; the anticipated 16-station cascadia registry is
entirely `CC.*`/`CN.*` and routes 12 `single` + 4 `outside_fallback`, with zero overlap routings.
This amendment exists so the **realized** selection at the registered 2026-09-02 cutoff cannot
encounter an unresolvable station at prestart, when there would be no time to rule on it.
