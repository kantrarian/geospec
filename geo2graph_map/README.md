# geo2graph-map

An interactive **geometry-only** map of three seismic-network graphs built by the geo2graph
pipeline: Istanbul/Marmara, Southern California (Coachella), and Türkiye
(Kahramanmaraş / East Anatolian).

**[Open the map](https://kantrarian.github.io/geo2graph-map/)**

## What this shows

- **Stations** — the station pool (110) and the selected analysis registry (35), colored per
  network region. Coordinates are reproduced from the sources and retrieval records named in
  `layers_manifest.json` (FDSN station services: SCEDC/Caltech for network CI; KOERI/Boğaziçi
  University EIDA for network KO — exact query URLs and receipt hashes in the manifest). Every plotted coordinate is provider-confirmed.
- **Proximity edges** — k-nearest-neighbor geometric edges. Distances are the projected
  Euclidean distance in metres in the listed per-carrier UTM CRS. These are *geometric* edges,
  **not** measurement edges.
- **Segment boxes** — coarse station-grouping polygons. They are *not* fault traces.
- **Typed absence** — one selected station (KO.KHMN, Pazarcık) has no published coordinate in
  the provider metadata; it is listed, never plotted at an invented location.

Every layer carries an explicit `claim_status` in `layers_manifest.json`.

## What this does NOT show

**No measurement data of any kind.** No waveform data, no coherence or correlation values, no
alarms, no time series. This page makes **no earthquake forecast, precursor, or displacement
claims**. The underlying research method's validation status is **inconclusive**, and this map is
network geometry only — the instrument, not a result.

No measurement-valued source or measurement artifact is included in this publication. Source
artifacts not included here cannot be independently reconstructed from these hashes alone.

## Files

| File | Content |
|---|---|
| `index.html` | Single-page Leaflet client; requires network access to the pinned Leaflet CDN and OpenStreetMap tiles |
| `geo2graph_geometry.geojson` | All 325 features (stations, proximity edges, segment boxes) |
| `layers_manifest.json` | Per-layer claim status, coordinate provenance receipts, CRS, typed absences, non-claims |
| `LICENSE` | MIT (scope below) |

## License and attribution

The code and original presentation in this repository are MIT-licensed (see `LICENSE`).
Third-party data and services retain their own terms: station metadata © their providers —
SCEDC (Caltech) and KOERI/Boğaziçi University EIDA; map tiles © OpenStreetMap contributors;
Leaflet is BSD-2-Clause and loads from the unpkg CDN at view time.
