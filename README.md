# GeoSpec: Earthquake Precursor Detection System

**Prospective monitoring of seismic precursors using geodetic strain analysis.**

> Full methodology patent-pending. Available under NDA for research partnerships, licensing discussions, and institutional due diligence.

---

## Current Monitoring Status

*Last updated: 2025-12-27 (NGL data latency: ~14 days)*

| Region | Stations | Status | Ratio | Days Elevated |
|--------|----------|--------|-------|---------------|
| Southern California - Mojave | 35 | Normal | 0.64× | 0 |
| Southern California - Coachella | 36 | Normal | 1.85× | 0 |
| Northern California - Hayward | 23 | Normal | 0.00× | 0 |
| Tokyo - Kanto | 41 | Normal | 0.21× | 0 |
| Istanbul - Marmara | 5 | Normal | 0.04× | 0 |
| Cascadia | 30 | Normal | 0.05× | 0 |

**Alert Tiers:** Normal (<2×) → Watch (2-5×) → Elevated (5-10×) → Critical (≥10×)

---

## Historical Detection Performance

Retrospective validation against major earthquakes (M6.8+):

| Event | Magnitude | Lead Time | Amplification | Detected |
|-------|-----------|-----------|---------------|----------|
| Tohoku, Japan 2011 | M9.0 | 143 hours | 7,999× | **YES** |
| Chile 2010 | M8.8 | 187 hours | 485× | **YES** |
| Turkey 2023 | M7.8 | 140 hours | 1,336× | **YES** |
| Ridgecrest 2019 | M7.1 | 141 hours | 5,489× | **YES** |
| Morocco 2023 | M6.8 | — | 2.8× | NO |

**Detection Rate: 80% (4/5)** on plate boundary events with dense GPS coverage.

### Non-Detection Analysis

Morocco 2023 was not detected due to:
- Sparse GPS network in Atlas Mountains region
- Intraplate setting with diffuse deformation
- Insufficient station density for spatial analysis

---

## False Alarm Analysis

Analysis of Southern California during seismically quiet period (2020-2022, no M≥6.5):

| Threshold | Days Exceeded | Annual Rate |
|-----------|---------------|-------------|
| Watch (≥2×) | 7.9% | ~29 days/year |
| Elevated (≥5×) | 2.1% | ~8 days/year |
| Critical (≥10×) | 0.27% | ~1 day/year |

---

## Data Sources

Live monitoring uses publicly available GPS data from:
- Nevada Geodetic Laboratory (NGL) - IGS20 reference frame
- 17,000+ stations globally

---

## Methodology

The detection framework analyzes temporal evolution of geodetic strain fields to identify mechanical instabilities preceding fault rupture.

**What is disclosed:**
- Detection thresholds and alert tier definitions
- Historical validation results with timestamps
- False alarm rates during quiet periods
- Monitored regions and station counts

**What is confidential:**
- Core diagnostic algorithm
- Signal processing methodology
- Spatial coherence criteria
- Mathematical formulations

*Full methodology available under NDA.*

---

## Research Collaboration

For research partnerships, licensing inquiries, or institutional due diligence:

**R.J. Mathews**
Email: mail.rjmathews@gmail.com
ORCID: [0009-0003-8975-1352](https://orcid.org/0009-0003-8975-1352)

---

## License

Results and validation data: MIT License
Methodology and algorithms: Proprietary / Patent Pending

---

*Prospective monitoring began January 2026. Detection performance will be validated as events occur.*
