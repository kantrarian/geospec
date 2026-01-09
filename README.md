# GeoSpec: Earthquake Precursor Detection System

A novel earthquake precursor detection system based on geodetic strain tensor analysis.

## Concept

GeoSpec monitors the stability of principal strain directions derived from continuous GPS networks. When strain directions begin rotating rapidly, it signals stress regime instability that often precedes major earthquakes.

The system computes a diagnostic metric (Λ_geo) that quantifies the rate of eigenframe rotation in the geodetic strain-rate tensor field.

## Validation Results

Retrospective validation against 5 major earthquakes using real GPS data:

| Earthquake | Magnitude | Lead Time | Signal Amplification | Z-score | Detected |
|------------|-----------|-----------|---------------------|---------|----------|
| Tohoku 2011 | M9.0 | 143.5 hours | 7,999× | 21,235 | **YES** |
| Chile 2010 | M8.8 | 186.8 hours | 485× | 4,057 | **YES** |
| Turkey 2023 | M7.8 | 139.5 hours | 1,336× | 6,539 | **YES** |
| Ridgecrest 2019 | M7.1 | 141.3 hours | 5,489× | 14,303 | **YES** |
| Morocco 2023 | M6.8 | 208.6 hours | 2.8× | 1.7 | NO |

**Overall: 4/5 (80%) successful detections**

### Success Criteria

- Lead time: ≥ 24 hours before mainshock
- Signal amplification: ≥ 5× above baseline
- Statistical significance: Z-score ≥ 2.0

### Key Findings

- **Consistent lead times**: 139-208 hours (6-9 days) across successful detections
- **Strong signals**: Amplification factors of 485× to 7,999× in the 72 hours before rupture
- **Works across tectonic settings**: Subduction (Tohoku, Chile), transform (Ridgecrest), and continental (Turkey)
- **Morocco failure**: Sparse GPS coverage in region led to insufficient signal

## Data Sources

Validation used publicly available GPS data from:
- Nevada Geodetic Laboratory (NGL)
- UNAVCO/EarthScope
- GEONET (Japan)

## Target Monitoring Regions

The system is designed for prospective monitoring of high-risk fault zones with dense GPS coverage:
- Southern California (San Andreas Fault)
- San Francisco Bay Area (Hayward Fault)
- Tokyo/Kanto Region
- Cascadia Subduction Zone
- Istanbul (Marmara Segment)

## Status

- Retrospective validation complete
- Prospective monitoring system in development
- Shadow monitoring phase planned

## Author

**R.J. Mathews**
Email: mail.rjmathews@gmail.com
ORCID: 0009-0003-8975-1352

## License

MIT License - see LICENSE file

---

*For inquiries about collaboration or licensing, please contact the author.*
