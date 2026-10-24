# Backtest Data Sources for Historical Events

**Version**: 1.0.0
**Date**: January 14, 2026
**Purpose**: Document data sources for THD and Fault Correlation metrics for historical backtests

---

## Current Status

| Event | Lambda_geo | THD | Fault Correlation | Notes |
|-------|------------|-----|-------------------|-------|
| Ridgecrest 2019 | Real GPS | Real | Real | Fully populated |
| Tohoku 2011 | Real GPS | Pending | Pending | NIED Hi-net account available |
| Turkey 2023 | Real GPS | Pending | Pending | KOERI/IRIS sources |
| Chile 2010 | Real GPS | Pending | Pending | CSN/IRIS sources |
| Morocco 2023 | Real GPS | Pending | Limited | Sparse regional network |

---

## Data Sources by Event

### 1. Ridgecrest 2019 (M7.1) - COMPLETE

**Event Details**:
- Date: July 5, 2019 (M7.1), July 4, 2019 (M6.4 foreshock)
- Location: 35.77°N, 117.60°W
- Region: Eastern California

**THD Data Source**:
- Station: IU.TUC (Tucson, AZ)
- Network: Global Seismographic Network (IU)
- Access: IRIS FDSN Web Services
- Status: **OPERATIONAL**

**Fault Correlation Data Source**:
- Stations: CI network (Southern California)
- Segments: Ridgecrest fault system
- Access: SCEDC FDSN Web Services
- Status: **OPERATIONAL** (but limited CI availability)

---

### 2. Tohoku 2011 (M9.0) - PENDING

**Event Details**:
- Date: March 11, 2011
- Location: 38.30°N, 142.37°E
- Region: Japan Trench (Pacific coast of Tohoku)

**THD Data Source Options**:

| Source | Network | Stations | Access | Priority |
|--------|---------|----------|--------|----------|
| **NIED Hi-net** | N.* | ~800 stations | HinetPy (account approved) | **HIGH** |
| **NIED F-net** | F.* | ~70 broadband | HinetPy | Medium |
| IRIS | IU.MAJO | 1 station | Public FDSN | Fallback |

**Recommended THD Station**: N.KI2H (Kita-Ibaraki, 100Hz Hi-net) or IU.MAJO (200km from epicenter)

**Fault Correlation Data Source**:
- NIED Hi-net provides dense station coverage (~20km spacing)
- Can compute correlation across multiple fault segments
- Requires HinetPy data download for March 2011

**Data Access**:
```python
# Using HinetPy for Tohoku 2011 data
from HinetPy import Client
client = Client(username, password)
# Event window: 2011-02-25 to 2011-03-11 (14 days before)
client.get_continuous_waveform('0101', '201103010000', 24)  # Hi-net, daily
```

**Reference**: [HinetPy Documentation](https://seisman.github.io/HinetPy/)

---

### 3. Turkey 2023 (M7.8) - PENDING

**Event Details**:
- Date: February 6, 2023 (M7.8 at 04:17 local, M7.5 at 13:24 local)
- Location: 37.17°N, 37.03°E (first), 38.02°N, 37.20°E (second)
- Region: Kahramanmaras, SE Turkey

**THD Data Source Options**:

| Source | Network | Stations | Access | Priority |
|--------|---------|----------|--------|----------|
| IRIS | IU.ANTO | 1 (Ankara) | Public FDSN | **Operational** |
| KOERI | KO | ~242 stations | FDSN via KOERI | High |
| ORFEUS/EIDA | TU, GE | Regional | Public FDSN | Medium |
| AFAD | TK | Strong motion | API endpoint | Low |

**Recommended THD Station**: IU.ANTO (currently operational), augment with KOERI stations

**Fault Correlation Data Source**:
- KOERI Kandilli network has 242 stations
- East Anatolian Fault Zone needs multiple segment coverage
- Access via: http://www.koeri.boun.edu.tr/sismo/2/earthquake-catalog/

**Data Access**:
```python
# KOERI earthquake catalog
import requests
url = "http://udim.koeri.boun.edu.tr/zeqmap/xmlt/202302.xml"  # Feb 2023
response = requests.get(url)

# AFAD JSON API
afad_url = "https://deprem.afad.gov.tr/EventData/GetEventsByFilter"
```

**Reference**: [KOERI FDSN](https://www.fdsn.org/networks/detail/KO/)

---

### 4. Chile 2010 (M8.8) - PENDING

**Event Details**:
- Date: February 27, 2010
- Location: 35.85°S, 72.72°W
- Region: Maule, Central Chile

**THD Data Source Options**:

| Source | Network | Stations | Access | Priority |
|--------|---------|----------|--------|----------|
| IRIS | C, C1 | ~65 stations | Public FDSN | **HIGH** |
| CSN | C | Chilean National | IRIS DMC | High |
| IPOC | CX | GEOFON/GFZ | Public FDSN | Medium |
| II | II.LCO | Las Campanas | Public FDSN | Fallback |

**Important Note**: CSN network was significantly upgraded AFTER the 2010 earthquake. Coverage in February 2010 was limited. Best source is IRIS global stations.

**Recommended THD Station**: II.LCO (Las Campanas, Chile) or nearest IU station

**Fault Correlation Data Source**:
- Limited station coverage in 2010 (pre-upgrade)
- IPOC network (CX) provides some coverage
- May need to mark FC as "data unavailable" for this event

**Data Access**:
```python
# CSN data via IRIS
from obspy.clients.fdsn import Client
client = Client("IRIS")
# Network C for Chilean stations
st = client.get_waveforms("C", "*", "*", "BHZ",
                          "2010-02-13", "2010-02-27")
```

**Reference**: [CSN GPS Data Portal](https://gps.csn.uchile.cl/data/)

---

### 5. Morocco 2023 (M6.8) - LIMITED

**Event Details**:
- Date: September 8, 2023
- Location: 31.12°N, 8.43°W
- Region: Al Haouz Province, High Atlas Mountains

**THD Data Source Options**:

| Source | Network | Stations | Access | Priority |
|--------|---------|----------|--------|----------|
| Morocco INGM | MO | ~48 stations | FDSN (limited) | Uncertain |
| IRIS | II.* | Global network | Public FDSN | **Fallback** |
| GEOFON | GE | European | Public FDSN | Medium |

**Challenge**: Morocco's seismic network (INGM/CNRST) has limited international data sharing. The September 2023 earthquake was the largest in Morocco since 1960, so historical baseline data is sparse.

**Recommended THD Station**: Nearest IRIS station (possibly II.TAM in Tamanrasset, Algeria or GE stations in Europe)

**Fault Correlation Data Source**:
- Morocco network has 48 VSAT stations + 10-station array
- Strong motion network: 70 instruments
- **Data availability for international researchers is uncertain**
- May need to contact CNRST directly: http://cnrst.ma

**Data Access**:
```python
# Try FDSN for Morocco network (may be restricted)
from obspy.clients.fdsn import Client
try:
    client = Client("IRIS")  # MO network may not be on IRIS
    st = client.get_waveforms("MO", "*", "*", "BHZ",
                              "2023-08-25", "2023-09-08")
except:
    # Fall back to nearest global station
    st = client.get_waveforms("GE", "TAM", "*", "BHZ",
                              "2023-08-25", "2023-09-08")
```

**Reference**: [FDSN Morocco Network](https://www.fdsn.org/networks/detail/MO/)

---

## Implementation Priority

### Phase 1: THD Data (Highest Priority)
THD is more feasible to fetch than Fault Correlation because it only requires a single station.

| Event | Station | Network | Estimated Effort |
|-------|---------|---------|------------------|
| Tohoku 2011 | N.KI2H or IU.MAJO | Hi-net/IU | Medium (HinetPy) |
| Turkey 2023 | IU.ANTO | IU | Low (IRIS) |
| Chile 2010 | II.LCO or C.* | II/C | Low (IRIS) |
| Morocco 2023 | GE.TAM or II.* | GE/II | Medium (sparse) |

### Phase 2: Fault Correlation (Lower Priority)
FC requires multiple stations per segment - harder for historical events.

| Event | Feasibility | Notes |
|-------|-------------|-------|
| Tohoku 2011 | HIGH | Hi-net has dense coverage |
| Turkey 2023 | MEDIUM | KOERI has coverage, may be restricted |
| Chile 2010 | LOW | Pre-upgrade network was sparse |
| Morocco 2023 | LOW | Limited international data sharing |

---

## Data Fetch Script Requirements

### Script: `validation/fetch_historical_thd.py`

```python
"""
Fetch THD data for historical backtests.

Usage:
    python fetch_historical_thd.py --event tohoku_2011
    python fetch_historical_thd.py --event turkey_2023
    python fetch_historical_thd.py --all
"""

EVENTS = {
    'tohoku_2011': {
        'date': '2011-03-11',
        'lead_days': 14,
        'primary_station': ('IU', 'MAJO'),
        'backup_stations': [('N', 'KI2H')],  # Requires Hi-net account
        'region': 'tokyo_kanto'
    },
    'turkey_2023': {
        'date': '2023-02-06',
        'lead_days': 14,
        'primary_station': ('IU', 'ANTO'),
        'backup_stations': [('GE', 'ISP')],
        'region': 'turkey_kahramanmaras'
    },
    'chile_2010': {
        'date': '2010-02-27',
        'lead_days': 14,
        'primary_station': ('II', 'LCO'),
        'backup_stations': [('C', 'GO01')],
        'region': 'chile_maule'
    },
    'morocco_2023': {
        'date': '2023-09-08',
        'lead_days': 14,
        'primary_station': ('GE', 'TAM'),  # Algeria (nearest available)
        'backup_stations': [],
        'region': 'morocco_atlas'
    }
}
```

---

## References

### Data Centers
- [IRIS Data Services](https://ds.iris.edu/ds/nodes/dmc/data/) - Global seismic data
- [NIED Hi-net](https://www.hinet.bosai.go.jp/) - Japan high-sensitivity network
- [KOERI](https://www.koeri.boun.edu.tr/) - Turkey earthquake research
- [CSN Chile](https://gps.csn.uchile.cl/data/) - Chilean seismic/GPS data
- [FDSN Morocco](https://www.fdsn.org/networks/detail/MO/) - Morocco network info

### Python Libraries
- [HinetPy](https://github.com/seisman/HinetPy) - NIED data access
- [ObsPy](https://docs.obspy.org/) - FDSN web services client

---

*Document generated: January 14, 2026*
*GeoSpec Project - mail.rjmathews@gmail.com*
