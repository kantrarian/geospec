# GeoSpec API Reference: Geophysical Data Sources

**Version**: 1.0
**Date**: January 2026
**Purpose**: Step-by-step connection instructions for seismic and GNSS data APIs

---

## Table of Contents

1. [IRIS FDSN Web Service](#1-iris-fdsn-web-service)
2. [NGL (Nevada Geodetic Laboratory)](#2-ngl-nevada-geodetic-laboratory)
3. [IGS Real-Time Service (RTS)](#3-igs-real-time-service-rts)
4. [EarthScope NTRIP Caster](#4-earthscope-ntrip-caster)
5. [NIED F-net/Hi-net](#5-nied-f-nethi-net)
6. [ORFEUS/GEOFON FDSN](#6-orfeusgeofon-fdsn)
7. [GeoNet New Zealand](#7-geonet-new-zealand)
8. [Integration Priority](#8-integration-priority)

---

## 1. IRIS FDSN Web Service

**Purpose**: Global seismic waveforms for THD analysis and fault correlation
**Status**: Currently operational in GeoSpec

### Endpoint
- Base URL: `https://service.iris.edu/fdsnws/`
- Services: dataselect (waveforms), station (metadata), event (catalogs)

### Authentication
- **Public data**: No registration required
- **Restricted data**: Optional authenticated access via queryauth

### Connection (Python/ObsPy)

```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Initialize IRIS FDSN client
client = Client("IRIS")

# Define query parameters
start_time = UTCDateTime("2026-01-01T00:00:00")
end_time = UTCDateTime("2026-01-02T00:00:00")
network = "IU"
station = "TUC"
channel = "BHZ"

# Download waveforms
st = client.get_waveforms(
    network=network,
    station=station,
    location="*",
    channel=channel,
    starttime=start_time,
    endtime=end_time
)

# Save to file
st.write("waveforms.mseed", format="MSEED")
```

### Networks Available via IRIS

| Network | Region | Notes |
|---------|--------|-------|
| IU | Global | Primary network for GeoSpec (100% availability) |
| II | Global | Backup global network |
| CI | Southern California | 20-25% availability issues |
| BK | Northern California | Works via NCEDC |

### GeoSpec Integration
- File: `monitoring/src/seismic_thd.py`
- Function: `fetch_continuous_data_for_thd()`
- Current stations: IU.TUC, IU.COR, IU.MAJO, IU.ANTO, BK.BKS

---

## 2. NGL (Nevada Geodetic Laboratory)

**Purpose**: Primary source for GPS position time series used in Lambda_geo computation
**Status**: CURRENTLY OPERATIONAL - Primary GPS data source for GeoSpec

### Endpoint
- Base URL: `https://geodesy.unr.edu/`
- Data format: tenv3 (IGS20 reference frame)
- Solution: Daily final positions

### Authentication
- **Public access**: No registration required
- **Rate limiting**: None documented, but recommend caching

### Data Latency
- **Typical delay**: 10-14 days from observation to solution publication
- **Reason**: Requires final IGS orbits and clock products for high accuracy
- **Impact**: Real-time Lambda_geo requires supplementary sources (see IGS RTS)

### Connection (Python)

```python
import requests
from datetime import datetime

def fetch_ngl_tenv3(station_id: str) -> str:
    """
    Fetch GPS position time series from NGL.

    Args:
        station_id: 4-character station ID (e.g., "KAIK", "P595")

    Returns:
        tenv3 file content as string
    """
    url = f"https://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/{station_id}.tenv3"
    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        return response.text
    elif response.status_code == 404:
        raise ValueError(f"Station {station_id} not found in NGL database")
    else:
        raise ConnectionError(f"NGL returned status {response.status_code}")

# Example: Fetch Kaikoura station
tenv3_data = fetch_ngl_tenv3("KAIK")

# Parse tenv3 format
for line in tenv3_data.strip().split('\n'):
    if line.startswith('#'):
        continue  # Skip header
    parts = line.split()
    # Columns: station, decimal_year, mjd, week, day, north, east, up, ...
    station = parts[0]
    decimal_year = float(parts[1])
    north_mm = float(parts[5])
    east_mm = float(parts[6])
    up_mm = float(parts[7])
```

### tenv3 File Format

| Column | Description | Units |
|--------|-------------|-------|
| 1 | Station ID | - |
| 2 | Decimal year | years |
| 3 | Modified Julian Date | days |
| 4 | GPS week | weeks |
| 5 | Day of week | 0-6 |
| 6 | North displacement | mm |
| 7 | East displacement | mm |
| 8 | Up displacement | mm |
| 9-11 | Sigma (N, E, U) | mm |
| 12-14 | Correlation coefficients | - |

### Station Coverage

| Region | Example Stations | Count |
|--------|------------------|-------|
| Southern California | P595, P597, P598 | 200+ |
| Northern California | BRIB, TIBB | 150+ |
| Cascadia | ALBH, SC02, P402 | 100+ |
| New Zealand | KAIK, WGTN, MAST | 30+ |
| Japan | Limited coverage | 10-20 |
| Alaska | AC58, AC48 | 50+ |

### GeoSpec Integration
- File: `monitoring/src/live_data.py`
- Function: `_fetch_ngl_data(station_id)` and `fetch_multi_station_gps_data()`
- Cache location: `monitoring/data/gps_cache/`
- Cache expiry: 7 days

### Limitations
- **10-14 day latency**: Not suitable for real-time monitoring
- **No Japan coverage**: Use GEONET instead
- **No Turkey coverage**: Limited Mediterranean stations

---

## 3. IGS Real-Time Service (RTS)

**Purpose**: Real-time GNSS corrections for Lambda_geo (addresses 2-14 day NGL latency)
**Status**: CREDENTIALS OBTAINED - Connection verified January 2026

### Connection Status (January 2026)
- **User**: devilldog (credentials stored in `.env`)
- **Verified casters**:
  - `igs-ip.net:2101` - 352 streams (IGS, EUREF, MGEX networks)
  - `products.igs-ip.net:2101` - 87 streams (corrections: IGS02, IGS03)
  - `euref-ip.net:2101` - 211 streams (European stations)
- **Test script**: `monitoring/src/test_ntrip_connection.py`

### Endpoint
- Caster URL: `products.igs-ip.net:2101`
- Protocol: NTRIP (Networked Transport of RTCM via Internet Protocol)

### Authentication
- **Registered**: January 2026 via BKG
- **Cost**: Free for non-commercial/research use
- **Expires**: After 6 months of inactivity

### Connection (BKG NTRIP Client)

```bash
# Using BNC (BKG NTRIP Client)
bnc --caster products.igs-ip.net:2101 \
    --user [your_username] \
    --password [your_password] \
    --mountpoint IGS03
```

### Connection (RTKLIB str2str)

```bash
# Save RTCM stream to file
str2str -in ntrip://[username]:[password]@products.igs-ip.net:2101/IGS03 \
        -out file::output.rtcm3
```

### Mountpoints

| Mountpoint | Content | Latency |
|------------|---------|---------|
| IGS03 | Multi-GNSS corrections | ~5 seconds |
| IGS01 | GPS-only corrections | ~5 seconds |

### Integration Requirements for GeoSpec
1. Register for IGS RTS access
2. Install RTKLIB or BNC
3. Create `live_gnss_fetcher.py` module
4. Process RTCM streams → position/velocity → strain tensors
5. Feed to Lambda_geo computation

### Expected Impact
- Reduce GPS latency from 2-14 days to seconds
- Enable real-time Lambda_geo monitoring

---

## 4. EarthScope NTRIP Caster

**Purpose**: Dense U.S. GNSS coverage for California/Cascadia (alternative to NGL)
**Status**: NOT YET INTEGRATED

### Endpoint
- Caster URL: Provided upon registration (modernized platform as of 2025)
- Contact: rtgps@earthscope.org

### Authentication
- **Required**: Re-registration for modernized system
- **Portal**: https://www.earthscope.org/
- **Cost**: Free for research

### Connection (RTKLIB)

```bash
str2str -in ntrip://[username]:[password]@[caster_url]:[port]/[mountpoint] \
        -out file::output.rtcm3
```

### Coverage
- Network of the Americas (formerly UNAVCO/GAGE)
- Dense California coverage
- Cascadia subduction zone
- 1 Hz data streams available

### Integration Requirements for GeoSpec
1. Email rtgps@earthscope.org for access
2. Obtain credentials and caster URL
3. Select mountpoints for target regions
4. Process with PPP software

---

## 5. NIED F-net/Hi-net

**Purpose**: Dense seismic coverage for Japan (Tokyo/Kanto region)
**Status**: Hi-net registration submitted January 2026

### Endpoints

| Network | URL | Format |
|---------|-----|--------|
| F-net | https://www.fnet.bosai.go.jp/ | WIN32, FDSN |
| Hi-net | https://www.hinet.bosai.go.jp/ | WIN32 |

### Authentication
- **Required**: Registration for full access
- **Approval**: Typically granted for researchers
- **Application**: Via NIED website

### Connection (HinetPy for Hi-net)

```python
from HinetPy import Client, win32_to_sac

# Initialize client with credentials
client = Client("your_username", "your_password")

# Request 24 hours of continuous waveforms
data, ctable = client.get_continuous_waveform(
    "BO",  # F-net network code
    "20260101_000000",  # Start time
    1440  # Duration in minutes (24 hours)
)

# Convert WIN32 to SAC format
win32_to_sac(data, ctable, outdir="output/")
```

### Connection (ObsPy for F-net FDSN)

```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# F-net may be accessible via NIED FDSN
client = Client("NIED")

st = client.get_waveforms(
    network="BO",
    station="*",
    location="*",
    channel="BHZ",
    starttime=UTCDateTime("2026-01-01"),
    endtime=UTCDateTime("2026-01-02")
)
```

### Integration Requirements for GeoSpec
1. Complete Hi-net registration (pending)
2. Install HinetPy: `pip install HinetPy`
3. Create `japan_seismic_fetcher.py` module
4. Add F-net stations to fault segment configuration

### Expected Impact
- Enable fault correlation for Tokyo region
- Improve THD spatial resolution (currently using distant IU.MAJO)

---

## 6. ORFEUS/GEOFON FDSN

**Purpose**: Seismic data for Turkey (Marmara, Kahramanmaras regions)
**Status**: NOT YET INTEGRATED (using IU.ANTO fallback)

### Endpoints

| Provider | URL | Networks |
|----------|-----|----------|
| ORFEUS | https://orfeus-eu.org/fdsnws/ | European networks |
| GEOFON | https://geofon.gfz-potsdam.de/fdsnws/ | GE network |

### Authentication
- **Public access**: No registration for open data

### Connection (ObsPy)

```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Initialize ORFEUS or GEOFON client
client = Client("ORFEUS")  # or "GEOFON"

# Query Turkish network
st = client.get_waveforms(
    network="TU",
    station="AYDN",
    location="*",
    channel="BHZ",
    starttime=UTCDateTime("2026-01-01"),
    endtime=UTCDateTime("2026-01-02")
)

st.write("turkey_waveforms.mseed", format="MSEED")
```

### Turkish Stations

| Station | Location | Network |
|---------|----------|---------|
| AYDN | Aydin | TU |
| BALB | Balikesir | TU |
| ISK | Istanbul | KO |

### Integration Requirements for GeoSpec
1. Test TU network availability via ORFEUS
2. Add TU stations to `seismic_thd.py` client routing
3. Update `run_ensemble_daily.py` region config

### Expected Impact
- Replace distant IU.ANTO (Ankara) with local stations
- Enable fault correlation for Marmara region

---

## 7. GeoNet New Zealand

**Purpose**: Low-latency GNSS data for New Zealand regions (addresses NGL 10-14 day latency for kaikoura)
**Status**: NOT YET INTEGRATED - Recommended for kaikoura region

### Overview

GeoNet is New Zealand's geological hazard monitoring system operated by GNS Science. It provides real-time and near-real-time GNSS data through multiple access methods, with significantly lower latency than NGL.

### Endpoints

| Service | URL | Latency | Format |
|---------|-----|---------|--------|
| Tilde API | `https://tilde.geonet.org.nz/v3/api/` | ~1 day | JSON/CSV |
| PositioNZ RTS | NTRIP caster (registration required) | Real-time | RTCM3 |
| AWS Open Data | `s3://geonet-open-data/` | ~1 day | RINEX |
| FITS API | `https://fits.geonet.org.nz/` | ~1 day | CSV |

### Authentication

| Service | Registration | Cost |
|---------|-------------|------|
| Tilde API | Not required | Free |
| FITS API | Not required | Free |
| AWS Open Data | AWS account | Free (requester pays) |
| PositioNZ RTS | Required (LINZ) | Free for research |

### Connection: Tilde API (Recommended for Daily Data)

**VERIFIED WORKING (January 2026)** - Data latency: ~3 days vs NGL's 10-14 days

```python
import requests
from datetime import datetime, timedelta

def fetch_geonet_gnss(station_id: str, start_date: datetime, end_date: datetime,
                       direction: str = "east") -> dict:
    """
    Fetch GNSS displacement time series from GeoNet Tilde API.

    Args:
        station_id: 4-character station ID (e.g., "KAIK", "WGTN")
        start_date: Start of query period
        end_date: End of query period
        direction: "east", "north", or "up"

    Returns:
        Dictionary with displacement observations
    """
    # Tilde v3 API endpoint format:
    # /v3/data/gnss/{station}/displacement/nil/1d/{direction}/{start}/{end}
    url = (
        f"https://tilde.geonet.org.nz/v3/data/gnss/{station_id}/"
        f"displacement/nil/1d/{direction}/"
        f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    )

    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        return response.json()
    else:
        raise ConnectionError(f"GeoNet returned status {response.status_code}")

# Example: Fetch Kaikoura east displacement (last 30 days)
kaik_data = fetch_geonet_gnss(
    "KAIK",
    datetime.now() - timedelta(days=30),
    datetime.now(),
    "east"
)

# Response format:
# {
#   "domain": "gnss",
#   "station": "KAIK",
#   "aspect": "east",
#   "observations": [
#     {"ts": "2026-01-13T11:59:00Z", "val": 0.33353, "err": 0.0027},
#     ...
#   ]
# }
```

**Verified Data Availability (January 16, 2026)**:
- KAIK (Kaikoura): Data through January 13, 2026 (3 days latency)
- NGL (same station): Data through January 3, 2026 (13 days latency)

### Connection: FITS API (DEPRECATED)

**Note**: FITS was deprecated on April 10, 2025. All data migrated to Tilde. Use Tilde API above.

```python
# DEPRECATED - Do not use for new development
# FITS was retired in April 2025
# All GNSS data is now available via Tilde API
```

### Connection: PositioNZ Real-Time Service

```bash
# Registration required at: https://www.linz.govt.nz/products-services/geodetic/positionz

# Using RTKLIB str2str
str2str -in ntrip://[username]:[password]@positionz.linz.govt.nz:2101/KAIK_32 \
        -out file::kaik_rtcm.rtcm3

# Mountpoint naming: [STATION]_[rate]
# Rate options: 32 (1 Hz), 33 (5 Hz)
```

### Key Stations for Kaikoura Region

| Station | Location | Lat/Lon | Notes |
|---------|----------|---------|-------|
| KAIK | Kaikoura | -42.43, 173.54 | Primary station |
| WGTN | Wellington | -41.29, 174.78 | Reference station |
| CHTI | Chatham Islands | -43.96, -176.54 | Offshore reference |
| CMBL | Campbell Island | -52.55, 169.14 | Subantarctic |
| MAST | Masterton | -40.98, 175.68 | North of fault |
| HAST | Hastings | -39.65, 176.84 | Coastal reference |

### Comparison: GeoNet vs NGL for New Zealand

| Metric | GeoNet | NGL |
|--------|--------|-----|
| **Latency** | ~1 day (Tilde), real-time (PositioNZ) | 10-14 days |
| **Station Count** | 200+ (PositioNZ network) | ~30 |
| **Reference Frame** | ITRF2014/NZGD2000 | IGS20 |
| **Solution Type** | Daily PPP (Tilde), RTK (PositioNZ) | Daily final |
| **API Access** | Yes (Tilde, FITS) | HTTP only |

### Integration Requirements for GeoSpec

1. **Add `geonet_gnss.py` module** for Tilde/FITS API access
2. **Update `live_data.py`** to check GeoNet before NGL for NZ stations
3. **Reference frame alignment**: Transform NZGD2000 → IGS20 if needed
4. **For real-time**: Register for PositioNZ RTS access

### Expected Impact
- Reduce kaikoura data latency from 10-14 days to ~1 day
- Enable near-real-time Lambda_geo for New Zealand
- Improve dashboard responsiveness for kaikoura region

### Resources

- GeoNet Data Portal: https://www.geonet.org.nz/data/types/geodetic
- Tilde API Docs: https://tilde.geonet.org.nz/v3/api/
- FITS API Docs: https://fits.geonet.org.nz/api-docs/
- PositioNZ Info: https://www.linz.govt.nz/products-services/geodetic/positionz
- AWS Open Data: https://registry.opendata.aws/geonet/

---

## 8. Integration Priority

### Current Operational Sources

| Source | Type | Status | Regions |
|--------|------|--------|---------|
| NGL | GPS (daily) | **OPERATIONAL** | California, Cascadia, NZ, Alaska |
| IRIS FDSN | Seismic | **OPERATIONAL** | Global |

### Recommended Integration Order

Based on impact and effort:

| Priority | Service | Impact | Effort | Addresses |
|----------|---------|--------|--------|-----------|
| **1** | **GeoNet Tilde API** | **High** | **Low** | **Kaikoura latency (10-14 days → 1 day)** |
| 2 | ORFEUS/GEOFON | Medium | Low | Turkey THD improvement |
| 3 | NIED F-net/Hi-net | High | Medium | Japan coverage (awaiting registration) |
| 4 | IGS RTS | Critical | High | Lambda_geo real-time (all regions) |
| 5 | EarthScope NTRIP | High | High | California/Cascadia real-time GNSS |
| 6 | GeoNet PositioNZ | High | Medium | New Zealand real-time GNSS |

### Immediate Action: GeoNet Integration

**Why GeoNet is Priority 1:**
- Kaikoura region currently shows flatline data after Jan 4 due to NGL 10-14 day latency
- GeoNet Tilde API requires NO registration and provides ~1 day latency
- Low integration effort: HTTP API similar to current NGL workflow

**Steps:**
1. Create `monitoring/src/geonet_gnss.py` module
2. Add GeoNet as fallback source for New Zealand stations in `live_data.py`
3. Test with KAIK, WGTN, MAST stations
4. Regenerate kaikoura ensemble data

### Near-Term Actions (This Week)

1. **Integrate GeoNet Tilde API** for New Zealand
   - No registration required
   - Add to `live_data.py` as NZ-specific source

2. **Test ORFEUS/GEOFON access** for TU network
   - No registration required
   - Add to `seismic_thd.py` client routing

### Medium-Term Actions (This Month)

3. **Follow up on NIED Hi-net registration**
   - Check email for approval
   - Test F-net FDSN access as backup

4. **Register for IGS RTS** (already obtained credentials)
   - Plan RTKLIB integration
   - Target: real-time Lambda_geo for all regions

5. **Contact EarthScope**
   - Email rtgps@earthscope.org
   - Request access to modernized NTRIP caster

---

## Appendix: Required Software

| Software | Purpose | Installation |
|----------|---------|--------------|
| ObsPy | Seismic data via FDSN | `pip install obspy` |
| HinetPy | NIED Hi-net/F-net access | `pip install HinetPy` |
| RTKLIB | GNSS NTRIP processing | https://www.rtklib.com/ |
| BNC | NTRIP client | https://igs.bkg.bund.de/ntrip/bnc |

---

*Document generated: January 2026*
*Last updated: January 16, 2026 - Added NGL and GeoNet sections*
*GeoSpec Project - mail.rjmathews@gmail.com*
