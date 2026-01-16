# GeoSpec API Reference: Geophysical Data Sources

**Version**: 1.0
**Date**: January 2026
**Purpose**: Step-by-step connection instructions for seismic and GNSS data APIs

---

## Table of Contents

1. [IRIS FDSN Web Service](#1-iris-fdsn-web-service)
2. [IGS Real-Time Service (RTS)](#2-igs-real-time-service-rts)
3. [EarthScope NTRIP Caster](#3-earthscope-ntrip-caster)
4. [NIED F-net/Hi-net](#4-nied-f-nethi-net)
5. [ORFEUS/GEOFON FDSN](#5-orfeusgeofon-fdsn)
6. [Integration Priority](#6-integration-priority)

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

## 2. IGS Real-Time Service (RTS)

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

## 3. EarthScope NTRIP Caster

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

## 4. NIED F-net/Hi-net

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

## 5. ORFEUS/GEOFON FDSN

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

## 6. Integration Priority

Based on impact and effort, recommended integration order:

| Priority | Service | Impact | Effort | Addresses |
|----------|---------|--------|--------|-----------|
| 1 | ORFEUS/GEOFON | Medium | Low | Turkey THD improvement |
| 2 | NIED F-net/Hi-net | High | Medium | Japan coverage (awaiting registration) |
| 3 | IGS RTS | Critical | High | Lambda_geo real-time (core method) |
| 4 | EarthScope NTRIP | High | High | California/Cascadia real-time GNSS |

### Near-Term Actions (This Week)

1. **Test ORFEUS/GEOFON access** for TU network
   - No registration required
   - Add to `seismic_thd.py` client routing

2. **Follow up on NIED Hi-net registration**
   - Check email for approval
   - Test F-net FDSN access as backup

### Medium-Term Actions (This Month)

3. **Register for IGS RTS**
   - Apply at https://igs.org/rts/user-access/
   - Plan RTKLIB integration

4. **Contact EarthScope**
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
*GeoSpec Project - mail.rjmathews@gmail.com*
