# GeoSpec WSL Processing Pipeline

This directory contains scripts for processing RTCM data in WSL Ubuntu using RTKLIB.

## Quick Start

### 1. Install RTKLIB in WSL (one-time)

Open **Ubuntu (WSL)** interactively:

```powershell
wsl -d Ubuntu
```

Then inside Ubuntu:

```bash
# Set password if needed: passwd
sudo -v

# Ensure universe repo (rtklib is in universe)
sudo add-apt-repository -y universe
sudo apt-get update
sudo apt-get install -y rtklib curl wget gzip

# Verify
which convbin rnx2rtkp str2str
```

**Troubleshooting**: If `apt-get update` hangs, try:
```bash
sudo apt-get update -o Acquire::ForceIPv4=true
```

### 2. Test with Captured RTCM (single station)

```bash
GEOSPEC="/mnt/c/GeoSpec/geospec_sprint"
RTCM="$GEOSPEC/monitoring/data/rtcm/COSO00USA0/2026-01-11/COSO00USA0_2026-01-11_1500Z.rtcm3"
OUT="$GEOSPEC/monitoring/data/positions/COSO00USA0/2026-01-11"
mkdir -p "$OUT"

# RTCM -> RINEX
convbin -r rtcm3 -o "$OUT/test.obs" -n "$OUT/test.nav" "$RTCM"

# Broadcast-only positioning (no products needed)
rnx2rtkp -p 0 -sys G -ti 30 -o "$OUT/test.pos" "$OUT/test.obs" "$OUT/test.nav"

# Check output
head -30 "$OUT/test.pos"
```

### 3. Process All Stations

```bash
bash /mnt/c/GeoSpec/geospec_sprint/wsl/process_rtcm.sh 2026-01-11
```

### 4. Run Position Adapter (from PowerShell)

```powershell
cd C:\GeoSpec\geospec_sprint
python -m monitoring.src.position_adapter --date 2026-01-11
```

---

## Architecture

```
Windows                          WSL Ubuntu
--------                         -----------
RTCM Capture (Python)     -->    RTKLIB Processing
  |                                  |
  v                                  v
monitoring/data/rtcm/            RINEX + .pos files
  |                                  |
  +----------------------------------+
                  |
                  v
          Position Adapter (Python)
                  |
                  v
          NGL-format CSV for Lambda_geo
```

## Two-Pass Processing Strategy

### Pass A: Broadcast-only (fast, no dependencies)

Uses `-p 0` (single point positioning) with broadcast ephemeris from the RTCM stream itself.

```bash
bash process_rtcm.sh 2026-01-11
```

**Accuracy**: ~2-5 meters (sufficient for proving pipeline works)

### Pass B: PPP with IGS products (accuracy upgrade)

Uses `-p 7` (PPP static) with precise orbits/clocks.

```bash
# Download IGS products first
bash download_igs_products.sh 2026-01-11

# Then process with --ppp flag
bash process_rtcm.sh 2026-01-11 --ppp
```

**Accuracy**: ~5-20 cm (depending on products and observation quality)

---

## IGS Product Filename Change

IGS moved to **long product filenames** around GPS week 2238.

| Type | Old Name | New Name |
|------|----------|----------|
| Rapid orbit | `igr22380.sp3.Z` | `IGS0OPSRAP_20223310000_01D_15M_ORB.SP3.gz` |
| Ultra-rapid | `igu22380_00.sp3.Z` | `IGS0OPSULT_20223310600_02D_15M_ORB.SP3.gz` |

The `download_igs_products.sh` script tries **both** naming schemes automatically.

**Product Sources:**
- IGS: `https://files.igs.org/pub/products/{WWWW}/`
- CDDIS: `https://cddis.nasa.gov/archive/gnss/products/{WWWW}/`

---

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_rtklib.sh` | One-time RTKLIB installation |
| `process_rtcm.sh` | Convert RTCM -> RINEX -> positions |
| `download_igs_products.sh` | Fetch IGS orbits/clocks (old + new names) |
| `verify_setup.sh` | Verification and testing |

---

## Position Adapter

The adapter (`monitoring/src/position_adapter.py`) converts RTKLIB `.pos` files to NGL format.

**Key feature**: Reference position is computed from **median of first 100 epochs** in the data, not hardcoded. This prevents false motion signals from reference frame mismatch.

**Output columns:**
- `refepoch`: decimal year
- `e`, `n`, `u`: east/north/up in meters (relative to computed reference)
- `se`, `sn`, `su`: standard errors
- `station`: 4-char station code

---

## Data Flow

```
IGS-IP NTRIP (igs-ip.net:2101)
    |
    v
COSO00USA0_2026-01-11_1500Z.rtcm3   (binary stream)
    |
    v [convbin -r rtcm3]
COSO00USA0_2026-01-11_1500Z.obs     (RINEX obs)
COSO00USA0_2026-01-11_1500Z.nav     (RINEX nav)
    |
    v [rnx2rtkp -p 0 or -p 7]
COSO00USA0_2026-01-11_1500Z.pos     (ECEF positions)
    |
    v [position_adapter.py]
rtcm_positions_2026-01-11.csv       (NGL ENU format)
    |
    v [existing Lambda_geo pipeline]
Strain tensors and alerts
```

---

## Troubleshooting

### RTKLIB not found
```bash
sudo apt-get update && sudo apt-get install -y rtklib
```

### No observation data in RINEX
- RTCM stream may only contain navigation messages
- Check file size: should be >100KB for 1 hour
- Try: `convbin -scan` to see message types

### Position output is empty
- Check if .obs file was created
- Verify satellite count in navigation file
- Try: `rnx2rtkp -trace 2` for debug output

### IGS products download fails
- Products have ~17-41 hour latency (rapid) or ~12-18 days (final)
- Use broadcast-only mode for recent data
- Check if GPS week directory exists at source

### Poor position accuracy
- Broadcast-only: expect 2-5m scatter
- PPP: need >30 min observation, good satellite geometry
- Check quality flag in .pos (column 6): 1=fix, 5=single
