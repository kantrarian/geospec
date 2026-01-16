#!/bin/bash
# download_igs_products.sh
# Download IGS precise orbit and clock products for PPP processing
# Handles both legacy (igrWWWWD.sp3.Z) and new long-name formats
# (IGS0OPSRAP_YYYYDDD0000_01D_15M_ORB.SP3.gz)
#
# Usage:
#   bash download_igs_products.sh YYYY-MM-DD

set -euo pipefail

DATE="${1:-}"
if [[ -z "$DATE" ]]; then
  echo "Usage: $0 YYYY-MM-DD"
  exit 1
fi

GEOSPEC_DIR="/mnt/c/GeoSpec/geospec_sprint"
PRODUCTS_DIR="$GEOSPEC_DIR/monitoring/data/products"
mkdir -p "$PRODUCTS_DIR"
cd "$PRODUCTS_DIR"

# Compute GPS week / DOW
DAYS_SINCE_REF=$(( ($(date -d "$DATE" +%s) - $(date -d "1980-01-06" +%s)) / 86400 ))
GPS_WEEK=$(( DAYS_SINCE_REF / 7 ))
GPS_DOW=$(( DAYS_SINCE_REF % 7 ))

YEAR="$(date -d "$DATE" +%Y)"
DOY="$(date -d "$DATE" +%j)"
CLEAN_DATE="${DATE//-/}"

# Where to pull from:
# - CDDIS: /archive/gnss/products/WWWW/
# - IGS mirror: files.igs.org/pub/products/WWWW/
BASE_CDDIS="https://cddis.nasa.gov/archive/gnss/products/${GPS_WEEK}"
BASE_IGS="https://files.igs.org/pub/products/${GPS_WEEK}"

fetch() {
  local url="$1"
  local out="$2"
  echo "  -> $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$out" "$url" 2>/dev/null || return 1
  else
    wget -q -O "$out" "$url" 2>/dev/null || return 1
  fi
}

maybe_unzip() {
  local f="$1"
  if [[ "$f" == *.gz ]]; then
    gunzip -f "$f" 2>/dev/null || true
  elif [[ "$f" == *.Z ]]; then
    # .Z is "compress"; gunzip can often handle, otherwise install 'ncompress'
    gunzip -f "$f" 2>/dev/null || true
  fi
}

echo "=================================================="
echo "  IGS Products Download"
echo "  Date: $DATE (GPS week=$GPS_WEEK dow=$GPS_DOW doy=$DOY)"
echo "=================================================="

# ---------
# ORBITS
# ---------
echo ""
echo "[1/2] Orbits"

# Legacy rapid + ultrarapid
LEG_RAP_SP3="igr${GPS_WEEK}${GPS_DOW}.sp3.Z"
LEG_ULT_SP3="igu${GPS_WEEK}${GPS_DOW}_00.sp3.Z"

# New long-name rapid/ultra-rapid (post-GPS week 2238)
NEW_RAP_SP3="IGS0OPSRAP_${YEAR}${DOY}0000_01D_15M_ORB.SP3.gz"
NEW_ULT_SP3="IGS0OPSULT_${YEAR}${DOY}0600_02D_15M_ORB.SP3.gz"

ORB_OK="false"

# Try legacy rapid first
if [[ ! -f "$LEG_RAP_SP3" && ! -f "${LEG_RAP_SP3%.Z}" ]]; then
  echo "  Trying legacy rapid: $LEG_RAP_SP3"
  fetch "${BASE_IGS}/${LEG_RAP_SP3}" "$LEG_RAP_SP3" || \
  fetch "${BASE_CDDIS}/${LEG_RAP_SP3}" "$LEG_RAP_SP3" || true
fi
if [[ -f "$LEG_RAP_SP3" ]]; then maybe_unzip "$LEG_RAP_SP3"; fi
if [[ -f "${LEG_RAP_SP3%.Z}" ]]; then
  ln -sf "${LEG_RAP_SP3%.Z}" "igs${CLEAN_DATE}.sp3"
  ORB_OK="true"
  echo "  OK: Legacy rapid orbit linked"
fi

# If legacy rapid failed, try new rapid
if [[ "$ORB_OK" != "true" ]]; then
  if [[ ! -f "$NEW_RAP_SP3" && ! -f "${NEW_RAP_SP3%.gz}" ]]; then
    echo "  Trying new rapid: $NEW_RAP_SP3"
    fetch "${BASE_IGS}/${NEW_RAP_SP3}" "$NEW_RAP_SP3" || \
    fetch "${BASE_CDDIS}/${NEW_RAP_SP3}" "$NEW_RAP_SP3" || true
  fi
  if [[ -f "$NEW_RAP_SP3" ]]; then maybe_unzip "$NEW_RAP_SP3"; fi
  if [[ -f "${NEW_RAP_SP3%.gz}" ]]; then
    ln -sf "${NEW_RAP_SP3%.gz}" "igs${CLEAN_DATE}.sp3"
    ORB_OK="true"
    echo "  OK: New rapid orbit linked"
  fi
fi

# If still nothing, try legacy ultrarapid (predicted)
if [[ "$ORB_OK" != "true" ]]; then
  if [[ ! -f "$LEG_ULT_SP3" && ! -f "${LEG_ULT_SP3%.Z}" ]]; then
    echo "  Trying legacy ultra-rapid: $LEG_ULT_SP3"
    fetch "${BASE_IGS}/${LEG_ULT_SP3}" "$LEG_ULT_SP3" || \
    fetch "${BASE_CDDIS}/${LEG_ULT_SP3}" "$LEG_ULT_SP3" || true
  fi
  if [[ -f "$LEG_ULT_SP3" ]]; then maybe_unzip "$LEG_ULT_SP3"; fi
  if [[ -f "${LEG_ULT_SP3%.Z}" ]]; then
    ln -sf "${LEG_ULT_SP3%.Z}" "igs${CLEAN_DATE}.sp3"
    ORB_OK="true"
    echo "  OK: Legacy ultra-rapid orbit linked"
  fi
fi

# Try new ultra-rapid as last resort
if [[ "$ORB_OK" != "true" ]]; then
  if [[ ! -f "$NEW_ULT_SP3" && ! -f "${NEW_ULT_SP3%.gz}" ]]; then
    echo "  Trying new ultra-rapid: $NEW_ULT_SP3"
    fetch "${BASE_IGS}/${NEW_ULT_SP3}" "$NEW_ULT_SP3" || \
    fetch "${BASE_CDDIS}/${NEW_ULT_SP3}" "$NEW_ULT_SP3" || true
  fi
  if [[ -f "$NEW_ULT_SP3" ]]; then maybe_unzip "$NEW_ULT_SP3"; fi
  if [[ -f "${NEW_ULT_SP3%.gz}" ]]; then
    ln -sf "${NEW_ULT_SP3%.gz}" "igs${CLEAN_DATE}.sp3"
    ORB_OK="true"
    echo "  OK: New ultra-rapid orbit linked"
  fi
fi

if [[ "$ORB_OK" != "true" ]]; then
  echo "  WARNING: No orbit products available"
  echo "  (PPP will fail - use broadcast-only mode)"
fi

# ---------
# CLOCKS (best-effort legacy for now)
# ---------
echo ""
echo "[2/2] Clocks (best-effort legacy)"
LEG_RAP_CLK="igr${GPS_WEEK}${GPS_DOW}.clk.Z"

CLK_OK="false"
if [[ ! -f "$LEG_RAP_CLK" && ! -f "${LEG_RAP_CLK%.Z}" ]]; then
  echo "  Trying legacy rapid clock: $LEG_RAP_CLK"
  fetch "${BASE_IGS}/${LEG_RAP_CLK}" "$LEG_RAP_CLK" || \
  fetch "${BASE_CDDIS}/${LEG_RAP_CLK}" "$LEG_RAP_CLK" || true
fi
if [[ -f "$LEG_RAP_CLK" ]]; then maybe_unzip "$LEG_RAP_CLK"; fi
if [[ -f "${LEG_RAP_CLK%.Z}" ]]; then
  ln -sf "${LEG_RAP_CLK%.Z}" "igs${CLEAN_DATE}.clk"
  CLK_OK="true"
  echo "  OK: Legacy rapid clock linked"
fi

if [[ "$CLK_OK" != "true" ]]; then
  echo "  WARNING: No clock products available"
  echo "  (PPP accuracy reduced - consider broadcast-only)"
fi

echo ""
echo "=================================================="
echo "  Summary"
echo "=================================================="
echo "  Products dir: $PRODUCTS_DIR"
ls -lh "igs${CLEAN_DATE}.sp3" 2>/dev/null || echo "  SP3: NOT AVAILABLE"
ls -lh "igs${CLEAN_DATE}.clk" 2>/dev/null || echo "  CLK: NOT AVAILABLE (OK for broadcast-only)"
