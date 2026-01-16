#!/bin/bash
# verify_setup.sh
# Verify RTKLIB installation and test processing

set -e

echo "=================================================="
echo "  GeoSpec WSL Setup Verification"
echo "=================================================="

# Check RTKLIB tools
echo ""
echo "[1/4] Checking RTKLIB installation..."
for tool in convbin rnx2rtkp str2str; do
    if command -v $tool &> /dev/null; then
        echo "  $tool: OK ($(which $tool))"
    else
        echo "  $tool: MISSING"
        echo ""
        echo "ERROR: RTKLIB not installed!"
        echo "Run: sudo apt-get update && sudo apt-get install -y rtklib"
        exit 1
    fi
done

# Check access to Windows files
echo ""
echo "[2/4] Checking Windows filesystem access..."
GEOSPEC_DIR="/mnt/c/GeoSpec/geospec_sprint"
if [[ -d "$GEOSPEC_DIR" ]]; then
    echo "  GeoSpec directory: OK"
else
    echo "  GeoSpec directory: NOT FOUND"
    echo "  Expected: $GEOSPEC_DIR"
    exit 1
fi

# Check RTCM files
echo ""
echo "[3/4] Checking RTCM data..."
RTCM_DIR="$GEOSPEC_DIR/monitoring/data/rtcm"
if [[ -d "$RTCM_DIR" ]]; then
    RTCM_COUNT=$(find "$RTCM_DIR" -name "*.rtcm3" | wc -l)
    echo "  RTCM files found: $RTCM_COUNT"
    if [[ $RTCM_COUNT -gt 0 ]]; then
        echo "  Latest files:"
        find "$RTCM_DIR" -name "*.rtcm3" -exec ls -lh {} \; 2>/dev/null | tail -5
    fi
else
    echo "  RTCM directory not found"
    echo "  Run capture first: start_rtcm_capture.ps1"
fi

# Test convbin with a small sample
echo ""
echo "[4/4] Testing RTCM conversion..."
SAMPLE_RTCM=$(find "$RTCM_DIR" -name "*.rtcm3" -size +100k 2>/dev/null | head -1)
if [[ -n "$SAMPLE_RTCM" && -f "$SAMPLE_RTCM" ]]; then
    TEMP_DIR=$(mktemp -d)
    BASENAME=$(basename "$SAMPLE_RTCM" .rtcm3)

    echo "  Sample file: $SAMPLE_RTCM"
    echo "  Converting to RINEX..."

    # Try conversion (limit to first 100KB)
    head -c 102400 "$SAMPLE_RTCM" > "$TEMP_DIR/sample.rtcm3"
    if convbin -r rtcm3 -o "$TEMP_DIR/test.obs" -n "$TEMP_DIR/test.nav" "$TEMP_DIR/sample.rtcm3" 2>/dev/null; then
        if [[ -f "$TEMP_DIR/test.obs" ]]; then
            OBS_LINES=$(wc -l < "$TEMP_DIR/test.obs")
            echo "  Observation file: $OBS_LINES lines"
            echo "  RINEX conversion: OK"
        else
            echo "  WARNING: No observation data extracted"
            echo "  (This is normal if RTCM only contains navigation data)"
        fi
    else
        echo "  WARNING: convbin returned error"
        echo "  (Try with a larger RTCM file)"
    fi

    rm -rf "$TEMP_DIR"
else
    echo "  No RTCM files found for testing"
    echo "  (Skipping conversion test)"
fi

echo ""
echo "=================================================="
echo "  Verification Complete"
echo "=================================================="
echo ""
echo "To process RTCM files:"
echo "  bash /mnt/c/GeoSpec/geospec_sprint/wsl/process_rtcm.sh 2026-01-11"
echo ""
