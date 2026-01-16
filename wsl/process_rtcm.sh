#!/bin/bash
# process_rtcm.sh
# Convert RTCM files to position solutions using RTKLIB
#
# Key insight: NAV coverage must be continuous across observation span.
# Solution: Build GLOBAL merged NAV from ALL stations (some streams are obs-only).
#
# Three-phase approach:
#   Phase 1: Convert all RTCM to RINEX (all stations)
#   Phase 2: Build GLOBAL merged NAV from all stations
#   Phase 3: Compute positions using global NAV
#
# Usage:
#   bash process_rtcm.sh [YYYY-MM-DD] [--ppp]

set -e

# Configuration
GEOSPEC_DIR="/mnt/c/GeoSpec/geospec_sprint"
RTCM_DIR="$GEOSPEC_DIR/monitoring/data/rtcm"
POS_DIR="$GEOSPEC_DIR/monitoring/data/positions"
PRODUCTS_DIR="$GEOSPEC_DIR/monitoring/data/products"

# Parse arguments
PROCESS_DATE=""
USE_PPP="false"

for arg in "$@"; do
  case $arg in
    --ppp)
      USE_PPP="true"
      ;;
    *)
      if [[ -z "$PROCESS_DATE" ]]; then
        PROCESS_DATE="$arg"
      fi
      ;;
  esac
done

# Default to today
PROCESS_DATE="${PROCESS_DATE:-$(date -u +%Y-%m-%d)}"
CLEAN_DATE="${PROCESS_DATE//-/}"

echo "=================================================="
echo "  GeoSpec RTCM -> Position Processing"
echo "  Date: $PROCESS_DATE"
echo "  Mode: $([ "$USE_PPP" = "true" ] && echo "PPP (precise)" || echo "Broadcast-only (fast)")"
echo "  Strategy: Global merged NAV across all stations"
echo "=================================================="

# Check RTKLIB installation
if ! command -v convbin &> /dev/null; then
    echo ""
    echo "ERROR: RTKLIB not installed!"
    echo "Run: sudo apt-get update && sudo apt-get install -y rtklib"
    exit 1
fi

# Check for IGS products if PPP requested
SP3_FILE="$PRODUCTS_DIR/igs${CLEAN_DATE}.sp3"
CLK_FILE="$PRODUCTS_DIR/igs${CLEAN_DATE}.clk"

if [[ "$USE_PPP" = "true" ]]; then
    if [[ ! -f "$SP3_FILE" ]]; then
        echo ""
        echo "WARNING: SP3 products not found for PPP"
        echo "  Expected: $SP3_FILE"
        echo "  Falling back to broadcast-only mode"
        USE_PPP="false"
    fi
fi

# Station list
STATIONS=("COSO00USA0" "GOLD00USA0" "JPLM00USA0")
PROCESSED=0
FAILED=0
TOTAL_EPOCHS=0

# =========================================================
# PHASE 1: Convert all RTCM to RINEX (all stations)
# =========================================================
echo ""
echo "========== PHASE 1: RTCM -> RINEX Conversion =========="

for STATION in "${STATIONS[@]}"; do
    STATION_RTCM_DIR="$RTCM_DIR/$STATION/$PROCESS_DATE"
    STATION_POS_DIR="$POS_DIR/$STATION/$PROCESS_DATE"

    if [[ ! -d "$STATION_RTCM_DIR" ]]; then
        echo "[$STATION] No RTCM data for $PROCESS_DATE"
        continue
    fi

    echo "[$STATION] Converting RTCM -> RINEX..."
    mkdir -p "$STATION_POS_DIR"

    for RTCM_FILE in "$STATION_RTCM_DIR"/*.rtcm3; do
        if [[ ! -f "$RTCM_FILE" ]]; then
            continue
        fi

        BASENAME=$(basename "$RTCM_FILE" .rtcm3)
        OBS_FILE="$STATION_POS_DIR/${BASENAME}.obs"
        NAV_FILE="$STATION_POS_DIR/${BASENAME}.nav"

        # Skip if already converted
        if [[ -f "$OBS_FILE" ]]; then
            continue
        fi

        FILE_SIZE=$(stat -c%s "$RTCM_FILE" 2>/dev/null || echo "0")
        if [[ "$FILE_SIZE" -lt 10000 ]]; then
            echo "  $BASENAME: skipping (too small: $FILE_SIZE bytes)"
            continue
        fi

        convbin -r rtcm3 -o "$OBS_FILE" -n "$NAV_FILE" "$RTCM_FILE" 2>/dev/null || {
            echo "  $BASENAME: convbin failed"
        }
    done
done

# =========================================================
# PHASE 2: Build GLOBAL merged NAV from ALL stations
# =========================================================
echo ""
echo "========== PHASE 2: Building Global Merged NAV =========="

GLOBAL_NAV="$POS_DIR/global_merged_${PROCESS_DATE}.nav"
echo "Collecting NAV files from all stations..."

# Remove old merged file
rm -f "$GLOBAL_NAV"

# Merge NAV files with proper RINEX header handling
# (first file: full header + body; subsequent: body only after END OF HEADER)
NAV_COUNT=0
FIRST_NAV=1
for STATION in "${STATIONS[@]}"; do
    STATION_POS_DIR="$POS_DIR/$STATION/$PROCESS_DATE"
    if [[ -d "$STATION_POS_DIR" ]]; then
        for NAV_FILE in "$STATION_POS_DIR"/*.nav; do
            if [[ -f "$NAV_FILE" && -s "$NAV_FILE" && "$NAV_FILE" != *"merged"* ]]; then
                if [[ $FIRST_NAV -eq 1 ]]; then
                    # First file: keep full header + body
                    cat "$NAV_FILE" > "$GLOBAL_NAV"
                    FIRST_NAV=0
                else
                    # Subsequent files: append only body (after END OF HEADER)
                    awk 'f{print} /END OF HEADER/{f=1}' "$NAV_FILE" >> "$GLOBAL_NAV"
                fi
                ((NAV_COUNT++)) || true
            fi
        done
    fi
done

if [[ ! -s "$GLOBAL_NAV" ]]; then
    echo "ERROR: No NAV data found from any station!"
    echo "Cannot process observations without ephemeris."
    exit 1
fi

GLOBAL_NAV_SIZE=$(stat -c%s "$GLOBAL_NAV" 2>/dev/null || echo "0")
echo "Global merged NAV: $GLOBAL_NAV_SIZE bytes from $NAV_COUNT files"

# =========================================================
# PHASE 3: Compute positions using global NAV
# =========================================================
echo ""
echo "========== PHASE 3: Computing Positions =========="

for STATION in "${STATIONS[@]}"; do
    STATION_POS_DIR="$POS_DIR/$STATION/$PROCESS_DATE"

    if [[ ! -d "$STATION_POS_DIR" ]]; then
        continue
    fi

    # Check if station has any OBS files
    OBS_COUNT=$(ls "$STATION_POS_DIR"/*.obs 2>/dev/null | wc -l || echo "0")
    if [[ "$OBS_COUNT" -eq 0 ]]; then
        echo "[$STATION] No observations to process"
        continue
    fi

    echo "[$STATION] Processing with global NAV..."

    for OBS_FILE in "$STATION_POS_DIR"/*.obs; do
        if [[ ! -f "$OBS_FILE" ]]; then
            continue
        fi

        # Skip merged files
        [[ "$OBS_FILE" == *"merged"* ]] && continue

        BASENAME=$(basename "$OBS_FILE" .obs)
        POS_FILE="$STATION_POS_DIR/${BASENAME}.pos"

        # Skip if already processed with good results
        if [[ -f "$POS_FILE" ]]; then
            EXISTING_LINES=$(grep -v "^%" "$POS_FILE" 2>/dev/null | wc -l || echo "0")
            if [[ "$EXISTING_LINES" -gt 30 ]]; then
                echo "  $BASENAME: already processed ($EXISTING_LINES epochs)"
                ((TOTAL_EPOCHS += EXISTING_LINES)) || true
                ((PROCESSED++)) || true
                continue
            fi
        fi

        OBS_SIZE=$(stat -c%s "$OBS_FILE" 2>/dev/null || echo "0")
        if [[ "$OBS_SIZE" -lt 1000 ]]; then
            continue
        fi

        # Positioning with multi-GNSS and GLOBAL merged NAV
        # -p 0 = single point (broadcast)
        # -sys GRE = GPS + GLONASS + Galileo
        # -ti 30 = 30-second output interval
        if [[ "$USE_PPP" = "true" && -f "$SP3_FILE" ]]; then
            # PPP mode
            if [[ -f "$CLK_FILE" ]]; then
                rnx2rtkp -p 7 -sys GRE -ti 30 \
                    -o "$POS_FILE" \
                    "$OBS_FILE" "$GLOBAL_NAV" \
                    "$SP3_FILE" "$CLK_FILE" 2>/dev/null || {
                        # Fallback to broadcast
                        rnx2rtkp -p 0 -sys GRE -ti 30 \
                            -o "$POS_FILE" \
                            "$OBS_FILE" "$GLOBAL_NAV" 2>/dev/null || true
                    }
            else
                rnx2rtkp -p 7 -sys GRE -ti 30 \
                    -o "$POS_FILE" \
                    "$OBS_FILE" "$GLOBAL_NAV" \
                    "$SP3_FILE" 2>/dev/null || {
                        rnx2rtkp -p 0 -sys GRE -ti 30 \
                            -o "$POS_FILE" \
                            "$OBS_FILE" "$GLOBAL_NAV" 2>/dev/null || true
                    }
            fi
        else
            # Broadcast-only mode with GLOBAL merged NAV
            rnx2rtkp -p 0 -sys GRE -ti 30 \
                -o "$POS_FILE" \
                "$OBS_FILE" "$GLOBAL_NAV" 2>/dev/null || true
        fi

        if [[ -f "$POS_FILE" ]]; then
            LINES=$(grep -v "^%" "$POS_FILE" 2>/dev/null | wc -l || echo "0")
            if [[ "$LINES" -gt 0 ]]; then
                echo "  $BASENAME: $LINES epochs"
                ((TOTAL_EPOCHS += LINES)) || true
                ((PROCESSED++)) || true
            else
                echo "  $BASENAME: no solutions"
                ((FAILED++)) || true
            fi
        else
            ((FAILED++)) || true
        fi
    done
done

echo ""
echo "=================================================="
echo "  Processing Complete"
echo "  Files processed: $PROCESSED"
echo "  Failed: $FAILED"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Output: $POS_DIR"
echo "=================================================="

# Create processing summary
SUMMARY_FILE="$POS_DIR/processing_summary_${PROCESS_DATE}.json"
cat > "$SUMMARY_FILE" << EOF
{
  "date": "$PROCESS_DATE",
  "processed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mode": "$([ "$USE_PPP" = "true" ] && echo "ppp" || echo "broadcast")",
  "gnss_systems": "GRE",
  "nav_strategy": "merged_daily",
  "files_processed": $PROCESSED,
  "files_failed": $FAILED,
  "total_epochs": $TOTAL_EPOCHS,
  "stations": ["COSO00USA0", "GOLD00USA0", "JPLM00USA0"],
  "products_used": {
    "sp3": $([ -f "$SP3_FILE" ] && echo "true" || echo "false"),
    "clk": $([ -f "$CLK_FILE" ] && echo "true" || echo "false")
  }
}
EOF

echo "Summary: $SUMMARY_FILE"
