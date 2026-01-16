#!/bin/bash
# setup_rtklib.sh
# Run this script in WSL Ubuntu to install RTKLIB and configure processing
#
# Usage:
#   In WSL terminal: bash /mnt/c/GeoSpec/geospec_sprint/wsl/setup_rtklib.sh

set -e

echo "=================================================="
echo "  GeoSpec RTKLIB Setup for WSL Ubuntu"
echo "=================================================="

# Update package lists
echo "[1/4] Updating package lists..."
sudo apt-get update

# Install RTKLIB
echo "[2/4] Installing RTKLIB..."
sudo apt-get install -y rtklib

# Verify installation
echo "[3/4] Verifying installation..."
which rnx2rtkp && echo "  rnx2rtkp (RINEX to position): OK" || echo "  rnx2rtkp: MISSING"
which convbin && echo "  convbin (format converter): OK" || echo "  convbin: MISSING"
which str2str && echo "  str2str (stream server): OK" || echo "  str2str: MISSING"

# Create directories
echo "[4/4] Creating processing directories..."
GEOSPEC_DIR="/mnt/c/GeoSpec/geospec_sprint"
mkdir -p "$GEOSPEC_DIR/monitoring/data/positions"
mkdir -p "$GEOSPEC_DIR/monitoring/data/products"

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "RTKLIB version:"
rnx2rtkp -h 2>&1 | head -3 || echo "  (run rnx2rtkp -h for version)"
echo ""
echo "Next steps:"
echo "  1. Download IGS products (orbits, clocks) for PPP"
echo "  2. Run process_rtcm.sh to convert RTCM -> positions"
echo ""
