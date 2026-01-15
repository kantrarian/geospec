#!/usr/bin/env python3
"""
international_connectivity_test.py
Verify access to international seismic networks (Turkey, Japan, GEOFON).
"""

from seismic_waveform_fetcher import SeismicWaveformFetcher
from obspy import UTCDateTime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_international_connectivity():
    fetcher = SeismicWaveformFetcher()
    t_end = UTCDateTime.now() - 3600
    t_start = t_end - 600
    
    # 1. Turkey (TU network) - KOERI datacenter
    print("\n--- Testing Turkey (TU/KOERI) ---")
    # ISP is Isparta, a long-running station
    st = fetcher.get_waveforms("TU", "ISP", "*", "BHZ", t_start, t_end)
    if st:
        print("[SUCCESS] Turkey (TU.ISP):")
        print(st)
    else:
        print("[FAIL] Turkey (TU.ISP) - Check KOERI/IRIS")

    # 2. GEOFON (GE network) - GFZ datacenter
    print("\n--- Testing GEOFON (GE/GFZ) ---")
    # MORC is in Czech Republic, specialized GE station
    # Or try a Turkish one if GE covers it? 
    # Let's try 'WLF' (Walferdange) or 'EIL' (Eilat)
    st = fetcher.get_waveforms("GE", "WLF", "*", "BHZ", t_start, t_end)
    if st:
        print("[SUCCESS] GEOFON (GE.WLF):")
        print(st)
    else:
        print("[FAIL] GEOFON (GE.WLF) - Check GFZ/IRIS")

    # 3. Japan (JP/F-net)
    print("\n--- Testing Japan (JP/JMA/F-net) ---")
    # JMA network is 'JP'. F-net is often 'BO' (Bosai-Ken).
    # Checking JP (JMA) via IRIS is standard.
    # checking F-net via NIED might require special client, 
    # but let's try 'JP' first (JMA)
    st = fetcher.get_waveforms("JP", "JNU", "*", "BHZ", t_start, t_end)
    if st:
        print("[SUCCESS] Japan (JP.JNU - Nakatsue):")
        print(st)
    else:
        print("[FAIL] Japan (JP.JNU) - Check IRIS")
    
    # Try global station in Japan (IU.MAJO - Matsushiro)
    print("\n--- Testing Global Station in Japan (IU.MAJO) ---")
    st = fetcher.get_waveforms("IU", "MAJO", "00", "BHZ", t_start, t_end)
    if st:
        print("[SUCCESS] Japan (IU.MAJO):")
        print(st)
    else:
        print("[FAIL] Japan (IU.MAJO)")

if __name__ == "__main__":
    test_international_connectivity()
