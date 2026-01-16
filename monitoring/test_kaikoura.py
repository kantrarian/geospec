
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from gps_data_acquisition import NGLDataDownloader, KAIKOURA_STATIONS
from seismic_waveform_fetcher import SeismicWaveformFetcher

def test_kaikoura():
    print("=== TESTING KAIKOURA (NZ) IMPLEMENTATION ===")
    
    # 1. Test Seismic (GeoNet)
    print("\n[SEISMIC] Testing GeoNet fetch for NZ.HSES (Hanmer Springs)...")
    fetcher = SeismicWaveformFetcher()
    t_start = datetime(2016, 11, 13, 11, 0, 0) # Event was Nov 13 11:02 UTC
    t_end = datetime(2016, 11, 13, 11, 10, 0)
    
    try:
        st = fetcher.get_waveforms("NZ", "HSES", "10", "HHZ", t_start, t_end)
        if st and len(st) > 0:
            print(f"SUCCESS: Fetched {len(st)} traces from GeoNet")
            print(st)
        else:
            print("FAILURE: No traces returned for NZ.HSES")
            # Try 00 location code or *
            print("Retrying with wildcard location...")
            st = fetcher.get_waveforms("NZ", "HSES", "*", "HHZ", t_start, t_end)
            if st: print(f"SUCCESS (Wildcard): {st}")
            
    except Exception as e:
        print(f"ERROR fetching seismic: {e}")

    # 2. Test GPS (NGL)
    print("\n[GPS] Testing NGL download for Kaikoura 2016...")
    data_dir = project_root / "monitoring" / "data"
    downloader = NGLDataDownloader(data_dir / "cache" / "ngl")
    
    eq_config = {
        "name": "2016 Kaikoura M7.8",
        "date": "2016-11-13T11:02:56",
        "lat": -42.757,
        "lon": 173.077,
        "magnitude": 7.8,
        "data_window_days": 10
    }
    
    stations, _ = downloader.download_earthquake_data('kaikoura_2016', data_dir / "backtest", eq_config)
    
    if stations:
        print(f"\nSUCCESS: Downloaded {len(stations)} GPS stations")
        for code in stations:
             print(f"  - {code}: {len(stations[code])} days")
    else:
        print("FAILURE: No GPS stations downloaded")

if __name__ == "__main__":
    test_kaikoura()
