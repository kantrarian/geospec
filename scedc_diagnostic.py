
import requests
import time
from datetime import datetime, timedelta
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
except ImportError:
    print("Obspy not installed. Using requests only.")
    Client = None

def check_scedc_requests():
    print("\n--- Testing SCEDC via Requests ---")
    url = "http://service.scedc.caltech.edu/fdsnws/station/1/query"
    params = {
        "network": "CI",
        "station": "WBS",
        "level": "station",
        "format": "text"
    }
    try:
        t0 = time.time()
        print(f"GET {url} ...")
        resp = requests.get(url, params=params, timeout=10)
        dt = time.time() - t0
        print(f"Status: {resp.status_code}")
        print(f"Time: {dt:.2f}s")
        if resp.status_code == 200:
            print("Response preview:")
            print(resp.text[:200])
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def check_scedc_obspy():
    if not Client:
        return
    
    print("\n--- Testing SCEDC via Obspy ---")
    try:
        # SCEDC client
        client = Client("SCEDC")
        t = UTCDateTime.now() - 3600
        print(f"Querying waveforms for CI.WBS..BHZ for last hour...")
        st = client.get_waveforms("CI", "WBS", "*", "BHZ", t, t+10)
        print(f"Success! Got {len(st)} traces.")
        print(st)
    except Exception as e:
        print(f"Obspy query failed: {e}")

if __name__ == "__main__":
    print(f"Time: {datetime.now()}")
    check_scedc_requests()
    check_scedc_obspy()
