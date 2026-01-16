
import requests

def get_coords(stations):
    url = "https://geodesy.unr.edu/NGLStationPages/llh.out"
    print(f"Fetching {url}...")
    try:
        response = requests.get(url, verify=False, timeout=30)
        if response.status_code != 200:
            print(f"Failed to fetch: {response.status_code}")
            return

        lines = response.text.splitlines()
        found = {}
        
        for line in lines:
            parts = line.split()
            if not parts: continue
            name = parts[0].strip()
            if name in stations:
                # Format: SITE  LAT  LON  H
                lat = float(parts[1])
                lon = float(parts[2])
                found[name] = (lat, lon)
        
        for name in stations:
            if name in found:
                print(f"{name}: {found[name]}")
            else:
                print(f"{name}: Not found")

    except Exception as e:
        print(f"Error: {e}")

stations = ["KAIK", "HANM", "CULV", "WARD", "CMBL", "MQZG", "WHSB"] # Added MQZG, WHSB as potentials
get_coords(stations)
