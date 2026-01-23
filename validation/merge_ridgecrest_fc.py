"""
Merge ridgecrest_2019_fc.json into all_historical_fc.json

This script:
1. Loads the all_historical_fc.json file
2. Loads the ridgecrest_2019_fc.json file
3. Adds ridgecrest_2019 data to all_historical_fc.json using the event_key as the dictionary key
4. Saves the updated all_historical_fc.json
"""

import json
from pathlib import Path


def merge_ridgecrest_fc():
    """Merge ridgecrest_2019 FC data into the all_historical_fc.json file."""

    # Define file paths
    base_dir = Path(r"C:\GeoSpec\geospec_sprint\validation\results\fc_historical")
    all_historical_path = base_dir / "all_historical_fc.json"
    ridgecrest_path = base_dir / "ridgecrest_2019_fc.json"

    # Load all_historical_fc.json
    print(f"Loading {all_historical_path}...")
    with open(all_historical_path, "r", encoding="utf-8") as f:
        all_historical = json.load(f)

    print(f"  Found {len(all_historical)} existing events: {list(all_historical.keys())}")

    # Load ridgecrest_2019_fc.json
    print(f"Loading {ridgecrest_path}...")
    with open(ridgecrest_path, "r", encoding="utf-8") as f:
        ridgecrest_data = json.load(f)

    # Get the event key from the ridgecrest data
    event_key = ridgecrest_data.get("event_key", "ridgecrest_2019")
    print(f"  Event key: {event_key}")
    print(f"  Event name: {ridgecrest_data.get('event', {}).get('name')}")
    print(f"  Magnitude: {ridgecrest_data.get('event', {}).get('magnitude')}")

    # Check if ridgecrest_2019 already exists
    if event_key in all_historical:
        print(f"\nWARNING: {event_key} already exists in all_historical_fc.json!")
        print("  Overwriting existing entry...")

    # Add ridgecrest_2019 to all_historical
    all_historical[event_key] = ridgecrest_data

    print(f"\nAdded {event_key} to all_historical_fc.json")
    print(f"  Total events now: {len(all_historical)}")
    print(f"  Event keys: {list(all_historical.keys())}")

    # Save updated all_historical_fc.json
    print(f"\nSaving updated {all_historical_path}...")
    with open(all_historical_path, "w", encoding="utf-8") as f:
        json.dump(all_historical, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    merge_ridgecrest_fc()
