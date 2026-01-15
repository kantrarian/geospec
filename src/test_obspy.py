#!/usr/bin/env python3
"""
test_obspy.py
Test script to verify ObsPy installation and IRIS data access.

This script fetches seismic data from IRIS for the 2019 Ridgecrest foreshock
period to verify our data pipeline is working.

Author: R.J. Mathews
Date: January 2026
"""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import sys

def test_iris_connection():
    """Test basic IRIS connection."""
    print("=" * 60)
    print("ObsPy IRIS Connection Test")
    print("=" * 60)

    try:
        client = Client("IRIS")
        print("[OK] Successfully connected to IRIS")
        return client
    except Exception as e:
        print(f"[FAIL] Could not connect to IRIS: {e}")
        return None

def test_waveform_retrieval(client):
    """Test retrieving waveforms from IRIS."""
    print("\n" + "-" * 60)
    print("Testing waveform retrieval...")
    print("-" * 60)

    # Use IU.ANMO (Global Seismographic Network) - reliable on IRIS
    # CI network data is on SCEDC, not IRIS
    # Request 1 hour of data from the 2019 Ridgecrest M6.4 foreshock period
    try:
        st = client.get_waveforms(
            network="IU",
            station="ANMO",
            location="00",
            channel="BHZ",  # Broadband Horizontal Z (vertical)
            starttime=UTCDateTime("2019-07-04T00:00:00"),
            endtime=UTCDateTime("2019-07-04T01:00:00")
        )

        print(f"[OK] Retrieved {len(st)} trace(s) from IU.ANMO")
        for tr in st:
            print(f"     {tr.id}: {tr.stats.npts} samples @ {tr.stats.sampling_rate} Hz")
            print(f"     Start: {tr.stats.starttime}")
            print(f"     End:   {tr.stats.endtime}")

        # Note about CI network
        print("\n     NOTE: CI network data (Ridgecrest) is on SCEDC, not IRIS")
        print("     For California stations, use Client('SCEDC') or AWS S3")

        return st
    except Exception as e:
        print(f"[FAIL] Could not retrieve waveforms: {e}")
        return None

def test_station_inventory(client):
    """Test retrieving station inventory for SoCal region."""
    print("\n" + "-" * 60)
    print("Testing station inventory retrieval...")
    print("-" * 60)

    # Get CI network stations near Ridgecrest
    try:
        inventory = client.get_stations(
            network="CI",
            station="*",
            location="*",
            channel="BHZ",
            minlatitude=35.0,
            maxlatitude=36.5,
            minlongitude=-118.5,
            maxlongitude=-117.0,
            level="station"
        )

        station_count = sum(len(net) for net in inventory)
        print(f"[OK] Found {station_count} stations in Ridgecrest area")

        # Print first few stations
        for net in inventory[:1]:
            for sta in net[:5]:
                print(f"     {net.code}.{sta.code}: {sta.latitude:.3f}, {sta.longitude:.3f}")

        if station_count > 5:
            print(f"     ... and {station_count - 5} more")

        return inventory
    except Exception as e:
        print(f"[FAIL] Could not retrieve station inventory: {e}")
        return None

def test_ridgecrest_stations(client):
    """Test specific stations we plan to use for Ridgecrest monitoring."""
    print("\n" + "-" * 60)
    print("Testing Ridgecrest target stations...")
    print("-" * 60)

    # Key stations near Ridgecrest for fault correlation
    target_stations = ['WBS', 'SLA', 'CLC', 'LRL', 'TOW2']

    available = []
    for sta in target_stations:
        try:
            inv = client.get_stations(
                network="CI",
                station=sta,
                channel="BH*",
                level="channel"
            )
            if inv:
                available.append(sta)
                print(f"[OK] {sta}: Available")
        except Exception as e:
            print(f"[--] {sta}: Not available ({e})")

    print(f"\nAvailable stations: {len(available)}/{len(target_stations)}")
    return available

def test_event_catalog(client):
    """Test retrieving earthquake catalog around Ridgecrest."""
    print("\n" + "-" * 60)
    print("Testing earthquake catalog access...")
    print("-" * 60)

    try:
        # Get Ridgecrest sequence events (M >= 5)
        catalog = client.get_events(
            starttime=UTCDateTime("2019-07-01"),
            endtime=UTCDateTime("2019-07-10"),
            minlatitude=35.0,
            maxlatitude=36.5,
            minlongitude=-118.5,
            maxlongitude=-117.0,
            minmagnitude=5.0
        )

        print(f"[OK] Found {len(catalog)} events (M >= 5.0)")

        for event in catalog[:5]:
            origin = event.preferred_origin()
            mag = event.preferred_magnitude()
            print(f"     M{mag.mag:.1f} {origin.time.datetime.strftime('%Y-%m-%d %H:%M:%S')}")

        return catalog
    except Exception as e:
        print(f"[FAIL] Could not retrieve catalog: {e}")
        return None

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GeoSpec Seismic Data Access Verification")
    print("=" * 60)
    print("Testing ObsPy installation and IRIS FDSN web service access\n")

    # Test 1: Connection
    client = test_iris_connection()
    if not client:
        print("\n[ABORT] Cannot proceed without IRIS connection")
        sys.exit(1)

    # Test 2: Waveforms
    waveforms = test_waveform_retrieval(client)

    # Test 3: Station inventory
    inventory = test_station_inventory(client)

    # Test 4: Target stations
    stations = test_ridgecrest_stations(client)

    # Test 5: Event catalog
    catalog = test_event_catalog(client)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"IRIS Connection:     {'PASS' if client else 'FAIL'}")
    print(f"Waveform Retrieval:  {'PASS' if waveforms else 'FAIL'}")
    print(f"Station Inventory:   {'PASS' if inventory else 'FAIL'}")
    print(f"Target Stations:     {len(stations) if stations else 0}/5 available")
    print(f"Event Catalog:       {'PASS' if catalog else 'FAIL'}")
    print("=" * 60)

    if waveforms and inventory:
        print("\nSUCCESS: ObsPy is ready for seismic data integration!")
        return 0
    else:
        print("\nWARNING: Some tests failed. Check network and retry.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
