#!/usr/bin/env python3
"""
test_ntrip_connection.py
Test connectivity to IGS-IP NTRIP caster and list available mountpoints.

NTRIP (Networked Transport of RTCM via Internet Protocol) provides
real-time GNSS corrections that can reduce Lambda_geo latency from
2-14 days to seconds.

Usage:
    python test_ntrip_connection.py
"""

import os
import sys
import socket
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


def get_ntrip_sourcetable(caster: str, port: int, user: str, password: str) -> str:
    """
    Fetch the NTRIP sourcetable (list of available mountpoints).

    The sourcetable lists all available GNSS streams including:
    - Station coordinates
    - Data format (RTCM 3.x, etc.)
    - Update rate
    - Network/carrier
    """
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)

    try:
        # Connect to caster
        print(f"Connecting to {caster}:{port}...")
        sock.connect((caster, port))

        # Build NTRIP request (HTTP-like)
        auth = base64.b64encode(f"{user}:{password}".encode()).decode()
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {caster}\r\n"
            f"Ntrip-Version: Ntrip/2.0\r\n"
            f"User-Agent: NTRIP GeoSpec/1.0\r\n"
            f"Authorization: Basic {auth}\r\n"
            f"\r\n"
        )

        # Send request
        sock.send(request.encode())

        # Receive response
        response = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                # Sourcetable ends with ENDSOURCETABLE
                if b"ENDSOURCETABLE" in response:
                    break
            except socket.timeout:
                break

        return response.decode('utf-8', errors='replace')

    finally:
        sock.close()


def parse_sourcetable(sourcetable: str) -> dict:
    """Parse NTRIP sourcetable into structured data."""
    result = {
        'casters': [],
        'networks': [],
        'streams': []
    }

    for line in sourcetable.split('\n'):
        line = line.strip()
        if line.startswith('CAS;'):
            # Caster info
            parts = line.split(';')
            if len(parts) >= 3:
                result['casters'].append({
                    'host': parts[1],
                    'port': parts[2] if len(parts) > 2 else '',
                    'identifier': parts[3] if len(parts) > 3 else ''
                })
        elif line.startswith('NET;'):
            # Network info
            parts = line.split(';')
            if len(parts) >= 2:
                result['networks'].append({
                    'name': parts[1],
                    'operator': parts[3] if len(parts) > 3 else ''
                })
        elif line.startswith('STR;'):
            # Stream (mountpoint) info
            parts = line.split(';')
            if len(parts) >= 10:
                result['streams'].append({
                    'mountpoint': parts[1],
                    'identifier': parts[2],
                    'format': parts[3],
                    'format_details': parts[4],
                    'carrier': parts[5],
                    'nav_system': parts[6],
                    'network': parts[7],
                    'country': parts[8],
                    'latitude': parts[9],
                    'longitude': parts[10] if len(parts) > 10 else '',
                    'nmea': parts[11] if len(parts) > 11 else '',
                    'solution': parts[12] if len(parts) > 12 else '',
                })

    return result


def main():
    """Test NTRIP connection and display available streams."""
    # Get credentials from environment
    user = os.getenv('IGS_NTRIP_USER')
    password = os.getenv('IGS_NTRIP_PASSWORD')
    caster = os.getenv('IGS_NTRIP_CASTER', 'igs-ip.net')
    port = int(os.getenv('IGS_NTRIP_PORT', '2101'))

    if not user or not password:
        print("ERROR: IGS_NTRIP_USER and IGS_NTRIP_PASSWORD must be set in .env")
        sys.exit(1)

    print("=" * 60)
    print("  IGS-IP NTRIP Connection Test")
    print("=" * 60)
    print(f"  Caster: {caster}:{port}")
    print(f"  User: {user}")
    print("=" * 60)
    print()

    # Test all three casters
    casters = [
        ('igs-ip.net', 2101),
        ('products.igs-ip.net', 2101),
        ('euref-ip.net', 2101),
    ]

    for caster_host, caster_port in casters:
        print(f"\n{'='*60}")
        print(f"Testing: {caster_host}:{caster_port}")
        print('='*60)

        try:
            sourcetable = get_ntrip_sourcetable(caster_host, caster_port, user, password)

            # Check for errors
            if 'ICY 401' in sourcetable or 'Unauthorized' in sourcetable:
                print("  ERROR: Authentication failed")
                continue
            elif '200 OK' in sourcetable:
                print("  SUCCESS: Connected and authenticated")
            else:
                print(f"  Response: {sourcetable[:200]}...")
                continue

            # Parse sourcetable
            parsed = parse_sourcetable(sourcetable)

            print(f"\n  Networks: {len(parsed['networks'])}")
            for net in parsed['networks'][:5]:
                print(f"    - {net['name']}: {net.get('operator', '')}")

            print(f"\n  Streams (mountpoints): {len(parsed['streams'])}")

            # Filter for relevant streams (GPS, GLO, GAL)
            relevant = [s for s in parsed['streams']
                       if any(x in s.get('nav_system', '').upper()
                             for x in ['GPS', 'GLO', 'GAL', 'GNSS'])]

            print(f"  GNSS streams: {len(relevant)}")
            print("\n  Sample mountpoints:")
            for stream in relevant[:10]:
                print(f"    {stream['mountpoint']:12} | {stream['format']:8} | "
                      f"{stream['nav_system']:10} | {stream.get('country', '')}")

            if len(relevant) > 10:
                print(f"    ... and {len(relevant) - 10} more")

        except socket.timeout:
            print(f"  ERROR: Connection timed out")
        except socket.gaierror as e:
            print(f"  ERROR: DNS lookup failed: {e}")
        except ConnectionRefusedError:
            print(f"  ERROR: Connection refused")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("  Connection test complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
