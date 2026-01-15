#!/usr/bin/env python3
"""
capture_rtcm.py
Stream RTCM from an NTRIP caster and write rotating .rtcm3 files.

- Credentials are read from env vars (IGS_NTRIP_USER / IGS_NTRIP_PASSWORD)
- Never prints credentials
- Auto-reconnects
- Rotates output files hourly

Usage:
  python capture_rtcm.py --caster igs-ip.net --port 2101 --mount COSO00USA0 --out monitoring/data/rtcm --hours 24
"""

import argparse
import base64
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


def safe_print(msg: str) -> None:
    """Print safely on Windows (avoid Unicode console errors)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def ntrip_connect(caster: str, port: int, mount: str, user: str, password: str, timeout_s: int = 30) -> socket.socket:
    """Connect to NTRIP caster and return socket streaming RTCM."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    sock.connect((caster, port))

    auth = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    # NTRIP v2 style request
    req = (
        f"GET /{mount} HTTP/1.1\r\n"
        f"Host: {caster}\r\n"
        f"Ntrip-Version: Ntrip/2.0\r\n"
        f"User-Agent: GeoSpec-NTRIP/1.0\r\n"
        f"Authorization: Basic {auth}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    sock.sendall(req.encode("ascii"))

    # Read header
    header = b""
    while b"\r\n\r\n" not in header and len(header) < 8192:
        chunk = sock.recv(1024)
        if not chunk:
            break
        header += chunk

    header_text = header.decode("latin-1", errors="replace")
    if "401" in header_text or "Unauthorized" in header_text:
        sock.close()
        raise PermissionError("NTRIP authentication failed (401).")

    if "200" not in header_text:
        sock.close()
        raise RuntimeError(f"Unexpected NTRIP response: {header_text.splitlines()[:5]}")

    return sock


def rotate_path(base_dir: Path, mount: str, now: datetime) -> Path:
    """Generate hourly-rotated output path."""
    day = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")
    d = base_dir / mount / day
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mount}_{day}_{hour}00Z.rtcm3"


def stream_rtcm(args: argparse.Namespace) -> int:
    """Main streaming loop."""
    user = os.getenv("IGS_NTRIP_USER")
    password = os.getenv("IGS_NTRIP_PASSWORD")
    if not user or not password:
        safe_print("ERROR: Set IGS_NTRIP_USER and IGS_NTRIP_PASSWORD in .env (never hardcode).")
        return 2

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    end_time = time.time() + args.hours * 3600
    bytes_written = 0
    reconnects = 0
    disconnect_count = 0

    safe_print(f"Starting RTCM capture: caster={args.caster}:{args.port} mount={args.mount} hours={args.hours}")
    safe_print(f"Output dir: {out_dir}")

    while time.time() < end_time:
        sock = None
        current_file = None
        try:
            sock = ntrip_connect(args.caster, args.port, args.mount, user, password)
            reconnects += 1
            safe_print(f"[{datetime.now(timezone.utc).isoformat()}] Connected (session #{reconnects})")

            current_path = None
            next_rotate_key = None

            while time.time() < end_time:
                now = datetime.now(timezone.utc)
                rotate_key = (now.strftime("%Y-%m-%d"), now.strftime("%H"))

                if rotate_key != next_rotate_key:
                    if current_file:
                        current_file.flush()
                        current_file.close()
                    current_path = rotate_path(out_dir, args.mount, now)
                    current_file = open(current_path, "ab")
                    next_rotate_key = rotate_key
                    safe_print(f"Writing to: {current_path}")

                data = sock.recv(4096)
                if not data:
                    raise ConnectionError("Socket closed by caster")

                current_file.write(data)
                bytes_written += len(data)

        except (socket.timeout, ConnectionError, OSError) as e:
            disconnect_count += 1
            safe_print(f"[{datetime.now(timezone.utc).isoformat()}] Stream error: {type(e).__name__}: {e}")
            safe_print(f"  Disconnects: {disconnect_count}, Reconnecting in {args.retry_s}s...")
            time.sleep(args.retry_s)
        except PermissionError as e:
            safe_print(f"AUTH ERROR: {e}")
            return 3
        except KeyboardInterrupt:
            safe_print("\nInterrupted by user.")
            break
        except Exception as e:
            safe_print(f"FATAL: {type(e).__name__}: {e}")
            return 4
        finally:
            if current_file:
                try:
                    current_file.close()
                except Exception:
                    pass
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    safe_print(f"\nCapture complete.")
    safe_print(f"  Total bytes: {bytes_written:,}")
    safe_print(f"  Sessions: {reconnects}")
    safe_print(f"  Disconnects: {disconnect_count}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Capture RTCM from NTRIP caster")
    p.add_argument("--caster", default="igs-ip.net", help="NTRIP caster hostname")
    p.add_argument("--port", type=int, default=2101, help="NTRIP caster port")
    p.add_argument("--mount", required=True, help="Mountpoint name (e.g., COSO00USA0)")
    p.add_argument("--out", default="monitoring/data/rtcm", help="Output directory")
    p.add_argument("--hours", type=float, default=24, help="Hours to capture")
    p.add_argument("--retry-s", type=int, default=10, help="Seconds between reconnect attempts")
    args = p.parse_args()
    raise SystemExit(stream_rtcm(args))


if __name__ == "__main__":
    main()
