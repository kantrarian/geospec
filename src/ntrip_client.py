#!/usr/bin/env python3
"""
ntrip_client.py
Pure Python NTRIP 2.0 Client for IGS-IP Real-Time GNSS Data.

Implements the client side of Networked Transport of RTCM via Internet Protocol.
Used to fetch real-time GNSS observations or corrections for Lambda_geo.
"""

import socket
import base64
import time
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NtripClient:
    """
    NTRIP Client to fetch GNSS data from a Caster.
    """
    
    def __init__(self, caster_url: str, mountpoint: str, user: str = None, password: str = None):
        """
        Args:
            caster_url: e.g., "http://products.igs-ip.net:2101"
            mountpoint: Mountpoint name (e.g., "USHO00USA0")
            user: Username (optional)
            password: Password (optional)
        """
        self.url = caster_url
        self.mountpoint = mountpoint
        self.user = user
        self.password = password
        self.socket = None
        self.connected = False
        
        # Parse URL
        parsed = urlparse(caster_url)
        self.host = parsed.hostname
        self.port = parsed.port if parsed.port else 2101

    def connect(self):
        """Connect to NTRIP Caster."""
        logger.info(f"Connecting to NTRIP Caster {self.host}:{self.port}/{self.mountpoint}...")
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Construct HTTP Request
            # NTRIP 2.0 uses standard HTTP GET
            req = f"GET /{self.mountpoint} HTTP/1.1\r\n"
            req += f"Host: {self.host}\r\n"
            req += "Ntrip-Version: Ntrip/2.0\r\n"
            req += "User-Agent: GeoSpec/1.0 Python\r\n"
            req += "Accept: */*\r\n"
            req += "Connection: close\r\n"
            
            if self.user:
                auth_str = f"{self.user}:{self.password}"
                auth_b64 = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
                req += f"Authorization: Basic {auth_b64}\r\n"
                
            req += "\r\n"
            
            # Send Request
            self.socket.sendall(req.encode('ascii'))
            
            # Read Response Header
            response = b""
            while b"\r\n\r\n" not in response:
                chunk = self.socket.recv(1024)
                if not chunk:
                    raise ConnectionError("Connection closed by caster during handshake")
                response += chunk
                
            header, leftover = response.split(b"\r\n\r\n", 1)
            header_str = header.decode('ascii', errors='ignore')
            
            if "HTTP/1.1 200 OK" in header_str or "ICY 200 OK" in header_str:
                logger.info("Connected to Stream!")
                self.connected = True
                return leftover
            else:
                logger.error(f"NTRIP Handshake Failed:\n{header_str}")
                self.socket.close()
                self.socket = None
                return None
                
        except Exception as e:
            logger.error(f"Connection Error: {e}")
            if self.socket:
                self.socket.close()
            return None

    def read_stream(self, buffer_size=4096):
        """Generator to yield data chunks."""
        if not self.connected:
            return
            
        try:
            while True:
                data = self.socket.recv(buffer_size)
                if not data:
                    break
                yield data
        except Exception as e:
            logger.error(f"Stream Read Error: {e}")
        finally:
            self.close()

    def close(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
        self.connected = False
        logger.info("NTRIP Connection Closed")

if __name__ == "__main__":
    # Test with a public caster if possible, or just mock
    # IGS-IP usually requires registration.
    # RTK2GO is a free one often used for testing.
    # "http://rtk2go.com:2101", mountpoint usually needs to be known.
    
    print("NTRIP Client Module")
    print("Usage: client = NtripClient('http://host:port', 'MOUNT')")
