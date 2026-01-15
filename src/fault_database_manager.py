#!/usr/bin/env python3
"""
Fault Database Manager
=====================

Comprehensive fault database management system for cataloging and managing
fault systems with seismic history. This system provides the foundation
for fault movement prediction using the monodromy principle.

Author: R.J. Mathews
Email: mail.rjmathews@gmail.com
Date: 2025
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import hashlib
import uuid

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FaultSystem:
    """Fault system data structure."""
    fault_id: str
    name: str
    location: Tuple[float, float, float]  # (latitude, longitude, depth)
    fault_type: str  # 'strike-slip', 'reverse', 'normal', 'oblique'
    length_km: float
    width_km: float
    dip_degrees: float
    strike_degrees: float
    last_major_event: Optional[datetime]
    seismic_history_count: int
    data_sources: List[str]
    quality_score: float
    created_at: datetime
    updated_at: datetime

@dataclass
class SeismicEvent:
    """Seismic event data structure."""
    event_id: str
    fault_id: str
    time: datetime
    magnitude: float
    latitude: float
    longitude: float
    depth: float
    event_type: str  # 'main_event', 'aftershock', 'foreshock', 'swarm'
    source: str  # 'USGS', 'EMSC', 'local_network', etc.
    quality_score: float
    created_at: datetime

@dataclass
class FaultNetwork:
    """Fault network data structure."""
    network_id: str
    name: str
    region: str
    fault_systems: List[str]  # List of fault IDs
    network_type: str  # 'regional', 'local', 'global'
    total_length_km: float
    total_events: int
    last_activity: Optional[datetime]
    created_at: datetime

# ============================================================================
# Fault Database Manager
# ============================================================================

class FaultDatabaseManager:
    """
    Comprehensive fault database management system.
    
    This system provides cataloging, data management, and quality control
    for fault systems with seismic history.
    """
    
    def __init__(self, database_path: str = "fault_database.db"):
        """
        Initialize fault database manager.
        
        Args:
            database_path: Path to SQLite database for storing fault data
        """
        self.database_path = database_path
        self._init_database()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.max_faults = 10000
        self.max_events_per_fault = 50000
        self.data_retention_days = 365 * 10  # 10 years
        
        # Quality thresholds
        self.min_quality_score = 0.5
        self.min_events_for_analysis = 10
        self.max_magnitude_gap = 2.0  # Maximum gap in magnitude sequence
    
    def _init_database(self):
        """Initialize fault database with all required tables."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create fault systems table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fault_systems (
                    fault_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    depth REAL NOT NULL,
                    fault_type TEXT NOT NULL,
                    length_km REAL NOT NULL,
                    width_km REAL NOT NULL,
                    dip_degrees REAL NOT NULL,
                    strike_degrees REAL NOT NULL,
                    last_major_event TEXT,
                    seismic_history_count INTEGER DEFAULT 0,
                    data_sources_json TEXT,
                    quality_score REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Create seismic events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS seismic_events (
                    event_id TEXT PRIMARY KEY,
                    fault_id TEXT NOT NULL,
                    time TEXT NOT NULL,
                    magnitude REAL NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    depth REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fault_id) REFERENCES fault_systems (fault_id)
                )
            ''')
            
            # Create fault networks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fault_networks (
                    network_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    region TEXT NOT NULL,
                    fault_systems_json TEXT NOT NULL,
                    network_type TEXT NOT NULL,
                    total_length_km REAL DEFAULT 0.0,
                    total_events INTEGER DEFAULT 0,
                    last_activity TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create data sources table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    url TEXT,
                    api_key TEXT,
                    last_update TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    metric_id TEXT PRIMARY KEY,
                    fault_id TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (fault_id) REFERENCES fault_systems (fault_id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fault_location ON fault_systems (latitude, longitude)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fault_type ON fault_systems (fault_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_fault_id ON seismic_events (fault_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON seismic_events (time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_magnitude ON seismic_events (magnitude)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_source ON seismic_events (source)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing fault database: {e}")
            raise
    
    def add_fault_system(self, fault_data: Dict) -> str:
        """
        Add a new fault system to the database.
        
        Args:
            fault_data: Fault system data
        
        Returns:
            Fault ID
        """
        try:
            # Generate fault ID
            fault_id = self._generate_fault_id(fault_data)
            
            # Validate fault data
            self._validate_fault_data(fault_data)
            
            # Calculate quality score
            quality_score = self._calculate_fault_quality_score(fault_data)
            
            # Prepare data for insertion
            now = datetime.utcnow()
            fault_system = {
                'fault_id': fault_id,
                'name': fault_data.get('name', f'Fault_{fault_id}'),
                'latitude': fault_data['latitude'],
                'longitude': fault_data['longitude'],
                'depth': fault_data.get('depth', 0.0),
                'fault_type': fault_data.get('fault_type', 'strike-slip'),
                'length_km': fault_data.get('length_km', 10.0),
                'width_km': fault_data.get('width_km', 5.0),
                'dip_degrees': fault_data.get('dip_degrees', 90.0),
                'strike_degrees': fault_data.get('strike_degrees', 0.0),
                'last_major_event': fault_data.get('last_major_event'),
                'seismic_history_count': 0,
                'data_sources_json': json.dumps(fault_data.get('data_sources', [])),
                'quality_score': quality_score,
                'created_at': now.isoformat(),
                'updated_at': now.isoformat()
            }
            
            # Insert into database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fault_systems (
                    fault_id, name, latitude, longitude, depth, fault_type,
                    length_km, width_km, dip_degrees, strike_degrees,
                    last_major_event, seismic_history_count, data_sources_json,
                    quality_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fault_system['fault_id'],
                fault_system['name'],
                fault_system['latitude'],
                fault_system['longitude'],
                fault_system['depth'],
                fault_system['fault_type'],
                fault_system['length_km'],
                fault_system['width_km'],
                fault_system['dip_degrees'],
                fault_system['strike_degrees'],
                fault_system['last_major_event'],
                fault_system['seismic_history_count'],
                fault_system['data_sources_json'],
                fault_system['quality_score'],
                fault_system['created_at'],
                fault_system['updated_at']
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added fault system: {fault_system['name']} ({fault_id})")
            return fault_id
            
        except Exception as e:
            self.logger.error(f"Error adding fault system: {e}")
            raise
    
    def add_seismic_event(self, event_data: Dict, fault_id: str) -> str:
        """
        Add a seismic event to a fault system.
        
        Args:
            event_data: Seismic event data
            fault_id: Target fault system ID
        
        Returns:
            Event ID
        """
        try:
            # Generate event ID
            event_id = self._generate_event_id(event_data, fault_id)
            
            # Validate event data
            self._validate_event_data(event_data)
            
            # Calculate quality score
            quality_score = self._calculate_event_quality_score(event_data)
            
            # Prepare data for insertion
            now = datetime.utcnow()
            seismic_event = {
                'event_id': event_id,
                'fault_id': fault_id,
                'time': event_data['time'],
                'magnitude': event_data['magnitude'],
                'latitude': event_data['latitude'],
                'longitude': event_data['longitude'],
                'depth': event_data['depth'],
                'event_type': event_data.get('event_type', 'earthquake'),
                'source': event_data.get('source', 'unknown'),
                'quality_score': quality_score,
                'created_at': now.isoformat()
            }
            
            # Insert into database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO seismic_events (
                    event_id, fault_id, time, magnitude, latitude, longitude,
                    depth, event_type, source, quality_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                seismic_event['event_id'],
                seismic_event['fault_id'],
                seismic_event['time'],
                seismic_event['magnitude'],
                seismic_event['latitude'],
                seismic_event['longitude'],
                seismic_event['depth'],
                seismic_event['event_type'],
                seismic_event['source'],
                seismic_event['quality_score'],
                seismic_event['created_at']
            ))
            
            # Update fault system statistics
            cursor.execute('''
                UPDATE fault_systems 
                SET seismic_history_count = seismic_history_count + 1,
                    updated_at = ?
                WHERE fault_id = ?
            ''', (now.isoformat(), fault_id))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added seismic event: M{event_data['magnitude']:.1f} to fault {fault_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Error adding seismic event: {e}")
            raise
    
    def get_fault_system(self, fault_id: str) -> Optional[FaultSystem]:
        """
        Get fault system by ID.
        
        Args:
            fault_id: Fault system ID
        
        Returns:
            Fault system data or None
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM fault_systems WHERE fault_id = ?
            ''', (fault_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse data sources
            data_sources = json.loads(row[12]) if row[12] else []
            
            # Parse last major event
            last_major_event = None
            if row[10]:
                last_major_event = datetime.fromisoformat(row[10])
            
            fault_system = FaultSystem(
                fault_id=row[0],
                name=row[1],
                location=(row[2], row[3], row[4]),
                fault_type=row[5],
                length_km=row[6],
                width_km=row[7],
                dip_degrees=row[8],
                strike_degrees=row[9],
                last_major_event=last_major_event,
                seismic_history_count=row[11],
                data_sources=data_sources,
                quality_score=row[13],
                created_at=datetime.fromisoformat(row[14]),
                updated_at=datetime.fromisoformat(row[15])
            )
            
            conn.close()
            return fault_system
            
        except Exception as e:
            self.logger.error(f"Error getting fault system {fault_id}: {e}")
            return None
    
    def get_seismic_events(self, fault_id: str, limit: int = 1000) -> List[SeismicEvent]:
        """
        Get seismic events for a fault system.
        
        Args:
            fault_id: Fault system ID
            limit: Maximum number of events to return
        
        Returns:
            List of seismic events
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM seismic_events 
                WHERE fault_id = ? 
                ORDER BY time DESC 
                LIMIT ?
            ''', (fault_id, limit))
            
            events = []
            for row in cursor.fetchall():
                event = SeismicEvent(
                    event_id=row[0],
                    fault_id=row[1],
                    time=datetime.fromisoformat(row[2]),
                    magnitude=row[3],
                    latitude=row[4],
                    longitude=row[5],
                    depth=row[6],
                    event_type=row[7],
                    source=row[8],
                    quality_score=row[9],
                    created_at=datetime.fromisoformat(row[10])
                )
                events.append(event)
            
            conn.close()
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting seismic events for fault {fault_id}: {e}")
            return []
    
    def get_fault_systems_by_region(self, min_lat: float, max_lat: float, 
                                   min_lon: float, max_lon: float) -> List[FaultSystem]:
        """
        Get fault systems in a geographic region.
        
        Args:
            min_lat, max_lat: Latitude bounds
            min_lon, max_lon: Longitude bounds
        
        Returns:
            List of fault systems in region
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM fault_systems 
                WHERE latitude BETWEEN ? AND ? 
                AND longitude BETWEEN ? AND ?
                ORDER BY quality_score DESC
            ''', (min_lat, max_lat, min_lon, max_lon))
            
            fault_systems = []
            for row in cursor.fetchall():
                # Parse data sources
                data_sources = json.loads(row[12]) if row[12] else []
                
                # Parse last major event
                last_major_event = None
                if row[10]:
                    last_major_event = datetime.fromisoformat(row[10])
                
                fault_system = FaultSystem(
                    fault_id=row[0],
                    name=row[1],
                    location=(row[2], row[3], row[4]),
                    fault_type=row[5],
                    length_km=row[6],
                    width_km=row[7],
                    dip_degrees=row[8],
                    strike_degrees=row[9],
                    last_major_event=last_major_event,
                    seismic_history_count=row[11],
                    data_sources=data_sources,
                    quality_score=row[13],
                    created_at=datetime.fromisoformat(row[14]),
                    updated_at=datetime.fromisoformat(row[15])
                )
                fault_systems.append(fault_system)
            
            conn.close()
            return fault_systems
            
        except Exception as e:
            self.logger.error(f"Error getting fault systems by region: {e}")
            return []
    
    def get_fault_systems_by_type(self, fault_type: str) -> List[FaultSystem]:
        """
        Get fault systems by type.
        
        Args:
            fault_type: Type of fault ('strike-slip', 'reverse', 'normal', 'oblique')
        
        Returns:
            List of fault systems of specified type
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM fault_systems 
                WHERE fault_type = ? 
                ORDER BY quality_score DESC
            ''', (fault_type,))
            
            fault_systems = []
            for row in cursor.fetchall():
                # Parse data sources
                data_sources = json.loads(row[12]) if row[12] else []
                
                # Parse last major event
                last_major_event = None
                if row[10]:
                    last_major_event = datetime.fromisoformat(row[10])
                
                fault_system = FaultSystem(
                    fault_id=row[0],
                    name=row[1],
                    location=(row[2], row[3], row[4]),
                    fault_type=row[5],
                    length_km=row[6],
                    width_km=row[7],
                    dip_degrees=row[8],
                    strike_degrees=row[9],
                    last_major_event=last_major_event,
                    seismic_history_count=row[11],
                    data_sources=data_sources,
                    quality_score=row[13],
                    created_at=datetime.fromisoformat(row[14]),
                    updated_at=datetime.fromisoformat(row[15])
                )
                fault_systems.append(fault_system)
            
            conn.close()
            return fault_systems
            
        except Exception as e:
            self.logger.error(f"Error getting fault systems by type {fault_type}: {e}")
            return []
    
    def update_fault_quality_score(self, fault_id: str):
        """Update quality score for a fault system based on its seismic history."""
        try:
            # Get fault system
            fault_system = self.get_fault_system(fault_id)
            if not fault_system:
                return
            
            # Get seismic events
            events = self.get_seismic_events(fault_id)
            
            # Calculate new quality score
            quality_score = self._calculate_fault_quality_score_from_events(fault_system, events)
            
            # Update database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE fault_systems 
                SET quality_score = ?, updated_at = ?
                WHERE fault_id = ?
            ''', (quality_score, datetime.utcnow().isoformat(), fault_id))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Updated quality score for fault {fault_id}: {quality_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating fault quality score: {e}")
    
    def get_database_statistics(self) -> Dict:
        """Get comprehensive database statistics."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get fault system statistics
            cursor.execute('SELECT COUNT(*) FROM fault_systems')
            fault_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(quality_score) FROM fault_systems')
            avg_quality = cursor.fetchone()[0] or 0.0
            
            cursor.execute('SELECT COUNT(*) FROM seismic_events')
            event_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(magnitude) FROM seismic_events')
            avg_magnitude = cursor.fetchone()[0] or 0.0
            
            cursor.execute('SELECT MAX(magnitude) FROM seismic_events')
            max_magnitude = cursor.fetchone()[0] or 0.0
            
            # Get fault type distribution
            cursor.execute('''
                SELECT fault_type, COUNT(*) 
                FROM fault_systems 
                GROUP BY fault_type
            ''')
            fault_type_distribution = dict(cursor.fetchall())
            
            # Get source distribution
            cursor.execute('''
                SELECT source, COUNT(*) 
                FROM seismic_events 
                GROUP BY source
            ''')
            source_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'fault_count': fault_count,
                'event_count': event_count,
                'avg_quality_score': avg_quality,
                'avg_magnitude': avg_magnitude,
                'max_magnitude': max_magnitude,
                'fault_type_distribution': fault_type_distribution,
                'source_distribution': source_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            return {}
    
    def _generate_fault_id(self, fault_data: Dict) -> str:
        """Generate unique fault ID."""
        try:
            # Create hash from fault characteristics
            location_str = f"{fault_data['latitude']:.6f},{fault_data['longitude']:.6f}"
            name_str = fault_data.get('name', 'Unknown')
            fault_type_str = fault_data.get('fault_type', 'strike-slip')
            
            # Create hash
            hash_input = f"{location_str}_{name_str}_{fault_type_str}"
            hash_obj = hashlib.md5(hash_input.encode())
            hash_hex = hash_obj.hexdigest()[:8]
            
            return f"fault_{hash_hex}"
            
        except Exception as e:
            self.logger.error(f"Error generating fault ID: {e}")
            return f"fault_{uuid.uuid4().hex[:8]}"
    
    def _generate_event_id(self, event_data: Dict, fault_id: str) -> str:
        """Generate unique event ID."""
        try:
            # Create hash from event characteristics
            time_str = event_data['time'].isoformat()
            magnitude_str = f"{event_data['magnitude']:.2f}"
            location_str = f"{event_data['latitude']:.6f},{event_data['longitude']:.6f}"
            
            # Create hash
            hash_input = f"{fault_id}_{time_str}_{magnitude_str}_{location_str}"
            hash_obj = hashlib.md5(hash_input.encode())
            hash_hex = hash_obj.hexdigest()[:8]
            
            return f"event_{hash_hex}"
            
        except Exception as e:
            self.logger.error(f"Error generating event ID: {e}")
            return f"event_{uuid.uuid4().hex[:8]}"
    
    def _validate_fault_data(self, fault_data: Dict):
        """Validate fault data before insertion."""
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in fault_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate latitude
        if not -90 <= fault_data['latitude'] <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        
        # Validate longitude
        if not -180 <= fault_data['longitude'] <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        
        # Validate fault type
        valid_types = ['strike-slip', 'reverse', 'normal', 'oblique']
        if fault_data.get('fault_type') not in valid_types:
            raise ValueError(f"Invalid fault type. Must be one of: {valid_types}")
    
    def _validate_event_data(self, event_data: Dict):
        """Validate event data before insertion."""
        required_fields = ['time', 'magnitude', 'latitude', 'longitude', 'depth']
        for field in required_fields:
            if field not in event_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate magnitude
        if not 0 <= event_data['magnitude'] <= 10:
            raise ValueError("Magnitude must be between 0 and 10")
        
        # Validate latitude
        if not -90 <= event_data['latitude'] <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        
        # Validate longitude
        if not -180 <= event_data['longitude'] <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        
        # Validate depth
        if not 0 <= event_data['depth'] <= 1000:
            raise ValueError("Depth must be between 0 and 1000 km")
    
    def _calculate_fault_quality_score(self, fault_data: Dict) -> float:
        """Calculate quality score for fault data."""
        try:
            score = 0.0
            
            # Base score for required fields
            if 'latitude' in fault_data and 'longitude' in fault_data:
                score += 0.3
            
            # Additional fields
            if 'name' in fault_data and fault_data['name']:
                score += 0.1
            
            if 'fault_type' in fault_data:
                score += 0.1
            
            if 'length_km' in fault_data and fault_data['length_km'] > 0:
                score += 0.1
            
            if 'width_km' in fault_data and fault_data['width_km'] > 0:
                score += 0.1
            
            if 'dip_degrees' in fault_data:
                score += 0.1
            
            if 'strike_degrees' in fault_data:
                score += 0.1
            
            if 'data_sources' in fault_data and fault_data['data_sources']:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating fault quality score: {e}")
            return 0.0
    
    def _calculate_event_quality_score(self, event_data: Dict) -> float:
        """Calculate quality score for event data."""
        try:
            score = 0.0
            
            # Base score for required fields
            if 'time' in event_data:
                score += 0.2
            
            if 'magnitude' in event_data:
                score += 0.2
            
            if 'latitude' in event_data and 'longitude' in event_data:
                score += 0.2
            
            if 'depth' in event_data:
                score += 0.1
            
            if 'event_type' in event_data:
                score += 0.1
            
            if 'source' in event_data:
                score += 0.1
            
            # Quality based on magnitude
            magnitude = event_data.get('magnitude', 0)
            if magnitude > 5.0:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating event quality score: {e}")
            return 0.0
    
    def _calculate_fault_quality_score_from_events(self, fault_system: FaultSystem, events: List[SeismicEvent]) -> float:
        """Calculate fault quality score based on seismic events."""
        try:
            if not events:
                return fault_system.quality_score
            
            # Base score from fault system
            base_score = fault_system.quality_score
            
            # Event count factor
            event_count = len(events)
            count_factor = min(event_count / 100.0, 1.0)  # Normalize to 0-1
            
            # Magnitude range factor
            magnitudes = [event.magnitude for event in events]
            mag_range = max(magnitudes) - min(magnitudes)
            range_factor = min(mag_range / 3.0, 1.0)  # Normalize to 0-1
            
            # Temporal span factor
            times = [event.time for event in events]
            time_span = (max(times) - min(times)).total_seconds() / (365.25 * 24 * 3600)  # years
            time_factor = min(time_span / 10.0, 1.0)  # Normalize to 0-1
            
            # Quality factor
            avg_quality = np.mean([event.quality_score for event in events])
            quality_factor = avg_quality
            
            # Calculate final score
            final_score = (base_score * 0.3 + 
                          count_factor * 0.2 + 
                          range_factor * 0.2 + 
                          time_factor * 0.15 + 
                          quality_factor * 0.15)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating fault quality score from events: {e}")
            return fault_system.quality_score

# ============================================================================
# Example Usage and Testing
# ============================================================================

def test_fault_database_manager():
    """Test the fault database manager."""
    print("="*60)
    print("TESTING FAULT DATABASE MANAGER")
    print("="*60)
    
    # Initialize database manager
    db_manager = FaultDatabaseManager("test_fault_database.db")
    print("✓ Database manager initialized")
    
    # Test adding fault system
    fault_data = {
        'name': 'Test Fault System',
        'latitude': 34.2,
        'longitude': -118.5,
        'depth': 15.0,
        'fault_type': 'strike-slip',
        'length_km': 50.0,
        'width_km': 20.0,
        'dip_degrees': 90.0,
        'strike_degrees': 45.0,
        'data_sources': ['USGS', 'EMSC']
    }
    
    fault_id = db_manager.add_fault_system(fault_data)
    print(f"✓ Added fault system: {fault_id}")
    
    # Test adding seismic events
    events_data = [
        {
            'time': datetime.now() - timedelta(days=365),
            'magnitude': 6.5,
            'latitude': 34.2,
            'longitude': -118.5,
            'depth': 15.0,
            'event_type': 'main_event',
            'source': 'USGS'
        },
        {
            'time': datetime.now() - timedelta(days=300),
            'magnitude': 4.2,
            'latitude': 34.21,
            'longitude': -118.49,
            'depth': 16.0,
            'event_type': 'aftershock',
            'source': 'USGS'
        },
        {
            'time': datetime.now() - timedelta(days=200),
            'magnitude': 3.8,
            'latitude': 34.19,
            'longitude': -118.51,
            'depth': 14.0,
            'event_type': 'aftershock',
            'source': 'USGS'
        }
    ]
    
    for event_data in events_data:
        event_id = db_manager.add_seismic_event(event_data, fault_id)
        print(f"✓ Added seismic event: {event_id}")
    
    # Test retrieving fault system
    fault_system = db_manager.get_fault_system(fault_id)
    if fault_system:
        print(f"✓ Retrieved fault system: {fault_system.name}")
        print(f"  Quality score: {fault_system.quality_score:.3f}")
        print(f"  Seismic history count: {fault_system.seismic_history_count}")
    
    # Test retrieving seismic events
    events = db_manager.get_seismic_events(fault_id)
    print(f"✓ Retrieved {len(events)} seismic events")
    
    # Test database statistics
    stats = db_manager.get_database_statistics()
    print(f"✓ Database statistics:")
    print(f"  Fault count: {stats.get('fault_count', 0)}")
    print(f"  Event count: {stats.get('event_count', 0)}")
    print(f"  Average quality: {stats.get('avg_quality_score', 0.0):.3f}")
    print(f"  Average magnitude: {stats.get('avg_magnitude', 0.0):.3f}")
    
    print("\n✓ Fault database manager test completed successfully!")

if __name__ == "__main__":
    """Run fault database manager test."""
    test_fault_database_manager()
