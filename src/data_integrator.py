#!/usr/bin/env python3
"""
Multi-Source Data Integrator
============================

Comprehensive data integration system for fault movement prediction.
Integrates data from multiple sources including USGS, EMSC, local networks,
and research databases.

Author: R.J. Mathews
Email: mail.rjmathews@gmail.com
Date: 2025
"""

import requests
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3

# Import our database manager
from fault_database_manager import FaultDatabaseManager, FaultSystem, SeismicEvent

# ============================================================================
# Data Source Definitions
# ============================================================================

@dataclass
class DataSource:
    """Data source configuration."""
    source_id: str
    name: str
    type: str  # 'usgs', 'emsc', 'local_network', 'research'
    url: str
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    last_update: Optional[datetime]
    status: str  # 'active', 'inactive', 'error'
    priority: int  # 1-10, higher is more important

@dataclass
class DataIntegrationResult:
    """Result of data integration operation."""
    source_id: str
    success: bool
    events_added: int
    faults_added: int
    errors: List[str]
    processing_time: float
    timestamp: datetime

# ============================================================================
# Multi-Source Data Integrator
# ============================================================================

class MultiSourceDataIntegrator:
    """
    Multi-source data integration system for fault movement prediction.
    
    This system integrates data from multiple sources including USGS, EMSC,
    local networks, and research databases to provide comprehensive fault data.
    """
    
    def __init__(self, database_path: str = "fault_database.db"):
        """
        Initialize multi-source data integrator.
        
        Args:
            database_path: Path to SQLite database for storing integrated data
        """
        self.database_path = database_path
        self.db_manager = FaultDatabaseManager(database_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Data sources configuration
        self.data_sources = self._initialize_data_sources()
        
        # Integration settings
        self.max_concurrent_sources = 5
        self.request_timeout = 30  # seconds
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds
        
        # Rate limiting
        self.rate_limits = {}
        self.last_request_times = {}
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize data source configurations."""
        sources = {}
        
        # USGS data source
        sources['usgs'] = DataSource(
            source_id='usgs',
            name='USGS Earthquake Catalog',
            type='usgs',
            url='https://earthquake.usgs.gov/fdsnws/event/1/query',
            api_key=None,
            rate_limit=1000,  # requests per minute
            last_update=None,
            status='active',
            priority=10
        )
        
        # EMSC data source
        sources['emsc'] = DataSource(
            source_id='emsc',
            name='EMSC Earthquake Catalog',
            type='emsc',
            url='https://www.seismicportal.eu/fdsnws/event/1/query',
            api_key=None,
            rate_limit=1000,  # requests per minute
            last_update=None,
            status='active',
            priority=9
        )
        
        # Local network data source (example)
        sources['local_network'] = DataSource(
            source_id='local_network',
            name='Local Seismic Network',
            type='local_network',
            url='http://localhost:8080/api/earthquakes',
            api_key='local_api_key',
            rate_limit=100,  # requests per minute
            last_update=None,
            status='inactive',  # Not available by default
            priority=5
        )
        
        # Research database source (example)
        sources['research_db'] = DataSource(
            source_id='research_db',
            name='Research Database',
            type='research',
            url='https://research.example.com/api/earthquakes',
            api_key='research_api_key',
            rate_limit=50,  # requests per minute
            last_update=None,
            status='inactive',  # Not available by default
            priority=3
        )
        
        return sources
    
    def integrate_all_sources(self, time_window_hours: int = 24) -> List[DataIntegrationResult]:
        """
        Integrate data from all active sources.
        
        Args:
            time_window_hours: Time window for data integration (hours)
        
        Returns:
            List of integration results
        """
        try:
            self.logger.info(f"Starting data integration from all sources (last {time_window_hours} hours)")
            
            # Get active sources
            active_sources = [source for source in self.data_sources.values() 
                            if source.status == 'active']
            
            if not active_sources:
                self.logger.warning("No active data sources found")
                return []
            
            # Calculate time window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Integrate from all sources concurrently
            results = []
            with ThreadPoolExecutor(max_workers=self.max_concurrent_sources) as executor:
                # Submit integration tasks
                future_to_source = {
                    executor.submit(self._integrate_source, source, start_time, end_time): source
                    for source in active_sources
                }
                
                # Collect results
                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Integration from {source.name}: {result.events_added} events, {result.faults_added} faults")
                    except Exception as e:
                        self.logger.error(f"Error integrating from {source.name}: {e}")
                        results.append(DataIntegrationResult(
                            source_id=source.source_id,
                            success=False,
                            events_added=0,
                            faults_added=0,
                            errors=[str(e)],
                            processing_time=0.0,
                            timestamp=datetime.utcnow()
                        ))
            
            # Log summary
            total_events = sum(result.events_added for result in results)
            total_faults = sum(result.faults_added for result in results)
            successful_sources = sum(1 for result in results if result.success)
            
            self.logger.info(f"Integration complete: {total_events} events, {total_faults} faults from {successful_sources} sources")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in data integration: {e}")
            return []
    
    def integrate_source(self, source_id: str, time_window_hours: int = 24) -> DataIntegrationResult:
        """
        Integrate data from a specific source.
        
        Args:
            source_id: Data source ID
            time_window_hours: Time window for data integration (hours)
        
        Returns:
            Integration result
        """
        try:
            if source_id not in self.data_sources:
                raise ValueError(f"Unknown data source: {source_id}")
            
            source = self.data_sources[source_id]
            if source.status != 'active':
                raise ValueError(f"Data source {source_id} is not active")
            
            # Calculate time window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            return self._integrate_source(source, start_time, end_time)
            
        except Exception as e:
            self.logger.error(f"Error integrating from source {source_id}: {e}")
            return DataIntegrationResult(
                source_id=source_id,
                success=False,
                events_added=0,
                faults_added=0,
                errors=[str(e)],
                processing_time=0.0,
                timestamp=datetime.utcnow()
            )
    
    def _integrate_source(self, source: DataSource, start_time: datetime, end_time: datetime) -> DataIntegrationResult:
        """
        Integrate data from a specific source.
        
        Args:
            source: Data source configuration
            start_time: Start time for data integration
            end_time: End time for data integration
        
        Returns:
            Integration result
        """
        start_processing = time.time()
        errors = []
        events_added = 0
        faults_added = 0
        
        try:
            # Check rate limiting
            self._check_rate_limit(source)
            
            # Integrate based on source type
            if source.type == 'usgs':
                events_added, faults_added = self._integrate_usgs_data(source, start_time, end_time)
            elif source.type == 'emsc':
                events_added, faults_added = self._integrate_emsc_data(source, start_time, end_time)
            elif source.type == 'local_network':
                events_added, faults_added = self._integrate_local_network_data(source, start_time, end_time)
            elif source.type == 'research':
                events_added, faults_added = self._integrate_research_data(source, start_time, end_time)
            else:
                raise ValueError(f"Unknown source type: {source.type}")
            
            # Update source last update time
            source.last_update = datetime.utcnow()
            
        except Exception as e:
            errors.append(str(e))
            self.logger.error(f"Error integrating from {source.name}: {e}")
        
        processing_time = time.time() - start_processing
        
        return DataIntegrationResult(
            source_id=source.source_id,
            success=len(errors) == 0,
            events_added=events_added,
            faults_added=faults_added,
            errors=errors,
            processing_time=processing_time,
            timestamp=datetime.utcnow()
        )
    
    def _integrate_usgs_data(self, source: DataSource, start_time: datetime, end_time: datetime) -> Tuple[int, int]:
        """Integrate data from USGS."""
        try:
            # Format times for USGS API
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
            end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Build request parameters
            params = {
                'format': 'geojson',
                'starttime': start_time_str,
                'endtime': end_time_str,
                'minmagnitude': 3.0,
                'orderby': 'time-desc',
                'limit': 1000
            }
            
            # Make request
            response = requests.get(source.url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            events_added = 0
            faults_added = 0
            
            for feature in features:
                try:
                    # Extract event data
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    coordinates = geometry.get('coordinates', [])
                    
                    if len(coordinates) < 3:
                        continue
                    
                    # Extract coordinates (longitude, latitude, depth)
                    longitude, latitude, depth = coordinates[:3]
                    
                    # Extract properties
                    magnitude = properties.get('mag', 0.0)
                    time_str = properties.get('time', '')
                    place = properties.get('place', '')
                    
                    # Parse time
                    try:
                        event_time = datetime.fromtimestamp(time_str / 1000)  # USGS uses milliseconds
                    except:
                        event_time = datetime.utcnow()
                    
                    # Create event data
                    event_data = {
                        'time': event_time,
                        'magnitude': magnitude,
                        'latitude': latitude,
                        'longitude': longitude,
                        'depth': depth,
                        'event_type': 'earthquake',
                        'source': 'USGS'
                    }
                    
                    # Find or create fault system
                    fault_id = self._find_or_create_fault_system(latitude, longitude, depth, place)
                    if fault_id:
                        # Add seismic event
                        self.db_manager.add_seismic_event(event_data, fault_id)
                        events_added += 1
                        
                        # Check if this is a new fault
                        if self._is_new_fault(fault_id):
                            faults_added += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing USGS event: {e}")
                    continue
            
            return events_added, faults_added
            
        except Exception as e:
            self.logger.error(f"Error integrating USGS data: {e}")
            return 0, 0
    
    def _integrate_emsc_data(self, source: DataSource, start_time: datetime, end_time: datetime) -> Tuple[int, int]:
        """Integrate data from EMSC."""
        try:
            # Format times for EMSC API
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
            end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Build request parameters
            params = {
                'format': 'json',
                'starttime': start_time_str,
                'endtime': end_time_str,
                'minmagnitude': 3.0,
                'orderby': 'time-desc',
                'limit': 1000
            }
            
            # Make request
            response = requests.get(source.url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            events_added = 0
            faults_added = 0
            
            for feature in features:
                try:
                    # Extract event data
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    coordinates = geometry.get('coordinates', [])
                    
                    if len(coordinates) < 3:
                        continue
                    
                    # Extract coordinates (longitude, latitude, depth)
                    longitude, latitude, depth = coordinates[:3]
                    
                    # Extract properties
                    magnitude = properties.get('mag', 0.0)
                    time_str = properties.get('time', '')
                    place = properties.get('place', '')
                    
                    # Parse time
                    try:
                        event_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    except:
                        event_time = datetime.utcnow()
                    
                    # Create event data
                    event_data = {
                        'time': event_time,
                        'magnitude': magnitude,
                        'latitude': latitude,
                        'longitude': longitude,
                        'depth': depth,
                        'event_type': 'earthquake',
                        'source': 'EMSC'
                    }
                    
                    # Find or create fault system
                    fault_id = self._find_or_create_fault_system(latitude, longitude, depth, place)
                    if fault_id:
                        # Add seismic event
                        self.db_manager.add_seismic_event(event_data, fault_id)
                        events_added += 1
                        
                        # Check if this is a new fault
                        if self._is_new_fault(fault_id):
                            faults_added += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing EMSC event: {e}")
                    continue
            
            return events_added, faults_added
            
        except Exception as e:
            self.logger.error(f"Error integrating EMSC data: {e}")
            return 0, 0
    
    def _integrate_local_network_data(self, source: DataSource, start_time: datetime, end_time: datetime) -> Tuple[int, int]:
        """Integrate data from local network (SCEDC)."""
        try:
            from obspy.clients.fdsn import Client
            from obspy import UTCDateTime
        except ImportError:
            self.logger.error("ObsPy not installed. Cannot integrate local network data.")
            return 0, 0

        try:
            client = Client("SCEDC")
            
            # Format times for ObsPy
            t1 = UTCDateTime(start_time)
            t2 = UTCDateTime(end_time)
            
            # Query events (SCEDC)
            # Use a slightly lower magnitude threshold for local network
            events = client.get_events(
                starttime=t1,
                endtime=t2,
                minlatitude=32.0, maxlatitude=37.0,
                minlongitude=-121.0, maxlongitude=-115.0,
                minmagnitude=2.0
            )
            
            events_added = 0
            faults_added = 0
            
            for event in events:
                try:
                    origin = event.preferred_origin() or event.origins[0]
                    mag_obj = event.preferred_magnitude() or event.magnitudes[0]
                    
                    event_data = {
                        'time': origin.time.datetime,
                        'magnitude': mag_obj.mag,
                        'latitude': origin.latitude,
                        'longitude': origin.longitude,
                        'depth': origin.depth / 1000.0 if origin.depth else 0.0,
                        'event_type': 'earthquake',
                        'source': 'SCEDC'
                    }
                    
                    # Find or create fault system
                    # Note: SCEDC sometimes provides region names we could use
                    place = event.event_descriptions[0].text if event.event_descriptions else "Local Event"
                    
                    fault_id = self._find_or_create_fault_system(
                        origin.latitude, origin.longitude, event_data['depth'], place
                    )
                    
                    if fault_id:
                        self.db_manager.add_seismic_event(event_data, fault_id)
                        events_added += 1
                        
                        if self._is_new_fault(fault_id):
                            faults_added += 1
                            
                except Exception as e:
                    continue

            return events_added, faults_added

        except Exception as e:
            self.logger.error(f"Error integrating SCEDC data: {e}")
            return 0, 0
    
    def _integrate_research_data(self, source: DataSource, start_time: datetime, end_time: datetime) -> Tuple[int, int]:
        """Integrate data from research database."""
        try:
            # This is a placeholder for research database integration
            # In practice, this would connect to research databases
            self.logger.info("Research database integration not implemented (placeholder)")
            return 0, 0
            
        except Exception as e:
            self.logger.error(f"Error integrating research data: {e}")
            return 0, 0
    
    def _find_or_create_fault_system(self, latitude: float, longitude: float, depth: float, place: str) -> Optional[str]:
        """Find existing fault system or create new one."""
        try:
            # Search for existing fault systems in the area
            search_radius = 0.1  # degrees
            existing_faults = self.db_manager.get_fault_systems_by_region(
                latitude - search_radius, latitude + search_radius,
                longitude - search_radius, longitude + search_radius
            )
            
            if existing_faults:
                # Return the closest fault system
                closest_fault = min(existing_faults, 
                                 key=lambda f: np.sqrt((f.location[0] - latitude)**2 + (f.location[1] - longitude)**2))
                return closest_fault.fault_id
            
            # Create new fault system
            fault_data = {
                'name': f'Fault near {place}',
                'latitude': latitude,
                'longitude': longitude,
                'depth': depth,
                'fault_type': 'strike-slip',  # Default type
                'length_km': 10.0,  # Default length
                'width_km': 5.0,  # Default width
                'dip_degrees': 90.0,  # Default dip
                'strike_degrees': 0.0,  # Default strike
                'data_sources': ['automatic_detection']
            }
            
            return self.db_manager.add_fault_system(fault_data)
            
        except Exception as e:
            self.logger.error(f"Error finding or creating fault system: {e}")
            return None
    
    def _is_new_fault(self, fault_id: str) -> bool:
        """Check if fault system is new (has no previous events)."""
        try:
            events = self.db_manager.get_seismic_events(fault_id, limit=1)
            return len(events) == 1  # Only the event we just added
            
        except Exception as e:
            self.logger.error(f"Error checking if fault is new: {e}")
            return False
    
    def _check_rate_limit(self, source: DataSource):
        """Check and enforce rate limiting for data source."""
        try:
            current_time = time.time()
            source_id = source.source_id
            
            # Initialize rate limiting for source
            if source_id not in self.rate_limits:
                self.rate_limits[source_id] = []
                self.last_request_times[source_id] = 0
            
            # Clean old requests (older than 1 minute)
            cutoff_time = current_time - 60
            self.rate_limits[source_id] = [t for t in self.rate_limits[source_id] if t > cutoff_time]
            
            # Check if we're at rate limit
            if len(self.rate_limits[source_id]) >= source.rate_limit:
                # Calculate wait time
                oldest_request = min(self.rate_limits[source_id])
                wait_time = 60 - (current_time - oldest_request)
                
                if wait_time > 0:
                    self.logger.info(f"Rate limit reached for {source.name}, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
            
            # Record this request
            self.rate_limits[source_id].append(current_time)
            self.last_request_times[source_id] = current_time
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
    
    def get_integration_status(self) -> Dict:
        """Get status of all data sources."""
        try:
            status = {}
            
            for source_id, source in self.data_sources.items():
                status[source_id] = {
                    'name': source.name,
                    'type': source.type,
                    'status': source.status,
                    'priority': source.priority,
                    'last_update': source.last_update.isoformat() if source.last_update else None,
                    'rate_limit': source.rate_limit
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting integration status: {e}")
            return {}
    
    def update_source_status(self, source_id: str, status: str):
        """Update status of a data source."""
        try:
            if source_id not in self.data_sources:
                raise ValueError(f"Unknown data source: {source_id}")
            
            valid_statuses = ['active', 'inactive', 'error']
            if status not in valid_statuses:
                raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
            
            self.data_sources[source_id].status = status
            self.logger.info(f"Updated source {source_id} status to {status}")
            
        except Exception as e:
            self.logger.error(f"Error updating source status: {e}")
    
    def get_integration_statistics(self) -> Dict:
        """Get integration statistics."""
        try:
            # Get database statistics
            db_stats = self.db_manager.get_database_statistics()
            
            # Get source status
            source_status = self.get_integration_status()
            
            # Calculate active sources
            active_sources = sum(1 for source in source_status.values() if source['status'] == 'active')
            
            return {
                'database_statistics': db_stats,
                'source_status': source_status,
                'active_sources': active_sources,
                'total_sources': len(self.data_sources)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting integration statistics: {e}")
            return {}

# ============================================================================
# Example Usage and Testing
# ============================================================================

def test_data_integrator():
    """Test the multi-source data integrator."""
    print("="*60)
    print("TESTING MULTI-SOURCE DATA INTEGRATOR")
    print("="*60)
    
    # Initialize data integrator
    integrator = MultiSourceDataIntegrator("test_fault_database.db")
    print("✓ Data integrator initialized")
    
    # Test source status
    status = integrator.get_integration_status()
    print(f"✓ Data sources configured: {len(status)}")
    for source_id, source_info in status.items():
        print(f"  {source_id}: {source_info['name']} ({source_info['status']})")
    
    # Test integration from all sources (last 24 hours)
    print("\nTesting data integration from all sources...")
    results = integrator.integrate_all_sources(time_window_hours=24)
    
    if results:
        print(f"✓ Integration completed from {len(results)} sources")
        for result in results:
            if result.success:
                print(f"  {result.source_id}: {result.events_added} events, {result.faults_added} faults")
            else:
                print(f"  {result.source_id}: FAILED - {', '.join(result.errors)}")
    else:
        print("⚠ No integration results (sources may be inactive)")
    
    # Test integration statistics
    stats = integrator.get_integration_statistics()
    print(f"\n✓ Integration statistics:")
    print(f"  Active sources: {stats.get('active_sources', 0)}")
    print(f"  Total sources: {stats.get('total_sources', 0)}")
    
    db_stats = stats.get('database_statistics', {})
    print(f"  Fault count: {db_stats.get('fault_count', 0)}")
    print(f"  Event count: {db_stats.get('event_count', 0)}")
    print(f"  Average quality: {db_stats.get('avg_quality_score', 0.0):.3f}")
    
    print("\n✓ Multi-source data integrator test completed successfully!")

if __name__ == "__main__":
    """Run data integrator test."""
    test_data_integrator()
