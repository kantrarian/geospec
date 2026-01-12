"""
fault_segments.py
Fault segment definitions for seismic correlation analysis.

Each fault segment represents a section of a fault system that can be monitored
independently. The correlation between segments is a key diagnostic:
- Normal state: Segments are correlated (stress distributed across fault)
- Pre-earthquake: Segments decouple (stress concentrates before rupture)

Author: R.J. Mathews
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path


@dataclass
class SeismicStation:
    """A seismic station for monitoring."""
    network: str  # Network code (e.g., 'CI', 'NC')
    code: str     # Station code (e.g., 'WBS', 'SLA')
    lat: float    # Latitude
    lon: float    # Longitude
    name: str = ""  # Station name (optional)
    channels: List[str] = field(default_factory=lambda: ['BHZ', 'HHZ'])

    @property
    def nslc(self) -> str:
        """Network.Station.Location.Channel format."""
        return f"{self.network}.{self.code}"


@dataclass
class FaultSegment:
    """
    A segment of a fault system for correlation monitoring.

    Attributes:
        name: Unique identifier for the segment
        region: Parent region name
        stations: List of seismic stations in/near this segment
        polygon: Geographic boundary as list of (lat, lon) tuples
        strike: Approximate fault strike in degrees (0-360)
        dip: Approximate fault dip in degrees (0-90)
        rake: Typical slip rake (-180 to 180)
        notes: Additional information
    """
    name: str
    region: str
    stations: List[SeismicStation]
    polygon: List[Tuple[float, float]]  # [(lat, lon), ...]
    strike: float = 0.0
    dip: float = 90.0
    rake: float = 0.0
    notes: str = ""

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within the segment polygon (simple ray casting)."""
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            yi, xi = self.polygon[i]
            yj, xj = self.polygon[j]
            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


# =============================================================================
# RIDGECREST FAULT SEGMENTS
# =============================================================================

RIDGECREST_SEGMENTS = [
    FaultSegment(
        name="ridgecrest_mainshock",
        region="ridgecrest",
        stations=[
            SeismicStation("CI", "WBS", 35.726, -117.943, "Wells Road"),
            SeismicStation("CI", "SLA", 35.891, -117.283, "Searles Lake"),
            SeismicStation("CI", "CLC", 35.816, -117.598, "China Lake"),
        ],
        polygon=[
            (35.7, -117.65),   # SW
            (35.7, -117.45),   # SE
            (36.0, -117.45),   # NE
            (36.0, -117.65),   # NW
        ],
        strike=320,  # NW-SE trending fault
        dip=80,
        rake=180,    # Right-lateral strike-slip
        notes="M7.1 mainshock rupture zone"
    ),
    FaultSegment(
        name="airport_lake",
        region="ridgecrest",
        stations=[
            SeismicStation("CI", "LRL", 35.480, -117.682, "Little Lake"),
            SeismicStation("CI", "CCC", 35.525, -117.365, "Coso Volcanic"),
            SeismicStation("CI", "TOW2", 35.812, -117.768, "Towers"),
        ],
        polygon=[
            (35.5, -117.75),
            (35.5, -117.55),
            (35.75, -117.55),
            (35.75, -117.75),
        ],
        strike=45,   # NE-SW trending conjugate
        dip=85,
        rake=180,    # Strike-slip
        notes="M6.4 foreshock zone - conjugate fault"
    ),
    FaultSegment(
        name="little_lake",
        region="ridgecrest",
        stations=[
            SeismicStation("CI", "LRL", 35.480, -117.682, "Little Lake"),
            SeismicStation("CI", "WBS", 35.726, -117.943, "Wells Road"),
            SeismicStation("CI", "JRC2", 35.982, -117.808, "Jawbone Ranch"),
        ],
        polygon=[
            (35.3, -117.9),
            (35.3, -117.6),
            (35.6, -117.6),
            (35.6, -117.9),
        ],
        strike=340,
        dip=80,
        rake=175,
        notes="Southern extension of Ridgecrest system"
    ),
]


# =============================================================================
# SOUTHERN SAN ANDREAS - MOJAVE SEGMENTS
# =============================================================================

SOCAL_SAF_MOJAVE_SEGMENTS = [
    FaultSegment(
        name="mojave_north",
        region="socal_saf_mojave",
        stations=[
            SeismicStation("CI", "GSC", 35.302, -116.806, "Goldstone"),
            SeismicStation("CI", "HEC", 34.829, -116.335, "Hector Mine"),
        ],
        polygon=[
            (35.0, -117.5),
            (35.0, -116.5),
            (35.5, -116.5),
            (35.5, -117.5),
        ],
        strike=305,  # SAF strike in Mojave
        dip=90,
        rake=180,
        notes="Northern Mojave segment"
    ),
    FaultSegment(
        name="mojave_central",
        region="socal_saf_mojave",
        stations=[
            SeismicStation("CI", "VTV", 34.567, -117.333, "Victorville"),
            SeismicStation("CI", "HEC", 34.829, -116.335, "Hector Mine"),
        ],
        polygon=[
            (34.5, -117.5),
            (34.5, -116.5),
            (35.0, -116.5),
            (35.0, -117.5),
        ],
        strike=300,
        dip=90,
        rake=180,
        notes="Central Mojave - includes 1857 rupture extent"
    ),
    FaultSegment(
        name="mojave_south",
        region="socal_saf_mojave",
        stations=[
            SeismicStation("CI", "BEL", 34.001, -117.752, "Bell Canyon"),
            SeismicStation("CI", "PFO", 33.611, -116.459, "Pinon Flat Observatory"),
        ],
        polygon=[
            (34.0, -118.0),
            (34.0, -117.0),
            (34.5, -117.0),
            (34.5, -118.0),
        ],
        strike=295,
        dip=90,
        rake=175,
        notes="San Bernardino Mountains junction"
    ),
    FaultSegment(
        name="san_bernardino",
        region="socal_saf_mojave",
        stations=[
            SeismicStation("CI", "BEL", 34.001, -117.752, "Bell Canyon"),
            SeismicStation("CI", "DGR", 33.650, -116.101, "Durmid Hill"),
        ],
        polygon=[
            (33.8, -117.5),
            (33.8, -116.5),
            (34.2, -116.5),
            (34.2, -117.5),
        ],
        strike=290,
        dip=85,
        rake=170,
        notes="San Bernardino segment"
    ),
]


# =============================================================================
# COACHELLA VALLEY SEGMENTS
# =============================================================================

SOCAL_COACHELLA_SEGMENTS = [
    FaultSegment(
        name="coachella_north",
        region="socal_coachella",
        stations=[
            SeismicStation("CI", "PSR", 33.818, -116.555, "Palm Springs"),
            SeismicStation("CI", "BTC", 33.990, -116.518, "Butterfly Creek"),
        ],
        polygon=[
            (33.6, -116.7),
            (33.6, -116.3),
            (34.0, -116.3),
            (34.0, -116.7),
        ],
        strike=310,
        dip=70,
        rake=160,
        notes="Northern Coachella - high slip deficit"
    ),
    FaultSegment(
        name="coachella_south",
        region="socal_coachella",
        stations=[
            SeismicStation("CI", "BOR", 33.263, -116.406, "Borrego"),
            SeismicStation("CI", "TRO", 33.523, -116.425, "Toro Peak"),
            SeismicStation("CI", "CTW", 33.393, -116.638, "Coyote Wells"),
        ],
        polygon=[
            (33.0, -116.8),
            (33.0, -116.2),
            (33.6, -116.2),
            (33.6, -116.8),
        ],
        strike=305,
        dip=65,
        rake=155,
        notes="Southern Coachella - highest hazard"
    ),
    FaultSegment(
        name="brawley_seismic_zone",
        region="socal_coachella",
        stations=[
            SeismicStation("CI", "WMC", 33.573, -116.071, "Wiley Well"),
            SeismicStation("CI", "RXH", 33.182, -115.624, "Ripley"),
            SeismicStation("CI", "BC3", 33.655, -115.455, "Blythe"),
        ],
        polygon=[
            (32.8, -116.0),
            (32.8, -115.2),
            (33.6, -115.2),
            (33.6, -116.0),
        ],
        strike=320,
        dip=85,
        rake=180,
        notes="Brawley seismic zone - frequent swarms"
    ),
]


# =============================================================================
# HAYWARD-RODGERS CREEK SEGMENTS
# =============================================================================

NORCAL_HAYWARD_SEGMENTS = [
    FaultSegment(
        name="hayward_north",
        region="norcal_hayward",
        stations=[
            SeismicStation("BK", "BKS", 37.876, -122.236, "Berkeley"),
            SeismicStation("NC", "WENL", 37.622, -121.757, "Wente Vineyards"),
        ],
        polygon=[
            (37.6, -122.4),
            (37.6, -122.0),
            (38.0, -122.0),
            (38.0, -122.4),
        ],
        strike=325,
        dip=75,
        rake=175,
        notes="Northern Hayward - creeping section"
    ),
    FaultSegment(
        name="hayward_south",
        region="norcal_hayward",
        stations=[
            SeismicStation("NC", "JRSC", 37.404, -122.239, "Jasper Ridge"),
            SeismicStation("NC", "PACP", 37.008, -121.287, "Pacheco Peak"),
        ],
        polygon=[
            (37.0, -122.4),
            (37.0, -121.8),
            (37.6, -121.8),
            (37.6, -122.4),
        ],
        strike=330,
        dip=70,
        rake=170,
        notes="Southern Hayward - 1868 rupture zone"
    ),
    FaultSegment(
        name="rodgers_creek",
        region="norcal_hayward",
        stations=[
            SeismicStation("NC", "MCCM", 38.145, -122.880, "Marin County"),
            SeismicStation("NC", "HOPS", 38.994, -123.072, "Hopland"),
        ],
        polygon=[
            (38.0, -123.0),
            (38.0, -122.4),
            (38.5, -122.4),
            (38.5, -123.0),
        ],
        strike=335,
        dip=80,
        rake=180,
        notes="Rodgers Creek - northern extension"
    ),
    FaultSegment(
        name="calaveras",
        region="norcal_hayward",
        stations=[
            SeismicStation("NC", "PACP", 37.008, -121.287, "Pacheco Peak"),
            SeismicStation("BK", "CMB", 38.035, -120.386, "Columbia"),
        ],
        polygon=[
            (37.0, -122.0),
            (37.0, -121.2),
            (37.8, -121.2),
            (37.8, -122.0),
        ],
        strike=340,
        dip=85,
        rake=180,
        notes="Calaveras fault - connects to SAF"
    ),
]


# =============================================================================
# CASCADIA SEGMENTS
# =============================================================================

CASCADIA_SEGMENTS = [
    FaultSegment(
        name="vancouver_island",
        region="cascadia",
        stations=[
            SeismicStation("CN", "PHC", 50.709, -127.429, "Port Hardy"),
            SeismicStation("CN", "WALA", 49.071, -125.437, "Walters Point"),
            SeismicStation("CN", "LLLB", 49.475, -123.233, "Lillooet Lake"),
        ],
        polygon=[
            (48.5, -128.0),
            (48.5, -123.5),
            (51.0, -123.5),
            (51.0, -128.0),
        ],
        strike=320,  # Subduction interface
        dip=10,
        rake=90,     # Thrust
        notes="Northern Cascadia megathrust"
    ),
    FaultSegment(
        name="puget_sound",
        region="cascadia",
        stations=[
            SeismicStation("PB", "ALBH", 48.390, -123.488, "Albert Head"),
            SeismicStation("CN", "GOBB", 49.317, -121.878, "Goble Creek"),
        ],
        polygon=[
            (47.0, -124.0),
            (47.0, -122.0),
            (48.5, -122.0),
            (48.5, -124.0),
        ],
        strike=330,
        dip=12,
        rake=90,
        notes="Puget Sound - includes Seattle fault"
    ),
    FaultSegment(
        name="olympic_peninsula",
        region="cascadia",
        stations=[
            SeismicStation("UW", "PCEP", 47.050, -122.890, "Pacific Beach"),
            SeismicStation("UW", "LON", 46.750, -121.810, "Longmire"),
        ],
        polygon=[
            (46.5, -125.0),
            (46.5, -122.5),
            (48.0, -122.5),
            (48.0, -125.0),
        ],
        strike=335,
        dip=15,
        rake=90,
        notes="Olympic Peninsula segment"
    ),
    FaultSegment(
        name="columbia_river",
        region="cascadia",
        stations=[
            SeismicStation("UW", "STOR", 45.894, -122.206, "St. Helens"),
            SeismicStation("UW", "LON", 46.750, -121.810, "Longmire"),
        ],
        polygon=[
            (45.0, -124.5),
            (45.0, -121.5),
            (46.5, -121.5),
            (46.5, -124.5),
        ],
        strike=345,
        dip=18,
        rake=85,
        notes="Southern Cascadia - near Columbia River"
    ),
]


# =============================================================================
# ISTANBUL / MARMARA SEGMENTS
# Updated January 2026 with verified available stations from KO network
# =============================================================================

ISTANBUL_MARMARA_SEGMENTS = [
    FaultSegment(
        name="marmara_west",
        region="istanbul_marmara",
        stations=[
            # Marmara array stations - verified 99%+ availability via KOERI
            SeismicStation("KO", "NMR6", 40.70, 27.27, "Marmara Array 6"),
            SeismicStation("KO", "NMR3", 40.74, 27.29, "Marmara Array 3"),
            SeismicStation("KO", "NMR2", 40.72, 27.31, "Marmara Array 2"),
            SeismicStation("KO", "BOTS", 40.99, 27.98, "Bostanli"),
        ],
        polygon=[
            (40.5, 27.0),
            (40.5, 28.0),
            (41.1, 28.0),
            (41.1, 27.0),
        ],
        strike=265,  # North Anatolian Fault
        dip=88,
        rake=180,
        notes="Western Marmara - seismic gap, Marmara OBS array"
    ),
    FaultSegment(
        name="marmara_central",
        region="istanbul_marmara",
        stations=[
            # Verified available stations
            SeismicStation("KO", "NMR5", 40.70, 27.50, "Marmara Array 5"),
            SeismicStation("KO", "NMR7", 40.68, 27.45, "Marmara Array 7"),
            SeismicStation("KO", "NMR8", 40.66, 27.55, "Marmara Array 8"),
            SeismicStation("KO", "NMR9", 40.64, 27.60, "Marmara Array 9"),
        ],
        polygon=[
            (40.3, 27.5),
            (40.3, 29.0),
            (40.8, 29.0),
            (40.8, 27.5),
        ],
        strike=270,
        dip=85,
        rake=180,
        notes="Central Marmara basin - densest coverage"
    ),
    FaultSegment(
        name="izmit",
        region="istanbul_marmara",
        stations=[
            # Verified available
            SeismicStation("KO", "SAUV", 40.74, 30.33, "Sakarya University"),
            SeismicStation("KO", "GAZK", 40.50, 29.50, "Gazikoy"),  # 78% - marginal but usable
        ],
        polygon=[
            (40.5, 29.5),
            (40.5, 31.0),
            (41.0, 31.0),
            (41.0, 29.5),
        ],
        strike=275,
        dip=90,
        rake=180,
        notes="Izmit segment - ruptured 1999, now relocked"
    ),
]


# =============================================================================
# TURKEY / KAHRAMANMARAS SEGMENTS (2023 M7.8)
# Updated January 2026 with verified waveform-available stations via KOERI
# Scan verified 17 stations with HHZ/HNZ data access
# =============================================================================

TURKEY_KAHRAMANMARAS_SEGMENTS = [
    FaultSegment(
        name="east_anatolian_north",
        region="turkey_kahramanmaras",
        stations=[
            # Northern EAF: Malatya-Elazig zone (38.0-39.0N)
            # All verified via KOERI scan 2026-01-11
            SeismicStation("KO", "MLTY", 38.322, 38.441, "Malatya"),
            SeismicStation("KO", "AKDG", 38.362, 37.970, "Akcadag, Malatya"),
            SeismicStation("KO", "ARGN", 38.776, 38.264, "Arguvan, Malatya"),
            SeismicStation("KO", "SVRC", 38.377, 39.306, "Sivrice, Elazig"),
        ],
        polygon=[
            (38.0, 37.0),
            (38.0, 40.0),
            (39.0, 40.0),
            (39.0, 37.0),
        ],
        strike=65,  # East Anatolian Fault strike
        dip=85,
        rake=180,
        notes="Northern EAF - Malatya-Elazig zone, 2023 northern rupture terminus"
    ),
    FaultSegment(
        name="east_anatolian_central",
        region="turkey_kahramanmaras",
        stations=[
            # Central EAF: Epicentral zone (37.0-38.0N)
            # KHMN is Pazarcik - very close to 2023 M7.8 epicenter!
            SeismicStation("KO", "NURH", 37.969, 37.436, "Nurhak, Kahramanmaras"),
            SeismicStation("KO", "KMRS", 37.509, 36.900, "Kahramanmaras City"),
            SeismicStation("KO", "KHMN", 37.392, 37.157, "Pazarcik, Kahramanmaras"),
            SeismicStation("TU", "ANDN", 37.580, 36.345, "Andirin"),
        ],
        polygon=[
            (37.0, 35.5),
            (37.0, 38.0),
            (38.0, 38.0),
            (38.0, 35.5),
        ],
        strike=60,
        dip=85,
        rake=180,
        notes="Central EAF - 2023 M7.8 epicentral region (Pazarcik)"
    ),
    FaultSegment(
        name="east_anatolian_south",
        region="turkey_kahramanmaras",
        stations=[
            # Southern EAF: Gaziantep-Hatay zone (36.0-37.0N)
            # Extended coverage for rupture propagation zone
            SeismicStation("KO", "GAZ", 37.172, 37.210, "Gaziantep"),
            SeismicStation("KO", "CEYT", 37.011, 35.748, "Ceyhan, Adana"),
            SeismicStation("KO", "TAHT", 36.376, 36.185, "Tahtakopru, Hatay"),
        ],
        polygon=[
            (36.0, 35.0),
            (36.0, 38.0),
            (37.2, 38.0),
            (37.2, 35.0),
        ],
        strike=55,
        dip=80,
        rake=175,
        notes="Southern EAF - Gaziantep-Hatay, 2023 rupture propagation zone"
    ),
]


# =============================================================================
# JAPAN / TOHOKU SEGMENTS (2011 M9.0)
# =============================================================================

JAPAN_TOHOKU_SEGMENTS = [
    FaultSegment(
        name="japan_trench_north",
        region="japan_tohoku",
        stations=[
            SeismicStation("JP", "JMM", 37.869, 140.790, "Minamisoma"),
            SeismicStation("JP", "JSD", 38.040, 138.257, "Sado"),
        ],
        polygon=[
            (38.0, 140.0),
            (38.0, 145.0),
            (40.0, 145.0),
            (40.0, 140.0),
        ],
        strike=195,  # Subduction interface
        dip=15,
        rake=90,  # Thrust
        notes="Northern Japan Trench megathrust"
    ),
    FaultSegment(
        name="japan_trench_central",
        region="japan_tohoku",
        stations=[
            SeismicStation("IU", "MAJO", 36.546, 138.204, "Matsushiro"),
            SeismicStation("PS", "TSK", 36.211, 140.110, "Tsukuba"),
            SeismicStation("JP", "JYT", 36.231, 140.191, "Yatabe"),
        ],
        polygon=[
            (36.0, 140.0),
            (36.0, 145.0),
            (38.0, 145.0),
            (38.0, 140.0),
        ],
        strike=195,
        dip=12,
        rake=90,
        notes="Central segment - 2011 maximum slip zone"
    ),
    FaultSegment(
        name="japan_trench_south",
        region="japan_tohoku",
        stations=[
            SeismicStation("PS", "TSK", 36.211, 140.110, "Tsukuba"),
            SeismicStation("IU", "MAJO", 36.546, 138.204, "Matsushiro"),
        ],
        polygon=[
            (34.0, 140.0),
            (34.0, 144.0),
            (36.0, 144.0),
            (36.0, 140.0),
        ],
        strike=200,
        dip=20,
        rake=85,
        notes="Southern Japan Trench"
    ),
]


# =============================================================================
# CHILE / MAULE SEGMENTS (2010 M8.8)
# =============================================================================

CHILE_MAULE_SEGMENTS = [
    FaultSegment(
        name="chile_maule_north",
        region="chile_maule",
        stations=[
            SeismicStation("C", "PEL1", -33.144, -70.675, "Peldehue"),
            SeismicStation("G", "PEL", -33.144, -70.675, "Peldehue-G"),
        ],
        polygon=[
            (-35.0, -74.0),
            (-35.0, -70.0),
            (-33.0, -70.0),
            (-33.0, -74.0),
        ],
        strike=10,  # Chile subduction
        dip=18,
        rake=90,
        notes="Northern Maule segment"
    ),
    FaultSegment(
        name="chile_maule_central",
        region="chile_maule",
        stations=[
            SeismicStation("C", "CL2C", -33.396, -70.537, "Santiago"),
            SeismicStation("C", "CLCH", -33.396, -70.537, "Santiago-CH"),
        ],
        polygon=[
            (-37.0, -75.0),
            (-37.0, -71.0),
            (-35.0, -71.0),
            (-35.0, -75.0),
        ],
        strike=15,
        dip=15,
        rake=90,
        notes="Central Maule - 2010 maximum slip"
    ),
    FaultSegment(
        name="chile_maule_south",
        region="chile_maule",
        stations=[
            SeismicStation("YM", "V04", -39.388, -71.941, "Villarrica"),
            SeismicStation("YM", "V07", -39.508, -71.954, "Villarrica-7"),
        ],
        polygon=[
            (-40.0, -75.0),
            (-40.0, -71.0),
            (-37.0, -71.0),
            (-37.0, -75.0),
        ],
        strike=8,
        dip=12,
        rake=88,
        notes="Southern Maule segment"
    ),
]


# =============================================================================
# CAMPI FLEGREI SEGMENTS (Active volcanic caldera - bradyseismic unrest)
# Added January 2026 - Pilot region for Method 2 validation
# =============================================================================

CAMPI_FLEGREI_SEGMENTS = [
    FaultSegment(
        name="solfatara_crater",
        region="campi_flegrei",
        stations=[
            # Dense coverage around Solfatara - epicenter of current unrest
            SeismicStation("IV", "CSFT", 40.829, 14.140, "Solfatara"),
            SeismicStation("IV", "CSOB", 40.827, 14.144, "Solfatara Bordo Cratere"),
            SeismicStation("IV", "CSTH", 40.829, 14.149, "Solfatara Tennis Hotel"),
            SeismicStation("IV", "CPIS", 40.829, 14.147, "Pisciarelli Fumarole"),
        ],
        polygon=[
            (40.82, 14.13),
            (40.82, 14.16),
            (40.84, 14.16),
            (40.84, 14.13),
        ],
        strike=0,  # Caldera - no preferred strike
        dip=90,
        rake=0,
        notes="Solfatara crater - main degassing and uplift center"
    ),
    FaultSegment(
        name="pozzuoli_bay",
        region="campi_flegrei",
        stations=[
            # Coastal stations around Pozzuoli Bay
            SeismicStation("IV", "CPOZ", 40.821, 14.119, "Darsena Pozzuoli"),
            SeismicStation("IV", "CAAM", 40.820, 14.142, "Accademia Aeronautica"),
            SeismicStation("IV", "COLB", 40.819, 14.145, "Monte Olibano"),
        ],
        polygon=[
            (40.80, 14.10),
            (40.80, 14.15),
            (40.83, 14.15),
            (40.83, 14.10),
        ],
        strike=0,
        dip=90,
        rake=0,
        notes="Pozzuoli Bay - maximum historical uplift zone"
    ),
    FaultSegment(
        name="western_caldera",
        region="campi_flegrei",
        stations=[
            # Western caldera rim stations
            SeismicStation("IV", "CBAC", 40.811, 14.081, "Castello di Baia"),
            SeismicStation("IV", "CFMN", 40.833, 14.090, "Monte Nuovo"),
            SeismicStation("IV", "CAFL", 40.844, 14.094, "Arco Felice"),
        ],
        polygon=[
            (40.80, 14.05),
            (40.80, 14.10),
            (40.85, 14.10),
            (40.85, 14.05),
        ],
        strike=0,
        dip=90,
        rake=0,
        notes="Western caldera - Monte Nuovo (last eruption 1538)"
    ),
    FaultSegment(
        name="eastern_caldera",
        region="campi_flegrei",
        stations=[
            # Eastern caldera stations
            SeismicStation("IV", "CBAG", 40.812, 14.175, "Bagnoli"),
            SeismicStation("IV", "CMSN", 40.838, 14.182, "Monte S.Angelo"),
            SeismicStation("IV", "CAWE", 40.840, 14.139, "Astroni Ovest"),
        ],
        polygon=[
            (40.80, 14.15),
            (40.80, 14.20),
            (40.85, 14.20),
            (40.85, 14.15),
        ],
        strike=0,
        dip=90,
        rake=0,
        notes="Eastern caldera rim - Astroni crater"
    ),
]


# =============================================================================
# FAULT SEGMENT REGISTRY
# =============================================================================

FAULT_SEGMENTS: Dict[str, List[FaultSegment]] = {
    'ridgecrest': RIDGECREST_SEGMENTS,
    'socal_saf_mojave': SOCAL_SAF_MOJAVE_SEGMENTS,
    'socal_coachella': SOCAL_COACHELLA_SEGMENTS,
    'norcal_hayward': NORCAL_HAYWARD_SEGMENTS,
    'cascadia': CASCADIA_SEGMENTS,
    'istanbul_marmara': ISTANBUL_MARMARA_SEGMENTS,
    'turkey_kahramanmaras': TURKEY_KAHRAMANMARAS_SEGMENTS,
    'japan_tohoku': JAPAN_TOHOKU_SEGMENTS,
    'chile_maule': CHILE_MAULE_SEGMENTS,
    'campi_flegrei': CAMPI_FLEGREI_SEGMENTS,
}


def get_segments_for_region(region: str) -> List[FaultSegment]:
    """Get fault segments for a specific region."""
    if region not in FAULT_SEGMENTS:
        raise ValueError(f"Unknown region: {region}. Valid: {list(FAULT_SEGMENTS.keys())}")
    return FAULT_SEGMENTS[region]


def get_all_stations_for_region(region: str) -> List[SeismicStation]:
    """Get all unique stations for a region's fault segments."""
    segments = get_segments_for_region(region)
    seen = set()
    stations = []
    for seg in segments:
        for sta in seg.stations:
            key = f"{sta.network}.{sta.code}"
            if key not in seen:
                seen.add(key)
                stations.append(sta)
    return stations


def get_segment_by_name(name: str) -> Optional[FaultSegment]:
    """Find a segment by name across all regions."""
    for region, segments in FAULT_SEGMENTS.items():
        for seg in segments:
            if seg.name == name:
                return seg
    return None


def summarize_segments():
    """Print a summary of all defined fault segments."""
    print("=" * 70)
    print("GeoSpec Fault Segment Inventory")
    print("=" * 70)

    for region, segments in FAULT_SEGMENTS.items():
        print(f"\n{region.upper()}")
        print("-" * 40)
        for seg in segments:
            n_stations = len(seg.stations)
            print(f"  {seg.name:<25} {n_stations} stations, strike={seg.strike}°")


if __name__ == "__main__":
    summarize_segments()

    # Test lookup
    print("\n\nTest: Get Ridgecrest stations")
    stations = get_all_stations_for_region('ridgecrest')
    for sta in stations:
        print(f"  {sta.nslc}: {sta.name} ({sta.lat:.3f}, {sta.lon:.3f})")
