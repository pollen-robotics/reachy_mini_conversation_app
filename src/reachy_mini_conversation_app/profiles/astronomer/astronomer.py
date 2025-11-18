"""Astronomer profile: celestial object catalog and position calculations."""

import csv
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from datetime import UTC, datetime

import astropy.units as u
from rapidfuzz import fuzz, process
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord, EarthLocation, get_body

# Hardcoded location: Paris, France
LATITUDE = 48.8666
LONGITUDE = 2.3333

# ======================================================================================================================

# -----------------
#  OBJECTS CATALOG
# -----------------

CATALOG_FILE = "celestial_objects.csv"

SOLAR_SYSTEM_OBJECTS = {'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'}


class CelestialObject:
    """Represents a celestial object with its coordinates."""

    def __init__(self, name: str, aliases: List[str], ra_hours: float,
                 dec_degrees: float, obj_type: str, category: str = 'fixed', classic: bool = False):
        """Initialize celestial object."""
        self.name = name
        self.aliases = aliases
        self.ra_hours = ra_hours
        self.dec_degrees = dec_degrees
        self.type = obj_type
        self.category = category
        self.classic = classic

    def __repr__(self):
        """Return string representation of the celestial object."""
        return f"CelestialObject('{self.name}', ra={self.ra_hours:.2f}h, dec={self.dec_degrees:.2f}°)"


class Catalog:
    """Manages the catalog of celestial objects."""

    def __init__(self, csv_path: Optional[Path] = None):
        """Initialize catalog from CSV file."""
        if csv_path is None:
            csv_path = Path(__file__).parent / CATALOG_FILE

        self.objects: List[CelestialObject] = []
        self.name_index: Dict[str, CelestialObject] = {}

        self._load_catalog(csv_path)

    def _load_catalog(self, csv_path: Path):
        """Load celestial objects from CSV file."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                aliases = []
                if row['aliases']:
                    aliases = [a.strip() for a in row['aliases'].split('|')]

                category = row.get('category', 'fixed')
                classic = row.get('classic', 'no') == 'yes'

                obj = CelestialObject(
                    name=row['name'],
                    aliases=aliases,
                    ra_hours=float(row['ra_hours']),
                    dec_degrees=float(row['dec_degrees']),
                    obj_type=row['type'],
                    category=category,
                    classic=classic
                )

                self.objects.append(obj)
                self.name_index[obj.name.lower()] = obj

                for alias in aliases:
                    self.name_index[alias.lower()] = obj

    def get_by_exact_name(self, name: str) -> Optional[CelestialObject]:
        """Get object by exact name match (case-insensitive)."""
        return self.name_index.get(name.lower())

    def get_all_names(self) -> List[str]:
        """Get all indexed names (primary names and aliases)."""
        return list(self.name_index.keys())

    def __len__(self):
        """Return the number of celestial objects in the catalog."""
        return len(self.objects)

# ======================================================================================================================

# ------------------------
#  COORDINATE CONVERSIONS
# ------------------------

'''
For far away celestial objects (stars, galaxies), we use their Right Ascension (RA) and Declination (Dec) to compute
their Altitude and Azimuth from a given Earth location and time.

For solar system objects (planets, moon, sun), we use astropy's get_body function to get their current position.
'''

def _ra_dec_to_alt_az(ra_hours: float, dec_degrees: float, latitude: float, longitude: float, time: datetime) -> Tuple[float, float]:
    """Convert (absolute) Right Ascension and Declination to (relative time and location dependent) Altitude and Azimuth."""
    celestial_coord = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_degrees * u.deg, frame='icrs')
    observer_location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
    obs_time = Time(time)
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
    altaz_coord = celestial_coord.transform_to(altaz_frame)
    return (altaz_coord.az.degree, altaz_coord.alt.degree)


def _get_solar_system_position(name: str, latitude: float, longitude: float, time: datetime) -> Tuple[float, float]:
    """Calculate azimuth and altitude for solar system objects."""
    observer_location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
    obs_time = Time(time)
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
    body_coord = get_body(name.lower(), obs_time, observer_location)
    altaz_coord = body_coord.transform_to(altaz_frame)
    return (altaz_coord.az.degree, altaz_coord.alt.degree)


# ======================================================================================================================

# --------------------------
#  OBJECT SEARCH IN CATALOG
# --------------------------

'''
We use fuzzy string matching to find the best matching celestial object name in the catalog.
As several objects may have multiple aliases, we index all names and aliases for matching.
(for instance M31 can be matched with "Andromeda Galaxy", "M31", "Messier 31" or "NGC 224")
'''

def _find_object(query: str, catalog: Catalog, threshold: int = 70) -> Optional[Tuple[CelestialObject, int]]:
    """Find a celestial object by name using fuzzy matching."""
    # Try exact match first
    exact_match = catalog.get_by_exact_name(query)
    if exact_match:
        return (exact_match, 100)

    # Fuzzy matching
    all_names = catalog.get_all_names()
    if not all_names:
        return None

    result = process.extractOne(query.lower(), all_names, scorer=fuzz.WRatio)

    if result is None:
        return None

    matched_name, score, _ = result

    if score >= threshold:
        obj = catalog.get_by_exact_name(matched_name)
        return (obj, int(score))

    return None


# ======================================================================================================================

_catalog: Optional[Catalog] = None


def get_catalog() -> Catalog:
    """Get or initialize the global catalog."""
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog


def find_celestial_angles(name: str, latitude: float = LATITUDE, longitude: float = LONGITUDE, time: Optional[datetime] = None,
                          search_threshold: int = 70) -> Dict[str, Any]:
    """Find a celestial object and calculate its azimuth and altitude."""
    if time is None:
        time = datetime.now(UTC)

    catalog = get_catalog()
    search_result = _find_object(name, catalog, threshold=search_threshold)

    if search_result is None:
        return {'found': False, 'error': f"No celestial object found matching '{name}'"}

    obj, score = search_result

    # Calculate position based on object category
    if obj.category == 'solar_system':
        azimuth, altitude = _get_solar_system_position(obj.name, latitude, longitude, time)
    else:
        azimuth, altitude = _ra_dec_to_alt_az(obj.ra_hours, obj.dec_degrees, latitude, longitude, time)

    return {'found': True, 'object_name': obj.name, 'match_score': score,
            'azimuth': azimuth, 'altitude': altitude, 'type': obj.type,
            'ra_hours': obj.ra_hours, 'dec_degrees': obj.dec_degrees}
