"""
OpenStreetMap Parking & Amenity Service
Uses OSM Overpass API for parking, gas stations, and EV charging (FREE - No API key needed)
"""
import requests
from typing import Dict, List

class OSMParkingService:
    def __init__(self):
        # OSM Overpass API (public, no key required)
        self.overpass_url = "https://overpass-api.de/api/interpreter"

        # Default to San Francisco for demo
        self.default_coords = {"lat": 37.7749, "lon": -122.4194}

        print("🅿️ OSM Parking/Amenity Service initialized (Real data only, no fallbacks)")

        print("🅿️ OSM Parking Service initialized (using free Overpass API)")

    def get_parking_areas(self, lat: float, lon: float, radius: int = 1000) -> List[Dict]:
        """
        Get parking areas from OpenStreetMap
        No API key required - uses public Overpass API
        """
        try:
            # Calculate bounding box (rough approximation)
            # 1 degree ≈ 111km, so radius in degrees
            radius_deg = radius / 111000

            bbox = {
                'south': lat - radius_deg,
                'north': lat + radius_deg,
                'west': lon - radius_deg,
                'east': lon + radius_deg
            }

            # Overpass QL query for parking
            query = f"""
[out:json][timeout:15];
(
  node["amenity"="parking"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="parking"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  node["amenity"="parking_space"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out center 100;
"""

            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=20,
                headers={'User-Agent': 'UrbanMonitoringPlatform/1.0'}
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_osm_parking(data)
            elif response.status_code == 429:
                print("⚠️ OSM rate limit - waiting and retrying...")
                print("   Please wait a moment before requesting parking again")
                return []
            else:
                print(f"⚠️ OSM API error {response.status_code}")
                print("   Returning empty parking data (no fallback)")
                return []

        except Exception as e:
            print(f"❌ Error fetching OSM parking: {e}")
            print("⚠️ Returning empty parking data (no demo fallback)")
            return []

    def get_gas_stations(self, lat: float, lon: float, radius: int = 5000) -> List[Dict]:
        """Get gas/fuel stations from OpenStreetMap"""
        try:
            radius_deg = radius / 111000
            bbox = {
                'south': lat - radius_deg,
                'north': lat + radius_deg,
                'west': lon - radius_deg,
                'east': lon + radius_deg
            }

            query = f"""
[out:json][timeout:15];
(
  node["amenity"="fuel"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="fuel"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out center 50;
"""

            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=20,
                headers={'User-Agent': 'UrbanMonitoringPlatform/1.0'}
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_osm_fuel(data)
            else:
                print(f"⚠️ OSM API returned status {response.status_code}, returning empty gas station data")
                return []

        except Exception as e:
            print(f"❌ Error fetching OSM fuel stations: {e}")
            print("⚠️ Returning empty gas station data (no demo fallback)")
            return []

    def get_ev_charging_stations(self, lat: float, lon: float, radius: int = 5000) -> List[Dict]:
        """Get EV charging stations from OpenStreetMap"""
        try:
            radius_deg = radius / 111000
            bbox = {
                'south': lat - radius_deg,
                'north': lat + radius_deg,
                'west': lon - radius_deg,
                'east': lon + radius_deg
            }

            query = f"""
[out:json][timeout:15];
(
  node["amenity"="charging_station"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="charging_station"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out center 50;
"""

            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=20,
                headers={'User-Agent': 'UrbanMonitoringPlatform/1.0'}
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_osm_ev_charging(data)
            else:
                print(f"⚠️ OSM API returned status {response.status_code}, returning empty EV charging data")
                return []

        except Exception as e:
            print(f"❌ Error fetching OSM EV stations: {e}")
            print("⚠️ Returning empty EV charging data (no demo fallback)")
            return []

    def _parse_osm_parking(self, data: Dict) -> List[Dict]:
        """Parse OSM parking data"""
        results = []

        for element in data.get('elements', []):
            # Get coordinates
            if 'lat' in element and 'lon' in element:
                lat, lon = element['lat'], element['lon']
            elif 'center' in element:
                lat, lon = element['center']['lat'], element['center']['lon']
            else:
                continue

            tags = element.get('tags', {})

            results.append({
                'id': f"osm_{element['id']}",
                'name': tags.get('name', 'Parking'),
                'location': {'lat': lat, 'lon': lon},
                'type': 'parking',
                'capacity': tags.get('capacity', 'Unknown'),
                'parking_type': tags.get('parking', 'surface'),
                'fee': tags.get('fee', 'Unknown'),
                'access': tags.get('access', 'Unknown'),
                'status': 'available',  # Default to available (OSM doesn't have real-time data)
                'source': 'openstreetmap'
            })

        return results

    def _parse_osm_fuel(self, data: Dict) -> List[Dict]:
        """Parse OSM fuel station data"""
        results = []

        for element in data.get('elements', []):
            if 'lat' in element and 'lon' in element:
                lat, lon = element['lat'], element['lon']
            elif 'center' in element:
                lat, lon = element['center']['lat'], element['center']['lon']
            else:
                continue

            tags = element.get('tags', {})

            results.append({
                'id': f"osm_fuel_{element['id']}",
                'name': tags.get('name', tags.get('brand', 'Gas Station')),
                'location': {'lat': lat, 'lon': lon},
                'type': 'gas_station',
                'brand': tags.get('brand'),
                'operator': tags.get('operator'),
                'address': tags.get('addr:full', tags.get('addr:street', 'Unknown')),
                'fuel_types': self._parse_fuel_types(tags),
                'source': 'openstreetmap'
            })

        return results

    def _parse_osm_ev_charging(self, data: Dict) -> List[Dict]:
        """Parse OSM EV charging station data"""
        results = []

        for element in data.get('elements', []):
            if 'lat' in element and 'lon' in element:
                lat, lon = element['lat'], element['lon']
            elif 'center' in element:
                lat, lon = element['center']['lat'], element['center']['lon']
            else:
                continue

            tags = element.get('tags', {})

            results.append({
                'id': f"osm_ev_{element['id']}",
                'name': tags.get('name', tags.get('operator', 'EV Charging Station')),
                'location': {'lat': lat, 'lon': lon},
                'type': 'ev_charging',
                'operator': tags.get('operator'),
                'network': tags.get('network'),
                'capacity': tags.get('capacity', 'Unknown'),
                'socket_types': self._parse_socket_types(tags),
                'fee': tags.get('fee', 'Unknown'),
                'access': tags.get('access', 'Unknown'),
                'source': 'openstreetmap'
            })

        return results

    def _parse_fuel_types(self, tags: Dict) -> List[str]:
        """Extract fuel types from OSM tags"""
        fuel_types = []
        for key in ['fuel:diesel', 'fuel:octane_91', 'fuel:octane_95', 'fuel:e85']:
            if tags.get(key) == 'yes':
                fuel_types.append(key.replace('fuel:', ''))
        return fuel_types if fuel_types else ['unknown']

    def _parse_socket_types(self, tags: Dict) -> List[str]:
        """Extract EV socket types from OSM tags"""
        socket_types = []
        for key in ['socket:type2', 'socket:chademo', 'socket:type1']:
            if tags.get(key):
                socket_types.append(key.replace('socket:', ''))
        return socket_types if socket_types else ['unknown']



# Singleton instance
_osm_parking_service = None

def get_osm_parking_service():
    global _osm_parking_service
    if _osm_parking_service is None:
        _osm_parking_service = OSMParkingService()
    return _osm_parking_service

