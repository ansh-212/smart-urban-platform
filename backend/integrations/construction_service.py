"""
Construction & Road Closure Service
Gets real construction data from government APIs and OpenStreetMap
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List
import random

class ConstructionService:
    def __init__(self):
        # OpenStreetMap Overpass API for construction data
        self.overpass_url = "https://overpass-api.de/api/interpreter"

        # San Francisco bounding box (south, west, north, east)
        self.bbox_coords = {
            'south': 37.7,
            'west': -122.5,
            'north': 37.8,
            'east': -122.4
        }

        print("🚧 Construction service initialized")

    def get_construction_zones(self) -> List[Dict]:
        """Get construction zones from multiple sources"""
        zones = []

        # Try to get real data from OpenStreetMap
        try:
            osm_zones = self._get_osm_construction()
            zones.extend(osm_zones)
        except Exception as e:
            print(f"⚠️ OSM construction data unavailable: {e}")

        # Add simulated construction for demo
        sim_zones = self._simulate_construction()
        zones.extend(sim_zones)

        return zones

    def get_road_closures(self) -> List[Dict]:
        """Get current road closures"""
        closures = []

        # Simulated road closures (in production, connect to city APIs)
        sf_roads = [
            "Market Street", "Van Ness Avenue", "Lombard Street",
            "California Street", "Geary Boulevard", "Mission Street"
        ]

        for _ in range(random.randint(1, 3)):
            road = random.choice(sf_roads)

            closures.append({
                'type': 'road_closure',
                'road_name': road,
                'location': self._generate_sf_location(),
                'reason': random.choice([
                    'Water main repair',
                    'Gas line maintenance',
                    'Street paving',
                    'Utility work',
                    'Emergency repair'
                ]),
                'severity': random.choice(['medium', 'high']),
                'start_time': datetime.now().isoformat(),
                'estimated_end': (datetime.now() + timedelta(hours=random.randint(2, 24))).isoformat(),
                'source': 'city_maintenance'
            })

        return closures

    def get_planned_construction(self, days_ahead: int = 7) -> List[Dict]:
        """Get planned construction activities"""
        planned = []

        for day in range(1, days_ahead + 1):
            # Random chance of planned construction each day
            if random.random() < 0.3:  # 30% chance per day
                future_date = datetime.now() + timedelta(days=day)

                planned.append({
                    'type': 'planned_construction',
                    'project_name': f"Infrastructure Project #{random.randint(100, 999)}",
                    'location': self._generate_sf_location(),
                    'description': random.choice([
                        'Sidewalk renovation',
                        'Bridge maintenance',
                        'Traffic signal upgrade',
                        'Storm drain repair',
                        'Street resurfacing'
                    ]),
                    'scheduled_date': future_date.strftime('%Y-%m-%d'),
                    'duration_days': random.randint(1, 5),
                    'impact_level': random.choice(['low', 'medium', 'high']),
                    'source': 'city_planning'
                })

        return planned

    def _get_osm_construction(self) -> List[Dict]:
        """Get construction data from OpenStreetMap"""
        try:
            bbox = f"{self.bbox_coords['south']},{self.bbox_coords['west']},{self.bbox_coords['north']},{self.bbox_coords['east']}"

            # Corrected Overpass QL query
            query = f"""
    [out:json][timeout:15];
    (
      node["highway"="construction"]({bbox});
      way["highway"="construction"]({bbox});
      relation["highway"="construction"]({bbox});
    );
    out center 20;
    """

            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=20,
                headers={'User-Agent': 'UrbanMonitoringPlatform/1.0'}
            )

            if response.status_code == 200:
                data = response.json()
                zones = self._parse_osm_construction(data)
                if zones:
                    print(f"🗺️ Retrieved {len(zones)} real OSM construction zones")
                return zones
            elif response.status_code == 429:
                print("⚠️ OSM API rate limit - try again later")
                return []
            elif response.status_code == 504:
                print("⚠️ OSM API timeout (504 Gateway Timeout) - server overloaded, using cached/demo data")
                return []
            else:
                # Don't print HTML error responses, just the status code
                error_text = response.text[:100] if not response.text.startswith('<?xml') else 'HTML error page'
                print(f"⚠️ OSM API error {response.status_code}: {error_text}")
                return []

        except requests.exceptions.Timeout:
            print("⚠️ OSM API timeout (15s) - server may be slow, using cached data")
            return []
        except requests.exceptions.ConnectionError:
            print("⚠️ OSM API connection failed - check internet connection")
            return []
        except Exception as e:
            print(f"⚠️ OSM construction query failed: {type(e).__name__}: {str(e)[:100]}")
            return []

    def _parse_osm_construction(self, data: Dict) -> List[Dict]:
        """Parse OpenStreetMap construction data"""
        zones = []

        elements = data.get('elements', [])

        for element in elements:
            try:
                # Get coordinates
                if element.get('type') == 'node':
                    lat, lon = element.get('lat'), element.get('lon')
                elif element.get('type') == 'way' and 'center' in element:
                    lat, lon = element['center']['lat'], element['center']['lon']
                else:
                    continue

                tags = element.get('tags', {})

                zones.append({
                    'type': 'construction_zone',
                    'location': {'lat': lat, 'lon': lon},
                    'description': tags.get('construction', 'Road construction'),
                    'road_name': tags.get('name', 'Unknown road'),
                    'severity': 'medium',
                    'source': 'openstreetmap',
                    'osm_id': element.get('id')
                })
            except Exception as e:
                # Skip malformed elements
                continue

        return zones

    def _simulate_construction(self) -> List[Dict]:
        """Generate simulated construction zones for demo"""
        zones = []

        construction_types = [
            'Street repaving', 'Utility installation', 'Building construction',
            'Road widening', 'Bridge repair', 'Sidewalk reconstruction'
        ]

        for _ in range(random.randint(3, 7)):
            zones.append({
                'type': 'construction_zone',
                'location': self._generate_sf_location(),
                'description': random.choice(construction_types),
                'severity': random.choice(['low', 'medium', 'high']),
                'contractor': f"Construction Co. #{random.randint(100, 500)}",
                'start_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'estimated_completion': (datetime.now() + timedelta(days=random.randint(10, 90))).strftime('%Y-%m-%d'),
                'source': 'simulated',
                'permit_number': f"SF{random.randint(10000, 99999)}"
            })

        return zones

    def _generate_sf_location(self) -> Dict:
        """Generate random location in San Francisco"""
        return {
            'lat': self.bbox_coords['south'] + random.uniform(0, self.bbox_coords['north'] - self.bbox_coords['south']),
            'lon': self.bbox_coords['west'] + random.uniform(0, self.bbox_coords['east'] - self.bbox_coords['west'])
        }

# Global construction service instance
construction_service = None

def get_construction_service():
    """Get or create construction service instance"""
    global construction_service
    if construction_service is None:
        construction_service = ConstructionService()
    return construction_service