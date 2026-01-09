"""
Emergency Facilities Service
Uses OSM Overpass API to fetch real emergency facilities (hospitals, police, fire stations)
"""
import requests
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class EmergencyService:
    def __init__(self):
        # OSM Overpass API (public, no key required)
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        print("🚨 Emergency Facilities Service initialized (using free Overpass API)")

    def get_emergency_facilities(self, lat: float, lon: float, radius: int = 5000) -> Dict[str, List[Dict]]:
        """
        Get emergency facilities from OpenStreetMap
        Returns hospitals, police stations, and fire stations

        Args:
            lat: Latitude of the center point
            lon: Longitude of the center point
            radius: Search radius in meters (default 5km)

        Returns:
            Dictionary with 'hospitals', 'police', and 'fire_stations' lists
        """
        try:
            # Calculate bounding box
            radius_deg = radius / 111000  # 1 degree ≈ 111km

            bbox = {
                'south': lat - radius_deg,
                'north': lat + radius_deg,
                'west': lon - radius_deg,
                'east': lon + radius_deg
            }

            # Overpass QL query for emergency facilities
            query = f"""
[out:json][timeout:25];
(
  // Hospitals
  node["amenity"="hospital"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="hospital"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  node["amenity"="clinic"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="clinic"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  
  // Police Stations
  node["amenity"="police"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="police"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  
  // Fire Stations
  node["amenity"="fire_station"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["amenity"="fire_station"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out center 50;
"""

            logger.info(f"Fetching emergency facilities near ({lat}, {lon}) within {radius}m")

            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=30,
                headers={'User-Agent': 'Urban-Monitoring-Platform/1.0'}
            )

            if response.status_code != 200:
                logger.error(f"Overpass API error: {response.status_code}")
                return self._get_empty_facilities()

            data = response.json()
            elements = data.get('elements', [])

            # Categorize facilities
            hospitals = []
            police = []
            fire_stations = []

            for element in elements:
                amenity_type = element.get('tags', {}).get('amenity')

                # Get coordinates
                if 'lat' in element and 'lon' in element:
                    facility_lat = element['lat']
                    facility_lon = element['lon']
                elif 'center' in element:
                    facility_lat = element['center']['lat']
                    facility_lon = element['center']['lon']
                else:
                    continue

                # Calculate distance from center point
                distance = self._calculate_distance(lat, lon, facility_lat, facility_lon)

                facility_info = {
                    'name': element.get('tags', {}).get('name', f'{amenity_type.replace("_", " ").title()}'),
                    'location': {'lat': facility_lat, 'lon': facility_lon},
                    'distance': round(distance, 2),  # in km
                    'address': element.get('tags', {}).get('addr:street', 'Address not available'),
                    'phone': element.get('tags', {}).get('phone', 'N/A'),
                    'emergency': element.get('tags', {}).get('emergency', 'yes')
                }

                # Categorize by type
                if amenity_type in ['hospital', 'clinic']:
                    facility_info['type'] = 'hospital'
                    facility_info['emergency_room'] = element.get('tags', {}).get('emergency', 'yes')
                    hospitals.append(facility_info)
                elif amenity_type == 'police':
                    facility_info['type'] = 'police'
                    police.append(facility_info)
                elif amenity_type == 'fire_station':
                    facility_info['type'] = 'fire_station'
                    fire_stations.append(facility_info)

            # Sort by distance
            hospitals.sort(key=lambda x: x['distance'])
            police.sort(key=lambda x: x['distance'])
            fire_stations.sort(key=lambda x: x['distance'])

            logger.info(f"Found {len(hospitals)} hospitals, {len(police)} police stations, {len(fire_stations)} fire stations")

            return {
                'hospitals': hospitals[:5],  # Return top 5 closest
                'police': police[:5],
                'fire_stations': fire_stations[:5]
            }

        except requests.Timeout:
            logger.error("Overpass API timeout")
            return self._get_empty_facilities()
        except Exception as e:
            logger.error(f"Error fetching emergency facilities: {e}")
            return self._get_empty_facilities()

    def get_nearest_facilities(self, lat: float, lon: float, limit: int = 3) -> List[Dict]:
        """
        Get the nearest emergency facilities of all types combined

        Returns a mixed list of the closest facilities regardless of type
        """
        all_facilities_dict = self.get_emergency_facilities(lat, lon)

        # Combine all facilities into one list
        all_facilities = []
        all_facilities.extend(all_facilities_dict['hospitals'])
        all_facilities.extend(all_facilities_dict['police'])
        all_facilities.extend(all_facilities_dict['fire_stations'])

        # Sort by distance
        all_facilities.sort(key=lambda x: x['distance'])

        return all_facilities[:limit]

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        Returns distance in kilometers
        """
        from math import radians, cos, sin, asin, sqrt

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers
        r = 6371

        return c * r

    def _get_empty_facilities(self) -> Dict[str, List]:
        """Return empty facilities structure"""
        return {
            'hospitals': [],
            'police': [],
            'fire_stations': []
        }

