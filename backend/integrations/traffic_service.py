"""
Traffic Data Service
Simulates real traffic conditions and incidents
(In production, connect to traffic APIs like HERE, TomTom, or Google)
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List
import random

class TrafficService:
    def __init__(self):
        # San Francisco major roads
        self.major_roads = [
            {"name": "US-101 North", "type": "highway"},
            {"name": "US-101 South", "type": "highway"},
            {"name": "I-280 North", "type": "highway"},
            {"name": "I-280 South", "type": "highway"},
            {"name": "Bay Bridge", "type": "bridge"},
            {"name": "Golden Gate Bridge", "type": "bridge"},
            {"name": "Market Street", "type": "street"},
            {"name": "Van Ness Avenue", "type": "street"},
            {"name": "19th Avenue", "type": "street"},
            {"name": "Geary Boulevard", "type": "street"}
        ]

        self.city_bounds = {
            'lat_min': 37.7, 'lat_max': 37.8,
            'lon_min': -122.5, 'lon_max': -122.4
        }

        print("🚗 Traffic service initialized")

    def get_traffic_incidents(self) -> List[Dict]:
        """Get current traffic incidents"""
        incidents = []

        # Generate realistic traffic incidents
        incident_types = [
            {'type': 'accident', 'severity': 'high', 'duration': 60},
            {'type': 'stalled_vehicle', 'severity': 'medium', 'duration': 30},
            {'type': 'debris', 'severity': 'medium', 'duration': 20},
            {'type': 'road_work', 'severity': 'low', 'duration': 180},
            {'type': 'flooding', 'severity': 'high', 'duration': 120},
        ]

        # Generate 1-4 incidents
        for _ in range(random.randint(1, 4)):
            incident = random.choice(incident_types)
            road = random.choice(self.major_roads)

            incidents.append({
                'type': 'traffic_incident',
                'incident_type': incident['type'],
                'location': self._generate_road_location(),
                'road_name': road['name'],
                'severity': incident['severity'],
                'description': self._get_incident_description(incident['type']),
                'reported_time': datetime.now().isoformat(),
                'estimated_clearance': (datetime.now() + timedelta(minutes=incident['duration'])).isoformat(),
                'lanes_affected': random.randint(1, 3),
                'source': 'traffic_monitoring'
            })

        return incidents

    def get_traffic_flow(self) -> List[Dict]:
        """Get traffic flow conditions for major roads"""
        flow_data = []

        # Time-based traffic patterns
        current_hour = datetime.now().hour

        # Rush hour detection
        is_morning_rush = 7 <= current_hour <= 9
        is_evening_rush = 17 <= current_hour <= 19
        is_rush_hour = is_morning_rush or is_evening_rush

        for road in self.major_roads:
            # Base traffic level
            if is_rush_hour:
                base_congestion = random.uniform(0.6, 0.9)  # Heavy traffic
                avg_speed_reduction = random.uniform(0.3, 0.7)
            elif 22 <= current_hour or current_hour <= 6:
                base_congestion = random.uniform(0.1, 0.3)  # Light traffic
                avg_speed_reduction = random.uniform(0.0, 0.2)
            else:
                base_congestion = random.uniform(0.3, 0.6)  # Normal traffic
                avg_speed_reduction = random.uniform(0.2, 0.4)

            # Road type affects capacity
            if road['type'] == 'highway':
                speed_limit = random.randint(55, 75)
                capacity = random.randint(4, 8)  # lanes
            elif road['type'] == 'bridge':
                speed_limit = random.randint(45, 55)
                capacity = random.randint(4, 6)
            else:  # street
                speed_limit = random.randint(25, 35)
                capacity = random.randint(2, 4)

            current_speed = speed_limit * (1 - avg_speed_reduction)

            flow_data.append({
                'road_name': road['name'],
                'road_type': road['type'],
                'speed_limit': speed_limit,
                'current_speed': round(current_speed, 1),
                'congestion_level': round(base_congestion, 2),
                'volume': round(base_congestion * capacity * 1000),  # vehicles per hour
                'travel_time_index': round(1 + avg_speed_reduction, 2),  # 1.0 = normal
                'status': self._get_traffic_status(base_congestion),
                'last_updated': datetime.now().isoformat()
            })

        return flow_data

    def get_congestion_hotspots(self) -> List[Dict]:
        """Identify current traffic congestion hotspots"""
        hotspots = []

        # Generate 2-5 congestion areas
        for _ in range(random.randint(2, 5)):
            congestion_level = random.uniform(0.6, 1.0)  # High congestion only

            hotspots.append({
                'type': 'congestion_hotspot',
                'location': self._generate_road_location(),
                'congestion_level': round(congestion_level, 2),
                'affected_area': f"{random.randint(500, 2000)} meters",
                'cause': random.choice([
                    'Heavy volume', 'Signal timing', 'Lane reduction',
                    'Incident nearby', 'Event traffic', 'Construction'
                ]),
                'delay_minutes': random.randint(5, 25),
                'severity': 'high' if congestion_level > 0.8 else 'medium',
                'detected_time': datetime.now().isoformat()
            })

        return hotspots

    def get_predicted_traffic(self, hours_ahead: int = 3) -> List[Dict]:
        """Predict traffic conditions for next few hours"""
        predictions = []

        for hour in range(1, hours_ahead + 1):
            future_time = datetime.now() + timedelta(hours=hour)
            future_hour = future_time.hour

            # Predict based on typical patterns
            if 7 <= future_hour <= 9 or 17 <= future_hour <= 19:
                predicted_level = 'heavy'
                congestion = random.uniform(0.7, 0.9)
            elif 22 <= future_hour or future_hour <= 6:
                predicted_level = 'light'
                congestion = random.uniform(0.1, 0.3)
            else:
                predicted_level = 'moderate'
                congestion = random.uniform(0.4, 0.6)

            predictions.append({
                'time': future_time.strftime('%Y-%m-%d %H:00'),
                'predicted_level': predicted_level,
                'congestion_score': round(congestion, 2),
                'confidence': round(random.uniform(0.75, 0.95), 2),
                'peak_areas': [
                    random.choice(self.major_roads)['name']
                    for _ in range(random.randint(2, 4))
                ]
            })

        return predictions

    def _generate_road_location(self) -> Dict:
        """Generate location along a road"""
        return {
            'lat': random.uniform(self.city_bounds['lat_min'], self.city_bounds['lat_max']),
            'lon': random.uniform(self.city_bounds['lon_min'], self.city_bounds['lon_max'])
        }

    def _get_incident_description(self, incident_type: str) -> str:
        """Get description for incident type"""
        descriptions = {
            'accident': 'Multi-vehicle accident blocking lanes',
            'stalled_vehicle': 'Vehicle breakdown in travel lane',
            'debris': 'Debris in roadway requiring cleanup',
            'road_work': 'Active construction affecting traffic flow',
            'flooding': 'Water over roadway due to heavy rain'
        }
        return descriptions.get(incident_type, 'Traffic incident reported')

    def _get_traffic_status(self, congestion_level: float) -> str:
        """Convert congestion level to status"""
        if congestion_level < 0.3:
            return 'free_flow'
        elif congestion_level < 0.6:
            return 'moderate'
        elif congestion_level < 0.8:
            return 'congested'
        else:
            return 'severely_congested'

# Global traffic service instance
traffic_service = None

def get_traffic_service():
    """Get or create traffic service instance"""
    global traffic_service
    if traffic_service is None:
        traffic_service = TrafficService()
    return traffic_service
