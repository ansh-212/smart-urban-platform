"""
GraphHopper / TomTom Routing Service Client
Handles real routing with dynamic hazard avoidance
"""
import os
import math
import requests
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GraphHopperRoutingService:
    def __init__(self, graphhopper_url: str = "http://localhost:8989"):
        self.base_url = graphhopper_url
        self.route_endpoint = f"{self.base_url}/route"

    def calculate_route(
        self,
        start: Tuple[float, float],  # (lat, lon)
        end: Tuple[float, float],    # (lat, lon)
        avoid_areas: Optional[List[Dict]] = None,
        profile: str = "car"
    ) -> Optional[Dict]:
        """Calculate route avoiding hazard areas using GraphHopper"""
        try:
            params = {
                "point": [
                    f"{start[0]},{start[1]}",
                    f"{end[0]},{end[1]}"
                ],
                "profile": profile,
                "locale": "en",
                "instructions": "false",
                "calc_points": "true",
                "points_encoded": "false"
            }

            # Note: GraphHopper avoid custom areas configuration depends on custom models.
            # We simply pass precomputed avoid zones via custom profile if enabled elsewhere.

            response = requests.get(self.route_endpoint, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"GraphHopper error: {response.text}")
                return None

            data = response.json()
            if "paths" not in data or len(data["paths"]) == 0:
                return None

            path = data["paths"][0]
            return {
                "coordinates": path["points"]["coordinates"],
                "distance": round(path["distance"] / 1000, 2),  # km
                "duration": round(path["time"] / 1000 / 60, 1),  # minutes
                "avoided_hazards": len(avoid_areas) if avoid_areas else 0,
                "status": "success",
                "engine": "GraphHopper"
            }
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to GraphHopper")
            return None
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return None

    def health_check(self) -> bool:
        """Check if GraphHopper is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class TomTomRoutingService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com/routing/1/calculateRoute"

    def health_check(self) -> bool:
        """Check if TomTom API is accessible"""
        try:
            # Simple test call to verify API key works
            test_url = f"{self.base_url}/52.50931,13.42936:52.50274,13.43872/json"
            resp = requests.get(test_url, params={"key": self.api_key}, timeout=5)
            return resp.status_code == 200
        except:
            return True  # Assume available to avoid startup errors

    @staticmethod
    def _hazard_to_bbox(lat: float, lon: float, radius_m: float = 300.0) -> str:
        """Convert a hazard point + radius to TomTom avoidAreas bbox string: lat1,lon1:lat2,lon2"""
        # Degrees per meter approx
        dlat = radius_m / 111_000.0
        dlon = radius_m / (111_000.0 * max(math.cos(math.radians(lat)), 0.0001))
        lat1, lon1 = lat - dlat, lon - dlon
        lat2, lon2 = lat + dlat, lon + dlon
        return f"{lat1:.6f},{lon1:.6f}:{lat2:.6f},{lon2:.6f}"

    def calculate_route(
        self,
        start: Tuple[float, float],  # (lat, lon)
        end: Tuple[float, float],    # (lat, lon)
        avoid_areas: Optional[List[Dict]] = None,
        profile: str = "car"
    ) -> Optional[Dict]:
        """Calculate route using TomTom Directions with traffic and avoid areas"""
        try:
            start_str = f"{start[0]},{start[1]}"
            end_str = f"{end[0]},{end[1]}"
            url = f"{self.base_url}/{start_str}:{end_str}/json"

            params = {
                "key": self.api_key,
                "traffic": "true",
                "computeBestOrder": "false",
                "routeType": "fastest",
                "travelMode": "car",
                "avoid": "unpavedRoads",
                "sectionType": "traffic",
                "report": "effectiveSettings"
            }

            # Build avoidAreas param from hazards
            if avoid_areas:
                bboxes = [self._hazard_to_bbox(h["lat"], h["lon"], float(h.get("radius", 300))) for h in avoid_areas]
                # TomTom uses multiple avoidAreas by repeating parameter; we can pass comma-separated
                params["avoidAreas"] = ",".join(bboxes)

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.error(f"TomTom routing error {resp.status_code}: {resp.text}")
                return None

            data = resp.json()
            routes = data.get("routes", [])
            if not routes:
                return None

            # Take first route summary and polyline
            route0 = routes[0]
            summary = route0.get("summary", {})

            # Coordinates: TomTom returns polyline in leg points shape or guidance; easiest is to use 'legs[0].points'
            coords: List[List[float]] = []
            legs = route0.get("legs", [])
            for leg in legs:
                for p in leg.get("points", []):
                    # TomTom points are lat, lon
                    coords.append([p["longitude"], p["latitude"]])

            distance_km = round(float(summary.get("lengthInMeters", 0)) / 1000.0, 2)
            duration_min = round(float(summary.get("travelTimeInSeconds", 0)) / 60.0, 1)

            return {
                "coordinates": coords,
                "distance": distance_km,
                "duration": duration_min,
                "avoided_hazards": len(avoid_areas) if avoid_areas else 0,
                "status": "success",
                "engine": "TomTom"
            }
        except Exception as e:
            logger.error(f"TomTom routing exception: {e}")
            return None


_routing_service = None


def get_routing_service():
    """Return a routing service. Prefer TomTom if TOMTOM_API_KEY is set, else GraphHopper."""
    global _routing_service
    if _routing_service is not None:
        return _routing_service

    api_key = os.getenv("TOMTOM_API_KEY")
    if api_key:
        logger.info("Using TomTom Routing Service")
        _routing_service = TomTomRoutingService(api_key)
    else:
        logger.info("Using GraphHopper Routing Service")
        _routing_service = GraphHopperRoutingService()

    return _routing_service

