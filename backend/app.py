import os
import json
from flask import Flask, request, jsonify
import requests
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)

GRAPHOPPER_URL = os.environ.get("GRAPHOPPER_URL", "http://localhost:8989")
GRAPHOPPER_KEY = os.environ.get("GRAPHOPPER_KEY", None)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def haversine_distance(lat1, lon1, lat2, lon2):
    # returns distance in meters
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/route", methods=["POST"])
def route():
    """
    Expects JSON:
    {
      "points": [[lat, lon], [lat, lon], ...],
      "profile": "car",                # optional
      "avoid_polygons": [<geojson polygon objects>]  # optional - forwarded to GraphHopper as 'avoid_polygon'
    }
    Forwards request to GraphHopper routing endpoint and returns GraphHopper response.
    """
    try:
        body = request.get_json(force=True)
    except:
        return jsonify({"error": "invalid json"}), 400

    points = body.get("points") or []
    profile = body.get("profile", "car")
    avoid_polygons = body.get("avoid_polygons")

    if not points or len(points) < 2:
        return jsonify({"error": "need at least two points"}), 400

    params = []
    for p in points:
        lat, lon = p
        params.append(("point", f"{lat},{lon}"))
    params.append(("profile", profile))
    params.append(("calc_points", "true"))

    if GRAPHOPPER_KEY:
        params.append(("key", GRAPHOPPER_KEY))

    # If avoid_polygons provided, forward as avoid_polygon param(s).
    if avoid_polygons:
        for poly in avoid_polygons:
            params.append(("avoid_polygon", json.dumps(poly)))

    try:
        gh_resp = requests.get(f"{GRAPHOPPER_URL}/route", params=params, timeout=15)
        gh_resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": "graphhopper request failed", "detail": str(e)}), 502

    return (gh_resp.content, gh_resp.status_code, {"Content-Type": gh_resp.headers.get("Content-Type", "application/json")})

@app.route("/sos", methods=["POST"])
def sos():
    """
    Expects JSON:
    {
      "lat": 12.34,
      "lon": 56.78,
      "radius": 2000   # meters, optional (default 2000)
    }
    Returns list of nearby emergency places (hospital, police, fire_station) with type and coordinates.
    """
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"error": "invalid json"}), 400

    lat = data.get("lat")
    lon = data.get("lon")
    radius = int(data.get("radius", 2000))

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon required"}), 400

    # Overpass QL: search for amenity=hospital|police|fire_station within radius
    q = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})[amenity=hospital];
      way(around:{radius},{lat},{lon})[amenity=hospital];
      relation(around:{radius},{lat},{lon})[amenity=hospital];
      node(around:{radius},{lat},{lon})[amenity=police];
      way(around:{radius},{lat},{lon})[amenity=police];
      relation(around:{radius},{lat},{lon})[amenity=police];
      node(around:{radius},{lat},{lon})[amenity=fire_station];
      way(around:{radius},{lat},{lon})[amenity=fire_station];
      relation(around:{radius},{lat},{lon})[amenity=fire_station];
    );
    out center;
    """

    try:
        resp = requests.post(OVERPASS_URL, data={"data": q}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return jsonify({"error": "overpass query failed", "detail": str(e)}), 502

    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        amenity = tags.get("amenity")
        name = tags.get("name", "unknown")

        # For nodes use lat/lon, for ways/relations Overpass returns a 'center'
        if el.get("type") == "node":
            el_lat = el.get("lat")
            el_lon = el.get("lon")
        else:
            center = el.get("center") or {}
            el_lat = center.get("lat")
            el_lon = center.get("lon")

        if el_lat is None or el_lon is None:
            continue

        dist = haversine_distance(lat, lon, el_lat, el_lon)

        results.append({
            "id": el.get("id"),
            "name": name,
            "type": amenity,
            "lat": el_lat,
            "lon": el_lon,
            "distance_m": int(dist)
        })

    # sort by distance
    results.sort(key=lambda x: x["distance_m"])
    return jsonify({"places": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

