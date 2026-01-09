# Smart Urban Platform Architecture

## Overview
The Smart Urban Platform is an integrated system designed to monitor urban hazards, provide safe routing for vehicles, and assist in emergency situations. It combines real-time AI computer vision, geospatial data analysis, and routing algorithms to enhance urban safety.

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Web Framework**: Flask (for API endpoints)
- **Real-time Communication**: Python `websockets` library
- **Database**: SQLite (`urban_monitoring.db`) for lightweight local storage.
- **AI/ML**:
    - **YOLOv8** (Ultralytics): For real-time object detection (vehicles, pedestrians, hazards).
    - **Transformers** (Hugging Face): For scene classification (e.g., detecting flooded streets).
    - **OpenCV**: For image processing (pothole detection, waterlogging analysis).

### Frontend
- **Language**: JavaScript (ES6+), HTML5, CSS3
- **Map Interface**: Leaflet.js / OpenStreetMap (presumed based on typical GraphHopper usage).
- **Client Communication**: Native WebSocket API for real-time updates.

### Services & Integrations
- **Routing Engine**: GraphHopper (running in Docker) for calculating routes and handling obstacle avoidance.
- **Emergency Data**: Overpass API (OpenStreetMap) to locate real-time emergency facilities (hospitals, police, fire stations).
- **Weather/Traffic**: Modular service integrations (`weather_service.py`, `traffic_service.py`).

## Architecture Components

### 1. Hazard Detection System
The core is the `SmartHazardDetector` (in `hazard_detector.py` and `backend/models/detector.py`).
- **Input**: Images/Video frames from urban cameras.
- **Processing**:
    - **YOLOv8**: Detects objects (cars, accidents, people).
    - **CV Algorithms**: Detects road anomalies like potholes (texture analysis) and waterlogging (color/reflection analysis).
- **Output**: JSON objects describing hazards with bounding boxes and severity.

### 2. Routing Service (`/route`)
Located in `backend/app.py`.
- Acts as a proxy to the GraphHopper instance.
- **Dynamic Avoidance**: Accepts `avoid_polygons` (geo-fenced hazard zones).
- **Mechanism**: When a user requests a route, the backend checks known hazards, creates polygons around them, and instructs GraphHopper to avoid these areas. GraphHopper's algorithms (like A* or Dijkstra with dynamic weights) penalize edges within these polygons, effectively routing around them.

### 3. Emergency SOS (`/sos`)
Located in `backend/app.py`.
- **Function**: Finds the nearest *real* emergency amenities.
- **Mechanism**:
    1.  User sends current coordinates.
    2.  Backend constructs an **Overpass QL** query.
    3.  Queries the public OpenStreetMap Overpass API.
    4.  Returns sorted list of hospitals, police stations, and fire brigades within a radius.

### 4. Real-time Updates (WebSockets)
- The backend runs a WebSocket server (`websocket_server.py`) that pushes new hazard alerts to connected frontend clients immediately, ensuring the dashboard map is always up-to-date.

## Workflow Example: Hazard Avoidance
1.  **Detection**: A camera detects a "major accident" at Location A.
2.  **Storage**: The system logs this hazard in the database and broadcasts it via WebSocket.
3.  **Routing**: A user requests a route from Location X to Y.
4.  **Optimization**: The backend retrieves the "major accident" at Location A, defines a buffer zone (polygon) around it.
5.  **Calculation**: This polygon is sent to GraphHopper as a "blocked area".
6.  **Result**: GraphHopper returns a route that physically circumvents Location A, which the frontend then renders.

