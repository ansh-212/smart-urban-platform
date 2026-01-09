# Smart Urban Platform

A comprehensive urban safety system that leverages AI, Computer Vision, and Geospatial data to detect hazards in real-time, optimize vehicle routing around obstacles, and provide emergency assistance.

## 🚀 Features

*   **Real-time Hazard Detection**: Uses YOLOv8 and computer vision to detect accidents, potholes, waterlogging, and traffic congestion from camera feeds.
*   **Smart Routing**: Integrates with GraphHopper to calculate safe routes that automatically avoid detected hazard zones.
*   **Emergency SOS**: Locates the nearest *real* operational hospitals, police stations, and fire brigades using OpenStreetMap data.
*   **Live Dashboard**: Interactive web frontend displaying real-time alerts and map updates.

## 🛠️ Architecture

For a deep dive into the system design, tech stack, and components, please read [ARCHITECTURE.md](./ARCHITECTURE.md).

## 📋 Prerequisites

*   **Python 3.11+**
*   **GraphHopper**: A running instance of GraphHopper (locally or via Docker) for routing.
*   **Node.js** (Optional, if you expand the frontend).

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ansh-212/smart-urban-platform.git
    cd smart-urban-platform
    ```

2.  **Set up Python Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up GraphHopper (Required for Routing)**
    Build and run GraphHopper using Docker or the Java jar.
    ```bash
    cd routing_service
    docker-compose up -d
    ```
    *Note: Ensure GraphHopper is accessible at `http://localhost:8989`.*

## 🏃‍♂️ Usage

### 1. Start the Backend
The backend handles API requests, hazard processing, and WebSocket events.

```bash
cd backend
python app.py
```
*The server will start on `http://localhost:5000`.*

### 2. Launch the Frontend
Open `frontend/index.html` in your web browser. You can use a simple HTTP server for better experience:

```bash
cd frontend
python -m http.server 8000
```
Then visit `http://localhost:8000`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License.

