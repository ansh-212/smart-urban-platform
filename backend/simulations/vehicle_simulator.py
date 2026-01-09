"""
Vehicle Simulator for Real-time Traffic Monitoring
Simulates moving vehicles on the map
"""
import random
import time
import threading
from datetime import datetime
from typing import Dict, List
import math
import logging

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleSimulator:
    def __init__(self, socketio, event_manager):
        self.socketio = socketio
        self.event_manager = event_manager
        self.vehicles = {}
        self.running = False
        self.simulation_thread = None

        # SF Bay Area bounds
        self.bounds = {
            'lat_min': 37.70,
            'lat_max': 37.80,
            'lon_min': -122.50,
            'lon_max': -122.40
        }

        # Vehicle types with emojis
        self.vehicle_types = [
            {'type': 'car', 'icon': '🚗', 'speed': 0.0005, 'color': '#3498db'},
            {'type': 'bus', 'icon': '🚌', 'speed': 0.0003, 'color': '#e74c3c'},
            {'type': 'truck', 'icon': '🚛', 'speed': 0.0002, 'color': '#95a5a6'},
            {'type': 'emergency', 'icon': '🚑', 'speed': 0.0008, 'color': '#e67e22'},
            {'type': 'police', 'icon': '🚓', 'speed': 0.0006, 'color': '#2ecc71'}
        ]

        print("VehicleSimulator initialized")

    def generate_vehicle(self) -> Dict:
        """Generate a new random vehicle"""
        vehicle_type = random.choice(self.vehicle_types)
        vehicle_id = f"vehicle_{random.randint(1000, 9999)}"

        vehicle = {
            'id': vehicle_id,
            'type': vehicle_type['type'],
            'icon': vehicle_type['icon'],
            'color': vehicle_type['color'],
            'location': {
                'lat': random.uniform(self.bounds['lat_min'], self.bounds['lat_max']),
                'lon': random.uniform(self.bounds['lon_min'], self.bounds['lon_max'])
            },
            'speed': vehicle_type['speed'],
            'heading': random.randint(0, 360),
            'status': random.choice(['moving', 'stopped']),
            'created_at': datetime.now().isoformat()
        }

        print(f"Generated vehicle: {vehicle_id}")
        return vehicle

    def move_vehicle(self, vehicle: Dict):
        """Move vehicle in a realistic pattern"""
        if vehicle['status'] != 'moving':
            # Sometimes resume movement
            if random.random() < 0.3:
                vehicle['status'] = 'moving'
            return

        # Get current position
        lat = vehicle['location']['lat']
        lon = vehicle['location']['lon']
        speed = vehicle['speed']
        heading = vehicle['heading']

        # Calculate movement
        lat_change = speed * math.cos(math.radians(heading))
        lon_change = speed * math.sin(math.radians(heading))

        # Update position
        new_lat = lat + lat_change
        new_lon = lon + lon_change

        # Boundary checking with realistic bouncing
        if new_lat < self.bounds['lat_min'] or new_lat > self.bounds['lat_max']:
            vehicle['heading'] = (180 - vehicle['heading']) % 360
            new_lat = max(self.bounds['lat_min'], min(self.bounds['lat_max'], new_lat))

        if new_lon < self.bounds['lon_min'] or new_lon > self.bounds['lon_max']:
            vehicle['heading'] = (360 - vehicle['heading']) % 360
            new_lon = max(self.bounds['lon_min'], min(self.bounds['lon_max'], new_lon))

        # Update vehicle position
        vehicle['location']['lat'] = new_lat
        vehicle['location']['lon'] = new_lon

        # Random direction changes (like real traffic)
        if random.random() < 0.1:  # 10% chance to change direction
            vehicle['heading'] = (vehicle['heading'] + random.randint(-45, 45)) % 360

        # Random status changes
        if random.random() < 0.05:  # 5% chance to stop
            vehicle['status'] = random.choice(['moving', 'stopped'])

    def start_simulation(self, num_vehicles: int = 8):
        """Start the vehicle simulation"""
        if self.running:
            print("Simulation already running")
            return False

        try:
            self.running = True
            self.vehicles.clear()

            # Generate initial vehicles
            for i in range(num_vehicles):
                vehicle = self.generate_vehicle()
                self.vehicles[vehicle['id']] = vehicle

            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()

            print(f"🚗 Vehicle simulation started with {num_vehicles} vehicles")
            return True

        except Exception as e:
            print(f"Failed to start vehicle simulation: {e}")
            self.running = False
            return False

    def stop_simulation(self):
        """Stop the vehicle simulation"""
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2)

        self.vehicles.clear()
        print("🚗 Vehicle simulation stopped")

    def _simulation_loop(self):
        """Main simulation loop - runs in separate thread"""
        print("Starting vehicle simulation loop")

        while self.running:
            try:
                # Move all vehicles
                for vehicle in list(self.vehicles.values()):
                    if vehicle['status'] == 'moving':
                        self.move_vehicle(vehicle)

                # Broadcast vehicle updates via WebSocket
                if self.vehicles:
                    vehicle_data = {
                        'vehicles': list(self.vehicles.values()),
                        'total_count': len(self.vehicles)
                    }

                    # Use socketio directly for broadcasting
                    try:
                        self.socketio.emit('vehicle_update', {
                            'type': 'vehicle_update',
                            'data': vehicle_data,
                            'timestamp': datetime.now().isoformat()
                        })

                        print(f"Broadcasting {len(self.vehicles)} vehicle updates")
                    except Exception as e:
                        print(f"Error broadcasting vehicle updates: {e}")

                # Occasionally add/remove vehicles for realism
                if random.random() < 0.1:  # 10% chance every cycle
                    if len(self.vehicles) < 12 and random.random() < 0.7:
                        # Add vehicle (70% chance if under limit)
                        vehicle = self.generate_vehicle()
                        self.vehicles[vehicle['id']] = vehicle
                        print(f"Added new vehicle: {vehicle['id']}")
                    elif len(self.vehicles) > 4:
                        # Remove vehicle (if above minimum)
                        vehicle_id = random.choice(list(self.vehicles.keys()))
                        del self.vehicles[vehicle_id]
                        print(f"Removed vehicle: {vehicle_id}")

                # Sleep for update interval
                time.sleep(2)  # Update every 2 seconds

            except Exception as e:
                print(f"❌ Vehicle simulation error: {e}")
                time.sleep(1)

        print("Vehicle simulation loop ended")

    def get_vehicles(self) -> List[Dict]:
        """Get current vehicle list"""
        return list(self.vehicles.values())

    def is_running(self) -> bool:
        """Check if simulation is running"""
        return self.running

# Global simulator instance
vehicle_simulator = None

def get_vehicle_simulator():
    global vehicle_simulator
    return vehicle_simulator

def init_vehicle_simulator(socketio, event_manager):
    global vehicle_simulator
    vehicle_simulator = VehicleSimulator(socketio, event_manager)
    print("Vehicle simulator initialized")
    return vehicle_simulator